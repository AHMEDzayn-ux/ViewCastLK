"""Export snapshot partitions to Parquet nightly; drop them only under pressure.

video_snapshots is partitioned by day. The Supabase free tier stops at 500 MB,
so old partitions eventually have to go — but there is no reason to give up
queryable history while there is room. Exporting and dropping are therefore two
separate decisions:

  EXPORT  every completed day, every night, unconditionally. Data reaches
          Google Drive as Parquet long before anything is removed from
          Postgres, so a drop is never the first time a partition has been
          copied anywhere.

  DROP    only when the database exceeds --drop-above-mb, and then only the
          oldest partitions, only ones verified present on the remote, only
          ones older than the --retain-days floor, and only until the database
          is back under --drop-target-mb.

Nothing is thinned: every snapshot the collector took is preserved at full
fidelity in the exports. Postgres holds the working set; the Parquet files are
the authoritative long-term dataset.

Labels are materialised before any drop, so the observation nearest each
horizon is captured while its partition still exists. Training reads
video_horizon_labels and never the raw partitions.

Usage:
    python scripts/archive_snapshots.py                       # export only, at 108 MB
    python scripts/archive_snapshots.py --drop-above-mb 380   # export, drop if over
    python scripts/archive_snapshots.py --dry-run
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import psycopg2

# pandas warns on every psycopg2 connection because it prefers SQLAlchemy; the
# queries here are plain SELECTs and the warning would drown the run log.
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import connect, ensure_snapshot_partitions  # noqa: E402

PARTITION_RE = re.compile(r"^video_snapshots_(\d{8})$")
REMOTE = os.environ.get("ARCHIVE_REMOTE", "gdrive:ViewCastLK/archives")
HORIZONS = (7, 14, 21, 30)

# How far back label extraction looks. Wide enough to absorb several missed
# runs, narrow enough that the statement stays well inside the timeout.
LABEL_LOOKBACK_DAYS = int(os.environ.get("LABEL_LOOKBACK_DAYS", "4"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def db_size_mb(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_database_size(current_database())")
        return cur.fetchone()[0] / 1e6


def partitions(conn):
    """Every daily partition with its day and row count, oldest first."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.relname
              FROM pg_class c
              JOIN pg_inherits i ON i.inhrelid = c.oid
             WHERE i.inhparent = 'video_snapshots'::regclass
             ORDER BY c.relname
        """)
        names = [r[0] for r in cur.fetchall()]
    out = []
    for n in names:
        m = PARTITION_RE.match(n)
        if not m:
            continue                                   # the default partition
        with conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM public."{n}"')
            rows = cur.fetchone()[0]
        out.append((n, datetime.strptime(m.group(1), "%Y%m%d").date(), rows))
    return out


def check_default(conn):
    """A non-empty default partition means partition creation fell behind. It
    also blocks that day's real partition from being created, so this is loud
    on purpose."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM video_snapshots_default")
        n = cur.fetchone()[0]
    if n:
        print(f"WARNING: default partition holds {n:,} rows. Partition creation "
              f"has fallen behind; those days cannot get a real partition until "
              f"the rows are moved out.")
    return n


def extract_labels(conn, lookback_days=LABEL_LOOKBACK_DAYS):
    """Materialise the observation nearest each horizon, for every video.

    Runs every night, so a label is recorded long before its partition becomes
    a drop candidate. The conflict clause keeps whichever observation is
    closest to the mark, so re-running can only improve a label, never degrade
    one.

    Only recent snapshots are considered. Scanning the whole table re-derives
    labels that earlier runs already settled, and the cost grows with the
    archive: at 1.9 million rows it exceeded the two-minute statement timeout
    and took the whole job down with it -- including the export and drop phases
    that had not run yet, so nothing was freed at exactly the moment space was
    short. A snapshot can only become the closest observation to a horizon on
    the run that records it, so a window a few days wide covers every new label
    plus several missed runs.
    """
    with conn.cursor() as cur:
        # Belt and braces: this statement is far smaller now, but a night that
        # follows a long outage still has more to do than usual.
        cur.execute("SET LOCAL statement_timeout = '10min'")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_horizon_labels (
                video_id      text NOT NULL REFERENCES videos(video_id),
                horizon_days  int  NOT NULL,
                captured_at   timestamptz NOT NULL,
                hours_off     numeric NOT NULL,
                view_count    bigint,
                like_count    bigint,
                comment_count bigint,
                PRIMARY KEY (video_id, horizon_days)
            )""")
        cur.execute("""
            INSERT INTO video_horizon_labels
                (video_id, horizon_days, captured_at, hours_off,
                 view_count, like_count, comment_count)
            SELECT DISTINCT ON (s.video_id, h.horizon)
                   s.video_id, h.horizon, s.captured_at,
                   EXTRACT(EPOCH FROM (s.captured_at - v.published_at))/3600.0
                       - h.horizon * 24,
                   s.view_count, s.like_count, s.comment_count
              FROM video_snapshots s
              JOIN videos v USING (video_id)
        CROSS JOIN (VALUES (7), (14), (21), (30)) AS h(horizon)
             WHERE s.captured_at >= now() - make_interval(days => %s)
               AND abs(EXTRACT(EPOCH FROM (s.captured_at - v.published_at))/3600.0
                       - h.horizon * 24) <= 12
             ORDER BY s.video_id, h.horizon,
                      abs(EXTRACT(EPOCH FROM (s.captured_at - v.published_at))/3600.0
                          - h.horizon * 24)
                ON CONFLICT (video_id, horizon_days) DO UPDATE
                   SET captured_at   = EXCLUDED.captured_at,
                       hours_off     = EXCLUDED.hours_off,
                       view_count    = EXCLUDED.view_count,
                       like_count    = EXCLUDED.like_count,
                       comment_count = EXCLUDED.comment_count
                 WHERE abs(EXCLUDED.hours_off) < abs(video_horizon_labels.hours_off)
        """, (lookback_days,))
        n = cur.rowcount
    conn.commit()
    return n


class _NoRclone:
    """Stands in for a CompletedProcess when the binary is not installed, so
    callers passing check=False see an ordinary failure rather than an
    exception from the process launcher."""
    returncode = 127
    stdout = ""
    stderr = "rclone is not installed or not on PATH"


def rclone(*args, check=True):
    try:
        r = subprocess.run(["rclone", *args], capture_output=True, text=True)
    except (FileNotFoundError, OSError) as e:
        if check:
            raise RuntimeError(f"rclone could not be run: {e}") from e
        return _NoRclone()
    if check and r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
    return r


def remote_size(dest, filename):
    """Byte size of a file on the remote, or None if it is not there."""
    r = rclone("lsjson", f"{dest}/{filename}", check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)[0]["Size"]
    except (ValueError, IndexError, KeyError):
        return None


def report_remote_usage():
    """How much the archive occupies, and how much room is left on the remote.

    Worth logging every run rather than checking by hand: the archive is the
    authoritative dataset, so it filling up is a quieter failure than the
    database filling up -- collection would keep working while nothing new
    could be preserved.
    """
    base = REMOTE.split(":", 1)[0]
    size = rclone("size", REMOTE, "--json", check=False)
    if size.returncode == 0 and size.stdout.strip():
        try:
            d = json.loads(size.stdout)
            print(f"archive on {base}: {d['bytes']/1e6:,.0f} MB "
                  f"across {d['count']:,} files")
        except (ValueError, KeyError):
            pass
    about = rclone("about", f"{base}:", "--json", check=False)
    if about.returncode == 0 and about.stdout.strip():
        try:
            d = json.loads(about.stdout)
            used, total = d.get("used"), d.get("total")
            if used is not None and total:
                print(f"{base} storage: {used/1e9:.2f} GB of {total/1e9:.2f} GB "
                      f"used ({100*used/total:.1f}%), {d.get('free', 0)/1e9:.2f} GB free")
        except (ValueError, KeyError):
            pass


def check_remote():
    """Prove the remote is reachable and authenticated, without writing.

    Runs on dry runs too. Everything else a dry run exercises — the database,
    the Parquet export — is local and reliable; the credential and OAuth path
    to Drive is the part that actually breaks, so a rehearsal that skipped it
    would give exactly the wrong kind of confidence.
    """
    remote = REMOTE.split(":", 1)[0]
    r = rclone("lsd", f"{remote}:", check=False)
    if r.returncode != 0:
        print(f"REMOTE UNREACHABLE: rclone cannot list '{remote}:' — "
              f"{r.stderr.strip()[:200]}")
        return False
    print(f"remote '{remote}:' reachable")
    return True


def export_partition(conn, name, day, rows, work, dry_run):
    """Write one partition to Parquet and upload it.

    Returns (size, sha256) on success or None if skipped. The checksum is
    returned rather than only logged because it is the only thing that can
    later prove a downloaded archive is intact, and a run log is not a durable
    record. Already-uploaded partitions are left alone, which makes re-runs
    cheap and the job safe to retry."""
    dest = f"{REMOTE}/{day:%Y/%m}"
    fname = f"{name}.parquet"

    if not dry_run:
        already = remote_size(dest, fname)
        if already is not None:
            return already, None          # already recorded by an earlier run

    df = pd.read_sql_query(f'SELECT * FROM public."{name}"', conn)
    if len(df) != rows:
        print(f"  {name}: SKIPPED — read {len(df):,} rows, expected {rows:,}")
        return None

    local = os.path.join(work, fname)
    df.to_parquet(local, index=False, compression="zstd")
    size = os.path.getsize(local)

    if len(pd.read_parquet(local)) != rows:
        print(f"  {name}: SKIPPED — parquet re-read row count disagrees")
        return None

    if dry_run:
        print(f"  {name}: would upload {size/1e6:.2f} MB ({rows:,} rows) -> {dest}")
        return None

    digest = sha256(local)
    rclone("copy", local, dest, "--checksum", "--retries", "3")
    confirmed = remote_size(dest, fname)
    if confirmed != size:
        print(f"  {name}: upload UNVERIFIED (local {size}, remote {confirmed})")
        return None
    print(f"  {name}: exported {rows:,} rows, {size/1e6:.2f} MB, sha {digest[:12]}")
    return confirmed, digest


def main():
    ap = argparse.ArgumentParser()
    # Three days, down from five on 16 August. At five the job could only ever
    # free one partition a night while collection added a slightly larger one,
    # so the database climbed to 479 MB of the 500 MB tier -- about two days
    # from failing writes. Nothing is lost by holding less: every partition is
    # on Drive before it can be dropped, and the day 7/14/21/30 labels the
    # model trains on live in video_horizon_labels, which is never dropped.
    # The cost is tolerance for consecutive failed runs, which this buys back
    # by keeping the database far enough below the ceiling to have room to
    # recover in.
    ap.add_argument("--retain-days", type=int, default=3,
                    help="never drop a partition newer than this, whatever the size")
    ap.add_argument("--drop-above-mb", type=float, default=380.0,
                    help="only start dropping once the database exceeds this")
    ap.add_argument("--drop-target-mb", type=float, default=300.0,
                    help="stop dropping once the database is back under this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    if not check_remote():
        print("aborting: nothing is exported and nothing is dropped when the "
              "archive destination cannot be verified")
        sys.exit(1)

    conn = connect()
    made = ensure_snapshot_partitions(14)
    if made:
        print(f"created {made} upcoming partition(s)")
    check_default(conn)

    size_before = db_size_mb(conn)
    print(f"database: {size_before:.0f} MB "
          f"(drop threshold {args.drop_above_mb:.0f} MB)")

    # A label failure must not take the whole job down. Exporting is always
    # safe -- it only copies -- and blocking it achieved nothing except leaving
    # the database full. Dropping is different: a partition must not be removed
    # before the observations it holds have been materialised, so that stays
    # gated on labels having succeeded.
    labels_ok = True
    if args.dry_run:
        print("dry run: skipping label extraction")
    else:
        try:
            print(f"horizon labels materialised/updated: {extract_labels(conn):,}")
        except psycopg2.Error as e:
            conn.rollback()
            labels_ok = False
            print(f"WARNING: label extraction failed ({type(e).__name__}: "
                  f"{str(e).strip()[:120]}). Exporting anyway; dropping is "
                  f"skipped this run because a partition must not be removed "
                  f"before its observations are materialised.")

    work = args.workdir or tempfile.mkdtemp(prefix="vcl-archive-")
    os.makedirs(work, exist_ok=True)

    # ------------------------------------------------------------- export
    today = date.today()
    floor = today - timedelta(days=args.retain_days)
    complete = [(n, d, r) for n, d, r in partitions(conn) if d < today]
    empty = [(n, d) for n, d, r in complete if r == 0]
    with_rows = [(n, d, r) for n, d, r in complete if r > 0]

    print(f"\nexporting {len(with_rows)} completed partition(s) "
          f"({len(empty)} empty, nothing to export):")
    exported, newly = {}, []
    for name, day, rows in with_rows:
        got = export_partition(conn, name, day, rows, work, args.dry_run)
        if got is not None:
            size, digest = got
            exported[name] = (day, rows, size)
            if digest:
                newly.append(dict(partition=name, day=day.isoformat(), rows=rows,
                                  bytes=size, sha256=digest,
                                  exported_at=datetime.now(timezone.utc).isoformat()))
    print(f"  {len(exported)} partition(s) safely on the remote"
          f"{f', {len(newly)} newly uploaded' if newly else ' (all already there)'}")

    # One manifest per run that uploaded anything. This is the only durable
    # record of each file's checksum -- the run log is not one -- and it is what
    # verify_archives.py checks a downloaded copy against.
    if newly and not args.dry_run:
        mpath = os.path.join(work, f"exported_{today:%Y%m%d}.json")
        pd.DataFrame(newly).to_json(mpath, orient="records", indent=1)
        rclone("copy", mpath, f"{REMOTE}/manifests")
        print(f"  manifest for {len(newly)} file(s) uploaded")

    # An empty partition holds nothing to preserve, so it is dropped on age
    # alone rather than waiting for space pressure. Left alone they accumulate
    # one per day the collector produced no rows, and a growing partition count
    # costs query planning time for no benefit.
    stale_empty = [(n, d) for n, d in empty if d < floor]
    for name, day in stale_empty:
        if args.dry_run:
            print(f"  {name}: empty and older than the floor, would drop")
            continue
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE public."{name}"')
        conn.commit()
        print(f"  {name}: empty and older than the floor, dropped")

    # --------------------------------------------------------------- drop
    if not labels_ok:
        print("\nno drops: label extraction failed, so it cannot be shown that "
              "every partition's observations are safely materialised.")
        report_remote_usage()
        conn.close()
        return

    if size_before < args.drop_above_mb:
        print(f"\nno drops: {size_before:.0f} MB is below the "
              f"{args.drop_above_mb:.0f} MB threshold. History stays queryable "
              f"in Postgres until space actually runs short.")
        report_remote_usage()
        conn.close()
        return

    candidates = [(n, d) for n, (d, _, _) in sorted(exported.items(), key=lambda kv: kv[1][0])
                  if d < floor]
    print(f"\nover threshold — dropping oldest exported partitions until under "
          f"{args.drop_target_mb:.0f} MB ({len(candidates)} eligible)")

    dropped, manifest = 0, []
    for name, day in candidates:
        if db_size_mb(conn) <= args.drop_target_mb:
            break
        if args.dry_run:
            print(f"  {name}: would drop")
            continue
        _, rows, size = exported[name]
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE public."{name}"')
        conn.commit()
        dropped += 1
        manifest.append(dict(partition=name, day=day.isoformat(), rows=rows,
                             bytes=size,
                             dropped_at=datetime.now(timezone.utc).isoformat()))
        print(f"  {name}: dropped ({rows:,} rows)")

    if manifest and not args.dry_run:
        mpath = os.path.join(work, f"dropped_{today:%Y%m%d}.json")
        pd.DataFrame(manifest).to_json(mpath, orient="records", indent=1)
        rclone("copy", mpath, f"{REMOTE}/manifests")

    print(f"\ndropped {dropped} partition(s); database "
          f"{size_before:.0f} -> {db_size_mb(conn):.0f} MB")
    report_remote_usage()
    conn.close()


if __name__ == "__main__":
    main()
