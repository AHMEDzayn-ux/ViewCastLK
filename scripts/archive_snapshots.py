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

# pandas warns on every psycopg2 connection because it prefers SQLAlchemy; the
# queries here are plain SELECTs and the warning would drown the run log.
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import connect, ensure_snapshot_partitions  # noqa: E402

PARTITION_RE = re.compile(r"^video_snapshots_(\d{8})$")
REMOTE = os.environ.get("ARCHIVE_REMOTE", "gdrive:ViewCastLK/archives")
HORIZONS = (7, 14, 21, 30)


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


def extract_labels(conn):
    """Materialise the observation nearest each horizon, for every video.

    Runs against whatever is currently in Postgres, every night, so a label is
    already recorded long before its partition becomes a drop candidate. The
    conflict clause keeps whichever observation is closest to the mark, so
    re-running can only improve a label, never degrade one."""
    with conn.cursor() as cur:
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
             WHERE abs(EXTRACT(EPOCH FROM (s.captured_at - v.published_at))/3600.0
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
        """)
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
    """Write one partition to Parquet and upload it. Returns its remote size on
    success, or None if it was skipped. Already-uploaded partitions are left
    alone, which makes re-runs cheap and the job safe to retry."""
    dest = f"{REMOTE}/{day:%Y/%m}"
    fname = f"{name}.parquet"

    if not dry_run:
        already = remote_size(dest, fname)
        if already is not None:
            return already

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

    rclone("copy", local, dest, "--checksum", "--retries", "3")
    confirmed = remote_size(dest, fname)
    if confirmed != size:
        print(f"  {name}: upload UNVERIFIED (local {size}, remote {confirmed})")
        return None
    print(f"  {name}: exported {rows:,} rows, {size/1e6:.2f} MB, sha {sha256(local)[:12]}")
    return confirmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retain-days", type=int, default=7,
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

    if not args.dry_run:
        print(f"horizon labels materialised/updated: {extract_labels(conn):,}")
    else:
        print("dry run: skipping label extraction")

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
    exported = {}
    for name, day, rows in with_rows:
        got = export_partition(conn, name, day, rows, work, args.dry_run)
        if got is not None:
            exported[name] = (day, rows, got)
    print(f"  {len(exported)} partition(s) safely on the remote")

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
    if size_before < args.drop_above_mb:
        print(f"\nno drops: {size_before:.0f} MB is below the "
              f"{args.drop_above_mb:.0f} MB threshold. History stays queryable "
              f"in Postgres until space actually runs short.")
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
    conn.close()


if __name__ == "__main__":
    main()
