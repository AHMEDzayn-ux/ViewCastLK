"""Prove the archives are restorable, not merely present.

Downloading a file is not evidence that a backup works. This checks the three
things that actually matter, in order of how badly each would hurt:

  1. INTEGRITY  does every file's SHA-256 match what the archive job recorded
                when it uploaded? A silent corruption in transit or in Drive
                would otherwise surface only when the data was needed.
  2. COMPLETENESS  does every file hold exactly the rows the job counted, and
                do the days form an unbroken run?
  3. USABILITY  can a video's engagement trajectory be reconstructed across
                file boundaries, and does it agree with what Postgres still
                holds for the same rows?

Run against a folder of downloaded archives:
    python verify_archives.py restore_test/archives
"""
import argparse
import glob
import hashlib
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# What the archive run reported at upload time, from its own log.
REPORTED = {
    "20260717": (1980, "85f95a60e1c8"), "20260718": (2539, "4945f6a6aceb"),
    "20260720": (2249, "181c310b1904"), "20260721": (9481, "691acb5b1536"),
    "20260722": (7589, "6be268366b79"), "20260723": (18237, "fde6bbc12017"),
    "20260724": (21946, "e683a243bb62"), "20260725": (24922, "32ddfbf96288"),
    "20260726": (21118, "786a666857b4"), "20260727": (31013, "aa8ac57551f4"),
    "20260728": (26043, "42217af514f6"), "20260729": (37368, "55860a768fbc"),
    "20260730": (40463, "6236322a9c25"), "20260731": (43966, "cf5874f8ae4a"),
    "20260801": (46857, "d4591eb4510e"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="restore_test/archives")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {args.folder}")

    print(f"{'file':34}{'rows':>9}{'reported':>10}{'sha':>8}")
    frames, bad = [], 0
    for f in files:
        day = re.search(r"_(\d{8})\.parquet$", f).group(1)
        df = pd.read_parquet(f)
        frames.append(df)
        exp_rows, exp_sha = REPORTED.get(day, (None, None))
        got_sha = sha256(f)[:12]
        rows_ok = exp_rows is None or len(df) == exp_rows
        sha_ok = exp_sha is None or got_sha == exp_sha
        bad += (not rows_ok) or (not sha_ok)
        print(f"{os.path.basename(f):34}{len(df):>9,}"
              f"{'ok' if rows_ok else 'MISMATCH':>10}{'ok' if sha_ok else 'MISMATCH':>8}")

    all_rows = pd.concat(frames, ignore_index=True)
    print(f"\nintegrity: {'all files verified' if not bad else f'{bad} FILE(S) FAILED'}")
    print(f"total rows across archives: {len(all_rows):,}")
    print(f"columns: {list(all_rows.columns)}")

    # --- completeness: unbroken run of days
    days = sorted({d.date() for d in pd.to_datetime(all_rows.captured_at, utc=True)})
    span = (days[-1] - days[0]).days + 1
    print(f"\ndays covered: {days[0]} -> {days[-1]} "
          f"({len(days)} of {span}; 19 Jul absent, that day's collection failed)")

    # --- usability: reconstruct one trajectory across file boundaries
    counts = all_rows.groupby("video_id").size()
    vid = counts.idxmax()
    traj = (all_rows[all_rows.video_id == vid]
            .sort_values("captured_at")[["captured_at", "view_count"]])
    print(f"\nlongest trajectory: {vid}, {len(traj)} observations")
    print(traj.head(4).to_string(index=False))
    print("   ...")
    print(traj.tail(2).to_string(index=False))
    mono = traj.view_count.is_monotonic_increasing
    print(f"view count non-decreasing: {mono}"
          f"{'' if mono else '  <-- unexpected for a cumulative counter'}")

    # --- cross-check against what Postgres still holds
    from storage import connect
    conn = connect()
    db = pd.read_sql_query(
        "SELECT video_id, captured_at, view_count FROM video_snapshots "
        "WHERE captured_at >= %s AND captured_at < %s",
        conn, params=(str(days[0]), str(days[-1] + pd.Timedelta(days=1))))
    conn.close()

    a = all_rows[["video_id", "captured_at", "view_count"]].copy()
    a["captured_at"] = pd.to_datetime(a.captured_at, utc=True)
    db["captured_at"] = pd.to_datetime(db.captured_at, utc=True)
    merged = a.merge(db, on=["video_id", "captured_at"], how="inner",
                     suffixes=("_file", "_db"))
    # NaN != NaN, so a plain inequality reports every row where the API gave no
    # view count -- private or removed videos, and live broadcasts that hide
    # their statistics -- as a disagreement when both sides are equally null.
    both_null = merged.view_count_file.isna() & merged.view_count_db.isna()
    disagree = int((~((merged.view_count_file == merged.view_count_db) | both_null)).sum())
    nulls = int(both_null.sum())
    print(f"\ncross-check against Postgres over the same window:")
    print(f"  rows in archives {len(a):,}   rows still in Postgres {len(db):,}")
    print(f"  matched on (video_id, captured_at): {len(merged):,}")
    print(f"  rows null on both sides: {nulls} (no view count from the API)")
    print(f"  view_count disagreements: {disagree}"
          f"{'  -- archives are faithful' if disagree == 0 else '  <-- INVESTIGATE'}")

    ok = bad == 0 and disagree == 0
    print(f"\n{'RESTORE VERIFIED' if ok else 'VERIFICATION FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
