"""Write the activity sweep's verdict back onto the channels table.

sweep_last_upload.py establishes when each rostered channel last uploaded.
This records that on the row and clears the active flag for channels that have
stopped, so discovery stops paying for them.

Nothing is deleted. A flagged channel keeps its declared-country verification
and any videos it contributed keep their foreign key; only polling stops, and
setting the flag back reinstates it.

Dormant channels (31-60 days quiet) stay active on purpose. They are the
mid-frequency creators the specification names as the design centre, and
dropping them would bias the roster toward exactly the high-volume channels
the project is trying to dilute.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

from psycopg2.extras import execute_batch  # noqa: E402

from storage import connect  # noqa: E402

SWEEP = os.path.join(HERE, "channel_last_upload.csv")
INACTIVE_STATUSES = {"dead", "no_uploads"}


def main():
    d = pd.read_csv(SWEEP)
    d["last_upload"] = pd.to_datetime(d.last_upload, utc=True, errors="coerce")
    print(f"sweep rows: {len(d):,}")
    print(d.status.value_counts().to_string())

    rows = [
        (r.channel_id,
         None if pd.isna(r.last_upload) else r.last_upload.isoformat(),
         r.status not in INACTIVE_STATUSES)
        for r in d.itertuples()
    ]
    deactivating = sum(1 for _, _, a in rows if not a)
    print(f"\nmarking inactive: {deactivating:,}   keeping active: {len(rows)-deactivating:,}")

    conn = connect()
    try:
        with conn.cursor() as cur:
            # execute_batch, not executemany: the latter is one network
            # round-trip per row, which over the transaction pooler took several
            # minutes for 1,282 rows. This pipelines them into few round-trips.
            execute_batch(
                cur,
                """UPDATE channels
                      SET active = %s,
                          last_upload_at = %s::timestamptz,
                          activity_checked_at = now()
                    WHERE channel_id = %s""",
                [(a, ts, cid) for cid, ts, a in rows],
                page_size=200)
            conn.commit()
            cur.execute("""SELECT active, count(*),
                                  count(uploads_playlist_id) AS with_playlist
                             FROM channels GROUP BY active ORDER BY active DESC""")
            print(f"\n{'active':>8}{'channels':>10}{'pollable':>10}")
            for act, n, wp in cur.fetchall():
                print(f"{str(act):>8}{n:>10,}{wp:>10,}")
    finally:
        conn.close()

    saved = deactivating * 4
    print(f"\ndiscovery calls no longer made: {saved:,} units/day at four runs")


if __name__ == "__main__":
    main()
