"""What labelled data exists right now, cut three ways.

Computed from raw snapshots rather than video_horizon_labels, because that
table is materialised by the nightly archive run and has not yet seen the
24,067 videos added by today's backfill. The rule is the same one
build_training_table.py applies: the observation nearest N x 24 hours after a
video's own published_at, usable when it lands within +/-12 hours of the mark.

Three cuts, because they answer different questions:
  category          which categories can support a per-category metric
  channel size      whether the mid-range creator the SRS names as the design
                    centre is actually represented
  upload frequency  how much of the data comes from channels that publish at
                    industrial rates, since those dominate raw counts
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))

import pandas as pd  # noqa: E402
from storage import connect  # noqa: E402

HORIZONS = (7, 14, 21, 30)
TOL = 12.0

USABLE = """
SELECT v.video_id,
       COALESCE(v.category_name, '(none)') AS category,
       v.channel_id,
       h.horizon
  FROM videos v
  JOIN video_snapshots s USING (video_id)
 CROSS JOIN (VALUES (7), (14), (21), (30)) AS h(horizon)
 WHERE v.duration IS NOT NULL AND v.duration <> 'P0D'
   AND abs(EXTRACT(EPOCH FROM (s.captured_at - v.published_at))/3600.0
           - h.horizon * 24) <= %s
 GROUP BY 1, 2, 3, 4
"""


def band_table(df, band_col, order, title):
    t = df.pivot_table(index=band_col, columns="horizon", values="video_id",
                       aggfunc="count", fill_value=0)
    t = t.reindex(order).fillna(0).astype(int)
    t.columns = [f"day {c}" for c in t.columns]
    print(f"\n{title}")
    print(t.to_string())
    return t


def main():
    conn = connect()
    lab = pd.read_sql_query(USABLE, conn, params=(TOL,))

    chan = pd.read_sql_query("""
        SELECT c.channel_id,
               COALESCE(s.subscriber_count, 0) AS subs
          FROM channels c
          LEFT JOIN LATERAL (
              SELECT subscriber_count FROM channel_snapshots
               WHERE channel_id = c.channel_id
               ORDER BY captured_at DESC LIMIT 1) s ON true
    """, conn)

    freq = pd.read_sql_query("""
        SELECT channel_id,
               count(*) / GREATEST(
                   EXTRACT(EPOCH FROM (max(published_at) - min(published_at)))
                   / 86400.0, 1) AS per_day
          FROM videos GROUP BY 1
    """, conn)
    conn.close()

    print(f"usable labels: {len(lab):,} video-horizon pairs "
          f"across {lab.video_id.nunique():,} videos")
    per_h = lab.groupby("horizon").video_id.nunique()
    print("  " + "   ".join(f"day {h}: {per_h.get(h, 0):,}" for h in HORIZONS))

    # ---------------------------------------------------------- by category
    t = lab.pivot_table(index="category", columns="horizon", values="video_id",
                        aggfunc="count", fill_value=0)
    t.columns = [f"day {c}" for c in t.columns]
    t = t.sort_values("day 7", ascending=False)
    t["total"] = t.sum(axis=1)
    print("\nBY CATEGORY")
    print(t.to_string())
    thin = t[t["day 7"] < 100]
    if len(thin):
        print(f"\n  under 100 day-7 rows (too thin for a per-category metric): "
              f"{', '.join(thin.index)}")

    # ------------------------------------------------------- by channel size
    lab = lab.merge(chan, on="channel_id", how="left")
    bands = [(-1, 1e3, "<1k"), (1e3, 1e4, "1k-10k"), (1e4, 1e5, "10k-100k"),
             (1e5, 1e6, "100k-1M"), (1e6, 1e12, ">1M")]
    lab["size"] = pd.cut(lab.subs.fillna(0),
                         [b[0] for b in bands] + [1e12],
                         labels=[b[2] for b in bands])
    band_table(lab, "size", [b[2] for b in bands], "BY CHANNEL SIZE (subscribers)")

    # ---------------------------------------------------- by upload frequency
    lab = lab.merge(freq, on="channel_id", how="left")
    fb = [(-1, 0.14, "<1/week"), (0.14, 0.5, "1-3/week"), (0.5, 2, "0.5-2/day"),
          (2, 10, "2-10/day"), (10, 1e6, ">10/day")]
    lab["freq"] = pd.cut(lab.per_day.fillna(0),
                         [b[0] for b in fb] + [1e6],
                         labels=[b[2] for b in fb])
    t = band_table(lab, "freq", [b[2] for b in fb], "BY UPLOAD FREQUENCY")
    share = 100 * t.loc[">10/day"] / t.sum()
    print("\n  share of each horizon coming from channels publishing >10/day:")
    print("  " + "   ".join(f"{c}: {share[c]:.0f}%" for c in t.columns))


if __name__ == "__main__":
    main()
