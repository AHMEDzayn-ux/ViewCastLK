"""Find frequently-posting, API-verified Sri Lankan channels in the previous
batch's dataset that our roster does not already track.

Their release is a snapshot, so posting frequency has to be inferred from the
publication dates of the videos it happens to contain, over the window it
covers. That is a floor, not a measurement: a channel posting daily will show
up as such, but a channel the snapshot only caught once may still be active.
Ranking on it is safe; treating it as a rate is not.

The country column in their file is not usable - every row says LK because it
came from a region-filtered search. Analysis/prior_channel_countries.csv holds
the result of querying all 41,127 channels against channels.list, and only the
21.6% that actually declare LK are considered here.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "Reference Datasets")
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

CATEGORY_NAMES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 19: "Travel & Events", 20: "Gaming", 22: "People & Blogs",
    23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
    27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism",
}


def main():
    frames = []
    for fn in ["FinalProcessedDataset_05 (1).csv", "youtube_video_data_lk.csv"]:
        d = pd.read_csv(os.path.join(SRC, fn), low_memory=False,
                        usecols=lambda c: c in {"video_id", "channel_id", "channel_name",
                                                "publish_date", "category_id",
                                                "subscriber_count"})
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(subset="video_id")
    df["publish_date"] = pd.to_datetime(df.publish_date, utc=True,
                                        errors="coerce", format="mixed")
    df = df.dropna(subset=["channel_id", "publish_date"])
    print(f"prior-batch videos (deduped): {len(df):,}  "
          f"channels: {df.channel_id.nunique():,}")
    print(f"window: {df.publish_date.min().date()} -> {df.publish_date.max().date()}")

    # --- keep only channels the API confirms are LK
    cc = pd.read_csv(os.path.join(HERE, "prior_channel_countries.csv"))
    lk = cc[cc.country == "LK"]
    print(f"API-verified LK channels in their set: {len(lk):,} of {len(cc):,}")
    df = df[df.channel_id.isin(set(lk.channel_id))]
    print(f"videos from verified-LK channels: {len(df):,}")

    # --- exclude what we already track
    from storage import connect
    conn = connect()
    ours = set(pd.read_sql_query("select channel_id from channels", conn).channel_id)
    conn.close()
    print(f"our roster: {len(ours):,}")

    span_days = max((df.publish_date.max() - df.publish_date.min()).days, 1)
    g = df.groupby("channel_id").agg(
        videos=("video_id", "count"),
        first=("publish_date", "min"),
        last=("publish_date", "max"),
        name=("channel_name", "first"),
        subs=("subscriber_count", "max"),
        top_cat=("category_id", lambda s: s.mode().iat[0] if len(s.mode()) else None),
    ).reset_index()
    active = (g["last"] - g["first"]).dt.days.clip(lower=1)
    g["per_week"] = (g.videos / active * 7).round(1)
    g["already_tracked"] = g.channel_id.isin(ours)
    g["category"] = g.top_cat.map(CATEGORY_NAMES).fillna("(unknown)")

    overlap = int(g.already_tracked.sum())
    print(f"\noverlap with our roster: {overlap:,} of {len(g):,} verified-LK channels")
    new = g[~g.already_tracked].copy()
    print(f"candidates not yet tracked: {len(new):,}")

    # a channel the snapshot caught many times, across a span, is genuinely active
    cand = new[(new.videos >= 5)].sort_values("videos", ascending=False)
    print(f"of those, with >=5 videos in their window: {len(cand):,}")

    out = os.path.join(HERE, "candidate_channels.csv")
    cand[["channel_id", "name", "subs", "videos", "per_week", "category",
          "first", "last"]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {out}")

    print("\ncategory mix of the >=5-video candidates:")
    mix = cand.category.value_counts()
    for k, v in mix.items():
        print(f"    {k:24} {v:5,}")

    print("\ntop 25 by video count in their window:")
    show = cand.head(25)[["name", "subs", "videos", "per_week", "category"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
