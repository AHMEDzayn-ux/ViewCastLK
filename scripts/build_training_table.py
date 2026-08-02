"""Build the ViewCastLK model-training table from the Supabase warehouse.

Produces ONE ROW PER VIDEO containing only information that would have been
available BEFORE that video was published, plus the day-7/14/21/30 view
targets. Feature engineering and modelling happen downstream of this file.

Three correctness rules are enforced here so they cannot be got wrong later:

1. LABELS COME FROM ELAPSED TIME, NOT SNAPSHOT ORDER.
   "Day 7" is the snapshot nearest to 168 hours after *that video's* own
   published_at. Polls run roughly six-hourly and drift, so counting rows
   would silently mislabel. Each label carries the actual offset in hours
   (d7_hours_off) and a usability flag (d7_usable) that is false when the
   nearest snapshot is further from the mark than TOLERANCE_HOURS — which is
   what happens when a collection gap swallowed the target moment.

2. CHANNEL FEATURES ARE POINT-IN-TIME.
   Subscriber/view/video counts are taken from the newest channel snapshot at
   or before published_at, never the current value. Using today's subscriber
   count to predict a video published weeks ago leaks the outcome into the
   feature: a video that did well grew the channel. Where no snapshot predates
   the video (videos published before tracking began) the earliest available
   snapshot is substituted and channel_stats_backfilled is set to True so
   those rows can be dropped or down-weighted.

3. NO POST-PUBLICATION ENGAGEMENT IS EXPORTED AS A FEATURE.
   No view/like/comment value at any age appears among the feature columns —
   they exist only as prediction targets. This is the project's whole claim:
   forecasting before publication, unlike prior work that consumes observed
   early engagement. Adding an early-engagement feature would silently void it.

Usage:  python scripts/build_training_table.py [--out PATH] [--tolerance HOURS]
"""
import argparse
import math
import os
import re
import sys
from datetime import timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata_changes  # noqa: E402
from storage import connect  # noqa: E402

HORIZONS = (7, 14, 21, 30)
TOLERANCE_HOURS = 12.0
COLOMBO = "Asia/Colombo"

SINHALA = re.compile(r"[඀-෿]")
TAMIL = re.compile(r"[஀-௿]")
ISO_DUR = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


# ----------------------------------------------------------------- helpers
def duration_seconds(iso):
    """ISO-8601 duration -> seconds. Live/unfinished videos report 'P0D';
    absent durations arrive from the DB as NaN, not as a string."""
    if not isinstance(iso, str) or not iso or iso == "P0D":
        return None
    m = ISO_DUR.fullmatch(iso)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    total = d * 86400 + h * 3600 + mi * 60 + s
    return total or None


def title_script(text: str) -> str:
    """Which ALPHABET the title is written in — not which language it is in.

    Values are suffixed '_script' deliberately: 'latin' alone would read as the
    Latin language, when what is meant is the a-z Latin alphabet. An English
    title and a romanised Sinhala title ("Man Adarei") are both latin_script.
    """
    t = text or ""
    si, ta = bool(SINHALA.search(t)), bool(TAMIL.search(t))
    if si and ta:
        return "mixed_script"
    if si:
        return "sinhala_script"
    if ta:
        return "tamil_script"
    return "latin_script"


def cyc(series, period):
    rad = 2 * math.pi * series / period
    return rad.apply(math.sin), rad.apply(math.cos)


# ----------------------------------------------------------------- queries
VIDEOS_SQL = """
-- description is summarised server-side rather than shipped. At ~786 bytes
-- across 49,502 rows it is ~39 MB on the wire, and pulling it through the
-- transaction pooler -- which exists for short transactions -- reliably
-- dropped the connection mid-query. Only its length and its fingerprint are
-- ever used, and both are cheaper to compute in the database.
SELECT v.video_id, v.channel_id, v.published_at, v.title,
       length(v.description) AS description_length,
       -- convert_to(), not a ::bytea cast. Casting text to bytea runs the
       -- bytea input parser, which reads a leading backslash-x as hex and
       -- treats backslashes as escapes, so any description containing them
       -- raises "invalid input syntax for type bytea". convert_to() encodes
       -- the characters as they are.
       encode(sha256(convert_to(coalesce(v.description, '') || chr(31), 'UTF8')),
              'hex') AS description_sha,
       v.tags, v.category_id, v.category_name, v.duration, v.definition,
       v.caption, v.made_for_kids, v.default_audio_language, v.default_language,
       c.country AS channel_country, c.channel_published_at,
       c.topic_categories
FROM videos v
JOIN channels c USING (channel_id)
"""

# Nearest observation to each horizon, per video.
#
# Read from video_horizon_labels, not from video_snapshots. The snapshot table
# is partitioned by day and the nightly archive exports partitions to Parquet
# and drops them once the database exceeds its size threshold, so raw snapshots
# only reach back a week or so. Scanning them would still run, still exit zero,
# and quietly return a label for only the most recent videos -- the failure
# would show up as a small training set rather than as an error.
#
# video_horizon_labels is materialised nightly against whatever is in Postgres
# at the time, keeping whichever observation is closest to each mark, and is
# never dropped. It is the durable form of exactly this query.
LABEL_SQL = """
SELECT video_id,
       view_count    AS d{h}_views,
       like_count    AS d{h}_likes,
       comment_count AS d{h}_comments,
       hours_off     AS d{h}_hours_off
FROM video_horizon_labels
WHERE horizon_days = {h}
"""

# Post-publication edits, so a contaminated row can be identified.
#
# videos holds what was seen at first discovery, and 73% of the corpus was
# first seen more than a day after publication. A title edited since then may
# already be a reaction to the video's performance, which would put the target
# on the feature side of a model whose whole claim is pre-publication
# forecasting. Measured across 49,739 videos: titles changed on 0.4%,
# descriptions or tags on 31.9%.
CHANGES_SQL = """
SELECT video_id, title AS new_title, description_sha, tags_sha
FROM video_metadata_changes
"""

# newest channel snapshot at or before the video was published
CHAN_PIT_SQL = """
SELECT DISTINCT ON (v.video_id)
       v.video_id,
       cs.subscriber_count AS ch_subs_at_publish,
       cs.view_count       AS ch_views_at_publish,
       cs.video_count      AS ch_videos_at_publish,
       cs.captured_at      AS ch_stats_as_of
FROM videos v
JOIN channel_snapshots cs
  ON cs.channel_id = v.channel_id AND cs.captured_at <= v.published_at
ORDER BY v.video_id, cs.captured_at DESC
"""

# earliest snapshot per channel, used only as a flagged fallback
CHAN_FIRST_SQL = """
SELECT DISTINCT ON (channel_id)
       channel_id,
       subscriber_count AS ch_subs_first,
       view_count       AS ch_views_first,
       video_count      AS ch_videos_first,
       captured_at      AS ch_first_as_of
FROM channel_snapshots
ORDER BY channel_id, captured_at ASC
"""


def read(conn, sql):
    return pd.read_sql_query(sql, conn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "training_data",
        "viewcastlk_training_table.csv"))
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_HOURS)
    args = ap.parse_args()

    conn = connect()
    try:
        df = read(conn, VIDEOS_SQL)
        print(f"videos in warehouse: {len(df):,}")
        for h in HORIZONS:
            lab = read(conn, LABEL_SQL.format(h=h))
            df = df.merge(lab, on="video_id", how="left")
        pit = read(conn, CHAN_PIT_SQL)
        first = read(conn, CHAN_FIRST_SQL)
        changes = read(conn, CHANGES_SQL)
    finally:
        conn.close()

    df = df.merge(pit, on="video_id", how="left").merge(first, on="channel_id", how="left")

    # ---- rule 2: point-in-time channel stats, with flagged fallback --------
    df["channel_stats_backfilled"] = df["ch_subs_at_publish"].isna()
    for a, b in (("ch_subs_at_publish", "ch_subs_first"),
                 ("ch_views_at_publish", "ch_views_first"),
                 ("ch_videos_at_publish", "ch_videos_first"),
                 ("ch_stats_as_of", "ch_first_as_of")):
        df[a] = df[a].fillna(df[b])
    df = df.drop(columns=["ch_subs_first", "ch_views_first",
                          "ch_videos_first", "ch_first_as_of"])

    # ---- post-publication edit flags --------------------------------------
    # Compared by hash, not by length: an edit that happens to preserve length
    # would otherwise pass as unchanged. The hash is imported from
    # metadata_changes rather than reimplemented, so the two cannot drift.
    #
    # Vectorised deliberately. A row-wise apply over fifty thousand videos, each
    # doing an index lookup, ran for minutes; this runs on every rebuild.
    if len(changes):
        latest = changes.drop_duplicates("video_id", keep="last").set_index("video_id")
        seen_title = df.video_id.map(latest.new_title)
        df["title_changed"] = seen_title.notna() & (seen_title != df.title)
        seen_desc = df.video_id.map(latest.description_sha)
        df["description_changed"] = (seen_desc.notna()
                                     & (seen_desc != df.description_sha))
    else:
        df["title_changed"] = False
        df["description_changed"] = False

    # ---- pre-publish features --------------------------------------------
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    local = df["published_at"].dt.tz_convert(COLOMBO)
    df["publish_hour_slt"] = local.dt.hour
    df["publish_dow_slt"] = local.dt.dayofweek           # 0 = Monday
    df["publish_is_weekend"] = df["publish_dow_slt"].isin([5, 6])
    df["publish_hour_sin"], df["publish_hour_cos"] = cyc(df["publish_hour_slt"], 24)
    df["publish_dow_sin"], df["publish_dow_cos"] = cyc(df["publish_dow_slt"], 7)

    df["duration_seconds"] = df["duration"].apply(duration_seconds)
    df["is_short"] = df["duration_seconds"].le(60)

    title = df["title"].fillna("")
    df["title_length"] = title.str.len()
    df["title_word_count"] = title.str.split().str.len()
    df["title_has_number"] = title.str.contains(r"\d", regex=True)
    df["title_has_question"] = title.str.contains(r"\?", regex=True)
    df["title_has_exclaim"] = title.str.contains("!", regex=False)
    df["title_upper_ratio"] = title.apply(
        lambda t: sum(c.isupper() for c in t) / len(t) if t else 0.0)
    df["title_script"] = title.apply(title_script)

    df["description_length"] = df["description_length"].fillna(0).astype(int)
    df["tag_count"] = df["tags"].fillna("").apply(lambda s: len(s.split("|")) if s else 0)

    ch_pub = pd.to_datetime(df["channel_published_at"], utc=True, errors="coerce")
    df["channel_age_days_at_publish"] = (df["published_at"] - ch_pub).dt.total_seconds() / 86400

    # ---- labels: usability from actual offset, not row order ---------------
    for h in HORIZONS:
        off = df[f"d{h}_hours_off"].abs()
        df[f"d{h}_usable"] = off.le(args.tolerance) & df[f"d{h}_views"].notna()

    # ---- eligibility ------------------------------------------------------
    df["is_live_broadcast"] = df["duration"].eq("P0D")
    df["eligible"] = (~df["is_live_broadcast"]) & df["duration_seconds"].notna()

    FEATURES = [
        "category_id", "category_name", "duration_seconds", "is_short",
        "definition", "caption", "made_for_kids",
        "default_audio_language", "default_language",
        "publish_hour_slt", "publish_dow_slt", "publish_is_weekend",
        "publish_hour_sin", "publish_hour_cos", "publish_dow_sin", "publish_dow_cos",
        "title_length", "title_word_count", "title_has_number", "title_has_question",
        "title_has_exclaim", "title_upper_ratio", "title_script",
        "description_length", "tag_count",
        "ch_subs_at_publish", "ch_views_at_publish", "ch_videos_at_publish",
        "channel_age_days_at_publish", "channel_country", "topic_categories",
    ]
    LABELS = [f"d{h}_{k}" for h in HORIZONS
              for k in ("views", "likes", "comments", "hours_off", "usable")]
    KEYS = ["video_id", "channel_id", "published_at", "title"]
    META = ["eligible", "is_live_broadcast", "channel_stats_backfilled", "ch_stats_as_of",
            "title_changed", "description_changed"]

    # rule 3 guard: no engagement value may sit in the feature list
    leaked = [c for c in FEATURES
              if re.search(r"(view|like|comment)_count|^d\d+_", c)
              and not c.startswith("ch_")]
    if leaked:
        raise SystemExit(f"ABORT — post-publication engagement in features: {leaked}")

    out = df[KEYS + FEATURES + LABELS + META]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    # ---- summary ----------------------------------------------------------
    el = out[out.eligible]
    print(f"\nwrote {os.path.abspath(args.out)}")
    print(f"rows: {len(out):,}   eligible: {len(el):,}   "
          f"features: {len(FEATURES)}   labels: {len(LABELS)}")
    print(f"tolerance: +/-{args.tolerance:g}h from each horizon\n")
    print(f"{'horizon':>8} {'usable':>8} {'median |offset|':>16}")
    for h in HORIZONS:
        u = el[el[f"d{h}_usable"]]
        med = u[f"d{h}_hours_off"].abs().median()
        print(f"{'day ' + str(h):>8} {len(u):>8,} "
              f"{(f'{med:.1f} h' if len(u) else '-'):>16}")
    print(f"\nchannel stats backfilled (no snapshot before publish): "
          f"{int(el.channel_stats_backfilled.sum()):,} of {len(el):,}")
    print(f"edited since first seen — title: {int(el.title_changed.sum()):,}   "
          f"description: {int(el.description_changed.sum()):,}")
    print(f"shorts (<=60s): {int(el.is_short.sum()):,}   "
          f"long-form: {int((~el.is_short).sum()):,}")
    print("\ntitle script mix:")
    for k, v in el.title_script.value_counts().items():
        print(f"    {k:8} {v:,}")


if __name__ == "__main__":
    main()
