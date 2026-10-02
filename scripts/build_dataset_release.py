"""Build the public ViewCastLK dataset from a training table build.

The release is the training table minus every column that can be rebuilt
exactly from the ones kept (title statistics, tag count, local publish time,
the model's cyclical encodings, the label usability flags), plus the channel's
display name. The data dictionary below is the single source of truth: it is
written into the dashboard's dataset page, so the page and the files cannot
describe different columns.

Writes, for a table whose newest video was published on YYYY-MM-DD:
    <out>/viewcastlk_dataset_YYYYMMDD.parquet
    <out>/viewcastlk_dataset_YYYYMMDD.csv.gz
    dashboard/src/data/dataset.json

The files are uploaded to a GitHub Release tagged dataset-YYYY-MM-DD; the page
links there, because the site's host refuses files over 25 MB.

Usage:
    python scripts/build_dataset_release.py
    python scripts/build_dataset_release.py --table ../Dataset/viewcastlk_training_table.parquet
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from storage import connect, read_df

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DSEP = os.path.dirname(REPO)
PAGE_DATA = os.path.join(REPO, "dashboard", "src", "data", "dataset.json")
RELEASE_BASE = "https://github.com/AHMEDzayn-ux/ViewCastLK/releases/download"
HORIZONS = (7, 14, 21, 30)
TOLERANCE_HOURS = 12

YOUTUBE, COLLECTED, COMPUTED = "youtube", "collected", "computed"

# name, type, group, meaning, source, how
COLUMNS = [
    ("video_id", "string", "Identifiers", "YouTube video ID. The video is at youtube.com/watch?v=<video_id>.",
     YOUTUBE, "videos.list id."),
    ("channel_id", "string", "Identifiers", "YouTube channel ID of the uploader.",
     YOUTUBE, "videos.list snippet.channelId."),
    ("channel_title", "string", "Identifiers", "The channel's display name.",
     YOUTUBE, "channels.list snippet.title, as first collected."),

    ("published_at", "timestamp (UTC)", "Video", "When the video was published.",
     YOUTUBE, "videos.list snippet.publishedAt."),
    ("title", "string", "Video", "Video title.",
     YOUTUBE, "videos.list snippet.title, as first collected. See title_changed."),
    ("tags", "string", "Video", "Uploader's tags, separated by |. Empty when the video has none.",
     YOUTUBE, "videos.list snippet.tags, joined with |."),
    ("category_id", "string", "Video", "YouTube category number.",
     YOUTUBE, "videos.list snippet.categoryId."),
    ("category_name", "string", "Video", "YouTube category name.",
     COMPUTED, "Looked up from YouTube's category list for Sri Lanka. That list leaves out "
               "category 29, which we fill in as Nonprofits & Activism."),
    ("duration_seconds", "number", "Video", "Length in seconds. Empty when YouTube reports no "
     "usable duration.", YOUTUBE, "videos.list contentDetails.duration, converted from ISO 8601."),
    ("is_short", "boolean", "Video", "Whether the video is a YouTube Short.",
     COMPUTED, "The API has no Shorts flag. True when the player is vertical or square and the "
               "video is at most 180 seconds. Where the player shape is unknown (is_vertical "
               "empty), true when the video is at most 60 seconds."),
    ("is_vertical", "boolean", "Video", "Whether the video player is taller than or as tall as it "
     "is wide. Empty when the shape could not be read.",
     COLLECTED, "Embed width and height from YouTube's oEmbed endpoint."),
    ("is_live_broadcast", "boolean", "Video", "Whether the video is a live broadcast (YouTube "
     "reports its duration as zero).", COMPUTED, "contentDetails.duration equals P0D."),
    ("definition", "string", "Video", "hd or sd.",
     YOUTUBE, "videos.list contentDetails.definition."),
    ("caption", "boolean", "Video", "Whether the uploader added captions.",
     YOUTUBE, "videos.list contentDetails.caption."),
    ("made_for_kids", "boolean", "Video", "Whether the video is marked as made for kids.",
     YOUTUBE, "videos.list status.madeForKids."),
    ("default_audio_language", "string", "Video", "Spoken language the uploader set, as a "
     "language code. Often empty.", YOUTUBE, "videos.list snippet.defaultAudioLanguage."),
    ("default_language", "string", "Video", "Language of the title and description the uploader "
     "set, as a language code.", YOUTUBE, "videos.list snippet.defaultLanguage."),
    ("description_length", "integer", "Video", "Number of characters in the description. 0 when "
     "there is none.", COMPUTED, "Length of videos.list snippet.description. The description "
                                 "itself is not included."),
    ("topic_categories", "string", "Video", "The channel's topics, as Wikipedia URLs separated by |.",
     YOUTUBE, "channels.list topicDetails.topicCategories."),

    ("ch_subs_at_publish", "integer", "Channel when the video was published",
     "Channel subscribers.", COLLECTED,
     "From our channel snapshot taken at or most recently before published_at."),
    ("ch_views_at_publish", "integer", "Channel when the video was published",
     "Channel total views.", COLLECTED, "Same snapshot as ch_subs_at_publish."),
    ("ch_videos_at_publish", "integer", "Channel when the video was published",
     "Number of videos on the channel.", COLLECTED, "Same snapshot as ch_subs_at_publish."),
    ("channel_age_days_at_publish", "number", "Channel when the video was published",
     "Days between the channel's creation and the video's publication.",
     COMPUTED, "published_at minus channels.list snippet.publishedAt."),
    ("ch_stats_as_of", "timestamp (UTC)", "Channel when the video was published",
     "When the channel snapshot above was taken.", COLLECTED,
     "If it is after published_at, no snapshot predated the video and the earliest one was "
     "used instead, so the channel figures include some growth after publication."),
]
for h in HORIZONS:
    grp = f"Day {h}"
    COLUMNS += [
        (f"d{h}_views", "integer", grp, f"Total views at day {h}.", COLLECTED,
         f"From our poll nearest to {h * 24} hours after published_at."),
        (f"d{h}_likes", "integer", grp, f"Total likes at day {h}. Empty when likes are hidden.",
         COLLECTED, f"Same poll as d{h}_views."),
        (f"d{h}_comments", "integer", grp, f"Total comments at day {h}. Empty when comments are "
         "off.", COLLECTED, f"Same poll as d{h}_views."),
        (f"d{h}_hours_off", "number", grp, f"How far that poll was from the exact day-{h} mark, in "
         "hours. Positive means after the mark.", COMPUTED,
         f"Poll time minus (published_at + {h * 24} hours). Use the day-{h} figures only when "
         f"this is within ±{TOLERANCE_HOURS} and d{h}_views is present."),
    ]
COLUMNS += [
    ("title_changed", "boolean", "Edits after publishing", "Whether a later poll saw a different "
     "title from the one in this file.", COLLECTED, "Compared on every poll."),
    ("description_changed", "boolean", "Edits after publishing", "Whether a later poll saw a "
     "different description.", COLLECTED, "Compared by fingerprint on every poll."),
]
NAMES = [c[0] for c in COLUMNS]

INTEGERS = [c[0] for c in COLUMNS if c[1] == "integer"]
BOOLEANS = [c[0] for c in COLUMNS if c[1] == "boolean"]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.map({"true": True, "false": False}).astype("boolean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(DSEP, "Dataset", "viewcastlk_training_table.parquet"))
    ap.add_argument("--out", default=os.path.join(DSEP, "Dataset", "release"))
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    with connect(session_pooler=True) as conn:
        titles = read_df(conn, "SELECT channel_id, title AS channel_title FROM channels")
    df = df.merge(titles, on="channel_id", how="left")
    missing = df.channel_title.isna().sum()
    if missing:
        raise SystemExit(f"{missing} videos have no channel title; is the warehouse reachable?")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["ch_stats_as_of"] = pd.to_datetime(df["ch_stats_as_of"], utc=True)
    df["tags"] = df["tags"].fillna("")
    for h in HORIZONS:
        df[f"d{h}_hours_off"] = pd.to_numeric(df[f"d{h}_hours_off"], errors="coerce")
    df["channel_age_days_at_publish"] = df["channel_age_days_at_publish"].round(3)
    for col in INTEGERS:
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    for col in BOOLEANS:
        df[col] = as_bool(df[col])
    out = df[NAMES].sort_values(["published_at", "video_id"]).reset_index(drop=True)

    cutoff = out.published_at.max()
    stamp = cutoff.strftime("%Y%m%d")
    tag = f"dataset-{cutoff:%Y-%m-%d}"
    os.makedirs(args.out, exist_ok=True)
    parquet = os.path.join(args.out, f"viewcastlk_dataset_{stamp}.parquet")
    csv_gz = os.path.join(args.out, f"viewcastlk_dataset_{stamp}.csv.gz")
    out.to_parquet(parquet, index=False)
    csv_out = out.copy()
    for col in ("published_at", "ch_stats_as_of"):
        csv_out[col] = csv_out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_out.to_csv(csv_gz, index=False, compression={"method": "gzip", "mtime": 0})

    def usable(h):
        return int(((out[f"d{h}_hours_off"].abs() <= TOLERANCE_HOURS)
                    & out[f"d{h}_views"].notna()).sum())

    files = []
    for path, fmt in ((parquet, "Parquet"), (csv_gz, "CSV (gzip)")):
        name = os.path.basename(path)
        files.append(dict(name=name, format=fmt, bytes=os.path.getsize(path),
                          sha256=sha256(path), url=f"{RELEASE_BASE}/{tag}/{name}"))

    page = dict(
        generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        releaseTag=tag,
        videos=int(len(out)),
        channels=int(out.channel_id.nunique()),
        periodStart=out.published_at.min().date().isoformat(),
        periodEnd=cutoff.date().isoformat(),
        toleranceHours=TOLERANCE_HOURS,
        labelled={f"day{h}": usable(h) for h in HORIZONS},
        files=files,
        columns=[dict(name=n, type=t, group=g, meaning=m, source=s, how=w,
                      emptyPct=round(float(out[n].isna().mean() * 100), 1))
                 for n, t, g, m, s, w in COLUMNS],
    )
    os.makedirs(os.path.dirname(PAGE_DATA), exist_ok=True)
    with open(PAGE_DATA, "w", encoding="utf-8") as f:
        json.dump(page, f, indent=1, ensure_ascii=False)

    print(f"{len(out):,} videos, {out.channel_id.nunique():,} channels, {len(NAMES)} columns, "
          f"published {page['periodStart']} to {page['periodEnd']}")
    for f_ in files:
        print(f"  {f_['name']}: {f_['bytes'] / 1e6:.1f} MB  sha256 {f_['sha256'][:12]}…")
    print(f"  labelled within ±{TOLERANCE_HOURS}h: {page['labelled']}")
    print(f"upload both files to the GitHub Release tagged {tag}")


if __name__ == "__main__":
    main()
