"""Convert the previous batch's public dataset into the ViewCastLK training-table
schema, as a SEPARATE file — not merged with our own data.

Source: HuggingFace madhushankhades/Sri-Lankan-YouTube-Channel-Data (Apache 2.0).
Attribution is required if this contributes to published results.

Why separate: this data differs from ours in ways that matter, and merging it
silently would hide that. Every row is tagged source='prior_batch_2025' so the
distinction survives any later concatenation. Whether to use it at all is a
modelling decision, not a data-engineering one.

Four differences the consumer must know about:

1. NOT VERIFIED SRI LANKAN AT SOURCE. Every row in the original is labelled
   country='LK', but that label comes from a region-filtered search rather than
   from checking each channel. Querying all 41,127 channels against the API
   showed only 21.6% actually declare LK, 40.8% declare somewhere else (India
   alone is 20.8%), and 29.9% declare nothing. This script keeps ONLY channels
   verified as LK, using Analysis/prior_channel_countries.csv.

2. CHANNEL STATS ARE NOT POINT-IN-TIME. subscriber_count is as of their
   snapshot, not as of publication — the leakage our own builder avoids. Rows
   are flagged channel_stats_backfilled=True. Drift is small because most of
   their videos were young when snapshotted, but it is not zero.

3. ONE OBSERVATION PER VIDEO. Their public release is a snapshot, so each video
   yields a label at exactly one horizon — whichever its age happened to match.
   There is no trajectory.

4. DIFFERENT PERIOD. September–October 2025 versus our July 2026 onwards.
   Expect distribution shift.

Also note their publish_hour is UTC; ours is Asia/Colombo. All timing features
are recomputed here from the raw timestamp rather than copied.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_training_table import (HORIZONS, COLOMBO, duration_seconds,  # noqa: E402
                                  title_script, cyc)

FILES = ["FinalProcessedDataset_05 (1).csv", "youtube_video_data_lk.csv"]


def find_up(name, start=None):
    """Walk up from the script (then the cwd) looking for a folder/file, so this
    works whether it is run from the repo, a copy, or anywhere else."""
    for base in filter(None, [start, os.path.dirname(os.path.abspath(__file__)), os.getcwd()]):
        d = os.path.abspath(base)
        for _ in range(6):
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    raise SystemExit(f"could not locate '{name}' — pass an explicit path")


SRC = find_up("Reference Datasets")
COUNTRIES = find_up(os.path.join("Analysis", "prior_channel_countries.csv"))

# their days_since_publish is whole-ish days, so +/-0.5 d == our +/-12 h
TOL_DAYS = 0.5

# youtube_video_data_lk.csv carries category_id but leaves category_name 100%
# empty, so names are reconstructed from the id. Standard YouTube taxonomy.
CATEGORY_NAMES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming",
    21: "Videoblogging", 22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "Howto & Style", 27: "Education",
    28: "Science & Technology", 29: "Nonprofits & Activism", 30: "Movies",
    31: "Anime/Animation", 32: "Action/Adventure", 33: "Classics", 34: "Comedy",
    35: "Documentary", 36: "Drama", 37: "Family", 38: "Foreign", 39: "Horror",
    40: "Sci-Fi/Fantasy", 41: "Thriller", 42: "Shorts", 43: "Shows", 44: "Trailers",
}


def load_source():
    frames = []
    for fn in FILES:
        d = pd.read_csv(os.path.join(SRC, fn), low_memory=False)
        d["__file"] = fn
        frames.append(d)
        print(f"  {fn}: {len(d):,} rows")
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(SRC), "ViewCastLK", "training_data", "prior_batch_training_table.csv"))
    args = ap.parse_args()

    print("loading source files:")
    df = load_source()
    print(f"combined: {len(df):,} rows")

    df = df.drop_duplicates(subset="video_id", keep="first")
    print(f"after dedupe on video_id: {len(df):,}")

    # ---- 1. keep only API-verified Sri Lankan channels ---------------------
    cc = pd.read_csv(COUNTRIES)
    lk = set(cc.loc[cc.country == "LK", "channel_id"])
    before = len(df)
    df = df[df.channel_id.isin(lk)].copy()
    print(f"LK-verified filter: {len(df):,} of {before:,} kept "
          f"({100*len(df)/before:.1f}%), {df.channel_id.nunique():,} channels")

    out = pd.DataFrame(index=df.index)

    # ---- keys --------------------------------------------------------------
    out["video_id"] = df.video_id
    out["channel_id"] = df.channel_id
    published = pd.to_datetime(df.publish_date, utc=True, errors="coerce", format="mixed")
    out["published_at"] = published
    out["title"] = df.title

    # ---- content features --------------------------------------------------
    out["category_id"] = pd.to_numeric(df.category_id, errors="coerce")
    out["category_name"] = df.category_name.where(
        df.category_name.notna(),
        out.category_id.map(CATEGORY_NAMES))
    secs = df.duration.apply(duration_seconds)
    secs = secs.fillna(pd.to_numeric(df.get("duration_minutes"), errors="coerce") * 60)
    out["duration_seconds"] = secs
    out["is_short"] = secs.le(60)
    out["definition"] = df.get("definition")
    out["caption"] = df.get("caption")
    out["made_for_kids"] = pd.NA                      # not collected by them
    out["default_audio_language"] = df.get("language")
    out["default_language"] = pd.NA

    # ---- timing: recomputed in Colombo, NOT their UTC publish_hour ----------
    local = published.dt.tz_convert(COLOMBO)
    out["publish_hour_slt"] = local.dt.hour
    out["publish_dow_slt"] = local.dt.dayofweek
    out["publish_is_weekend"] = out.publish_dow_slt.isin([5, 6])
    out["publish_hour_sin"], out["publish_hour_cos"] = cyc(out.publish_hour_slt.fillna(0), 24)
    out["publish_dow_sin"], out["publish_dow_cos"] = cyc(out.publish_dow_slt.fillna(0), 7)

    # ---- text: recomputed so definitions match ours exactly ----------------
    title = df.title.fillna("")
    out["title_length"] = title.str.len()
    out["title_word_count"] = title.str.split().str.len()
    out["title_has_number"] = title.str.contains(r"\d", regex=True)
    out["title_has_question"] = title.str.contains(r"\?", regex=True)
    out["title_has_exclaim"] = title.str.contains("!", regex=False)
    out["title_upper_ratio"] = title.apply(
        lambda t: sum(c.isupper() for c in t) / len(t) if t else 0.0)
    out["title_script"] = title.apply(title_script)
    out["description_length"] = df.description.fillna("").str.len()
    out["tag_count"] = df.video_tags.fillna("").apply(
        lambda s: len([x for x in str(s).replace("|", ",").split(",") if x.strip()]) if s else 0)

    # ---- channel features (limited; NOT point-in-time) ----------------------
    out["ch_subs_at_publish"] = pd.to_numeric(df.subscriber_count, errors="coerce")
    out["ch_views_at_publish"] = pd.NA
    out["ch_videos_at_publish"] = pd.NA
    out["channel_age_days_at_publish"] = pd.NA
    out["channel_country"] = "LK"
    out["topic_categories"] = pd.NA

    # ---- labels: one observation, assigned to its nearest horizon ----------
    age_days = pd.to_numeric(df.days_since_publish, errors="coerce")
    views = pd.to_numeric(df.view_count, errors="coerce")
    likes = pd.to_numeric(df.like_count, errors="coerce")
    comments = pd.to_numeric(df.comment_count, errors="coerce")
    for h in HORIZONS:
        hit = (age_days - h).abs().le(TOL_DAYS) & views.notna()
        out[f"d{h}_views"] = views.where(hit)
        out[f"d{h}_likes"] = likes.where(hit)
        out[f"d{h}_comments"] = comments.where(hit)
        out[f"d{h}_hours_off"] = ((age_days - h) * 24).where(hit)
        out[f"d{h}_usable"] = hit

    # ---- metadata ----------------------------------------------------------
    out["eligible"] = secs.notna() & secs.gt(0) & published.notna()
    out["is_live_broadcast"] = df.duration.eq("P0D")
    out["channel_stats_backfilled"] = True            # see docstring point 2
    out["ch_stats_as_of"] = pd.to_datetime(df.snapshot_date, utc=True,
                                           errors="coerce", format="mixed")
    out["source"] = "prior_batch_2025"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    el = out[out.eligible]
    print(f"\nwrote {os.path.abspath(args.out)}")
    print(f"rows: {len(out):,}   eligible: {len(el):,}   columns: {len(out.columns)}")
    print(f"\n{'horizon':>8} {'usable':>9} {'median views':>14}")
    for h in HORIZONS:
        u = el[el[f"d{h}_usable"]]
        med = f"{u[f'd{h}_views'].median():,.0f}" if len(u) else "-"
        print(f"{'day ' + str(h):>8} {len(u):>9,} {med:>14}")
    print(f"\nperiod: {el.published_at.min()}  ->  {el.published_at.max()}")
    print(f"shorts: {int(el.is_short.sum()):,}   long-form: {int((~el.is_short).sum()):,}")
    print("\ntitle script mix:")
    for k, v in el.title_script.value_counts().items():
        print(f"    {k:8} {v:,}")


if __name__ == "__main__":
    main()
