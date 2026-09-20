# ViewCastLK — training dataset

Current canonical snapshot: **18 September 2026**, built from the Supabase
warehouse by `ViewCastLK/scripts/build_training_table.py`. It contains
**123,766 rows, one per video**.

| file | size | use |
|---|---|---|
| `viewcastlk_training_table.parquet` | 22.6 MB | **model from this one** |
| `viewcastlk_training_table.csv` | 123 MB | inspection, Excel, non-Python tools |

Prefer the Parquet. CSV has no type information, so every read re-guesses it:
booleans come back as the strings `"True"`/`"False"`, `published_at` as text, and
any integer column containing a missing value silently becomes a float.

```python
import pandas as pd
df = pd.read_parquet("viewcastlk_training_table.parquet")
```

Current build notes and statistics:
[`Dataset latest/HANDOVER_20260918.md`](Dataset%20latest/HANDOVER_20260918.md).

Full statistics: `python Analysis/tools/dataset/dataset_stats.py --horizon 7`

---

## The one rule that must not be broken

**No post-publication engagement may become a feature.** Not views at 24 h, not
an early like count, not a derived ratio. The project's entire claim is
forecasting *before* publication, unlike prior work that consumes observed early
engagement. A single early-engagement feature voids that claim, and it will do it
while making your metrics look excellent.

The builder enforces this — it aborts if an engagement column appears in the
feature list — but it cannot see what you construct downstream. `d7_views` and
friends are **targets only**.

---

## Columns

### Keys and references
`video_id`, `channel_id`, `published_at` (UTC, tz-aware), `title`,
`thumbnail_url`

`thumbnail_url` is a reference, **not a feature** — the string is a CDN path
built from the video id and predicts nothing. It is here so thumbnail features
(faces, text, brightness, colour) can be derived from the actual image. Two
caveats: YouTube serves a *replaced* thumbnail from the same URL, so an image
downloaded today may not be the one published with the video, and that is also
why thumbnail changes cannot be detected at all.

### Features — all knowable before publication

**Content** — `category_id`, `category_name`, `duration_seconds`, `is_short`,
`is_short_source`, `is_vertical`, `definition`, `caption`, `made_for_kids`,
`default_audio_language`, `default_language`. `is_short` uses player shape plus
the current three-minute Shorts limit, with the old duration rule only as a
fallback when player shape is unavailable.

**Timing**, converted to Asia/Colombo because posting time matters locally —
`publish_hour_slt`, `publish_dow_slt` (0 = Monday), `publish_is_weekend`,
`publish_hour_sin/cos`, `publish_dow_sin/cos`

The sin/cos pairs exist so hour 23 sits next to hour 0. Use the cyclical pair
*or* the raw integer, not both.

**Title** — `title_length`, `title_word_count`, `title_has_number`,
`title_has_question`, `title_has_exclaim`, `title_upper_ratio`, `title_script`

`title_script` is the **alphabet**, not the language — an English title and a
romanised Sinhala one ("Man Adarei") are both `latin_script`.

**Text** — `tags` (pipe-separated, raw), `tag_count`, `description_length`

Description text is deliberately not exported, only its length: ~39 MB across
the corpus, and judged not to drive views.

**Channel, point-in-time** — `ch_subs_at_publish`, `ch_views_at_publish`,
`ch_videos_at_publish`, `channel_age_days_at_publish`, `channel_country`,
`topic_categories`

Taken from the newest channel snapshot **at or before** publication, never
today's value. Using a current subscriber count to predict a three-week-old
video leaks the outcome into the feature — a video that did well grew the
channel.

### Targets
Per horizon `h` ∈ {7, 14, 21, 30}: `dh_views`, `dh_likes`, `dh_comments`,
`dh_hours_off`, `dh_usable`

"Day 7" is the observation nearest 168 h after *that video's own*
`published_at` — not the seventh row. Polls run roughly six-hourly and drift.
`dh_hours_off` is the signed error in hours; `dh_usable` is true when it is
within ±12 h and a value exists. **Always filter on `dh_usable`** — a non-null
`d7_views` 40 hours off the mark is not a day-7 observation.

### Metadata — filters, not features
`eligible`, `is_live_broadcast`, `channel_stats_backfilled`, `ch_stats_as_of`,
`title_changed`, `description_changed`

---

## Shape

119,415 eligible rows across **2,561 channels**, published through
18 September 2026.

| horizon | usable | % of eligible | median offset |
|---|---|---|---|
| day 7 | 77,399 | 64.8% | 1.5 h |
| day 14 | 73,057 | 61.2% | 1.5 h |
| day 21 | 72,704 | 60.9% | 1.5 h |
| day 30 | 73,678 | 61.7% | 1.5 h |

---

## What will bite you

**1. Train and evaluate each horizon separately.**

Labels now overlap substantially: 43,765 videos have usable labels at all four
horizons. Separate exports are still required because each horizon has a
different target, eligible population, and observation window.

**2. Backfilled channel statistics are not point-in-time features.**

| horizon | usable | + true point-in-time channel stats |
|---|---:|---:|
| day 7 | 77,399 | **70,568** |
| day 14 | 73,057 | **61,254** |
| day 21 | 72,704 | **51,016** |
| day 30 | 73,678 | **37,994** |

Rows marked `channel_stats_backfilled = True` used the earliest available
snapshot because none predates the video. Exclude them when making the strict
pre-publication claim. There are now enough clean rows to do this at every
horizon; 37,033 videos have clean point-in-time statistics and usable labels at
all four horizons.

**3. The target is extremely heavy-tailed.**

Day-7 views have a median of 1,197, a 75th percentile of 6,068, and a 99th
percentile of 340,256. Fit on `log1p(views)`; RMSE on raw counts is dominated by
rare viral videos.

**4. Channel identity is a major source of leakage.**

Split by **channel**, not at random, or the same channel's videos land on both
sides and the model memorises channels instead of learning about videos. The
baseline to beat is a per-channel median computed from training rows only.

**5. Publication history is still short.** Avoid month or week-of-year
seasonality features until the collection window is much longer.

**6. Missing tags and languages are informative.** Encode missingness rather
than dropping rows.

**7. `title_changed` and `description_changed` mark post-publication edits.** A
change after a video took off may be a reaction to performance. Exclude changed
titles when using title features.

---

## Suggested starting point

```python
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_parquet("viewcastlk_training_table.parquet")

HORIZON = 7                                     # one model per horizon
d = df[df.eligible & df[f"d{HORIZON}_usable"]].copy()
d = d[~d.channel_stats_backfilled]              # strict point-in-time features at every horizon
d["y"] = np.log1p(d[f"d{HORIZON}_views"])

# group split — the same channel must not appear on both sides
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
              .split(d, groups=d.channel_id))
train, test = d.iloc[tr], d.iloc[te]

# baseline to beat, computed from TRAIN only
ch_median = train.groupby("channel_id").y.median()
baseline = test.channel_id.map(ch_median).fillna(train.y.median())
```

A model that cannot beat that baseline has learned nothing about the video
itself.

---

## Rebuilding

```bash
python scripts/build_training_table.py --out ../Dataset/viewcastlk_training_table.csv
```

Needs `SUPABASE_BACKUP_DB_URL` (session pooler, port 5432) in `.env`; writes
both CSV and Parquet; `--tolerance` changes the ±12 h window. Close the CSV in
Excel first — an open file makes the write fail at the very end.

Coverage grows daily as videos reach their horizons, so rebuild before a final
training run rather than reusing this snapshot.
