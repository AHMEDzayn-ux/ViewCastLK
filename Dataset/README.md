# ViewCastLK — training dataset

Built 12 August 2026 from the Supabase warehouse by
`ViewCastLK/scripts/build_training_table.py`. **64,515 rows, one per video.**

| file | size | use |
|---|---|---|
| `viewcastlk_training_table.parquet` | 10.4 MB | **model from this one** |
| `viewcastlk_training_table.csv` | 60.3 MB | inspection, Excel, non-Python tools |

Prefer the Parquet. CSV has no type information, so every read re-guesses it:
booleans come back as the strings `"True"`/`"False"`, `published_at` as text, and
any integer column containing a missing value silently becomes a float.

```python
import pandas as pd
df = pd.read_parquet("viewcastlk_training_table.parquet")
```

Full statistics: `python Analysis/dataset_stats.py --horizon 7`

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

**Content** — `category_id`, `category_name`, `duration_seconds`, `is_short`
(≤60 s), `definition`, `caption`, `made_for_kids`, `default_audio_language`,
`default_language`

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

63,126 eligible rows (1,389 live broadcasts and unparseable durations removed)
across **2,416 channels**, published 2 July – 11 Aug 2026 (40 days).
Median 7 videos per channel.

| horizon | usable | % of eligible | median offset |
|---|---|---|---|
| day 7 | 20,663 | 32.7% | 1.4 h |
| day 14 | 15,685 | 24.8% | 1.4 h |
| day 21 | 15,100 | 23.9% | 1.4 h |
| day 30 | 14,753 | 23.4% | 1.4 h |

---

## What will bite you

**1. Each horizon is a separate dataset — no video has all four.**

Collection began 17 July, so old videos are past their early marks and recent
ones have not reached their late ones. Coverage by publication week:

| published | videos | d7 | d14 | d21 | d30 |
|---|---|---|---|---|---|
| 29 Jun | 5,744 | 0 | 0 | 0 | 4,838 |
| 6 Jul | 10,930 | 0 | 0 | 1,627 | 9,930 |
| 13 Jul | 11,478 | 1,463 | 3,054 | 11,457 | 0 |
| 20 Jul | 10,917 | 6,539 | 10,817 | 2,086 | 0 |
| 27 Jul | 11,223 | 11,089 | 2,089 | 0 | 0 |
| 3 Aug | 11,448 | 2,235 | 0 | 0 | 0 |

**Train one model per horizon.** A multi-output model has nothing to learn from.

The empty cells are permanent. Every archived snapshot (17 July – 9 August,
2,148,774 rows on Google Drive) was re-processed on 12 August and produced
**zero** additional labels: the observations behind those cells were never
taken, because most of those channels only joined the roster in the August
expansion and their videos arrived through backfill with the early history
already missing. Coverage grows forward from here, never backward.

**2. Usable rows shrink sharply once you demand clean channel stats.**

| horizon | usable | + true point-in-time channel stats | + unedited title |
|---|---|---|---|
| day 7 | 20,663 | **13,832** | 13,683 |
| day 14 | 15,685 | **8,074** | 8,008 |
| day 21 | 15,100 | **2,571** | 2,558 |
| day 30 | 14,753 | **0** | 0 |

58.4% of eligible rows have `channel_stats_backfilled = True`: no channel
snapshot predates the video, so the earliest available one was substituted.
Because it is measured *after* publication it is mildly contaminated.

For **day 21 and day 30 you have no choice** — every labelled video there
predates channel tracking. Use the backfilled stats and say so, or restrict the
project's headline claim to day 7 and day 14.

**3. The target is extremely heavy-tailed.**

Day-7 views: median 1,146, mean 19,326 (17× the median), p99 303,539, max
17.4 M. The top 1% of videos hold **44.8%** of all day-7 views. Fit on
`log1p(views)` — on that scale skew is −0.09, near symmetric. RMSE on raw counts
is close to meaningless.

**4. Channel identity dominates, and channels are wildly unequal.**

The top 10 channels hold **25.7%** of all rows; the largest posts ~150 clips a
day. Split by **channel**, not at random, or the same channel's videos land on
both sides and the model memorises channels instead of learning about videos.

Subscribers correlate 0.423 with log views — only ~18% of variance — and the
medians saturate past 10K subscribers (10K–100K: 1,511; 1M+: 1,851). Channel
size is the strongest *simple* predictor, not a sufficient one. The baseline to
beat is a **per-channel median**.

**5. Only 40 days of publication history.** No seasonality is learnable; month
or week-of-year features would be noise.

**6. 29.9% of videos have no tags**, 13.3% no `default_audio_language`. Null
means the uploader set none — informative, so encode as a category rather than
dropping rows.

**7. `title_changed` (392) and `description_changed` (494)** mark videos edited
since first seen. Small, but a title edited after a video took off may be a
*reaction* to performance. Consider excluding them when using title features.

---

## Suggested starting point

```python
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_parquet("viewcastlk_training_table.parquet")

HORIZON = 7                                     # one model per horizon
d = df[df.eligible & df[f"d{HORIZON}_usable"]].copy()
d = d[~d.channel_stats_backfilled]              # drop for d7/d14; impossible for d21/d30
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
