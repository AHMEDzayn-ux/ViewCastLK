# ViewCastLK — Training Table Data Dictionary

Produced by `scripts/build_training_table.py`. **One row per video.**
Regenerate at any time; labels fill in as videos age.

```bash
python scripts/build_training_table.py --out training_data/viewcastlk_training_table.csv
```

---

## Read this first — three rules baked into the file

**1. A row contains only what was knowable *before* the video was published.**
No view, like or comment value appears as a feature. They exist only as
prediction targets. This is the project's core claim — forecasting from
pre-publication metadata, unlike prior work that consumes observed early
engagement. The build aborts if an engagement column reaches the feature list.

**2. "Day N" means elapsed time, not the Nth snapshot.**
Collection runs roughly every six hours and drifts, so the 7th row is not day 7.
Each label is taken from the snapshot nearest 168/336/504/720 hours after *that
video's own* `published_at`. `d{N}_hours_off` records how far off it landed;
`d{N}_usable` is false when that gap exceeds ±12 h, which is what happens when a
collection outage swallowed the target moment.

**3. Channel statistics are point-in-time.**
`ch_*_at_publish` come from the newest channel snapshot at or before the video
was published — never today's value. Using the current subscriber count would
leak the outcome, because a video that performed well grew its own channel.

---

## Keys

| Column | Type | Meaning |
|---|---|---|
| `video_id` | str | YouTube video ID. Unique — one row per video. |
| `channel_id` | str | YouTube channel ID. Join key for channel-level grouping. |
| `published_at` | timestamptz (UTC) | When the video went live. All ages derive from this. |
| `title` | str | Raw title. Kept for inspection; use the derived `title_*` features for modelling. |

## Features — content

| Column | Type | Meaning |
|---|---|---|
| `category_id` | int | YouTube numeric category (25 = News & Politics, 24 = Entertainment, …). |
| `category_name` | str | Human-readable category. |
| `duration_seconds` | float | Video length in seconds. Null for live/unfinished videos. |
| `is_short` | bool | `duration_seconds <= 60`. Shorts behave differently; consider separate models. |
| `definition` | str | `hd` or `sd`. |
| `caption` | str | `"true"` / `"false"` — whether captions are available. |
| `made_for_kids` | bool | Creator-declared. **Only 0.27 % are true — near-useless as a feature.** |
| `default_audio_language` | str | Creator-declared. **Unreliable** — Sinhala videos are often tagged `en`. Prefer `title_script`. |
| `default_language` | str | Creator-declared metadata language. Same unreliability. |

## Features — publish timing (Sri Lanka time, UTC+5:30)

| Column | Type | Meaning |
|---|---|---|
| `publish_hour_slt` | int 0–23 | Hour of day in **Asia/Colombo**, not UTC. |
| `publish_dow_slt` | int 0–6 | Day of week, **0 = Monday**, 6 = Sunday. |
| `publish_is_weekend` | bool | Saturday or Sunday in Sri Lanka time. |
| `publish_hour_sin` / `publish_hour_cos` | float | Cyclical encoding of hour, so 23:00 and 00:00 are adjacent rather than maximally distant. **Use these instead of raw hour in tree models that split linearly.** |
| `publish_dow_sin` / `publish_dow_cos` | float | Cyclical encoding of day of week. |

## Features — text

| Column | Type | Meaning |
|---|---|---|
| `title_length` | int | Characters in the title. |
| `title_word_count` | int | Whitespace-separated tokens. |
| `title_has_number` | bool | Title contains a digit. |
| `title_has_question` | bool | Title contains `?`. |
| `title_has_exclaim` | bool | Title contains `!`. |
| `title_upper_ratio` | float 0–1 | Share of uppercase characters — a rough shouting/clickbait proxy. |
| `title_script` | str | `sinhala` / `tamil` / `latin` / `mixed`, detected from Unicode ranges. **This is the trustworthy language signal**, unlike `default_audio_language`. |
| `description_length` | int | Characters in the description. |
| `tag_count` | int | Number of creator tags. |

## Features — channel, as of publish time

| Column | Type | Meaning |
|---|---|---|
| `ch_subs_at_publish` | int | Subscriber count at publication. |
| `ch_views_at_publish` | int | Channel lifetime views at publication. |
| `ch_videos_at_publish` | int | Channel video count at publication. |
| `channel_age_days_at_publish` | float | Days between channel creation and this video. |
| `channel_country` | str | Declared channel country. Every tracked channel is verified `LK`. |
| `topic_categories` | str | Pipe-separated Wikipedia topic URLs from the API. Sparse; optional. |

## Labels — prediction targets

For each horizon **N ∈ {7, 14, 21, 30}**:

| Column | Type | Meaning |
|---|---|---|
| `d{N}_views` | int | **Primary target.** Cumulative views at ≈ N days. |
| `d{N}_likes` | int | Secondary target. |
| `d{N}_comments` | int | Secondary target. |
| `d{N}_hours_off` | float | Signed hours between the chosen snapshot and the exact mark. Negative = early. |
| `d{N}_usable` | bool | **Filter on this.** True when `\|hours_off\| ≤ 12` and a view count exists. |

## Metadata

| Column | Type | Meaning |
|---|---|---|
| `eligible` | bool | **Filter on this.** False for live broadcasts and videos with no parseable duration. |
| `is_live_broadcast` | bool | Duration reported as `P0D` — live/unfinished, excluded from `eligible`. |
| `channel_stats_backfilled` | bool | True when no channel snapshot predated the video (published before tracking began) and the earliest available snapshot was substituted. **Mildly leaky — drop or down-weight these.** |
| `ch_stats_as_of` | timestamptz | When the channel statistics used in this row were actually captured. |

---

## Recommended usage

```python
import pandas as pd
df = pd.read_csv("training_data/viewcastlk_training_table.csv")

# day-7 modelling set
d7 = df[df.eligible & df.d7_usable & ~df.channel_stats_backfilled]

y = d7.d7_views
X = d7.drop(columns=[c for c in d7.columns if c.startswith("d7_")
                     or c.startswith("d14_") or c.startswith("d21_")
                     or c.startswith("d30_")])
```

**Split by time, not randomly.** Videos from the same channel published hours
apart are highly correlated; a random split leaks channel behaviour across the
boundary and flatters the score. Split on `published_at`.

**Log-transform the target.** `d7_views` spans 0 to 2.4 million with a median of
1,242 — the distribution is extremely heavy-tailed.

**Evaluate per category.** Day-7 medians differ ~7× between categories
(News & Politics ≈ 640, Entertainment ≈ 4,293), so a single global MAPE mostly
measures category mix rather than model quality.

**Baseline to beat:** the naive category-average growth curve — predict each
category's median day-N views. The model must beat this or it adds nothing.

---

## Current state and known gaps

| Horizon | Usable rows | Median offset |
|---|---|---|
| Day 7 | 3,168 | 1.4 h |
| Day 14 | fills from 30 Jul 2026 | — |
| Day 21 | fills from ~6 Aug 2026 | — |
| Day 30 | fills from ~15 Aug 2026 | — |

- 8,497 eligible videos of 8,796 collected; 894 have backfilled channel stats.
- Shorts 2,065 / long-form 6,432.
- Title script: Sinhala 7,246 · Latin 1,042 · Tamil 206 · mixed 3.
- **Not yet included:** channel historical performance (mean views of the
  channel's earlier videos). It requires only outcomes that had already matured
  before the target video was published, which the current 11-day window barely
  supports. Worth adding once collection is deeper.
