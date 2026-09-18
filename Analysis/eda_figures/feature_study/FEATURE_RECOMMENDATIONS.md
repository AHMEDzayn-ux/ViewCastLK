# Feature recommendations for the forecasting model

Prepared from the feature study on the 14 September 2026 training table
(`Analysis/build_feature_study.py`, full numbers in `feature_study_numbers.md`).
Everything below was measured, not assumed. It is evidence for the modelling
work, not a replacement for the model's own evaluation.

## How it was measured

* The deployed feature set was rebuilt exactly as `notebooks/01_horizon_isolation.ipynb`
  builds it, and every model used the team's own `HorizonDatasetPreprocessor`,
  `build_xgb_regressor` settings and 3-sigma inlier rule.
* Split by publication date, as SRS FR-75 requires: the last 20% of each
  horizon's rows are the test set (published from 28 Aug for day 7, 22 Aug for
  day 14, 15 Aug for day 21, 6 Aug for day 30). About 55,000 training and
  13,000 test videos per horizon.
* Error is mean absolute error on log views (0.69 means a factor of 2), plus the
  share of videos predicted within 2×.
* Seed-to-seed noise was measured by refitting the same model three times. It is
  0.002 to 0.005, so a change smaller than about 0.01 is not a real difference.

---

## 1. The deployed model is scored on inputs it never receives

The API cannot know a planned video's YouTube topics, and it never receives an
is_short flag. `feature_builder.py` therefore sends every `topic_*` flag as False
with `topic_missing` True, and `is_short` as missing, on every real forecast. In
training those features were present for 99% of rows.

| horizon | deployed model, training-style inputs | deployed model, inputs as actually served | model retrained on served inputs |
|---|---|---|---|
| day 7 | 1.188 (40.3% within 2×) | 1.241 (37.4%) | **1.197 (39.7%)** |
| day 14 | 1.198 (38.2%) | 1.302 (33.5%) | **1.214 (38.0%)** |
| day 21 | 1.184 (37.6%) | 1.272 (33.4%) | **1.190 (38.0%)** |
| day 30 | 1.234 (35.8%) | 1.401 (28.8%) | **1.253 (35.0%)** |

Any accuracy figure computed on the training-style inputs overstates what a
creator gets, by up to 7 percentage points of "within 2×" at day 30. Retraining
on the inputs serving actually has recovers nearly all of it.

The topics were only ever worth a little: removing them from the deployed model
on training-style inputs costs 0.013 to 0.025. Having them in training and
missing at serving costs 0.05 to 0.17.

---

## 2. Recommendations

### Keep

| group | cost of removing it (Δ log-MAE, day 7 / 14 / 21 / 30) | note |
|---|---|---|
| channel statistics (subscribers, video count, channel age, average views per video, subscriber tier) | +0.45 / +0.44 / +0.44 / +0.38 | by far the most important group |
| duration | +0.10 / +0.12 / +0.11 / +0.12 | second most important |
| category | +0.02 / +0.01 / +0.01 / +0.02 | small, above noise at 7, 14 and 30 |
| publish day and time bucket | +0.02 / +0.01 / +0.01 / +0.01 | small, above noise at 7 and 14; optional on the form, which is fine |
| language | +0.01 / +0.01 / 0.00 / +0.01 | small, above noise at 7 and 14; see the fix below |

### Drop

| feature | evidence |
|---|---|
| all 19 `topic_*` flags and `topic_missing` | never available for a planned video; worth 0.013 to 0.025 when present, cost 0.05 to 0.17 as served |
| `is_live_broadcast`, `channel_country`, `caption`, `made_for_kids` | one value covers 100.0%, 99.9%, 99.8% and 99.5% of rows. Already excluded; keep them excluded. |

### Fix

* **Short or long-form: replace `is_short`, do not drop it.** The old flag was
  just `duration_seconds <= 60`, so it added nothing the tree could not already
  read from duration (Δ within noise at every horizon). That was a flaw in the
  flag, not a sign the format does not matter. Split by channel size, Shorts get
  6 to 9 times the day-7 views of long-form for channels under 10K subscribers,
  and only lose at 1M+ (`Analysis/eda_figures/simple_features/SF7_shorts_by_channel_size.png`).
  The ≤60 s rule also misses Shorts of 61 s to 3 min, allowed since October 2024.
  From 18 September the training table's `is_short` is the real format:
  vertical or square player and at most 3 minutes (`video_shapes` table,
  `is_short_source` says whether shape or the old rule decided it). At serving
  the creator says which they are making, so the form needs a
  **Short / long-form** field and `feature_builder.py` must send it instead of
  leaving `is_short` missing.
* **Language.** Training uses the video's `default_language` metadata, which has
  43 values including `en-US` and `en-GB`. Serving sends the creator's audio
  language mapped to `en`, `si`, `ta` or missing. Train on the same mapping the
  API applies, so the model learns the input it will see.
* **Train on served inputs.** Whatever the final feature list, fit on a frame
  built the way `feature_builder.py` builds it. That single change is worth more
  than any feature added below.

### Add, in order of value for effort

| candidate | gain (Δ log-MAE, day 7 / 14 / 21 / 30; negative helps) | what serving needs |
|---|---|---|
| **channel's recent record**: median day-7 log views of its last 10 uploads that were at least 7 days old | −0.021 / −0.014 / −0.010 / −0.020, above noise at every horizon | recent uploads' view counts, about 2 quota units per forecast |
| **title surface features**: length, has number, question, exclamation, upper-case ratio, script | −0.003 / −0.008 / −0.018 / −0.031, growing with horizon | nothing new, derived from the title already typed |
| **tags and description**: tag count, description length | −0.012 / −0.024 / −0.032 / −0.035, the largest single gain | two new form fields |
| exact hour and weekday | −0.002 / −0.008 / −0.001 / −0.018, inconsistent | nothing new |
| all candidates together | −0.046 / −0.052 / −0.049 / −0.067 | all of the above |

Do not add **posting cadence** (gap since the previous upload, uploads in the last
24 h): it hurts day 7 (+0.006), helps only day 30, and is 0.85 correlated with the
channel's video count, so it mostly repeats channel size.

If title features are added, keep `title_length` and drop `title_word_count`
(Spearman 0.83).

Caveat on the channel's recent record: training used each earlier upload's
day-7 views, but serving would read their current views. Check the two agree
closely enough before relying on it. It overlaps with average views per video
(Spearman 0.65) but still adds on top of it, because it reflects the channel's
recent uploads rather than its lifetime average.

---

## 3. Data decisions

**Keep rows whose channel statistics were recorded after publication.** The
training table already masks their average-views feature. Dropping the rows
entirely made every horizon worse on clean test videos, and day 30 much worse:

| horizon | trained on all rows | trained without them | share of training rows they make up |
|---|---|---|---|
| day 7 | 1.197 | 1.210 | 12% |
| day 14 | 1.214 | 1.230 | 23% |
| day 21 | 1.190 | 1.232 | 41% |
| day 30 | 1.253 | 1.447 | 67% |

They are every video published before 13 July, 87% of the week from 13 July,
about half of those from 20 July to 2 August, and none after. Day 30 training depends on that early period, so
removing them removes most of its data. No test video is affected.

**Complete the Gemini title-score backfill before deciding.** Only 8% of videos
have scores (9,323). On the 4,568 scored day-7 videos, adding the four scores
reduced error from 1.624 to 1.570 across five channel-grouped folds (paired
t = −6.06, p = 0.004). That is promising, but the subset is small and older than
the rest, so treat it as a reason to finish scoring, not yet as a result.
`title_urgency` and `title_seriousness` correlate at 0.64.

---

## 4. Limits of this study

* **Mostly known channels.** A date split keeps the same channels on both sides,
  and only 1% of test videos come from unseen channels. On those few, "within
  2×" fell to 18 to 23% at days 14 to 30 (about 120 to 145 videos each). New
  creators need their own evaluation with channel-grouped splits.
* **One date split per horizon.** Seed noise is measured; split-to-split
  variation is not.
* **Tags and description at serving** would be what the creator plans, which may
  differ from what they finally publish.
* **Not the SRS ablation.** This supplies the evidence FR-80 asks for, but the
  official ablation and metrics report belong with the released model.
