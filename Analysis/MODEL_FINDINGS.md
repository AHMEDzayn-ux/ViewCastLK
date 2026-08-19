# ViewCastLK — modelling findings

Measured 18 August 2026 on the 17 August dataset build: 73,809 videos,
71,930 eligible, 2,446 channels, published 2 July – 17 August 2026.

Everything here is reproducible from four scripts in this folder. Every figure
is a **mean over five splits**, because a single grouped split on this data
swings by ±0.1 R² depending only on which channels land in the test set — a
single number would be noise wearing a decimal point.

```bash
python Analysis/train_baseline.py                        # R², MAE, both regimes
python Analysis/evaluate_srs_metrics.py                  # MAPE and the SRS metrics
python Analysis/test_target_binning.py                   # bands vs regression
python Analysis/compare_encodings.py                     # correlations, encoding
```

---

## 1. Headline

**The model works.** Against the naive category-average baseline the SRS
requires it to beat, it is 5–6× more accurate, on every horizon, in both
regimes, on 5 of 5 splits.

| | naive baseline MAPE | model MAPE | p |
|---|---|---|---|
| warm start (channel known) | 1,730% | **300%** | < 0.0001 |
| cold start (channel unseen) | 1,559% | **634%** | 0.006 |

MAPE's absolute value should not be read as "wrong by 300%". It divides by the
true value, and 16% of videos — those under 100 views — contribute **91%** of
the total MAPE. A video that got 3 views and was predicted 960 is wrong by
32,000% on its own. **MedAPE is the figure that describes the typical video:
68% warm start.**

---

## 2. Two regimes, always reported together

| | split | what it answers |
|---|---|---|
| **cold start** | grouped by channel | a creator the system has never seen |
| **warm start** | random | a channel already on the roster |

Quoting one alone misleads in opposite directions. Warm start credits the
features with the channel's contribution; cold start hides that deployment
usually *does* know the channel.

### R² on log1p(views)

| horizon | cold | warm | per-channel median (warm) |
|---|---|---|---|
| day 7 | 0.380 ±0.092 | **0.692** | 0.551 |
| day 14 | 0.371 ±0.112 | **0.689** | 0.548 |
| day 21 | 0.449 ±0.094 | **0.699** | 0.569 |
| day 30 | 0.494 ±0.105 | **0.716** | 0.592 |

The model beats a per-channel median by a consistent **+0.13 R²**. Stability
across four horizons matters, because each is a largely different set of
videos — agreement is corroboration, not the same data reread.

### Proportional error (SRS primary metric)

| horizon | regime | MAPE | MedAPE | sMAPE |
|---|---|---|---|---|
| day 7 | warm | 300% | 68.4% | 81.7% |
| day 14 | warm | 290% | 68.1% | 81.2% |
| day 21 | warm | 315% | 66.3% | 79.6% |
| day 7 | cold | 634% | 93.6% | 113.8% |
| day 14 | cold | 540% | 88.5% | 111.3% |
| day 21 | cold | 483% | 86.1% | 109.5% |
| day 30 | cold | 630% | 87.3% | 108.6% |

Roughly 45 videos per split have exactly zero views, where MAPE is undefined.
They are excluded and counted, never quietly treated as small.

### What the accuracy means in practice

Warm start: median error factor **2.1×**, 65% of forecasts within 3× of
actual, Spearman **0.81**.

Predict 1,000 views and the truth is typically 480–2,100. That is not a view
counter. It **is** a reliable ranker, which is what the product needs:
"this will likely be one of your stronger videos", "19:00 beats 03:00",
"space your uploads".

---

## 3. The finding that changes the architecture

**Channel-history features must be dropped for cold start.** Two models are
needed, not one.

| horizon | cold without history | cold **with** history | cost |
|---|---|---|---|
| day 7 | **0.380** | 0.013 | −0.37 |
| day 14 | **0.371** | 0.143 | −0.23 |
| day 21 | **0.449** | 0.095 | −0.35 |
| day 30 | **0.494** | 0.121 | −0.37 |

`ch_target_enc`, `ch_history_n`, `dur_vs_channel` and `titlelen_vs_channel`
vary freely during training and then collapse to a constant for a channel the
model has never seen — the global median for the encoding, zero for the count.
The model learns to depend on something that will not be there.

This was found by accident: cold start dropped to 0.09 after the engineered
features were added, which looked like a bug and was the result.

---

## 4. Feature set

Target: `log1p(dN_views)`, back-transformed with `expm1` for reporting.

### Warm start — 20 features

| # | feature | type | notes |
|---|---|---|---|
| 1 | `duration_seconds` | numeric | |
| 2 | `title_length` | numeric | |
| 3 | `title_word_count` | numeric | |
| 4 | `title_upper_ratio` | numeric | 0–1 |
| 5 | `tag_count` | numeric | |
| 6 | `description_length` | numeric | |
| 7 | `publish_hour_slt` | numeric | 0–23, Asia/Colombo |
| 8 | `ch_subs_at_publish` | numeric | point-in-time |
| 9 | `ch_views_at_publish` | numeric | point-in-time |
| 10 | `ch_videos_at_publish` | numeric | point-in-time |
| 11 | `channel_age_days_at_publish` | numeric | derived |
| 12 | `gap_prev_h` | numeric | hours since channel's previous upload |
| 13 | `same_ch_24h_before` | numeric | uploads by same channel in prior 24 h |
| 14 | `category_name` | categorical | 14 levels |
| 15 | `title_script` | categorical | 4 levels |
| 16 | `title_has_number` | boolean | |
| 17 | `ch_target_enc` | numeric | **train-fold only** — channel median log-views |
| 18 | `ch_history_n` | numeric | **train-fold only** — how far to trust #17 |
| 19 | `dur_vs_channel` | numeric | **train-fold only** — duration ÷ channel median |
| 20 | `titlelen_vs_channel` | numeric | **train-fold only** — title length ÷ channel median |

**Cold start uses features 1–16 only.**

Optional thumbnail block adds 16 more (`brightness`, `contrast`, `saturation`,
`colourfulness`, `sharpness`, `edge_density`, `text_band_density`, `face_count`,
`has_face`, `face_area_ratio`, `warm_ratio`, `dark_share`, `bright_share`,
`hue_mean`, `img_w`, `img_h`).

### Deliberately excluded

`gap_next_h` was the strongest cadence signal in the EDA and **cannot be
known at prediction time** — the timing of the next upload is unknown when
the forecast is made.

Scored at or near zero and dropped: `is_short`, `publish_dow_slt`,
`publish_is_weekend`, `definition`, `made_for_kids`, `caption`,
`title_has_question`, `title_has_exclaim`, `default_audio_language`,
`channel_country`, `topic_categories`, raw `tags`.

### Leakage control

Features 17–20 are fitted on the **training fold only** and applied to the
test fold. Computing them across the whole frame first lets a channel's test
videos inform its own encoding — inflating warm-start scores substantially
while looking entirely reasonable in the code. SAD FR-76 already requires this.

---

## 5. What the model actually uses

Permutation importance on the real target (drop in R² when shuffled):

| feature | cold | warm |
|---|---|---|
| `ch_views_at_publish` | **0.699** | **0.858** |
| `ch_videos_at_publish` | 0.153 | 0.346 |
| `ch_subs_at_publish` | 0.106 | 0.231 |
| `duration_seconds` | 0.193 | 0.204 |
| `description_length` | 0.015 | 0.101 |
| `channel_age_days_at_publish` | 0.061 | 0.055 |
| `category_name` | **−0.018** | 0.025 |
| `tag_count` | −0.012 | 0.023 |
| `title_upper_ratio` | 0.005 | 0.024 |
| `publish_hour_slt` | 0.003 | 0.023 |

`ch_views_at_publish` — cumulative channel views at publication — is the single
strongest feature in the dataset and was missing from earlier drafts. Adding it
lifted cold-start R² from 0.339 to 0.422.

`category_name` and `tag_count` go **negative** cold start: their apparent
effect is channel-confounded, so on an unseen channel they mislead.

### Correlation ranks these almost backwards

| feature | vs views | vs residual | p (vs views) |
|---|---|---|---|
| `ch_target_enc` | 0.748 | 0.023 | <1e-300 |
| `ch_views_at_publish` | 0.388 | −0.054 | <1e-300 |
| `ch_subs_at_publish` | 0.381 | −0.055 | <1e-300 |
| `same_ch_24h_before` | 0.108 | −0.069 | 1.0e-80 |
| `dur_vs_channel` | 0.060 | 0.088 | 1.6e-25 |
| `title_upper_ratio` | −0.015 | **+0.077** | 0.011 |
| `duration_seconds` | 0.011 | 0.026 | 0.060 |
| `gap_prev_h` | 0.006 | **0.103** | **0.31** |

Three traps live in that table:

* **`duration_seconds` looks dead** (r = 0.011, p = 0.06) and is the second
  strongest feature in the model. Rank correlation assumes monotonicity;
  duration has a sweet spot.
* **`gap_prev_h` correlates 0.006 with views** (not significant) **and 0.103
  with the residual.** Cadence is entirely masked by channel — fast-posting
  channels are also the large ones — until the channel is removed.
* **`title_upper_ratio` flips sign**: −0.015 across channels, +0.077 within.
* Correlations for `category_name` and `title_script` measure the alphabetical
  accident of the ordinal encoding and mean nothing. `compare_encodings.py`
  prints them with a `meaningful` column marking them so, rather than omitting
  them, so nobody re-derives and believes them.

---

## 6. Questions asked and settled

### Does binning the target help? Marginally, and not worth it.

Day 7, five view bands (`<100`, `100-1k`, `1k-10k`, `10k-100k`, `100k+`):

| approach | warm accuracy | within 1 band | cold accuracy |
|---|---|---|---|
| majority-class floor | 32.1% | 79.7% | 27.9% |
| regress, then bin | 61.0% | **96.8%** | 40.5% |
| classify bands directly | **62.8%** | 94.2% | **42.0%** |

Training on bands wins by 1.8 points and costs three things: ranking within a
band disappears, the boundaries freeze into the artefact, and MAPE/R²/MAE/RMSE
stop being computable — and the SRS names MAPE primary. Regress-then-bin also
fails more gracefully: 96.8% of its errors land in an adjacent band against
94.2%, because a regressor missing by a little produces a neighbouring band
while a classifier can jump anywhere.

**Recommendation: keep the regressor, present banded output.**

### Is "will this beat the channel's own average?" predictable? No.

Predicting which quartile of its own channel's output a video lands in:
**37.4% warm, 28.4% cold, against 25% chance.**

The EDA reached the same conclusion from the other direction — predicting the
within-channel residual gives mean R² ≈ 0.00 (sd 0.11) across eight splits.
Two independent methods agree.

**This matters for the dashboard.** "This will be one of your better videos" is
not a claim the data supports. "Videos like this typically get 1k–10k views" is.

### Does ordinal encoding hurt? Almost not at all.

| | ordinal | LGBM native | one-hot |
|---|---|---|---|
| cold | **0.3801** | 0.3763 | 0.3744 |
| warm | 0.6919 | **0.6935** | 0.6929 |

Native beats ordinal by +0.0016 R² warm — significant (p = 0.006, 5/5 splits)
and practically irrelevant. With 14 and 4 levels, and 63 leaves over 800 trees,
the model routes around the false ordering. **Use native in the final artefact
anyway**: it is free and correct.

### Do thumbnails help? Only cold start, and only as a channel-style proxy.

| target | metadata | + thumbnails | gain | p |
|---|---|---|---|---|
| within-channel residual | −0.037 | −0.036 | +0.001 | 0.85 |
| log1p(views), cold start | +0.354 | **+0.375** | +0.021 | 0.054 |

They encode *what kind of channel this is* — a news desk looks different from a
vlog — not what makes a given video succeed. Worth including cold start; no
measurable value warm.

---

## 7. Limits to state in the report

* **46 days of publication history.** No seasonality is learnable. Every
  finding is conditional on July–August 2026 and on Sri Lanka.
* **Cold-start R² has a standard deviation of ±0.10.** Always quote the mean
  over splits.
* **51% of rows carry backfilled channel statistics** — no channel snapshot
  predates the video, so the earliest available one was substituted, which is
  measured after publication and mildly contaminated. Day 30 has only 782 rows
  with genuinely clean point-in-time stats.
* **Day-30 warm start scoring highest is mildly suspicious.** Those are the
  oldest videos, the ones most affected by the point above. Worth re-checking
  on the 782 clean rows.
* **The top 1% of videos hold 42.7% of all day-7 views.** Any metric on raw
  counts is dominated by them; any metric dividing by them is dominated by the
  smallest videos instead. This is why three proportional metrics are reported.

---

## 8. Next

1. **Switch categorical encoding to LightGBM native** in the final artefact.
2. **Train two artefacts per horizon** — warm (20 features) and cold (16) —
   and route on whether the channel has history.
3. **Test explicit interaction terms.** The EDA found nearly every effect flips
   sign by category; LightGBM captures interactions implicitly through splits,
   but `category × duration` and `category × is_short` were never tested
   explicitly.
4. **Re-extract thumbnails before final training** — `fetch_thumbnails.py` is
   resumable and only fetches what is new.
5. **Rebuild the dataset first.** Label coverage grows daily; the 12 August
   build was a third smaller than the 17 August one.
