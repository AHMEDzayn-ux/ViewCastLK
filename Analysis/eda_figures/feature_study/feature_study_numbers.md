# Feature study in numbers

Log-MAE is mean absolute error on log1p(views); 0.69 is a factor of 2. Date-based split: last 20% of each horizon's rows by publication time.

## Splits

| horizon | train | test | test published from | test rows from unseen channels |
|---|---|---|---|---|
| day 7 | 55,560 | 13,891 | 28 Aug | 1% |
| day 14 | 52,060 | 13,016 | 22 Aug | 1% |
| day 21 | 52,041 | 13,011 | 15 Aug | 1% |
| day 30 | 52,856 | 13,215 | 06 Aug | 1% |

## S1 Serving reality

| horizon | model | scored with | log-MAE | within 2× | MedAPE | R² (log) |
|---|---|---|---|---|---|---|
| day 7 | deployed inputs | training-style inputs | 1.188 | 40.3% | 70% | 0.632 |
| day 7 | deployed inputs | as served, day and hour given | 1.241 | 37.4% | 74% | 0.614 |
| day 7 | deployed inputs | as served, day and hour blank | 1.253 | 36.4% | 76% | 0.610 |
| day 7 | deployed without topics | training-style inputs | 1.205 | 39.5% | 70% | 0.623 |
| day 7 | serve-safe base | as served, day and hour given | 1.197 | 39.7% | 70% | 0.628 |
| day 7 | serve-safe base | as served, day and hour blank | 1.207 | 39.1% | 72% | 0.626 |
| day 14 | deployed inputs | training-style inputs | 1.198 | 38.2% | 75% | 0.617 |
| day 14 | deployed inputs | as served, day and hour given | 1.302 | 33.5% | 82% | 0.574 |
| day 14 | deployed inputs | as served, day and hour blank | 1.325 | 31.9% | 84% | 0.566 |
| day 14 | deployed without topics | training-style inputs | 1.212 | 38.0% | 76% | 0.609 |
| day 14 | serve-safe base | as served, day and hour given | 1.214 | 38.0% | 76% | 0.610 |
| day 14 | serve-safe base | as served, day and hour blank | 1.236 | 36.9% | 77% | 0.600 |
| day 21 | deployed inputs | training-style inputs | 1.184 | 37.6% | 78% | 0.639 |
| day 21 | deployed inputs | as served, day and hour given | 1.272 | 33.4% | 79% | 0.602 |
| day 21 | deployed inputs | as served, day and hour blank | 1.294 | 32.4% | 80% | 0.590 |
| day 21 | deployed without topics | training-style inputs | 1.209 | 37.1% | 79% | 0.627 |
| day 21 | serve-safe base | as served, day and hour given | 1.190 | 38.0% | 77% | 0.634 |
| day 21 | serve-safe base | as served, day and hour blank | 1.215 | 36.7% | 79% | 0.623 |
| day 30 | deployed inputs | training-style inputs | 1.234 | 35.8% | 81% | 0.626 |
| day 30 | deployed inputs | as served, day and hour given | 1.401 | 28.8% | 87% | 0.548 |
| day 30 | deployed inputs | as served, day and hour blank | 1.420 | 28.3% | 88% | 0.538 |
| day 30 | deployed without topics | training-style inputs | 1.247 | 35.2% | 82% | 0.617 |
| day 30 | serve-safe base | as served, day and hour given | 1.253 | 35.0% | 83% | 0.616 |
| day 30 | serve-safe base | as served, day and hour blank | 1.276 | 33.9% | 83% | 0.606 |

## Seed noise (serve-safe base)

| horizon | log-MAE mean | sd | 2 sd |
|---|---|---|---|
| day 7 | 1.200 | 0.0027 | 0.0055 |
| day 14 | 1.215 | 0.0023 | 0.0046 |
| day 21 | 1.193 | 0.0051 | 0.0101 |
| day 30 | 1.249 | 0.0047 | 0.0095 |

## S3 Ablation from the serve-safe base (Δ log-MAE; positive means the group helps)

| group | day 7 | day 14 | day 21 | day 30 |
|---|---|---|---|---|
| channel statistics | +0.454 * | +0.436 * | +0.439 * | +0.377 * |
| category | +0.019 * | +0.011 * | +0.009 | +0.015 * |
| duration and format | +0.103 * | +0.124 * | +0.110 * | +0.123 * |
| publish day and time | +0.017 * | +0.013 * | +0.007 | +0.009 |
| language | +0.011 * | +0.007 * | +0.003 | +0.008 |

## S4 Add-backs to the serve-safe base (Δ log-MAE; negative means it helps)

| group | needs at serving | day 7 | day 14 | day 21 | day 30 |
|---|---|---|---|---|---|
| is_short derived from duration <= 60s | nothing new: duration is on the form | +0.003 | -0.000 | +0.001 | -0.008 |
| is_short derived from duration <= 180s | nothing new: duration is on the form | -0.000 | -0.002 | +0.001 | +0.006 |
| title surface features | derived from the title the creator already types | -0.003 | -0.008 * | -0.018 * | -0.031 * |
| exact hour and weekday | the optional day/hour already on the form | -0.002 | -0.008 * | -0.001 | -0.018 * |
| tags and description | new form fields | -0.012 * | -0.024 * | -0.032 * | -0.035 * |
| posting cadence | channel's recent upload times (free RSS feed) plus planned time | +0.006 * | -0.004 | +0.001 | -0.020 * |
| channel's recent record | views of the channel's recent uploads (about 2 quota units) | -0.021 * | -0.014 * | -0.010 * | -0.020 * |
| all candidates together | all of the above | -0.046 * | -0.052 * | -0.049 * | -0.067 * |

`*` exceeds two standard deviations of seed-to-seed noise at that horizon.

## S5 Post-publication channel stats (scored on clean test rows)

| horizon | variant | log-MAE | within 2× | n |
|---|---|---|---|---|
| day 7 | trained on all rows | 1.197 | 39.7% | 13,891 |
| day 7 | trained without backfilled rows | 1.210 | 39.2% | 13,891 |
| day 14 | trained on all rows | 1.214 | 38.0% | 13,016 |
| day 14 | trained without backfilled rows | 1.230 | 37.6% | 13,016 |
| day 21 | trained on all rows | 1.190 | 38.0% | 13,011 |
| day 21 | trained without backfilled rows | 1.232 | 36.8% | 13,011 |
| day 30 | trained on all rows | 1.253 | 35.0% | 13,215 |
| day 30 | trained without backfilled rows | 1.447 | 30.0% | 13,215 |

## Known versus new channels (serve-safe base)

| horizon | subset | log-MAE | within 2× | n |
|---|---|---|---|---|
| day 7 | channels seen in training | 1.196 | 39.6% | 13,683 |
| day 7 | channels unseen in training | 1.213 | 45.2% | 208 |
| day 14 | channels seen in training | 1.209 | 38.2% | 12,894 |
| day 14 | channels unseen in training | 1.670 | 23.0% | 122 |
| day 21 | channels seen in training | 1.183 | 38.2% | 12,866 |
| day 21 | channels unseen in training | 1.829 | 19.3% | 145 |
| day 30 | channels seen in training | 1.249 | 35.2% | 13,094 |
| day 30 | channels unseen in training | 1.684 | 18.2% | 121 |

## Gemini title scores (indicative: 4,568 scored day-7 rows, channel-grouped 5-fold)

| variant | log-MAE mean | sd |
|---|---|---|
| with Gemini scores | 1.570 | 0.056 |
| without scores | 1.624 | 0.058 |
| paired difference | -0.054 | t=-6.06, p=0.004 |

## Redundant pairs (|Spearman| ≥ 0.7)

| a | b | ρ |
|---|---|---|
| log_gap_prev_h | log_uploads_prev_24h | -0.86 |
| ch_videos_at_publish | log_uploads_prev_24h | +0.85 |
| title_length | title_word_count | +0.83 |
| ch_subs_at_publish | ch_videos_at_publish | +0.82 |
| ch_videos_at_publish | log_gap_prev_h | -0.71 |
| ch_subs_at_publish | log_uploads_prev_24h | +0.70 |

## Dominant value share (near-constant columns)

| column | top value share |
|---|---|
| is_live_broadcast | 100.0% |
| channel_country | 99.9% |
| caption | 99.8% |
| made_for_kids | 99.5% |
| definition | 96.5% |
| title_has_question | 93.9% |
| title_has_exclaim | 92.9% |
| publish_is_weekend | 72.0% |
| is_short | 59.2% |
