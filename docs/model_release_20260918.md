# Model release: 18 September 2026 dataset

## Release

- Checkpoint: `checkpoint27_reconciled_latest_clean_20260918`
- Portable artifact: `viewcastlk_reconciled_latest_clean_20260918_v9.zip`
- Dataset SHA-256: `aa340358e82c4764d09f04ff1d8e1b0fc84cf8441c95e4cc5319ca5137258433`
- Artifact ZIP SHA-256: `03603476d31be12c26991956368dd837697a4a78b7ff9dda103d0eb833e6b037`
- Status: promoted after development OOF selection and reserved-channel release gate

The release trains an independent regressor for each horizon using every clean
point-in-time label available, then applies log-space isotonic reconciliation.
This guarantees `day 7 <= day 14 <= day 21 <= day 30` without allowing an early
overprediction to force every later forecast upward.

## Training rows

Rows with post-publication/backfilled channel statistics are excluded.

| horizon | clean rows |
|---|---:|
| day 7 | 69,477 |
| day 14 | 60,343 |
| day 21 | 50,226 |
| day 30 | 37,444 |

## Reconciliation selection

The reconciliation rule was selected from five-fold channel-grouped development
OOF predictions. Reserved channels were not used for selection.

| method | development mean RMSLE | monotonic |
|---|---:|---|
| Raw independent | 1.674943 | no |
| Forward cumulative maximum | 1.682896 | yes |
| Backward cumulative minimum | 1.670643 | yes |
| **Log-space isotonic** | **1.663485** | **yes** |

## Reserved-channel release gate

The common reserved partition contains 7,202 complete, naturally monotone
trajectories from 439 held-out channels.

| horizon | aligned clean baseline RMSLE | promoted RMSLE |
|---|---:|---:|
| day 7 | 1.665156 | **1.613746** |
| day 14 | 1.644738 | **1.612358** |
| day 21 | 1.635817 | **1.629868** |
| day 30 | 1.634941 | **1.629770** |
| **mean** | **1.645163** | **1.621435** |

Mean RMSLE improved by **1.44%** over the clean aligned baseline. The release
gate also checks finite, nonnegative, nondecreasing predictions and complete
development/reserved channel separation.

## Serving contract change

The forecast request now includes `isShort`. The dashboard asks explicitly for
YouTube Short versus standard video because duration alone cannot recover the
corrected dataset label. Shorts longer than three minutes are rejected.

## Validation

- Python: 80 tests passed, 5 expected skips.
- Dashboard: ESLint passed.
- Dashboard: optimized production build and TypeScript checks passed.
- Portable prediction CLI: checksum and monotonic smoke test passed.
