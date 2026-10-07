# ViewCastLK monotonic trajectory — experimental artifact

This artifact predicts cumulative view totals for days 7, 14, 21,
and 30 in one call. Its reconciliation strategy guarantees a
nondecreasing trajectory.

## Status

The model has been evaluated end to end through day 30 on a
channel-grouped holdout with complete four-horizon labels.
The included channel holdout has already been evaluated.

## Usage

```text
python -m pip install -r requirements.txt
python predict.py --input sample_input.csv --output predictions.csv
```

The output adds predicted_day_7_views, predicted_day_14_views,
predicted_day_21_views, and predicted_day_30_views. See manifest.json
and evaluation/ for the limitations and held-out results.

Source checkpoint: checkpoint30_reconciled_latest_clean_20261001
