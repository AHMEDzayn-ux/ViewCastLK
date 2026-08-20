# ViewCastLK monotonic trajectory — experimental artifact

This artifact predicts cumulative view totals for days 7, 14, 21,
and 30 in one call. Its day-7 base plus nonnegative growth
parameterization guarantees a nondecreasing trajectory.

## Status

Experimental only. No video in the frozen training dataset has all
four labels, so end-to-end day-30 accuracy is not measurable yet.
The included experimental channel holdout has already been evaluated.

## Usage

```text
python -m pip install -r requirements.txt
python predict.py --input sample_input.csv --output predictions.csv
```

The output adds predicted_day_7_views, predicted_day_14_views,
predicted_day_21_views, and predicted_day_30_views. See manifest.json
and evaluation/ for the limitations and held-out results.

Source checkpoint: checkpoint12_monotonic_trajectory
