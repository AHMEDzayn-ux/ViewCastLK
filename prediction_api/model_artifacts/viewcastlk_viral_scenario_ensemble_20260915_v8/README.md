# ViewCastLK two-scenario forecast

Returns one normal trajectory, one calibrated breakout probability,
and one conditional viral-upside trajectory.

```text
python predict.py --input sample_input.csv --output predictions.csv
```

The viral-upside value is conditional on a breakout; it is not a
second equally likely point prediction.
