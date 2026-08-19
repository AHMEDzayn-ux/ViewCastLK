# ViewCastLK MVP model candidate

This artifact predicts independent Day 7, 14, 21, or 30 view totals 
from pre-publication video and channel features.

## Status

Candidate only. The reserved test partition has not been evaluated. 
The development result shows weak viral-video performance and is not 
production-ready accuracy.

## Development evaluation

- Combined OOF WAPE: 93.01%
- Combined OOF RMSLE: 2.067
- Total view capture: 21.11%
- Top-decile view capture: 9.89%

## Usage

```text
python -m pip install -r requirements.txt
python predict.py --horizon 7 --input sample_input.csv --output predictions.csv
```

See `manifest.json` for the complete schema, checksums, selected model 
components, and per-horizon metrics.
