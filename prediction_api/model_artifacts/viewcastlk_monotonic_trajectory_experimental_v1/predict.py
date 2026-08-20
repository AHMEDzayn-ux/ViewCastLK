"""Predict one monotonic ViewCastLK view trajectory from CSV rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["model"]
    model_path = ROOT / record["model_path"]
    if sha256_file(model_path) != record["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {model_path.name}")
    return joblib.load(model_path), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, manifest = load_model()
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    predictions = np.asarray(model.predict_views(frame), dtype=float)
    horizons = manifest["supported_horizons_days"]
    if predictions.shape != (len(frame), len(horizons)):
        raise RuntimeError("Model returned an unexpected prediction shape")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise RuntimeError("Model returned invalid predictions")
    if (np.diff(predictions, axis=1) < -1e-12).any():
        raise RuntimeError("Model returned a decreasing cumulative trajectory")

    output = frame.copy()
    for position, horizon in enumerate(horizons):
        output[f"predicted_day_{horizon}_views"] = predictions[:, position]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} monotonic trajectories to {args.output}")


if __name__ == "__main__":
    main()
