"""Run a ViewCastLK horizon model against a CSV feature table."""

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


def load_model(horizon: int):
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["models"] if item["horizon_days"] == horizon]
    if not matches:
        supported = ", ".join(map(str, manifest["supported_horizons_days"]))
        raise ValueError(f"Unsupported horizon {horizon}; choose one of: {supported}")
    record = matches[0]
    model_path = ROOT / record["model_path"]
    actual_hash = sha256_file(model_path)
    if actual_hash != record["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {model_path.name}")
    return joblib.load(model_path), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, choices=(7, 14, 21, 30))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, manifest = load_model(args.horizon)
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    predictions = np.asarray(model.predict_views(frame), dtype=float)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise RuntimeError("Model returned invalid predictions")
    output = frame.copy()
    output["prediction_horizon_days"] = args.horizon
    output["predicted_views"] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} Day {args.horizon} predictions to {args.output}")


if __name__ == "__main__":
    main()
