"""Generate a normal forecast and conditional viral-upside scenario."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--display-horizon", type=int, choices=(7, 14, 21, 30), default=30
    )
    args = parser.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["model"]
    model_path = ROOT / record["model_path"]
    if sha256_file(model_path) != record["sha256"]:
        raise RuntimeError("Model checksum mismatch")
    model = joblib.load(model_path)
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    scenario = model.predict_scenario_frame(frame)
    normal_columns = [f"normal_day_{day}_views" for day in (7, 14, 21, 30)]
    viral_columns = [f"viral_upside_day_{day}_views" for day in (7, 14, 21, 30)]
    normal = scenario[normal_columns].to_numpy(dtype=float)
    viral = scenario[viral_columns].to_numpy(dtype=float)
    probability = scenario["breakout_probability"].to_numpy(dtype=float)
    if not np.isfinite(normal).all() or not np.isfinite(viral).all():
        raise RuntimeError("Model returned non-finite scenario views")
    if (np.diff(normal, axis=1) <= 0).any() or (np.diff(viral, axis=1) <= 0).any():
        raise RuntimeError("Model returned a flat or decreasing trajectory")
    if (viral <= normal).any():
        raise RuntimeError("Viral upside must be above the normal forecast")
    if ((probability < 0) | (probability > 1)).any():
        raise RuntimeError("Breakout probability is outside [0, 1]")

    horizon = args.display_horizon
    scenario["prediction_summary"] = [
        (
            f"Normally expected around {normal_value:,.0f} views by day {horizon}. "
            f"There is a {chance:.1%} breakout chance; if that happens, "
            f"around {viral_value:,.0f} views."
        )
        for normal_value, chance, viral_value in zip(
            scenario[f"normal_day_{horizon}_views"],
            probability,
            scenario[f"viral_upside_day_{horizon}_views"],
        )
    ]
    output = pd.concat([frame.reset_index(drop=True), scenario.reset_index(drop=True)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} two-scenario forecasts to {args.output}")


if __name__ == "__main__":
    main()
