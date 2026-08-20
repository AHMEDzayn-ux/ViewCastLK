"""Model registry and loader for the ViewCastLK trajectory artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "model_artifacts"
    / "viewcastlk_monotonic_trajectory_experimental_v1"
)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChecksumMismatchError(ValueError):
    """Raised when a model file checksum does not match its manifest."""


class ModelRegistry:
    """Verify, load, and serve the four-horizon trajectory model."""

    def __init__(self, artifact_dir: Path | str | None = None) -> None:
        self.artifact_dir = (
            Path(artifact_dir).resolve()
            if artifact_dir
            else DEFAULT_ARTIFACT_DIR.resolve()
        )
        self.manifest: dict[str, Any] | None = None
        self.model: Any | None = None

    def get_manifest(self) -> dict[str, Any]:
        """Read and cache manifest.json."""
        if self.manifest is None:
            manifest_path = self.artifact_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self.manifest

    def verify_checksum(self) -> str:
        """Verify the trajectory model against the checksum in manifest.json."""
        record = self.get_manifest().get("model", {})
        model_rel_path = record.get("model_path")
        expected_hash = record.get("sha256")
        if not model_rel_path or not expected_hash:
            raise ValueError("Manifest does not define the trajectory model checksum")

        model_path = self.artifact_dir / model_rel_path
        if not model_path.exists():
            raise FileNotFoundError(f"Trajectory model not found: {model_path}")

        actual_hash = sha256_file(model_path)
        if actual_hash != expected_hash:
            raise ChecksumMismatchError(
                f"SHA-256 mismatch for {model_path.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        return actual_hash

    def load_model(self) -> Any:
        """Verify and load the trajectory model once."""
        if self.model is not None:
            return self.model

        self.verify_checksum()

        # joblib needs the artifact package on sys.path to import viewcastlk_ml.
        artifact_dir_str = str(self.artifact_dir)
        if artifact_dir_str not in sys.path:
            sys.path.insert(0, artifact_dir_str)

        model_path = self.artifact_dir / self.get_manifest()["model"]["model_path"]
        self.model = joblib.load(model_path)
        return self.model

    def predict_trajectory(self, df: pd.DataFrame) -> np.ndarray:
        """Predict and validate Day 7/14/21/30 cumulative views in one call."""
        horizons = self.get_manifest().get("supported_horizons_days", [])
        predictions = np.asarray(self.load_model().predict_views(df), dtype=float)
        expected_shape = (len(df), len(horizons))

        if predictions.shape != expected_shape:
            raise ValueError(
                f"Model returned shape {predictions.shape}; expected {expected_shape}"
            )
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Model returned non-finite or negative predictions")
        if (np.diff(predictions, axis=1) < -1e-12).any():
            raise ValueError("Model returned a decreasing cumulative trajectory")
        return predictions

    def predict_views(self, horizon_days: int, df: pd.DataFrame) -> np.ndarray:
        """Return one horizon for callers using the previous registry interface."""
        horizons = self.get_manifest().get("supported_horizons_days", [])
        if horizon_days not in horizons:
            supported = ", ".join(map(str, horizons))
            raise ValueError(
                f"Unsupported horizon {horizon_days}; choose one of: {supported}"
            )
        position = horizons.index(horizon_days)
        return self.predict_trajectory(df)[:, position]

    def predict_sample(self) -> dict[int, float]:
        """Run the bundled sample row as a smoke test."""
        sample_path = self.artifact_dir / "sample_input.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample input not found at {sample_path}")

        horizons = self.get_manifest().get("supported_horizons_days", [])
        predictions = self.predict_trajectory(
            pd.read_csv(sample_path, low_memory=False)
        )[0]
        return {
            horizon: float(predictions[position])
            for position, horizon in enumerate(horizons)
        }


def main() -> None:
    """Developer CLI entry point."""
    registry = ModelRegistry()
    manifest = registry.get_manifest()
    print(f"Artifact Version: {manifest.get('artifact_version', 'unknown')}")
    print(f"SHA-256 Checksum Verified: {registry.verify_checksum()}")
    print("\nSample Predictions:")
    for horizon, prediction in registry.predict_sample().items():
        print(f"  Day {horizon:2d}: {prediction:.12f}")


if __name__ == "__main__":
    main()
