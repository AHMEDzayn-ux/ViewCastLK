"""Model registry and loader for ViewCastLK ML model artifacts."""

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
    / "viewcastlk_mvp_candidate_v1"
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChecksumMismatchError(ValueError):
    """Raised when a model file checksum does not match manifest."""


class ModelRegistry:
    """Safely loads and manages ViewCastLK horizon model bundles."""

    def __init__(self, artifact_dir: Path | str | None = None) -> None:
        self.artifact_dir = (
            Path(artifact_dir).resolve()
            if artifact_dir
            else DEFAULT_ARTIFACT_DIR.resolve()
        )
        self.manifest: dict[str, Any] | None = None
        self.models: dict[int, Any] = {}
        self._is_loaded = False

    def verify_checksums(self) -> dict[int, str]:
        """Verify SHA-256 checksums of all model files against manifest.json.

        Returns a dictionary mapping horizon_days to actual sha256 hashes.
        Raises ChecksumMismatchError if any hash fails verification.
        """
        manifest = self.get_manifest()
        verified_hashes: dict[int, str] = {}

        for record in manifest.get("models", []):
            horizon = record["horizon_days"]
            model_rel_path = record["model_path"]
            expected_hash = record["sha256"]
            file_path = self.artifact_dir / model_rel_path

            if not file_path.exists():
                raise FileNotFoundError(f"Model file for Day {horizon} not found: {file_path}")

            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                raise ChecksumMismatchError(
                    f"SHA-256 mismatch for Day {horizon} model ({file_path.name}): "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            verified_hashes[horizon] = actual_hash

        return verified_hashes

    def get_manifest(self) -> dict[str, Any]:
        """Read and cache manifest.json."""
        if self.manifest is None:
            manifest_path = self.artifact_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self.manifest

    def load_models(self) -> dict[int, Any]:
        """Verify checksums and load all four horizon models into memory."""
        if self._is_loaded:
            return self.models

        manifest = self.get_manifest()
        self.verify_checksums()

        # Ensure the artifact directory is in Python import path so joblib can import viewcastlk_ml
        artifact_dir_str = str(self.artifact_dir)
        if artifact_dir_str not in sys.path:
            sys.path.insert(0, artifact_dir_str)

        for record in manifest.get("models", []):
            horizon = record["horizon_days"]
            model_path = self.artifact_dir / record["model_path"]
            model_bundle = joblib.load(model_path)
            self.models[horizon] = model_bundle

        self._is_loaded = True
        return self.models

    def get_model(self, horizon_days: int) -> Any:
        """Get loaded model bundle for a specific horizon (7, 14, 21, 30)."""
        if not self._is_loaded:
            self.load_models()
        if horizon_days not in self.models:
            manifest = self.get_manifest()
            supported = ", ".join(map(str, manifest.get("supported_horizons_days", [])))
            raise ValueError(f"Unsupported horizon {horizon_days}; choose one of: {supported}")
        return self.models[horizon_days]

    def predict_views(self, horizon_days: int, df: pd.DataFrame) -> np.ndarray:
        """Run prediction for a specific horizon on input feature table."""
        model_bundle = self.get_model(horizon_days)
        return np.asarray(model_bundle.predict_views(df), dtype=float)

    def predict_sample(self) -> dict[int, float]:
        """Smoke test run against sample_input.csv."""
        if not self._is_loaded:
            self.load_models()
        sample_path = self.artifact_dir / "sample_input.csv"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample input not found at {sample_path}")
        sample_df = pd.read_csv(sample_path, low_memory=False)

        results: dict[int, float] = {}
        for horizon in (7, 14, 21, 30):
            predictions = self.predict_views(horizon, sample_df)
            results[horizon] = float(predictions[0])
        return results


def main() -> None:
    """Developer CLI entry point."""
    registry = ModelRegistry()
    manifest = registry.get_manifest()
    version = manifest.get("artifact_version", "unknown")
    print(f"Artifact Version: {version}")

    hashes = registry.verify_checksums()
    print("SHA-256 Checksums Verified:")
    for horizon, file_hash in hashes.items():
        print(f"  Day {horizon:2d}: {file_hash}")

    predictions = registry.predict_sample()
    print("\nSample Predictions:")
    for horizon in (7, 14, 21, 30):
        print(f"  Day {horizon:2d}: {predictions[horizon]:.12f}")


if __name__ == "__main__":
    main()
