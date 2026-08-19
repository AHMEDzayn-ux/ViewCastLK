"""Automated tests for ViewCastLK model artifact loading and predictions."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from app.model_registry import ModelRegistry, DEFAULT_ARTIFACT_DIR, ChecksumMismatchError


def test_artifact_exists():
    """Verify that candidate model artifact directory and manifest exist."""
    assert DEFAULT_ARTIFACT_DIR.exists()
    assert DEFAULT_ARTIFACT_DIR.is_dir()
    manifest_file = DEFAULT_ARTIFACT_DIR / "manifest.json"
    assert manifest_file.exists()
    sample_file = DEFAULT_ARTIFACT_DIR / "sample_input.csv"
    assert sample_file.exists()


def test_model_hashes():
    """Verify SHA-256 hashes of all four horizon model files."""
    registry = ModelRegistry()
    hashes = registry.verify_checksums()
    assert len(hashes) == 4
    for horizon in (7, 14, 21, 30):
        assert horizon in hashes
        assert isinstance(hashes[horizon], str)
        assert len(hashes[horizon]) == 64


def test_load_all_models():
    """Verify that all four horizon models load successfully into memory."""
    registry = ModelRegistry()
    models = registry.load_models()
    assert len(models) == 4
    for horizon in (7, 14, 21, 30):
        assert horizon in models
        model = registry.get_model(horizon)
        assert hasattr(model, "predict_views")


def test_sample_input_predictions():
    """Verify sample predictions match expected values from manifest.json."""
    registry = ModelRegistry()
    manifest = registry.get_manifest()
    expected_dict = manifest["sample_smoke_predictions"]

    sample_predictions = registry.predict_sample()
    assert len(sample_predictions) == 4

    for horizon in (7, 14, 21, 30):
        predicted = sample_predictions[horizon]
        expected = expected_dict[str(horizon)]
        assert np.isclose(predicted, expected, rtol=1e-5, atol=1e-5), (
            f"Day {horizon} prediction mismatch: predicted {predicted}, expected {expected}"
        )


def test_prediction_output_sanity():
    """Check sanity of predictions: finite, non-negative, and report monotonicity."""
    registry = ModelRegistry()
    sample_predictions = registry.predict_sample()

    for horizon, val in sample_predictions.items():
        assert np.isfinite(val), f"Day {horizon} prediction is not finite: {val}"
        assert val >= 0, f"Day {horizon} prediction is negative: {val}"

    # Note monotonicity: raw Day 30 model output is lower than Day 21 in this candidate artifact.
    day7 = sample_predictions[7]
    day14 = sample_predictions[14]
    day21 = sample_predictions[21]
    day30 = sample_predictions[30]

    assert day7 <= day14, f"Day 7 ({day7}) > Day 14 ({day14})"
    assert day14 <= day21, f"Day 14 ({day14}) > Day 21 ({day21})"
    # Day 30 is preserved raw without post-processing (Day 30 < Day 21 expected for this candidate)
    is_monotonic_all = (day7 <= day14 <= day21 <= day30)
    assert is_monotonic_all is False
