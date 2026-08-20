"""Tests for the deployed monotonic trajectory artifact."""

import json

import numpy as np
import pandas as pd

from app.model_registry import DEFAULT_ARTIFACT_DIR, ModelRegistry


def test_artifact_exists():
    assert DEFAULT_ARTIFACT_DIR.is_dir()
    assert (DEFAULT_ARTIFACT_DIR / "manifest.json").is_file()
    assert (DEFAULT_ARTIFACT_DIR / "sample_input.csv").is_file()


def test_model_checksum_matches_manifest():
    registry = ModelRegistry()
    expected = registry.get_manifest()["model"]["sha256"]
    assert registry.verify_checksum() == expected


def test_trajectory_model_loads():
    model = ModelRegistry().load_model()
    assert hasattr(model, "predict_views")
    assert tuple(model.horizons) == (7, 14, 21, 30)


def test_sample_predictions_match_manifest():
    registry = ModelRegistry()
    expected = registry.get_manifest()["sample_smoke_prediction"]
    predicted = registry.predict_sample()

    for horizon in (7, 14, 21, 30):
        assert np.isclose(
            predicted[horizon],
            expected[f"day_{horizon}_views"],
            rtol=1e-5,
            atol=1e-5,
        )


def test_sample_trajectory_is_finite_nonnegative_and_monotonic():
    predictions = ModelRegistry().predict_sample()
    values = np.asarray([predictions[horizon] for horizon in (7, 14, 21, 30)])

    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert (np.diff(values) >= 0).all()


def test_batch_prediction_has_expected_shape_and_monotonic_rows():
    registry = ModelRegistry()
    sample = pd.read_csv(DEFAULT_ARTIFACT_DIR / "sample_input.csv", low_memory=False)
    batch = pd.concat([sample, sample], ignore_index=True)
    predictions = registry.predict_trajectory(batch)

    assert predictions.shape == (2, 4)
    assert (np.diff(predictions, axis=1) >= 0).all()


def test_artifact_package_checksums_file_is_complete_and_valid():
    checksum_file = DEFAULT_ARTIFACT_DIR / "SHA256SUMS.txt"
    entries = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split(maxsplit=1)
        entries[relative_path] = expected

    # SHA256SUMS.txt intentionally does not include itself.
    packaged_files = {
        path.relative_to(DEFAULT_ARTIFACT_DIR).as_posix()
        for path in DEFAULT_ARTIFACT_DIR.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(entries) == packaged_files

    import hashlib

    for relative_path, expected in entries.items():
        digest = hashlib.sha256(
            (DEFAULT_ARTIFACT_DIR / relative_path).read_bytes()
        ).hexdigest()
        assert digest == expected, relative_path


def test_manifest_contract_is_the_expected_experimental_artifact():
    manifest = json.loads(
        (DEFAULT_ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_version"] == "viewcastlk_monotonic_trajectory_experimental_v1"
    assert manifest["supported_horizons_days"] == [7, 14, 21, 30]
    assert manifest["trajectory_guarantee"] == "day_7 <= day_14 <= day_21 <= day_30"
