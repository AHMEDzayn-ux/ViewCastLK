"""Tests for the deployed monotonic trajectory artifact."""

import json

import numpy as np
import pandas as pd
import pytest

from app.artifact import (
    BREAKOUT_ARTIFACT_DIR,
    BREAKOUT_ARTIFACT_VERSION,
    V8_ARTIFACT_DIR,
    V8_ARTIFACT_VERSION,
)
from app.model_registry import DEFAULT_ARTIFACT_DIR, ModelRegistry, ViralScenarioRegistry


def test_artifact_exists():
    assert DEFAULT_ARTIFACT_DIR.is_dir()
    assert (DEFAULT_ARTIFACT_DIR / "manifest.json").is_file()
    assert (DEFAULT_ARTIFACT_DIR / "sample_input.csv").is_file()


def test_training_video_ids_artifact_matches_manifest():
    import hashlib

    record = json.loads(
        (DEFAULT_ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")
    )["training_video_ids"]
    path = DEFAULT_ARTIFACT_DIR / record["path"]
    identifiers = path.read_text(encoding="utf-8").splitlines()

    assert len(identifiers) == record["count"]
    assert len(identifiers) == len(set(identifiers))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


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


def test_manifest_contract_is_the_served_artifact():
    from app.artifact import ACTIVE_ARTIFACT_VERSION

    manifest = json.loads(
        (DEFAULT_ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_version"] == ACTIVE_ARTIFACT_VERSION
    assert manifest["supported_horizons_days"] == [7, 14, 21, 30]
    assert manifest["trajectory_guarantee"] == "day_7 <= day_14 <= day_21 <= day_30"


def test_a_category_stored_blank_in_training_reaches_the_model_blank():
    """Every Nonprofits & Activism video reached training with a blank name, so
    a model trained on that data learned it as the blank category. Sending the
    name instead would discard that and fall back to the training average."""
    import math

    from app.model_registry import MISSING_CATEGORY_TOKEN, STORED_BLANK_CATEGORIES

    registry = ModelRegistry()
    fitted = registry._encoder_categories()
    if fitted is None or MISSING_CATEGORY_TOKEN not in fitted:
        pytest.skip("the served model has no blank category")
    for name in STORED_BLANK_CATEGORIES:
        if name in fitted:
            assert registry.model_category(name) == name
        else:
            assert math.isnan(registry.model_category(name))
        assert name in registry.known_categories()
    assert registry.model_category("Music") == "Music"
    assert registry.model_category("Made Up Category") == "Made Up Category"


def test_v8_breakout_artifact_checksum_and_sample_contract():
    registry = ViralScenarioRegistry(V8_ARTIFACT_DIR)
    manifest = registry.get_manifest()
    assert manifest["artifact_version"] == V8_ARTIFACT_VERSION
    assert registry.verify_checksum() == manifest["model"]["sha256"]

    sample = pd.read_csv(V8_ARTIFACT_DIR / "sample_input.csv", low_memory=False)
    scenario = registry.predict_scenario(sample)
    probability = float(scenario.iloc[0]["breakout_probability"])
    assert 0 <= probability <= 1
    for horizon in (7, 14, 21, 30):
        assert scenario.iloc[0][f"viral_upside_day_{horizon}_views"] > scenario.iloc[0][
            f"normal_day_{horizon}_views"
        ]


@pytest.mark.parametrize("artifact_dir", [V8_ARTIFACT_DIR, BREAKOUT_ARTIFACT_DIR])
def test_breakout_packaged_checksums_are_complete_and_valid(artifact_dir):
    import hashlib

    entries = {}
    for line in (artifact_dir / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative_path = line.split(maxsplit=1)
        entries[relative_path] = expected

    packaged_files = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(entries) == packaged_files
    for relative_path, expected in entries.items():
        assert (
            hashlib.sha256((artifact_dir / relative_path).read_bytes()).hexdigest()
            == expected
        ), relative_path


def test_v10_breakout_artifact_checksum_and_sample_contract():
    registry = ViralScenarioRegistry(BREAKOUT_ARTIFACT_DIR)
    manifest = registry.get_manifest()
    assert manifest["artifact_version"] == BREAKOUT_ARTIFACT_VERSION
    assert registry.verify_checksum() == manifest["model"]["sha256"]
    assert manifest["training_video_ids"]["count"] == 70_015

    sample = pd.read_csv(BREAKOUT_ARTIFACT_DIR / "sample_input.csv", low_memory=False)
    scenario = registry.predict_scenario(sample)
    assert 0 <= float(scenario.iloc[0]["breakout_probability"]) <= 1
