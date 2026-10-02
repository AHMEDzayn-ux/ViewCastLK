"""/accuracy publishes measured results for the served model, and only those."""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.accuracy import PUBLISHED_ACCURACY_PATH, load_published_accuracy
from app.artifact import ACTIVE_ARTIFACT_VERSION
from app.main import app


client = TestClient(app)

UNAVAILABLE = {
    "status": "unavailable",
    "modelName": ACTIVE_ARTIFACT_VERSION,
    "evaluatedAt": None,
    "evaluations": [],
    "dataSource": "prediction_api",
    "message": (
        "Evaluation results are not available yet. No approved held-out "
        "MAPE, baseline comparison, or accuracy values are published."
    ),
}


def _published():
    return json.loads(PUBLISHED_ACCURACY_PATH.read_text(encoding="utf-8"))


def test_published_results_are_served_for_the_model_they_describe():
    response = client.get("/accuracy")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "available"
    assert payload["modelName"] == ACTIVE_ARTIFACT_VERSION
    assert payload["dataSource"] == "prediction_api"
    segments = {evaluation["segment"] for evaluation in payload["evaluations"]}
    assert segments == {"tracked_channel", "new_channel"}


def test_every_published_figure_says_which_direction_is_better():
    for evaluation in client.get("/accuracy").json()["evaluations"]:
        assert evaluation["videos"] > 0
        for metric in evaluation["metrics"]:
            assert metric["betterWhen"] in {"higher", "lower"}
            assert metric["modelValue"] is not None


def test_another_models_results_are_never_shown(tmp_path):
    other = {**_published(), "modelName": "some_other_model"}
    path = tmp_path / "published_accuracy.json"
    path.write_text(json.dumps(other), encoding="utf-8")

    assert load_published_accuracy(ACTIVE_ARTIFACT_VERSION, path) is None


def test_missing_or_broken_results_fall_back_to_unavailable(tmp_path):
    missing = tmp_path / "absent.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps({**_published(), "evaluations": []}), encoding="utf-8"
    )

    for path in (missing, broken, malformed):
        assert load_published_accuracy(ACTIVE_ARTIFACT_VERSION, path) is None


def test_endpoint_says_unavailable_when_nothing_is_published():
    with patch("app.main.load_published_accuracy", return_value=None):
        response = client.get("/accuracy")

    assert response.status_code == 200
    assert response.json() == UNAVAILABLE
