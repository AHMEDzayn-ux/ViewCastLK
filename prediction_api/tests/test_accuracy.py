"""Tests for the honest unavailable accuracy response."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_accuracy_endpoint_returns_unavailable_status():
    response = client.get("/accuracy")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "modelName": "viewcastlk_monotonic_trajectory_experimental_v1",
        "evaluatedAt": None,
        "evaluations": [],
        "dataSource": "prediction_api",
        "message": (
            "Evaluation results are not available yet. No approved held-out "
            "MAPE, baseline comparison, or accuracy values are published."
        ),
    }


def test_accuracy_endpoint_does_not_publish_unapproved_values():
    payload = client.get("/accuracy").json()
    serialized = str(payload).lower()

    assert "baselineName" not in payload
    assert payload["evaluations"] == []
    assert "modelvalue" not in serialized
    assert "baselinevalue" not in serialized
    assert payload["evaluatedAt"] is None
