"""Tests for EDA guidance that remains separate from forecast inference."""

import hashlib
from pathlib import Path

from app.idea_optimizer import load_recommendation_artifact, optimize_idea
from app.schemas import ForecastRequest


def request(**overrides) -> ForecastRequest:
    values = {
        "title": "A creator's next video",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "isShort": False,
        "audioLanguage": "English",
        "channelIdentifier": "@creator",
        "plannedPublishDay": "Tuesday",
        "plannedPublishHour": 9,
    }
    values.update(overrides)
    return ForecastRequest(**values)


def by_type(recommendations: list[dict]) -> dict[str, dict]:
    return {item["type"]: item for item in recommendations}


def unavailable_types(unavailable) -> set[str]:
    return {item.type for item in unavailable}


def test_released_artifact_records_exact_eda_and_training_sources():
    artifact = load_recommendation_artifact()
    repo = Path(__file__).resolve().parents[2]
    insights = (repo / "dashboard" / "src" / "data" / "insights.json").read_bytes()
    # CRLF on a Windows checkout, LF on Linux: hash the same text on both.
    insights = insights.replace(b"\r\n", b"\n")

    assert artifact["source"]["insightsSha256"] == hashlib.sha256(insights).hexdigest()
    assert artifact["source"]["trainingTableSha256"] == (
        "65d7c0f20214ea8324a14f012ef4fd1113a0dd87270368bcd9b1d6f3193e5ab9"
    )


def test_compares_submitted_timing_with_supported_eda_window():
    recommendations, unavailable, version = optimize_idea(request())

    timing = by_type(recommendations)["timing"]
    assert timing["recommendedPublishingWindow"] == {
        "day": "Saturday",
        "startHour": 18,
        "endHour": 21,
        "timeZone": "Asia/Colombo",
    }
    assert "Tuesday" in timing["evidence"][2]["detail"]
    assert "09:00-12:00" in timing["evidence"][2]["detail"]
    assert timing["status"] == "change"
    assert by_type(recommendations)["duration"]["status"] == "benchmark"
    assert "duration" not in unavailable_types(unavailable)
    assert version == "idea_optimization_20261001_v1"


def test_duration_advice_stays_within_long_form_and_category():
    recommendations, _, _ = optimize_idea(
        request(
            category="Education",
            durationSeconds=2_000,
            plannedPublishDay="Saturday",
            plannedPublishHour=18,
        )
    )

    duration = by_type(recommendations)["duration"]
    assert duration["title"] == "Test the Over 1 hour duration range"
    assert "30-60 min" in duration["guidance"]
    assert "Education" in duration["guidance"]


def test_format_change_requires_significant_category_contrast():
    recommendations, _, _ = optimize_idea(
        request(
            category="Entertainment",
            durationSeconds=45,
            isShort=True,
        )
    )

    format_recommendation = by_type(recommendations)["format"]
    assert "standard video" in format_recommendation["title"]
    assert "-59.4%" in format_recommendation["evidence"][0]["detail"]


def test_every_supported_dimension_explains_aligned_or_inconclusive_evidence():
    recommendations, unavailable, _ = optimize_idea(
        request(
            category="Education",
            durationSeconds=1_000,
            plannedPublishDay="Saturday",
            plannedPublishHour=18,
        )
    )

    review = by_type(recommendations)
    assert review["timing"]["status"] == "aligned"
    assert review["duration"]["status"] == "benchmark"
    assert review["format"]["status"] == "benchmark"
    assert unavailable == []


def test_existing_strong_format_choice_is_reported_as_aligned():
    recommendations, _, _ = optimize_idea(
        request(category="Travel & Events", durationSeconds=45, isShort=True)
    )

    format_review = by_type(recommendations)["format"]
    assert format_review["status"] == "aligned"
    assert "keeping the current format" in format_review["guidance"]


def test_missing_timing_and_short_duration_are_explained_not_invented():
    recommendations, unavailable, _ = optimize_idea(
        request(
            plannedPublishDay=None,
            plannedPublishHour=None,
            durationSeconds=45,
            isShort=True,
        )
    )

    assert "timing" not in by_type(recommendations)
    assert "duration" not in by_type(recommendations)
    reasons = {item.type: item.reason for item in unavailable}
    assert "Choose both" in reasons["timing"]
    assert "within Shorts" in reasons["duration"]
