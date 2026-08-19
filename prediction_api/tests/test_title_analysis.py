import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import ChannelStatsResponse, ForecastRequest
from app.title_analysis import TitleToneAnalysisInternal, analyze_title_tone
from app.feature_builder import build_candidate_feature_frame

client = TestClient(app)

MOCK_CHANNEL_STATS = ChannelStatsResponse(
    subscriberCount=10000,
    totalViewCount=500000,
    videoCount=100,
    createdAt="2020-01-01T00:00:00Z",
    channelAgeDays=1000,
)

SAMPLE_VALID_GEMINI_JSON = json.dumps({
    "urgency": 0.4,
    "emotional_appeal": 0.6,
    "seriousness": 0.8,
    "curiosity_gap": 0.5,
    "summary": "Clear and informative title framing.",
    "suggestions": [
        "Keep the main subject easy to identify.",
        "Put the most useful context near the beginning."
    ]
})


def test_1_successful_allowed_four_score_analysis():
    with patch("app.title_analysis.GEMINI_API_KEY", "dummy-key"):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_VALID_GEMINI_JSON
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            guidance, internal = analyze_title_tone("Sri Lanka Travel Guide 2026")
            assert guidance is not None
            assert guidance.summary == "Clear and informative title framing."
            assert len(guidance.suggestions) == 2
            assert internal is not None
            assert internal.urgency == 0.4
            assert internal.emotional_appeal == 0.6
            assert internal.seriousness == 0.8
            assert internal.curiosity_gap == 0.5


def test_2_invalid_malformed_gemini_response():
    with patch("app.title_analysis.GEMINI_API_KEY", "dummy-key"):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is non-JSON free form text"
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            guidance, internal = analyze_title_tone("Test Title")
            assert guidance is None
            assert internal is None


def test_3_score_outside_allowed_range():
    invalid_json = json.dumps({
        "urgency": 1.5,  # Invalid: > 1.0
        "emotional_appeal": 0.5,
        "seriousness": 0.5,
        "curiosity_gap": 0.5,
        "summary": "Summary",
        "suggestions": []
    })
    with patch("app.title_analysis.GEMINI_API_KEY", "dummy-key"):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = invalid_json
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            guidance, internal = analyze_title_tone("Test Title")
            assert guidance is None
            assert internal is None


def test_4_gemini_unavailable_missing_api_key():
    with patch("app.title_analysis.GEMINI_API_KEY", ""):
        guidance, internal = analyze_title_tone("Test Title")
        assert guidance is None
        assert internal is None


def test_5_gemini_rate_limit_or_exception():
    with patch("app.title_analysis.GEMINI_API_KEY", "dummy-key"):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Rate limit exceeded")

        with patch("google.genai.Client", return_value=mock_client):
            guidance, internal = analyze_title_tone("Test Title")
            assert guidance is None
            assert internal is None


def test_6_forecast_succeeds_when_gemini_fails():
    payload = {
        "title": "Sri Lanka Travel Guide 2026",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "audioLanguage": "English",
        "channelIdentifier": "UCKA7SQOUkD1_z5lr016Q70w",
        "plannedPublishDay": None,
        "plannedPublishHour": None
    }
    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(None, None)):
            response = client.post("/forecast", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "estimates" in data
            assert len(data["estimates"]) == 4


def test_7_failed_gemini_causes_degraded_completeness():
    payload = {
        "title": "Sri Lanka Travel Guide 2026",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "audioLanguage": "English",
        "channelIdentifier": "UCKA7SQOUkD1_z5lr016Q70w",
        "plannedPublishDay": None,
        "plannedPublishHour": None
    }
    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(None, None)):
            response = client.post("/forecast", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["completeness"]["status"] == "degraded"
            issues = data["completeness"]["issues"]
            assert any(i["source"] == "title_analysis" for i in issues)


def test_8_successful_gemini_returns_plain_language_title_guidance():
    payload = {
        "title": "Sri Lanka Travel Guide 2026",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "audioLanguage": "English",
        "channelIdentifier": "UCKA7SQOUkD1_z5lr016Q70w",
        "plannedPublishDay": None,
        "plannedPublishHour": None
    }
    from app.schemas import TitleGuidance
    mock_guidance = TitleGuidance(
        summary="Clear and informative title framing.",
        suggestions=["Keep the main subject easy to identify."]
    )
    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(mock_guidance, None)):
            response = client.post("/forecast", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["completeness"]["status"] == "complete"
            assert data["titleGuidance"] is not None
            assert data["titleGuidance"]["summary"] == "Clear and informative title framing."
            assert data["titleGuidance"]["suggestions"] == ["Keep the main subject easy to identify."]


def test_9_raw_scores_not_present_in_public_response():
    payload = {
        "title": "Sri Lanka Travel Guide 2026",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "audioLanguage": "English",
        "channelIdentifier": "UCKA7SQOUkD1_z5lr016Q70w",
        "plannedPublishDay": None,
        "plannedPublishHour": None
    }
    from app.schemas import TitleGuidance
    mock_guidance = TitleGuidance(summary="Summary", suggestions=[])
    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(mock_guidance, None)):
            response = client.post("/forecast", json=payload)
            assert response.status_code == 200
            data = response.json()
            # Verify raw internal tone score fields are absent from response root and titleGuidance
            assert "urgency" not in data
            assert "emotional_appeal" not in data
            assert "seriousness" not in data
            assert "curiosity_gap" not in data
            if data.get("titleGuidance"):
                assert "urgency" not in data["titleGuidance"]
                assert "emotional_appeal" not in data["titleGuidance"]
                assert "seriousness" not in data["titleGuidance"]
                assert "curiosity_gap" not in data["titleGuidance"]


def test_10_candidate_feature_frame_remains_unchanged():
    request = ForecastRequest(
        title="Sri Lanka Travel Guide 2026",
        category="Travel & Events",
        durationSeconds=510,
        audioLanguage="English",
        channelIdentifier="UCKA7SQOUkD1_z5lr016Q70w",
        plannedPublishDay=None,
        plannedPublishHour=None
    )
    df = build_candidate_feature_frame(request, MOCK_CHANNEL_STATS)
    assert len(df.columns) == 30
    assert "urgency" not in df.columns
    assert "emotional_appeal" not in df.columns
    assert "seriousness" not in df.columns
    assert "curiosity_gap" not in df.columns


def test_11_model_predictions_remain_unchanged():
    payload = {
        "title": "Sri Lanka Travel Guide 2026",
        "category": "Travel & Events",
        "durationSeconds": 510,
        "audioLanguage": "English",
        "channelIdentifier": "UCKA7SQOUkD1_z5lr016Q70w",
        "plannedPublishDay": None,
        "plannedPublishHour": None
    }
    from app.schemas import TitleGuidance
    mock_guidance = TitleGuidance(summary="Summary", suggestions=[])

    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(None, None)):
            res_without_gemini = client.post("/forecast", json=payload).json()

    with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
        with patch("app.main.analyze_title_tone", return_value=(mock_guidance, None)):
            res_with_gemini = client.post("/forecast", json=payload).json()

    assert res_without_gemini["estimates"] == res_with_gemini["estimates"]
