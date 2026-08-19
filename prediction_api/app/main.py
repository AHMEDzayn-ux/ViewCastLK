from datetime import datetime, timezone
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from app.config import ALLOWED_ORIGINS, DASHBOARD_ORIGIN, YOUTUBE_API_KEY
from app.feature_builder import build_candidate_feature_frame
from app.model_registry import ModelRegistry
from app.schemas import (
    ChannelLookupRequest,
    ChannelStatsResponse,
    DataCompleteness,
    DataCompletenessIssue,
    ErrorResponse,
    ForecastEstimate,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelMetadata,
    TitleGuidance,
    UnavailableRecommendation,
)
from app.title_analysis import analyze_title_tone
from app.youtube import ChannelLookupException, fetch_channel_stats

app = FastAPI(
    title="ViewCastLK Prediction API",
    description="Backend API for ViewCastLK pre-publication YouTube channel analytics and forecasting",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global model registry singleton
model_registry = ModelRegistry()


@app.exception_handler(ChannelLookupException)
async def channel_lookup_exception_handler(
    request: Request, exc: ChannelLookupException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "code": exc.code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = "Invalid request input."
    code = "invalid_request"

    if errors:
        loc = errors[0].get("loc", [])
        msg = errors[0].get("msg", "")
        # Strip Pydantic prefix "Value error, " if present
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        first_msg = msg

        if "channelIdentifier" in loc or "invalid_channel_identifier" in msg:
            code = "invalid_channel_identifier"
            first_msg = "Enter a valid YouTube channel URL, handle, or channel ID."

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": first_msg, "code": code},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", service="viewcastlk-prediction-api")


@app.post(
    "/channel-lookup",
    response_model=ChannelStatsResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def channel_lookup(payload: ChannelLookupRequest):
    return fetch_channel_stats(payload.channelIdentifier, YOUTUBE_API_KEY)


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def create_forecast(payload: ForecastRequest):
    # 1. Resolve real YouTube channel statistics using reusable service
    channel_stats = fetch_channel_stats(payload.channelIdentifier, YOUTUBE_API_KEY)

    # 2. Analyze submitted title using Gemini server-side adapter
    title_guidance, internal_tone_analysis = analyze_title_tone(payload.title)

    # 3. Build 30-column model-ready raw feature frame
    try:
        df = build_candidate_feature_frame(payload, channel_stats)
    except Exception as err:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Failed to construct candidate model feature frame.", "code": "feature_building_error"},
        )

    # 4. Perform model inference across all 4 horizons using cached ModelRegistry
    try:
        model_registry.load_models()
    except Exception as err:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Candidate model artifacts are currently unavailable.", "code": "model_artifacts_unavailable"},
        )

    raw_predictions: dict[int, float] = {}
    estimates: list[ForecastEstimate] = []

    for horizon in (7, 14, 21, 30):
        try:
            preds = model_registry.predict_views(horizon, df)
            raw_val = float(preds[0])
            if not np.isfinite(raw_val) or np.isnan(raw_val):
                raise ValueError(f"Non-finite prediction for horizon {horizon}")
            raw_predictions[horizon] = raw_val
            # Round raw prediction to non-negative integer for API response
            rounded_views = max(0, int(round(raw_val)))
            estimates.append(
                ForecastEstimate(horizonDays=horizon, cumulativeViews=rounded_views)
            )
        except Exception as err:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"Model inference failed for Day {horizon}.", "code": "inference_error"},
            )

    # 5. Check trajectory monotonicity (day7 <= day14 <= day21 <= day30)
    is_monotonic = (
        raw_predictions[7] <= raw_predictions[14] <= raw_predictions[21] <= raw_predictions[30]
    )

    # 6. Build response metadata & documentation fields
    forecast_id = f"fc_{uuid.uuid4().hex[:12]}"
    manifest = model_registry.get_manifest()
    artifact_ver = manifest.get("artifact_version", "viewcastlk_mvp_candidate_v1")

    model_metadata = ModelMetadata(
        artifactVersion=artifact_ver,
        modelVersion=artifact_ver,
        generatedAt=datetime.now(timezone.utc).isoformat(),
        dataSource="prediction_api",
        status="candidate",
        trajectoryMonotonic=is_monotonic,
    )

    unavailable_recs = [
        UnavailableRecommendation(
            type="timing",
            reason="Recommendations are unavailable in candidate v1 model.",
        ),
        UnavailableRecommendation(
            type="duration",
            reason="Recommendations are unavailable in candidate v1 model.",
        ),
        UnavailableRecommendation(
            type="format",
            reason="Recommendations are unavailable in candidate v1 model.",
        ),
        UnavailableRecommendation(
            type="title",
            reason="Recommendations are unavailable in candidate v1 model.",
        ),
    ]

    issues: list[DataCompletenessIssue] = []
    if channel_stats.subscriberCount is None:
        issues.append(
            DataCompletenessIssue(
                source="channel_lookup",
                message="Subscriber count is hidden or unavailable for this channel.",
            )
        )
    if title_guidance is None:
        issues.append(
            DataCompletenessIssue(
                source="title_analysis",
                message="Title analysis is temporarily unavailable.",
            )
        )

    completeness = DataCompleteness(
        status="complete" if len(issues) == 0 else "degraded",
        issues=issues,
    )

    return ForecastResponse(
        forecastId=forecast_id,
        estimates=estimates,
        channelStats=channel_stats,
        recommendations=[],
        unavailableRecommendations=unavailable_recs,
        completeness=completeness,
        titleGuidance=title_guidance,
        model=model_metadata,
    )
