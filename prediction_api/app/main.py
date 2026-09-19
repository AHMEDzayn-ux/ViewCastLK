from datetime import datetime, timezone
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from app.config import ALLOWED_ORIGINS, DASHBOARD_ORIGIN, YOUTUBE_API_KEY
from app.auth import (
    AuthenticatedUser,
    AuthenticationException,
    require_authenticated_user,
)
from app.feature_builder import build_candidate_feature_frame
from app.model_registry import ModelRegistry
from app.schemas import (
    AccuracyResponse,
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
    YouTubeAuthorizationResponse,
    YouTubeConnectionResponse,
)
from app.title_analysis import analyze_title_tone
from app.youtube import ChannelLookupException, fetch_channel_stats
from app.creator_store import CreatorStore, CreatorStoreUnavailable
from app.youtube_oauth import (
    YouTubeOAuthException,
    build_authorization_url,
    encrypt_refresh_token,
    exchange_authorization_code,
    fetch_authenticated_channel,
    generate_oauth_state,
    hash_oauth_state,
    oauth_state_expiry,
)

app = FastAPI(
    title="ViewCastLK Prediction API",
    description="Backend API for ViewCastLK pre-publication YouTube channel analytics and forecasting",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Authorization", "Content-Type"],
)

# Global model registry singleton
model_registry = ModelRegistry()
creator_store = CreatorStore()


@app.exception_handler(AuthenticationException)
async def authentication_exception_handler(
    request: Request, exc: AuthenticationException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "code": exc.code},
    )


@app.exception_handler(ChannelLookupException)
async def channel_lookup_exception_handler(
    request: Request, exc: ChannelLookupException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "code": exc.code},
    )


@app.exception_handler(YouTubeOAuthException)
async def youtube_oauth_exception_handler(
    request: Request, exc: YouTubeOAuthException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "code": exc.code},
    )


@app.exception_handler(CreatorStoreUnavailable)
async def creator_store_exception_handler(
    request: Request, exc: CreatorStoreUnavailable
):
    return JSONResponse(
        status_code=503,
        content={
            "message": "YouTube channel connection is temporarily unavailable.",
            "code": "creator_storage_unavailable",
        },
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


@app.get("/accuracy", response_model=AccuracyResponse)
async def accuracy_status():
    manifest = model_registry.get_manifest()
    return AccuracyResponse(
        modelName=manifest.get(
            "artifact_version", "viewcastlk_monotonic_trajectory_experimental_v1"
        ),
        message=(
            "Evaluation results are not available yet. No approved held-out "
            "MAPE, baseline comparison, or accuracy values are published."
        ),
    )


@app.get(
    "/auth/youtube/start",
    response_model=YouTubeAuthorizationResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def start_youtube_oauth(
    authenticated_user: AuthenticatedUser = Depends(require_authenticated_user),
):
    state_value = generate_oauth_state()
    authorization_url = build_authorization_url(state_value)
    await creator_store.create_oauth_state(
        state_hash=hash_oauth_state(state_value),
        user_id=authenticated_user.id,
        expires_at=oauth_state_expiry(),
    )
    # A JSON URL lets the browser authenticate this API request with its bearer
    # token, then perform a normal top-level redirect without putting that token
    # into a query string.
    return YouTubeAuthorizationResponse(authorizationUrl=authorization_url)


@app.get(
    "/auth/youtube/callback",
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def youtube_oauth_callback(
    state_value: str = Query(alias="state", min_length=1, max_length=512),
    code: str | None = Query(default=None, max_length=4096),
    error: str | None = Query(default=None, max_length=256),
):
    state_record = await creator_store.consume_oauth_state(
        state_hash=hash_oauth_state(state_value)
    )
    if state_record is None:
        raise YouTubeOAuthException(
            status_code=400,
            message="This YouTube connection request is invalid or has expired.",
            code="invalid_oauth_state",
        )

    if error or not code:
        return RedirectResponse(
            url=f"{DASHBOARD_ORIGIN.rstrip('/')}/account?youtube=not_connected",
            status_code=303,
        )

    tokens = await exchange_authorization_code(code)
    channel = await fetch_authenticated_channel(tokens.access_token)
    encrypted_refresh_token = encrypt_refresh_token(
        tokens.refresh_token, user_id=state_record.user_id
    )
    await creator_store.upsert_youtube_connection(
        user_id=state_record.user_id,
        channel_id=channel.channel_id,
        channel_title=channel.title,
        encrypted_refresh_token=encrypted_refresh_token,
        scopes=tokens.scopes,
    )
    return RedirectResponse(
        url=f"{DASHBOARD_ORIGIN.rstrip('/')}/account?youtube=connected",
        status_code=303,
    )


@app.get(
    "/creator/youtube-connection",
    response_model=YouTubeConnectionResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def youtube_connection_status(
    authenticated_user: AuthenticatedUser = Depends(require_authenticated_user),
):
    connection = await creator_store.get_youtube_connection_status(
        user_id=authenticated_user.id
    )
    if connection is None:
        return YouTubeConnectionResponse(isConnected=False)
    return YouTubeConnectionResponse(
        isConnected=True,
        channelId=connection["channel_id"],
        channelTitle=connection["channel_title"],
        status=connection["status"],
        connectedAt=connection["connected_at"].isoformat(),
        lastRefreshOkAt=(
            connection["last_refresh_ok_at"].isoformat()
            if connection["last_refresh_ok_at"]
            else None
        ),
    )


@app.post(
    "/channel-lookup",
    response_model=ChannelStatsResponse,
    responses={
        401: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def channel_lookup(
    payload: ChannelLookupRequest,
    _authenticated_user: AuthenticatedUser = Depends(require_authenticated_user),
):
    return fetch_channel_stats(payload.channelIdentifier, YOUTUBE_API_KEY)


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        401: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_forecast(
    payload: ForecastRequest,
    _authenticated_user: AuthenticatedUser = Depends(require_authenticated_user),
):
    # 1. Resolve real YouTube channel statistics using reusable service
    channel_stats = fetch_channel_stats(payload.channelIdentifier, YOUTUBE_API_KEY)

    # 2. Analyze submitted title using Gemini server-side adapter
    title_guidance, internal_tone_analysis = analyze_title_tone(payload.title)

    # 3. Build 30-column model-ready raw feature frame
    try:
        df = build_candidate_feature_frame(payload, channel_stats)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Failed to construct candidate model feature frame.", "code": "feature_building_error"},
        )

    # 4. Perform one trajectory inference across all four horizons.
    try:
        trajectory = model_registry.predict_trajectory(df)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Trajectory model inference is currently unavailable.", "code": "inference_error"},
        )

    horizons = (7, 14, 21, 30)
    raw_predictions: dict[int, float] = {}
    estimates: list[ForecastEstimate] = []

    for position, horizon in enumerate(horizons):
        raw_val = float(trajectory[0, position])
        raw_predictions[horizon] = raw_val
        estimates.append(
            ForecastEstimate(
                horizonDays=horizon,
                cumulativeViews=max(0, int(round(raw_val))),
            )
        )

    # 5. Check trajectory monotonicity (day7 <= day14 <= day21 <= day30)
    is_monotonic = (
        raw_predictions[7] <= raw_predictions[14] <= raw_predictions[21] <= raw_predictions[30]
    )

    # 6. Build response metadata & documentation fields
    forecast_id = f"fc_{uuid.uuid4().hex[:12]}"
    manifest = model_registry.get_manifest()
    artifact_ver = manifest.get(
        "artifact_version", "viewcastlk_monotonic_trajectory_experimental_v1"
    )

    model_metadata = ModelMetadata(
        artifactVersion=artifact_ver,
        modelVersion=artifact_ver,
        generatedAt=datetime.now(timezone.utc).isoformat(),
        dataSource="prediction_api",
        status="experimental",
        trajectoryMonotonic=is_monotonic,
    )

    unavailable_recs = [
        UnavailableRecommendation(
            type="timing",
            reason="Recommendations are unavailable in the experimental trajectory model.",
        ),
        UnavailableRecommendation(
            type="duration",
            reason="Recommendations are unavailable in the experimental trajectory model.",
        ),
        UnavailableRecommendation(
            type="format",
            reason="Recommendations are unavailable in the experimental trajectory model.",
        ),
        UnavailableRecommendation(
            type="title",
            reason="Recommendations are unavailable in the experimental trajectory model.",
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
