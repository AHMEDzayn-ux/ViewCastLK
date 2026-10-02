from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from app.artifact import ACTIVE_ARTIFACT_VERSION


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "viewcastlk-prediction-api"


class AccuracyMetric(BaseModel):
    # View counts are heavy-tailed, so error is reported as a multiple and as
    # the share within a factor of two; one viral video would dominate an
    # average error in views, and a percentage error explodes on small counts.
    key: Literal["within_2x", "typical_factor", "rank_correlation"]
    label: str
    description: str
    unit: Literal["percent", "factor", "score"]
    betterWhen: Literal["higher", "lower"]
    modelValue: Optional[float] = None
    baselineValue: Optional[float] = None


class AccuracyEvaluation(BaseModel):
    scope: Literal["day_7", "day_14", "day_21", "day_30"]
    # Accuracy depends most on whether the channel's own history is known, so
    # the two are reported apart rather than averaged into one figure.
    segment: Literal["tracked_channel", "new_channel"]
    segmentLabel: str
    videos: int = Field(..., ge=1)
    baselineName: str
    metrics: List[AccuracyMetric] = Field(..., min_length=1)


class PendingHorizon(BaseModel):
    scope: Literal["day_7", "day_14", "day_21", "day_30"]
    measurableFrom: str


class AvailableAccuracyResponse(BaseModel):
    status: Literal["available"] = "available"
    modelName: str
    evaluatedAt: str
    periodStart: str
    periodEnd: str
    method: str
    evaluations: List[AccuracyEvaluation] = Field(..., min_length=1)
    notYetMeasured: List[PendingHorizon] = Field(default_factory=list)
    dataSource: Literal["prediction_api"] = "prediction_api"


class AccuracyResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    modelName: str = Field(..., description="Active model artifact name")
    evaluatedAt: None = Field(
        None, description="Null until approved held-out results are published"
    )
    evaluations: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Empty until approved evaluation results are published",
    )
    dataSource: Literal["prediction_api"] = "prediction_api"
    message: str = Field(..., description="Why evaluation results are unavailable")


class ChannelLookupRequest(BaseModel):
    channelIdentifier: str = Field(
        ...,
        description="YouTube channel handle (@handle), channel ID (UC...), or YouTube URL",
    )


class ChannelStatsResponse(BaseModel):
    channelId: Optional[str] = Field(
        None,
        description=(
            "The resolved YouTube channel ID. The forecast uses it to read the "
            "channel's own collected history from the public warehouse."
        ),
    )
    subscriberCount: Optional[int] = Field(
        None, description="Total subscribers or null if hidden/unavailable"
    )
    totalViewCount: Optional[int] = Field(
        None, description="Total channel view count or null if unavailable"
    )
    videoCount: Optional[int] = Field(
        None, description="Total video count or null if unavailable"
    )
    createdAt: Optional[str] = Field(
        None, description="Channel creation timestamp in ISO format"
    )
    channelAgeDays: Optional[int] = Field(
        None, description="Age of channel in full days"
    )
    topicCategories: Optional[List[str]] = Field(
        None,
        description=(
            "Wikipedia topic URLs YouTube assigns to the channel, or null when "
            "the channel has none"
        ),
    )


class ErrorResponse(BaseModel):
    message: str
    code: str


class YouTubeAuthorizationResponse(BaseModel):
    authorizationUrl: str


class YouTubeConnectionResponse(BaseModel):
    isConnected: bool
    channelId: Optional[str] = None
    channelTitle: Optional[str] = None
    status: Optional[str] = None
    connectedAt: Optional[str] = None
    lastRefreshOkAt: Optional[str] = None


class YouTubeDisconnectResponse(BaseModel):
    disconnected: Literal[True] = True


class ForecastRequest(BaseModel):
    title: str = Field(
        ..., description="Pre-publication video title (non-empty)"
    )
    category: str = Field(
        ..., description="YouTube category name (e.g. Music, Entertainment)"
    )
    durationSeconds: float = Field(
        ..., description="Planned video duration in seconds (must be > 0)"
    )
    isShort: bool | None = Field(
        None, description="Creator's own choice of Short or standard video format"
    )
    audioLanguage: str = Field(
        ..., description="Primary audio language (e.g. English, Sinhala, Tamil)"
    )
    channelIdentifier: str = Field(
        ..., description="YouTube channel handle (@handle), channel ID (UC...), or YouTube URL"
    )
    plannedPublishDay: Optional[str] = Field(
        None, description="Optional planned publish day of week"
    )
    plannedPublishHour: Optional[int] = Field(
        None, description="Optional planned publish hour (0-23)"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Enter a valid video title.")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Select a valid category.")
        return v.strip()

    @field_validator("durationSeconds")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Enter a positive video duration in seconds.")
        return float(v)

    @field_validator("channelIdentifier")
    @classmethod
    def validate_channel_identifier(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Enter a valid YouTube channel URL, handle, or channel ID.")
        return v.strip()

    @field_validator("plannedPublishHour")
    @classmethod
    def validate_hour(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (0 <= v <= 23):
            raise ValueError("Planned publish hour must be between 0 and 23.")
        return v


class ForecastEstimate(BaseModel):
    horizonDays: int = Field(..., description="Horizon in days (7, 14, 21, 30)")
    cumulativeViews: int = Field(
        ..., description="Predicted cumulative view count (rounded non-negative integer)"
    )


class PersonalizationAdjustment(BaseModel):
    horizonDays: Literal[7, 14, 21, 30]
    format: Literal["all", "short", "long"]
    factor: float
    nVideos: int


class ForecastPersonalization(BaseModel):
    applied: bool
    format: Literal["short", "long"]
    modelVersion: str
    sharedEstimates: List[ForecastEstimate]
    adjustments: List[PersonalizationAdjustment] = Field(default_factory=list)


class UnavailableRecommendation(BaseModel):
    type: str = Field(..., description="Recommendation category type")
    reason: str = Field(..., description="Reason recommendation is unavailable")


class DataCompletenessIssue(BaseModel):
    source: str = Field(..., description="Data source name")
    message: str = Field(..., description="Issue details")


class DataCompleteness(BaseModel):
    status: str = Field("complete", description="Data status: complete or degraded")
    issues: List[DataCompletenessIssue] = Field(
        default_factory=list, description="List of completeness issues"
    )


class ModelMetadata(BaseModel):
    artifactVersion: str = Field(
        ACTIVE_ARTIFACT_VERSION,
        description="Artifact version",
    )
    modelVersion: str = Field(
        ACTIVE_ARTIFACT_VERSION,
        description="Model version",
    )
    generatedAt: str = Field(..., description="Timestamp in ISO format")
    dataSource: str = Field("prediction_api", description="Data source identifier")
    status: str = Field("experimental", description="Model deployment status")
    trajectoryMonotonic: bool = Field(True, description="Whether horizon sequence is non-decreasing")


class TitleGuidance(BaseModel):
    summary: str = Field(..., description="Creator-facing summary of title tone")
    suggestions: List[str] = Field(
        default_factory=list, description="List of title phrasing suggestions"
    )


class ForecastResponse(BaseModel):
    forecastId: str = Field(..., description="Unique forecast execution ID")
    estimates: List[ForecastEstimate] = Field(
        ..., description="Forecast estimates for horizons 7, 14, 21, 30"
    )
    personalization: ForecastPersonalization
    channelStats: Optional[ChannelStatsResponse] = Field(
        None, description="Resolved channel statistics"
    )
    recommendations: List[dict[str, Any]] = Field(
        default_factory=list, description="Supported recommendations"
    )
    unavailableRecommendations: List[UnavailableRecommendation] = Field(
        default_factory=list, description="Unavailable recommendations documentation"
    )
    completeness: DataCompleteness = Field(
        default_factory=lambda: DataCompleteness(status="complete", issues=[]),
        description="Data completeness metadata",
    )
    titleGuidance: Optional[TitleGuidance] = Field(
        None, description="Creator-facing title guidance"
    )
    model: ModelMetadata = Field(..., description="Model artifact metadata")
