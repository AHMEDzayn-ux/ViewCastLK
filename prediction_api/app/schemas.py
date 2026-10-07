from typing import Any, List, Literal, Optional
import unicodedata
from pydantic import BaseModel, Field, field_validator

from app.artifact import ACTIVE_ARTIFACT_VERSION
from app.identifiers import parse_channel_identifier

AudioLanguage = Literal["Sinhala", "Tamil", "English", "Mixed / multilingual", "Other"]
PublishDay = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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
    model_config = {"extra": "forbid"}
    channelIdentifier: str = Field(
        ...,
        min_length=1, max_length=512,
        description="YouTube channel handle (@handle), channel ID (UC...), or YouTube URL",
    )

    @field_validator("channelIdentifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        parse_channel_identifier(value)
        return value.strip()


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
    state: str
    bindingNonce: str


class YouTubeOAuthCompletionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    state: str = Field(min_length=1, max_length=512)
    bindingNonce: str = Field(min_length=43, max_length=43, pattern=r"^[A-Za-z0-9_-]{43}$")
    code: str | None = Field(default=None, min_length=1, max_length=4096)
    denied: bool = False


class YouTubeOAuthCompletionResponse(BaseModel):
    connected: bool


class YouTubeConnectionResponse(BaseModel):
    isConnected: bool
    channelId: Optional[str] = None
    channelTitle: Optional[str] = None
    status: Optional[str] = None
    connectedAt: Optional[str] = None
    lastRefreshOkAt: Optional[str] = None


class CreatorEffect(BaseModel):
    """Views against the channel's own normal, with a bootstrap 95% range."""
    videos: int
    effectPct: float
    lowPct: float
    highPct: float


class CreatorGroupEffect(CreatorEffect):
    key: str
    label: str


class CreatorSpacing(BaseModel):
    uploadsCounted: int
    shareWithinHour: Optional[float] = None
    medianGapHours: Optional[float] = None
    buckets: List[CreatorGroupEffect] = Field(default_factory=list)


class CreatorFormat(BaseModel):
    shorts: int
    regular: int
    shortsVsRegular: Optional[CreatorEffect] = None


class CreatorGrowth(BaseModel):
    videos: int
    medianGrowthPct: float


class CreatorInsightsResponse(BaseModel):
    channelTitle: Optional[str] = None
    videosSynced: int
    videosMeasured: int
    periodStart: Optional[str] = None
    periodEnd: Optional[str] = None
    mainCategory: Optional[str] = None
    spacing: Optional[CreatorSpacing] = None
    timing: List[CreatorGroupEffect] = Field(default_factory=list)
    format: Optional[CreatorFormat] = None
    growth: Optional[CreatorGrowth] = None


class YouTubeDisconnectResponse(BaseModel):
    disconnected: Literal[True] = True


class ForecastRequest(BaseModel):
    model_config = {"extra": "forbid"}
    title: str = Field(
        ..., min_length=1, max_length=100, description="Pre-publication video title (non-empty)"
    )
    category: str = Field(
        ..., min_length=1, max_length=100,
        description="YouTube category name (e.g. Music, Entertainment)"
    )
    durationSeconds: float = Field(
        ..., gt=0, le=43200, allow_inf_nan=False,
        description="Planned video duration in seconds (must be > 0; at most 12 hours)"
    )
    isShort: bool | None = Field(
        None, description="Creator's own choice of Short or standard video format"
    )
    audioLanguage: AudioLanguage = Field(
        ..., description="Primary audio language (e.g. English, Sinhala, Tamil)"
    )
    channelIdentifier: str = Field(
        ..., min_length=1, max_length=512,
        description="YouTube channel handle (@handle), channel ID (UC...), or YouTube URL"
    )
    plannedPublishDay: Optional[PublishDay] = Field(
        None, description="Optional planned publish day of week"
    )
    plannedPublishHour: Optional[int] = Field(
        None, ge=0, le=23, strict=True, description="Optional planned publish hour (0-23)"
    )
    modelEngine: Literal["v8", "v9", "v10"] | None = Field(
        None,
        description="Optional model selection; defaults to the released v10 model",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip() or any(unicodedata.category(char) in {"Cc", "Cs"} for char in v):
            raise ValueError("Enter a valid video title.")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip() or not v.isprintable():
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
        parse_channel_identifier(v)
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


class BreakoutForecast(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    conditionalUpside: List[ForecastEstimate]
    definition: str


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


class GuidanceMetadata(BaseModel):
    artifactVersion: str = Field(..., description="Versioned historical EDA artifact")
    source: Literal["historical_eda"] = "historical_eda"
    isolatedFromForecast: Literal[True] = True
    associationWarning: str = Field(
        ...,
        description="Reminder that the historical comparisons are not causal claims",
    )


class ForecastResponse(BaseModel):
    forecastId: str = Field(..., description="Unique forecast execution ID")
    estimates: List[ForecastEstimate] = Field(
        ..., description="Forecast estimates for horizons 7, 14, 21, 30"
    )
    breakout: Optional[BreakoutForecast] = Field(
        None,
        description="Calibrated breakout chance and conditional upside trajectory",
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
    guidance: Optional[GuidanceMetadata] = Field(
        None,
        description="Provenance for guidance generated independently of model inference",
    )
    model: ModelMetadata = Field(..., description="Model artifact metadata")
