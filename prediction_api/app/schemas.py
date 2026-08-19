from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "viewcastlk-prediction-api"


class ChannelLookupRequest(BaseModel):
    channelIdentifier: str = Field(
        ...,
        description="YouTube channel handle (@handle), channel ID (UC...), or YouTube URL",
    )


class ChannelStatsResponse(BaseModel):
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


class ErrorResponse(BaseModel):
    message: str
    code: str


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
    artifactVersion: str = Field("viewcastlk_mvp_candidate_v1", description="Artifact version")
    modelVersion: str = Field("viewcastlk_mvp_candidate_v1", description="Model version")
    generatedAt: str = Field(..., description="Timestamp in ISO format")
    dataSource: str = Field("prediction_api", description="Data source identifier")
    status: str = Field("candidate", description="Model deployment status")
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
