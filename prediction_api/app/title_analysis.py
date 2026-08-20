import logging
import math
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from app.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL
from app.schemas import TitleGuidance

logger = logging.getLogger(__name__)

_RETRYABLE_GEMINI_STATUS_CODES = {404, 408, 429, 500, 502, 503, 504}


class TitleToneAnalysisInternal(BaseModel):
    urgency: float = Field(
        ..., ge=0.0, le=1.0, description="Urgency score (0.0 to 1.0)"
    )
    emotional_appeal: float = Field(
        ..., ge=0.0, le=1.0, description="Emotional appeal score (0.0 to 1.0)"
    )
    seriousness: float = Field(
        ..., ge=0.0, le=1.0, description="Seriousness of register score (0.0 to 1.0)"
    )
    curiosity_gap: float = Field(
        ..., ge=0.0, le=1.0, description="Curiosity gap score (0.0 to 1.0)"
    )
    summary: str = Field(
        ..., description="Neutral creator-facing summary of title tone"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Constructive creator-facing suggestions"
    )

    @field_validator("urgency", "emotional_appeal", "seriousness", "curiosity_gap")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError(f"Tone score {v} is outside allowed range [0.0, 1.0]")
        return float(v)


def _gemini_model_candidates() -> tuple[str, ...]:
    """Returns the configured Gemini models in order, without duplicates."""
    candidates: list[str] = []
    for model in (GEMINI_MODEL, *GEMINI_FALLBACK_MODELS):
        normalized = model.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _gemini_error_status_code(exc: Exception) -> Optional[int]:
    """Extracts an HTTP status code from google-genai or transport errors."""
    for value in (
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _should_try_next_model(exc: Exception) -> bool:
    """Only fail over when another model can plausibly complete the request."""
    return _gemini_error_status_code(exc) in _RETRYABLE_GEMINI_STATUS_CODES


def analyze_title_tone(
    title: str,
) -> Tuple[Optional[TitleGuidance], Optional[TitleToneAnalysisInternal]]:
    """Analyzes a YouTube video title using Google Gemini server-side.

    Returns:
        Tuple of (TitleGuidance for public API response, TitleToneAnalysisInternal for internal server state).
        If Gemini is unconfigured or fails, returns (None, None).
    """
    if not GEMINI_API_KEY:
        logger.info("GEMINI_API_KEY is not configured. Skipping title analysis.")
        return None, None

    if not title or not title.strip():
        return None, None

    prompt = (
        f"Analyze the following YouTube video title: '{title.strip()}'\n\n"
        "Instructions:\n"
        "- Analyze titles in Sinhala, Tamil, English, or mixed-script / multilingual format.\n"
        "- Score the title on EXACTLY four internal tone dimensions from 0.0 (lowest) to 1.0 (highest):\n"
        "  1. urgency\n"
        "  2. emotional_appeal\n"
        "  3. seriousness\n"
        "  4. curiosity_gap\n"
        "- Do NOT evaluate political, communal, ethnic, divisive, controversy, or misleading potential.\n"
        "- Do NOT recommend sensationalism or clickbait.\n"
        "- Provide a neutral, helpful creator-facing summary (e.g. 'Clear and informative title framing.').\n"
        "- Provide 2 to 3 constructive title phrasing suggestions to improve clarity.\n"
        "- Do NOT rewrite the title into clickbait or promise unverified video views.\n"
    )

    try:
        from google import genai
        from google.genai import types

        # The SDK retries transient failures several times by default. Use one
        # attempt per model so a depleted model does not delay trying the next.
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TitleToneAnalysisInternal,
        )
    except Exception as exc:
        logger.warning(
            "Gemini title analysis client setup failed (%s).",
            type(exc).__name__,
        )
        return None, None

    models = _gemini_model_candidates()
    for index, model in enumerate(models):
        has_fallback = index < len(models) - 1
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            status_code = _gemini_error_status_code(exc)
            if has_fallback and _should_try_next_model(exc):
                logger.warning(
                    "Gemini model %s failed with status %s; trying next model.",
                    model,
                    status_code,
                )
                continue
            logger.warning(
                "Gemini title analysis stopped at model %s (status=%s, error=%s).",
                model,
                status_code,
                type(exc).__name__,
            )
            return None, None

        if not response or not response.text:
            logger.warning("Gemini model %s returned an empty response.", model)
            if has_fallback:
                continue
            return None, None

        try:
            internal_analysis = TitleToneAnalysisInternal.model_validate_json(
                response.text
            )
        except Exception as exc:
            logger.warning(
                "Gemini model %s returned unusable structured output (%s).",
                model,
                type(exc).__name__,
            )
            if has_fallback:
                continue
            return None, None

        guidance = TitleGuidance(
            summary=internal_analysis.summary,
            suggestions=internal_analysis.suggestions,
        )
        logger.info("Gemini title analysis succeeded with model %s.", model)
        return guidance, internal_analysis

    return None, None
