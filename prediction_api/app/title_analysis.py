import logging
import math
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas import TitleGuidance

logger = logging.getLogger(__name__)


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

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TitleToneAnalysisInternal,
                temperature=0.2,
            ),
        )

        if not response or not response.text:
            logger.warning("Gemini returned empty title analysis response.")
            return None, None

        internal_analysis = TitleToneAnalysisInternal.model_validate_json(
            response.text
        )

        guidance = TitleGuidance(
            summary=internal_analysis.summary,
            suggestions=internal_analysis.suggestions,
        )

        return guidance, internal_analysis

    except Exception as exc:
        logger.warning("Gemini title analysis failed: %s", exc)
        return None, None
