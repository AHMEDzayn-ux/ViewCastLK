"""Title and timing features, derived the way the training table derives them.

scripts/build_training_table.py computes these from the video's title and its
publication time in Sri Lanka. Every one of them is knowable before the creator
publishes, because the creator supplies the title and the planned slot on the
forecast form, so serving can produce them instead of sending the model a gap.

Two columns are deliberately different. description_length and tag_count
describe fields the creator has not written when the forecast is requested.
Training fills a missing description or an empty tag list with zero, so zero is
a value the model met, and that is what is sent.
"""

from __future__ import annotations

import math
import re
from typing import Any

# The alphabet the title is written in, not the language. A romanised Sinhala
# title ("Man Adarei") is latin_script, the same as an English one.
SINHALA = re.compile(r"[඀-෿]")
TAMIL = re.compile(r"[஀-௿]")

TITLE_COLUMNS: tuple[str, ...] = (
    "title_length",
    "title_word_count",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
    "title_upper_ratio",
    "title_script",
    "description_length",
    "tag_count",
)

TIMING_COLUMNS: tuple[str, ...] = (
    "publish_hour_sin",
    "publish_hour_cos",
    "publish_dow_sin",
    "publish_dow_cos",
)

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def title_script(text: Any) -> str:
    value = text or ""
    has_sinhala = bool(SINHALA.search(value))
    has_tamil = bool(TAMIL.search(value))
    if has_sinhala and has_tamil:
        return "mixed_script"
    if has_sinhala:
        return "sinhala_script"
    if has_tamil:
        return "tamil_script"
    return "latin_script"


def derive_title_features(title: Any) -> dict[str, Any]:
    value = title if isinstance(title, str) else ""
    length = len(value)
    return {
        "title_length": float(length),
        "title_word_count": float(len(value.split())) if value else 0.0,
        "title_has_number": bool(re.search(r"\d", value)),
        "title_has_question": "?" in value,
        "title_has_exclaim": "!" in value,
        "title_upper_ratio": (
            sum(character.isupper() for character in value) / length if length else 0.0
        ),
        "title_script": title_script(value),
        # Neither is written yet at forecast time; training treats absence as 0.
        "description_length": 0.0,
        "tag_count": 0.0,
    }


def _cyclic(value: float, period: int) -> tuple[float, float]:
    radians = 2 * math.pi * value / period
    return math.sin(radians), math.cos(radians)


def derive_timing_features(*, publish_hour: Any = None, publish_day: Any = None) -> dict[str, float]:
    """Cyclic encodings of the planned slot, in Sri Lanka local time.

    The creator picks a local hour and weekday on the form, which is the same
    frame the training table uses after converting each publication time to
    Asia/Colombo. An unanswered field falls back to zero, matching the training
    builder's own fillna before the encoding.
    """
    try:
        hour = float(publish_hour) if publish_hour is not None else 0.0
    except (TypeError, ValueError):
        hour = 0.0

    day_index = 0.0
    if publish_day is not None:
        if isinstance(publish_day, str):
            day_index = float(WEEKDAYS.get(publish_day.strip().lower(), 0))
        else:
            try:
                day_index = float(publish_day)
            except (TypeError, ValueError):
                day_index = 0.0

    hour_sin, hour_cos = _cyclic(hour, 24)
    dow_sin, dow_cos = _cyclic(day_index, 7)
    return {
        "publish_hour_sin": hour_sin,
        "publish_hour_cos": hour_cos,
        "publish_dow_sin": dow_sin,
        "publish_dow_cos": dow_cos,
    }
