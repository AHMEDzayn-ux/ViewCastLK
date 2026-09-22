"""Title and timing features must carry the training table's definitions.

Where the training builder is importable these assert against its own
functions rather than against numbers copied by hand, so a change to the
definition on either side shows up here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from app.pre_publication_features import (
    derive_timing_features,
    derive_title_features,
    title_script,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _training_builder():
    scripts_dir = REPO_ROOT / "scripts"
    if not (scripts_dir / "build_training_table.py").exists():
        return None
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import build_training_table  # type: ignore
    except Exception:  # pragma: no cover - only when pipeline deps are absent
        return None
    return build_training_table


TITLES = [
    "Aluth Sindu 2026 | Best Hits!",
    "මගේ අලුත් වීඩියෝව",
    "எனது புதிய வீடியோ",
    "මගේ புதிய video",
    "ALL CAPS TITLE",
    "",
]


@pytest.mark.parametrize("title", TITLES)
def test_script_detection_matches_the_training_builder(title):
    training = _training_builder()
    if training is None:
        pytest.skip("training builder is not importable from this checkout")
    assert title_script(title) == training.title_script(title)


def test_title_measurements_follow_the_training_definitions():
    features = derive_title_features("Top 5 Hits! Best of 2026")
    assert features["title_length"] == 24
    assert features["title_word_count"] == 6
    assert features["title_has_number"] is True
    assert features["title_has_exclaim"] is True
    assert features["title_has_question"] is False
    # Three uppercase letters, T, H and B, over the full 24 characters.
    assert features["title_upper_ratio"] == pytest.approx(3 / 24)


def test_an_empty_title_does_not_divide_by_zero():
    features = derive_title_features("")
    assert features["title_upper_ratio"] == 0.0
    assert features["title_word_count"] == 0.0
    assert features["title_script"] == "latin_script"


def test_absent_title_is_treated_as_empty_rather_than_crashing():
    assert derive_title_features(None)["title_length"] == 0.0


def test_unwritten_description_and_tags_are_zero_as_training_filled_them():
    features = derive_title_features("Anything")
    assert features["description_length"] == 0.0
    assert features["tag_count"] == 0.0


def test_hour_and_day_are_encoded_on_the_circle():
    features = derive_timing_features(publish_hour=18, publish_day="Saturday")
    assert features["publish_hour_sin"] == pytest.approx(math.sin(2 * math.pi * 18 / 24))
    assert features["publish_hour_cos"] == pytest.approx(math.cos(2 * math.pi * 18 / 24))
    assert features["publish_dow_sin"] == pytest.approx(math.sin(2 * math.pi * 5 / 7))


def test_midnight_and_the_hour_before_it_are_neighbours_on_the_circle():
    midnight = derive_timing_features(publish_hour=0)
    late = derive_timing_features(publish_hour=23)
    distance = math.dist(
        (midnight["publish_hour_sin"], midnight["publish_hour_cos"]),
        (late["publish_hour_sin"], late["publish_hour_cos"]),
    )
    assert distance < 0.3, "the encoding lost the wrap from 23:00 to midnight"


def test_unanswered_timing_falls_back_the_way_training_filled_it():
    features = derive_timing_features(publish_hour=None, publish_day=None)
    assert features["publish_hour_sin"] == pytest.approx(0.0)
    assert features["publish_hour_cos"] == pytest.approx(1.0)


def test_numeric_weekday_is_accepted_as_well_as_a_name():
    by_name = derive_timing_features(publish_day="Wednesday")
    by_number = derive_timing_features(publish_day=2)
    assert by_name == by_number
