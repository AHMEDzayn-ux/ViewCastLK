"""The channel's YouTube topics must reach the model the way training folded them.

The model learned topic_* from the training table, where
scripts/prepare_model_datasets.py canonicalises YouTube's topic URLs and drops
a parent topic when one of its own children is present. Serving has to do the
same, or the feature means something different at prediction time.
"""

from app.feature_builder import (
    build_candidate_feature_frame,
    canonical_topic_labels,
    derive_topic_features,
)

WIKI = "https://en.wikipedia.org/wiki/"


def test_channel_without_topics_is_marked_missing():
    features = derive_topic_features(None)
    assert features["topic_missing"] is True
    assert not any(value for key, value in features.items() if key != "topic_missing")


def test_empty_list_is_treated_as_missing():
    assert derive_topic_features([])["topic_missing"] is True


def test_specific_music_topics_collapse_to_the_music_label():
    labels = canonical_topic_labels([WIKI + "Pop_music", WIKI + "Rock_music"])
    assert labels == {"Music"}
    features = derive_topic_features([WIKI + "Pop_music"])
    assert features["topic_music"] is True
    assert features["topic_missing"] is False


def test_url_escapes_are_decoded():
    assert canonical_topic_labels([WIKI + "Lifestyle_%28sociology%29"]) == {"Lifestyle"}


def test_parent_topic_is_dropped_when_a_child_is_present():
    # Society is the parent of Politics: a political channel counts once.
    labels = canonical_topic_labels([WIKI + "Society", WIKI + "Politics"])
    assert labels == {"Politics"}
    features = derive_topic_features([WIKI + "Society", WIKI + "Politics"])
    assert features["topic_politics"] is True
    assert features["topic_society"] is False


def test_parent_survives_without_a_child():
    assert canonical_topic_labels([WIKI + "Society"]) == {"Society"}


def test_unknown_topics_do_not_set_any_flag_but_clear_missing():
    features = derive_topic_features([WIKI + "Underwater_basket_weaving"])
    assert features["topic_missing"] is False
    assert not any(value for key, value in features.items() if key != "topic_missing")


def test_topics_reach_the_model_frame_from_the_channel_lookup():
    request = {
        "category": "Music",
        "durationSeconds": 480,
        "audioLanguage": "English",
        "plannedPublishDay": None,
        "plannedPublishHour": None,
    }
    channel_stats = {
        "subscriberCount": 11000,
        "totalViewCount": 500000,
        "videoCount": 120,
        "channelAgeDays": 900,
        "topicCategories": [WIKI + "Pop_music", WIKI + "Entertainment"],
    }
    frame = build_candidate_feature_frame(request, channel_stats)
    assert bool(frame.loc[0, "topic_music"]) is True
    assert bool(frame.loc[0, "topic_entertainment"]) is True
    assert bool(frame.loc[0, "topic_missing"]) is False


def test_frame_still_builds_when_the_channel_has_no_topics():
    request = {
        "category": "Music",
        "durationSeconds": 480,
        "audioLanguage": "English",
        "plannedPublishDay": None,
        "plannedPublishHour": None,
    }
    frame = build_candidate_feature_frame(request, {"subscriberCount": 11000})
    assert bool(frame.loc[0, "topic_missing"]) is True
