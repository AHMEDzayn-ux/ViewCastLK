from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd

from viewcastlk_ml.preprocessing import (
    LLM_SCORE_COLUMNS,
    RAW_INPUT_COLUMNS,
    USER_REMOVED_FEATURES,
    HorizonPreprocessor,
    SmoothedTargetEncoder,
    engineer_prepublication_features,
)


def example_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category_name": ["Music", "News", "Music", "Sports"],
            "publish_hour_slt": [0, 6, 15, 21],
            "duration_seconds": [180, 420, 45, 900],
            "is_short": [False, False, True, False],
            "made_for_kids": [False, False, False, True],
            "default_language": ["si", "en", "si", None],
            "publish_is_weekend": [False, True, False, True],
            "title_length": [20, 30, 15, 40],
            "title_word_count": [4, 6, 3, 8],
            "title_has_number": [False, True, False, False],
            "title_has_question": [False, False, True, False],
            "title_has_exclaim": [True, False, False, True],
            "title_upper_ratio": [0.1, 0.2, 0.0, 0.3],
            "title_script": ["sinhala", "latin", "sinhala", "latin"],
            "tag_count": [5, 10, 2, 8],
            "description_length": [100, 300, 50, 500],
            "ch_subs_at_publish": [1000, 2000, 0, 4000],
            "ch_views_at_publish": [100000, 500000, 20000, 900000],
            "ch_videos_at_publish": [100, 200, 0, 300],
            "channel_age_days_at_publish": [1000, 2000, 100, 3000],
        }
    )


class DeterministicFeatureEngineeringTests(unittest.TestCase):
    def test_publish_hour_boundaries(self) -> None:
        rows = pd.concat([example_rows().iloc[[0]]] * 9, ignore_index=True)
        rows["publish_hour_slt"] = [0, 5, 6, 14, 15, 20, 21, 23, np.nan]
        actual = (
            engineer_prepublication_features(rows)["publish_time_bucket"]
            .astype("string")
            .tolist()
        )
        self.assertEqual(
            actual,
            [
                "early_morning",
                "early_morning",
                "morning_afternoon",
                "morning_afternoon",
                "evening",
                "evening",
                "late_night",
                "late_night",
                pd.NA,
            ],
        )

    def test_zero_denominators_become_missing_not_infinity(self) -> None:
        engineered = engineer_prepublication_features(example_rows())
        ratios = engineered[
            ["ch_videos_per_day", "ch_views_per_video", "ch_views_per_sub"]
        ].to_numpy(dtype=float)
        self.assertFalse(np.isinf(ratios).any())
        self.assertTrue(np.isnan(engineered.loc[2, "ch_views_per_video"]))
        self.assertTrue(np.isnan(engineered.loc[2, "ch_views_per_sub"]))


class TargetEncoderTests(unittest.TestCase):
    def test_unseen_category_uses_training_global_mean(self) -> None:
        encoder = SmoothedTargetEncoder(smoothing=10).fit(
            pd.Series(["Music", "Music", "News"]),
            np.array([1.0, 2.0, 3.0]),
        )
        transformed = encoder.transform(pd.Series(["Unknown"]))
        self.assertAlmostEqual(transformed.iloc[0, 0], 2.0)


class HorizonPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = example_rows()
        self.y_log = np.log1p(np.array([1000, 2000, 300, 4000], dtype=float))

    def test_training_contract_requires_all_raw_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            HorizonPreprocessor().fit(self.X.drop(columns="duration_seconds"), self.y_log)

    def test_output_is_numeric_ordered_and_leakage_free(self) -> None:
        preprocessor = HorizonPreprocessor().fit(self.X, self.y_log)
        transformed = preprocessor.transform(self.X)
        self.assertEqual(list(transformed.columns), list(preprocessor.get_feature_names_out()))
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in transformed.dtypes))
        self.assertNotIn("publish_hour_slt", transformed.columns)
        self.assertNotIn("category_name", transformed.columns)
        self.assertNotIn("channel_id", transformed.columns)
        self.assertTrue(set(LLM_SCORE_COLUMNS).isdisjoint(transformed.columns))
        self.assertTrue(set(USER_REMOVED_FEATURES).isdisjoint(transformed.columns))
        self.assertNotIn("made_for_kids", transformed.columns)
        self.assertFalse(np.isinf(transformed.to_numpy()).any())

    def test_transform_accepts_omitted_optional_fields(self) -> None:
        preprocessor = HorizonPreprocessor().fit(self.X, self.y_log)
        request = self.X.iloc[[0]].drop(
            columns=["publish_hour_slt", "publish_is_weekend", "default_language"]
        )
        transformed = preprocessor.transform(request)
        self.assertEqual(transformed.shape[0], 1)
        self.assertEqual(transformed.shape[1], len(preprocessor.get_feature_names_out()))

    def test_removed_features_are_not_required_for_fit(self) -> None:
        reduced_input = self.X.drop(columns=list(USER_REMOVED_FEATURES))
        preprocessor = HorizonPreprocessor().fit(reduced_input, self.y_log)
        transformed = preprocessor.transform(reduced_input)
        self.assertTrue(set(USER_REMOVED_FEATURES).isdisjoint(transformed.columns))

    def test_serialised_component_preserves_feature_parity(self) -> None:
        preprocessor = HorizonPreprocessor().fit(self.X, self.y_log)
        expected = preprocessor.transform(self.X.iloc[[0]])
        restored = pickle.loads(pickle.dumps(preprocessor))
        actual = restored.transform(self.X.iloc[[0]])
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
