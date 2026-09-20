from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from viewcastlk_ml.horizon_preprocessing import (
    EXCLUDED_MODEL_COLUMNS,
    LLM_SCORE_COLUMNS,
    RARE_CATEGORY,
    TOPIC_COLUMNS,
    HorizonDatasetPreprocessor,
    subscriber_tier_from_count,
)


def example_frame() -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(
        {
            "category_name": ["Music", "News", "Music", None],
            "duration_seconds": [30, 300, 60, 900],
            "ch_subs_at_publish": [100, 1000, 0, 5000],
            "ch_avg_views_per_video_at_publish": [1000, 2500, np.nan, 1800],
            "ch_videos_at_publish": [10, 200, 0, 500],
            "channel_age_days_at_publish": [100, 1000, 0, 2000],
            "is_short": [True, False, True, False],
            "made_for_kids": [False, False, True, False],
            "publish_is_weekend": [False, True, False, True],
            "default_language": ["si", "en", "si", None],
            "subscriber_tier": [
                "under_1k", "1k_to_10k", "under_1k", "1k_to_10k"
            ],
            "publish_time_bucket": [
                "early_morning",
                "morning_afternoon",
                "evening",
                "late_night",
            ],
        }
    )
    for column in TOPIC_COLUMNS:
        frame[column] = False
    frame.loc[0, "topic_music"] = True
    frame.loc[1, "topic_politics"] = True
    return frame


class HorizonDatasetPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = example_frame()
        self.y = np.log1p([100, 1000, 500, 5000])

    def test_output_is_numeric_and_excludes_removed_fields(self) -> None:
        transformed = HorizonDatasetPreprocessor().fit_transform(
            self.X, self.y
        )
        self.assertIn("ch_subs_at_publish", transformed.columns)
        self.assertTrue(
            all(np.issubdtype(dtype, np.number) for dtype in transformed.dtypes)
        )
        self.assertTrue(set(EXCLUDED_MODEL_COLUMNS).isdisjoint(transformed))
        self.assertNotIn("made_for_kids", transformed.columns)
        self.assertTrue(set(LLM_SCORE_COLUMNS).isdisjoint(transformed))
        self.assertFalse(np.isinf(transformed.to_numpy()).any())

    def test_zero_channel_age_produces_missing_upload_rate(self) -> None:
        transformed = HorizonDatasetPreprocessor().fit_transform(
            self.X, self.y
        )
        self.assertTrue(np.isnan(transformed.loc[2, "ch_videos_per_day"]))

    def test_subscriber_tier_boundaries_and_model_columns(self) -> None:
        values = pd.Series([
            0, 999, 1000, 9999, 10000, 99999,
            100000, 249999, 250000, 499999,
            500000, 999999, 1000000, np.nan,
        ])
        self.assertEqual(
            subscriber_tier_from_count(values).tolist(),
            [
                "under_1k", "under_1k",
                "1k_to_10k", "1k_to_10k",
                "10k_to_100k", "10k_to_100k",
                "100k_to_250k", "100k_to_250k",
                "250k_to_500k", "250k_to_500k",
                "500k_to_1m", "500k_to_1m",
                "1m_plus", "missing",
            ],
        )
        transformed = HorizonDatasetPreprocessor().fit_transform(
            self.X, self.y
        )
        tier_columns = [
            column
            for column in transformed
            if column.startswith("subscriber_tier_")
        ]
        self.assertTrue(tier_columns)

    def test_unknown_category_uses_training_global_mean(self) -> None:
        preprocessor = HorizonDatasetPreprocessor().fit(self.X, self.y)
        request = self.X.iloc[[0]].copy()
        request["category_name"] = "Unseen category"
        transformed = preprocessor.transform(request)
        self.assertAlmostEqual(
            transformed.iloc[0]["category_encoded_log"],
            preprocessor.category_encoder_.global_mean_,
        )

    def test_transform_tolerates_missing_optional_inference_fields(self) -> None:
        preprocessor = HorizonDatasetPreprocessor().fit(self.X, self.y)
        request = self.X.iloc[[0]].drop(
            columns=["default_language", "ch_subs_at_publish"]
        )
        transformed = preprocessor.transform(request)
        self.assertEqual(transformed.shape[1], len(preprocessor.get_feature_names_out()))

    def test_removed_content_fields_are_not_required_for_training(self) -> None:
        removed = ["definition", "caption", "made_for_kids"]
        reduced = self.X.drop(
            columns=[column for column in removed if column in self.X],
        )
        transformed = HorizonDatasetPreprocessor().fit_transform(reduced, self.y)
        self.assertTrue(set(removed).isdisjoint(transformed.columns))

    def test_extended_metadata_and_history_are_model_features(self) -> None:
        frame = self.X.copy()
        frame["title_length"] = [10, 20, 30, 40]
        frame["title_script"] = [
            "latin_script", "sinhala_script", "tamil_script", "mixed_script"
        ]
        frame["prior_d7_median_views"] = [np.nan, 100, 200, 300]
        transformed = HorizonDatasetPreprocessor().fit_transform(frame, self.y)
        self.assertIn("title_length", transformed.columns)
        self.assertIn("prior_d7_median_views", transformed.columns)
        self.assertTrue(
            any(column.startswith("title_script_") for column in transformed)
        )

    def test_enhanced_history_features_are_opt_in(self) -> None:
        frame = self.X.copy()
        frame["prior_d7_view_count"] = [0, 2, 3, 4]
        frame["prior_d30_view_count"] = [0, 0, 1, 2]
        frame["prior_d7_median_views"] = [np.nan, 100, 200, 400]
        frame["prior_d7_recent5_median_views"] = [np.nan, 150, 100, 800]
        frame["prior_d7_last_views"] = [np.nan, 200, 100, 1000]
        frame["prior_same_category_d7_count"] = [0, 1, 2, 3]
        frame["prior_same_category_d7_median_views"] = [np.nan, 90, 250, 500]
        frame["prior_same_format_d7_count"] = [0, 1, 1, 2]
        frame["prior_same_format_d7_median_views"] = [np.nan, 80, 220, 600]
        frame["uploads_previous_7d"] = [0, 1, 2, 3]
        frame["uploads_previous_30d"] = [0, 2, 6, 12]
        transformed = HorizonDatasetPreprocessor(
            include_enhanced_features=True
        ).fit_transform(frame, self.y)
        self.assertIn("prior_d7_recent_to_all_log_ratio", transformed)
        self.assertIn("prior_same_format_d7_median_views", transformed)
        self.assertEqual(transformed.loc[0, "has_prior_d7_history"], 0)
        self.assertEqual(transformed.loc[1, "has_prior_d7_history"], 1)
        self.assertGreater(
            transformed.loc[1, "prior_d7_recent_to_all_log_ratio"], 0
        )

    def test_rare_languages_and_scripts_collapse_to_other(self) -> None:
        frame = self.X.copy()
        frame["default_language"] = ["si", "si", "en", "tiny"]
        frame["title_script"] = [
            "sinhala_script", "sinhala_script", "latin_script", "mixed_script"
        ]
        preprocessor = HorizonDatasetPreprocessor(
            collapse_rare_categories=True,
            rare_category_min_count=2,
        ).fit(frame, self.y)
        transformed = preprocessor.transform(frame)
        self.assertIn(f"default_language_{RARE_CATEGORY}", transformed)
        self.assertIn(f"title_script_{RARE_CATEGORY}", transformed)
        self.assertNotIn("default_language_tiny", transformed)
        unseen = frame.iloc[[0]].copy()
        unseen["default_language"] = "unseen"
        unseen_transformed = preprocessor.transform(unseen)
        self.assertEqual(
            unseen_transformed.iloc[0][f"default_language_{RARE_CATEGORY}"], 1
        )

    def test_drop_features_removes_ablation_columns(self) -> None:
        transformed = HorizonDatasetPreprocessor(
            drop_features=("publish_dow_sin", "title_has_question")
        ).fit_transform(self.X, self.y)
        self.assertNotIn("publish_dow_sin", transformed)
        self.assertNotIn("title_has_question", transformed)

    def test_llm_switch_rejects_incomplete_backfill(self) -> None:
        frame = self.X.copy()
        for column in LLM_SCORE_COLUMNS:
            frame[column] = 5.0
        frame.loc[0, LLM_SCORE_COLUMNS[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "complete consistent backfill"):
            HorizonDatasetPreprocessor(include_llm_scores=True).fit(
                frame, self.y
            )

    def test_llm_switch_accepts_complete_backfill(self) -> None:
        frame = self.X.copy()
        for column in LLM_SCORE_COLUMNS:
            frame[column] = 5.0
        transformed = HorizonDatasetPreprocessor(
            include_llm_scores=True
        ).fit_transform(frame, self.y)
        self.assertTrue(set(LLM_SCORE_COLUMNS).issubset(transformed.columns))


if __name__ == "__main__":
    unittest.main()
