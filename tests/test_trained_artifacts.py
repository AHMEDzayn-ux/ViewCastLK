from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "mvp_v2_reduced"
USER_REMOVED_FEATURES = {
    "title_length",
    "title_word_count",
    "title_upper_ratio",
    "tag_count",
    "description_length",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
}


@unittest.skipUnless(
    (ARTIFACT_ROOT / "training_manifest.json").exists(),
    "candidate training artifacts have not been generated",
)
class TrainedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ARTIFACT_ROOT / "training_manifest.json").read_text(encoding="utf-8")
        )
        cls.predictions = pd.read_csv(ARTIFACT_ROOT / "cv_predictions.csv")
        cls.assignments = pd.read_csv(ARTIFACT_ROOT / "holdout_assignments.csv")
        cls.summary = pd.read_csv(ARTIFACT_ROOT / "cv_summary_metrics.csv")

    def test_four_independent_bundles_are_complete_and_hash_verified(self) -> None:
        self.assertEqual({record["horizon_days"] for record in self.manifest["models"]}, {7, 14, 21, 30})
        for record in self.manifest["models"]:
            model_path = ARTIFACT_ROOT / record["model_path"]
            actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, record["model_sha256"])
            bundle = joblib.load(model_path)
            self.assertEqual(bundle.horizon_days, record["horizon_days"])
            feature_order = json.loads(
                (ARTIFACT_ROOT / record["feature_order_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(bundle.feature_names, feature_order)
            self.assertTrue(USER_REMOVED_FEATURES.isdisjoint(feature_order))

    def test_final_test_rows_never_appear_in_cv_predictions(self) -> None:
        for horizon in (7, 14, 21, 30):
            test_rows = set(
                self.assignments.loc[
                    (self.assignments["horizon_days"] == horizon)
                    & (self.assignments["partition"] == "test_untouched"),
                    "source_row_index",
                ]
            )
            predicted_rows = set(
                self.predictions.loc[
                    self.predictions["horizon_days"] == horizon,
                    "source_row_index",
                ]
            )
            self.assertTrue(test_rows.isdisjoint(predicted_rows))

    def test_cv_predictions_are_finite_and_non_negative(self) -> None:
        prediction_columns = [
            column for column in self.predictions if column.startswith("predicted_")
        ]
        values = self.predictions[prediction_columns].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())

    def test_xgboost_beats_strongest_baseline_on_combined_primary_metric(self) -> None:
        combined = self.summary[self.summary["horizon_days"].astype(str) == "combined"].set_index("model")
        self.assertLess(
            combined.loc["xgboost_mvp", "mape_nonzero_pct"],
            combined.loc["category_tier_median", "mape_nonzero_pct"],
        )
        self.assertLess(
            combined.loc["xgboost_mvp", "rmsle"],
            combined.loc["category_tier_median", "rmsle"],
        )

    def test_manifest_marks_test_as_untouched(self) -> None:
        self.assertEqual(
            self.manifest["status"], "candidate_untouched_test_not_evaluated"
        )
        self.assertEqual(
            set(self.manifest["user_removed_features"]), USER_REMOVED_FEATURES
        )


if __name__ == "__main__":
    unittest.main()
