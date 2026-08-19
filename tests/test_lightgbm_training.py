import unittest

import numpy as np
import pandas as pd

from scripts.train_lightgbm_models import (
    CANDIDATES,
    predictions_in_views,
    wape_eval,
)
from viewcastlk_ml.modeling import build_lgbm_regressor


class LightGBMTrainingTests(unittest.TestCase):
    def test_builder_fits_a_lightgbm_regressor(self):
        X = pd.DataFrame(
            {
                "feature_a": np.arange(20, dtype=float),
                "feature_b": np.arange(20, dtype=float) % 3,
            }
        )
        y = np.log1p(np.arange(20, dtype=float) * 10)
        model = build_lgbm_regressor(
            n_estimators=5,
            min_child_samples=1,
            n_jobs=1,
        ).fit(X, y)
        predictions = model.predict(X)
        self.assertTrue(model.__class__.__module__.startswith("lightgbm"))
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(np.isfinite(predictions).all())

    def test_wape_callback_uses_view_scale_for_log_targets(self):
        metric_name, value, higher_is_better = wape_eval("log1p")(
            np.log1p([100.0, 200.0]),
            np.log1p([90.0, 230.0]),
        )
        self.assertEqual(metric_name, "view_wape")
        self.assertAlmostEqual(value, 40.0 / 300.0)
        self.assertFalse(higher_is_better)

    def test_all_candidate_predictions_are_converted_to_non_negative_views(self):
        for candidate in CANDIDATES:
            target_values = (
                np.asarray([-1.0, 0.0, np.log1p(10.0)])
                if candidate.target_scale == "log1p"
                else np.asarray([-10.0, 0.0, 10.0])
            )
            converted = predictions_in_views(target_values, candidate.target_scale)
            self.assertTrue((converted >= 0).all())


if __name__ == "__main__":
    unittest.main()
