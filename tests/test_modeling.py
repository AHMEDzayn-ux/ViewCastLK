from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from viewcastlk_ml.modeling import (
    CategoryMedianBaseline,
    CategoryTierMedianBaseline,
    EnsembleHorizonModelBundle,
    GlobalMedianBaseline,
    HorizonModelBundle,
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    ScaleAwareHorizonModelBundle,
    log_target_inlier_mask,
    regression_metrics,
    views_from_log_predictions,
)
from viewcastlk_ml.preprocessing import HorizonPreprocessor


class MetricAndOutlierTests(unittest.TestCase):
    def test_metrics_report_zero_targets_without_infinite_mape(self) -> None:
        metrics = regression_metrics([0, 100, 200], [10, 110, 180])
        self.assertEqual(metrics["zero_target_rows"], 1)
        self.assertTrue(np.isfinite(metrics["mape_nonzero_pct"]))
        self.assertTrue(np.isfinite(metrics["smape_pct"]))
        self.assertTrue(np.isfinite(metrics["rmsle"]))
        self.assertTrue(np.isfinite(metrics["log_r2"]))
        self.assertTrue(np.isfinite(metrics["wape_pct"]))
        self.assertTrue(np.isfinite(metrics["top_decile_wape_pct"]))

    def test_wape_prioritises_large_absolute_misses(self) -> None:
        metrics = regression_metrics([2, 32_295], [28, 2_798])
        expected = (26 + 29_497) / (2 + 32_295) * 100
        self.assertAlmostEqual(metrics["wape_pct"], expected)
        self.assertGreater(29_497 / 26, 1_000)

    def test_log_target_filter_returns_bounds_and_mask(self) -> None:
        ordinary = np.linspace(1.0, 2.0, 100)
        values = np.append(ordinary, 20.0)
        mask, lower, upper = log_target_inlier_mask(values, sigma=3)
        self.assertEqual(mask.sum(), 100)
        self.assertLess(lower, upper)
        self.assertFalse(mask[-1])

    def test_log_prediction_inverse_is_non_negative(self) -> None:
        actual = views_from_log_predictions([-2.0, 0.0, np.log1p(100)])
        np.testing.assert_allclose(actual, [0.0, 0.0, 100.0])


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                "category_name": ["Music", "Music", "News", "News", "News", "Music"],
                "ch_subs_at_publish": [100, 1000, 100, 1000, 10000, 10000],
            }
        )
        self.y_log = np.log1p([100, 1000, 200, 2000, 20000, 10000])
        self.channels = ["a", "b", "c", "d", "e", "f"]

    def test_category_baseline_uses_global_fallback(self) -> None:
        baseline = CategoryMedianBaseline().fit(self.X, self.y_log)
        prediction = baseline.predict(
            pd.DataFrame({"category_name": ["Unknown"], "ch_subs_at_publish": [500]})
        )
        self.assertAlmostEqual(prediction[0], float(np.median(self.y_log)))

    def test_global_baseline_always_uses_training_median(self) -> None:
        baseline = GlobalMedianBaseline().fit(self.X, self.y_log)
        prediction = baseline.predict(self.X.iloc[:2])
        np.testing.assert_allclose(prediction, np.median(self.y_log))

    def test_category_tier_baseline_handles_unseen_category(self) -> None:
        baseline = CategoryTierMedianBaseline().fit(
            self.X, self.y_log, channel_ids=self.channels
        )
        prediction = baseline.predict(
            pd.DataFrame({"category_name": ["Unknown"], "ch_subs_at_publish": [500]})
        )
        self.assertAlmostEqual(prediction[0], float(np.median(self.y_log)))


class BundleTests(unittest.TestCase):
    def test_bundle_clips_negative_view_prediction(self) -> None:
        class FakePreprocessor:
            def transform(self, raw_features):
                return raw_features

            def get_feature_names_out(self):
                return np.asarray(["x"])

        class FakeRegressor:
            def predict(self, transformed):
                return np.full(len(transformed), -1.0)

        bundle = HorizonModelBundle(
            horizon_days=7,
            preprocessor=FakePreprocessor(),  # type: ignore[arg-type]
            regressor=FakeRegressor(),  # type: ignore[arg-type]
        )
        prediction = bundle.predict_views(pd.DataFrame({"x": [1, 2]}))
        np.testing.assert_array_equal(prediction, [0.0, 0.0])

    def test_ensemble_returns_weighted_view_prediction(self) -> None:
        class FakePreprocessor:
            def transform(self, raw_features):
                return raw_features

            def get_feature_names_out(self):
                return np.asarray(["x"])

        class FakeRegressor:
            def __init__(self, prediction):
                self.prediction = prediction

            def predict(self, transformed):
                return np.full(len(transformed), self.prediction)

        components = [
            ScaleAwareHorizonModelBundle(
                7, FakePreprocessor(), FakeRegressor(100.0), "views"
            ),
            ScaleAwareHorizonModelBundle(
                7, FakePreprocessor(), FakeRegressor(300.0), "views"
            ),
        ]
        ensemble = EnsembleHorizonModelBundle(7, components, [0.25, 0.75])
        prediction = ensemble.predict_views(pd.DataFrame({"x": [1, 2]}))
        np.testing.assert_array_equal(prediction, [250.0, 250.0])

    def test_increment_bundle_returns_nonnegative_view_growth(self) -> None:
        class FakePreprocessor:
            def transform(self, raw_features):
                return raw_features

            def get_feature_names_out(self):
                return np.asarray(["x"])

        class FakeRegressor:
            def predict(self, transformed):
                return np.asarray([-1.0, np.log1p(50.0)])

        bundle = NonnegativeIncrementModelBundle(
            7,
            14,
            FakePreprocessor(),
            FakeRegressor(),
        )
        prediction = bundle.predict_increment_views(
            pd.DataFrame({"x": [1, 2]})
        )
        np.testing.assert_allclose(prediction, [0.0, 50.0])

    def test_trajectory_bundle_is_monotonic_by_construction(self) -> None:
        class FakeBase:
            horizon_days = 7

            def predict_views(self, raw_features):
                return np.asarray([100.0, 200.0])

        class FakeIncrement:
            def __init__(self, start, end, values):
                self.from_horizon_days = start
                self.to_horizon_days = end
                self.values = np.asarray(values, dtype=float)

            def predict_increment_views(self, raw_features):
                return self.values

        bundle = MonotonicTrajectoryModelBundle(
            base_model=FakeBase(),
            increment_models=[
                FakeIncrement(7, 14, [50, 0]),
                FakeIncrement(14, 21, [20, 25]),
                FakeIncrement(21, 30, [10, 5]),
            ],  # type: ignore[list-item]
        )
        prediction = bundle.predict_views(pd.DataFrame({"x": [1, 2]}))
        np.testing.assert_array_equal(
            prediction,
            [[100.0, 150.0, 170.0, 180.0], [200.0, 200.0, 225.0, 230.0]],
        )
        self.assertTrue((np.diff(prediction, axis=1) >= 0).all())

    def test_trajectory_bundle_rejects_a_broken_increment_chain(self) -> None:
        class FakeBase:
            horizon_days = 7

        class FakeIncrement:
            def __init__(self, start, end):
                self.from_horizon_days = start
                self.to_horizon_days = end

        with self.assertRaisesRegex(ValueError, "7->14->21->30"):
            MonotonicTrajectoryModelBundle(
                base_model=FakeBase(),
                increment_models=[
                    FakeIncrement(7, 14),
                    FakeIncrement(14, 30),
                ],  # type: ignore[list-item]
            )


if __name__ == "__main__":
    unittest.main()
