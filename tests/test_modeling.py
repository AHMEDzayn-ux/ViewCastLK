from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from viewcastlk_ml.modeling import (
    CategoryMedianBaseline,
    CategoryTierMedianBaseline,
    EnsembleBreakoutClassifierBundle,
    EnsembleIncrementModelBundle,
    EnsembleHorizonModelBundle,
    GlobalMedianBaseline,
    HorizonModelBundle,
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    ProbabilityCalibratorBundle,
    ReconciledIndependentTrajectoryModelBundle,
    RelativePerformanceHorizonModelBundle,
    ScaleAwareHorizonModelBundle,
    TitleResidualCorrectionBundle,
    ViralScenarioTrajectoryModelBundle,
    breakout_threshold_views,
    channel_baseline_views,
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

    def test_ensemble_supports_geometric_blending(self) -> None:
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
                7, FakePreprocessor(), FakeRegressor(value), "views"
            )
            for value in (100.0, 300.0)
        ]
        ensemble = EnsembleHorizonModelBundle(
            7,
            components,
            [0.5, 0.5],
            blend_method="geometric",
        )

        prediction = ensemble.predict_views(pd.DataFrame({"x": [1]}))

        expected = np.expm1((np.log1p(100.0) + np.log1p(300.0)) / 2)
        np.testing.assert_allclose(prediction, [expected])

    def test_breakout_ensemble_blends_and_calibrates_probabilities(self) -> None:
        class FakeClassifier:
            def __init__(self, probability):
                self.probability = probability

            def predict_probability(self, raw_features):
                return np.full(len(raw_features), self.probability)

        ensemble = EnsembleBreakoutClassifierBundle(
            components=[FakeClassifier(0.2), FakeClassifier(0.6)],
            weights=[0.25, 0.75],
            calibrator=ProbabilityCalibratorBundle("identity"),
        )

        probability = ensemble.predict_probability(pd.DataFrame({"x": [1, 2]}))

        np.testing.assert_allclose(probability, [0.5, 0.5])

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

    def test_increment_ensemble_returns_weighted_view_growth(self) -> None:
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
            NonnegativeIncrementModelBundle(
                7,
                14,
                FakePreprocessor(),
                FakeRegressor(np.log1p(value)),
            )
            for value in (10.0, 30.0)
        ]
        ensemble = EnsembleIncrementModelBundle(
            7,
            14,
            components,
            [0.25, 0.75],
        )

        prediction = ensemble.predict_increment_views(
            pd.DataFrame({"x": [1, 2]})
        )

        np.testing.assert_allclose(prediction, [25.0, 25.0])

    def test_increment_ensemble_rejects_mismatched_transition(self) -> None:
        class FakePreprocessor:
            def transform(self, raw_features):
                return raw_features

            def get_feature_names_out(self):
                return np.asarray(["x"])

        class FakeRegressor:
            def predict(self, transformed):
                return np.zeros(len(transformed))

        component = NonnegativeIncrementModelBundle(
            14,
            21,
            FakePreprocessor(),
            FakeRegressor(),
        )
        with self.assertRaisesRegex(ValueError, "match the ensemble transition"):
            EnsembleIncrementModelBundle(
                7,
                14,
                [component],
                [1.0],
            )

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

    def test_trajectory_bundle_can_enforce_strictly_positive_growth(self) -> None:
        class FakeBase:
            horizon_days = 7

            def predict_views(self, raw_features):
                return np.asarray([100.0])

        class FakeIncrement:
            def __init__(self, start, end, value):
                self.from_horizon_days = start
                self.to_horizon_days = end
                self.value = value

            def predict_increment_views(self, raw_features):
                return np.asarray([self.value])

        bundle = MonotonicTrajectoryModelBundle(
            base_model=FakeBase(),
            increment_models=[
                FakeIncrement(7, 14, 0.0),
                FakeIncrement(14, 21, 5.0),
                FakeIncrement(21, 30, 0.0),
            ],  # type: ignore[list-item]
            minimum_increment_views=1.0,
        )
        prediction = bundle.predict_views(pd.DataFrame({"x": [1]}))
        np.testing.assert_array_equal(prediction, [[100.0, 101.0, 106.0, 107.0]])
        self.assertTrue((np.diff(prediction, axis=1) > 0).all())

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

    def test_independent_trajectory_reconciles_decreasing_predictions(self) -> None:
        class FakeHorizon:
            def __init__(self, horizon_days, values):
                self.horizon_days = horizon_days
                self.values = np.asarray(values, dtype=float)

            def predict_views(self, raw_features):
                return self.values

        bundle = ReconciledIndependentTrajectoryModelBundle(
            horizon_models=[
                FakeHorizon(7, [100, 10]),
                FakeHorizon(14, [90, 20]),
                FakeHorizon(21, [120, 15]),
                FakeHorizon(30, [110, 30]),
            ]
        )
        prediction = bundle.predict_views(pd.DataFrame({"x": [1, 2]}))
        np.testing.assert_array_equal(
            prediction,
            [[100, 100, 120, 120], [10, 20, 20, 30]],
        )
        self.assertTrue((np.diff(prediction, axis=1) >= 0).all())

    def test_independent_trajectory_can_enforce_strict_growth(self) -> None:
        class FakeHorizon:
            def __init__(self, horizon_days, values):
                self.horizon_days = horizon_days
                self.values = np.asarray(values, dtype=float)

            def predict_views(self, raw_features):
                return self.values

        bundle = ReconciledIndependentTrajectoryModelBundle(
            horizon_models=[
                FakeHorizon(7, [100]),
                FakeHorizon(14, [90]),
                FakeHorizon(21, [100]),
                FakeHorizon(30, [80]),
            ],
            minimum_increment_views=1.0,
        )

        prediction = bundle.predict_views(pd.DataFrame({"x": [1]}))

        np.testing.assert_array_equal(prediction, [[100, 101, 102, 103]])

    def test_independent_trajectory_can_reconcile_from_day_30_backward(self) -> None:
        class FakeHorizon:
            def __init__(self, horizon_days, values):
                self.horizon_days = horizon_days
                self.values = np.asarray(values, dtype=float)

            def predict_views(self, raw_features):
                return self.values

        bundle = ReconciledIndependentTrajectoryModelBundle(
            horizon_models=[
                FakeHorizon(7, [100, 10]),
                FakeHorizon(14, [90, 20]),
                FakeHorizon(21, [120, 15]),
                FakeHorizon(30, [110, 30]),
            ],
            reconciliation="backward_min",
        )

        prediction = bundle.predict_views(pd.DataFrame({"x": [1, 2]}))

        np.testing.assert_array_equal(
            prediction,
            [[90, 90, 110, 110], [10, 15, 15, 30]],
        )
        self.assertTrue((np.diff(prediction, axis=1) >= 0).all())

    def test_independent_trajectory_isotonic_log_reconciliation(self) -> None:
        class FakeHorizon:
            def __init__(self, horizon_days, values):
                self.horizon_days = horizon_days
                self.values = np.asarray(values, dtype=float)

            def predict_views(self, raw_features):
                return self.values

        bundle = ReconciledIndependentTrajectoryModelBundle(
            horizon_models=[
                FakeHorizon(7, [100]),
                FakeHorizon(14, [80]),
                FakeHorizon(21, [120]),
                FakeHorizon(30, [110]),
            ],
            reconciliation="isotonic_log",
        )

        prediction = bundle.predict_views(pd.DataFrame({"x": [1]}))

        self.assertTrue((np.diff(prediction, axis=1) >= 0).all())
        self.assertGreater(prediction[0, 0], 80)
        self.assertLess(prediction[0, 0], 100)
        self.assertGreater(prediction[0, 3], 110)
        self.assertLess(prediction[0, 3], 120)

    def test_channel_baseline_and_relative_model_reconstruct_views(self) -> None:
        frame = pd.DataFrame(
            {
                "prior_d7_view_count": [10, 2],
                "prior_d7_median_views": [1_000, 2_000],
                "prior_d30_view_count": [10, 2],
                "prior_d30_median_views": [4_000, 8_000],
                "ch_avg_views_per_video_at_publish": [5_000, 3_000],
            }
        )
        np.testing.assert_array_equal(
            channel_baseline_views(
                frame, horizon_days=7, strategy="d7_median"
            ),
            [1_000, 3_000],
        )
        np.testing.assert_allclose(
            channel_baseline_views(
                frame, horizon_days=30, strategy="interpolated_d7_d30"
            ),
            [4_000, 3_000],
        )

        class FakePreprocessor:
            def transform(self, raw_features):
                return raw_features

            def get_feature_names_out(self):
                return np.asarray(["x"])

        class FakeRegressor:
            def predict(self, transformed):
                return np.full(len(transformed), np.log(2.0))

        model = RelativePerformanceHorizonModelBundle(
            horizon_days=7,
            preprocessor=FakePreprocessor(),
            regressor=FakeRegressor(),
        )
        np.testing.assert_allclose(model.predict_views(frame), [2_001, 6_001])

    def test_title_residual_correction_adjusts_log_views_and_keeps_order(self) -> None:
        class FakePca:
            def transform(self, values):
                return values

        class FakeScaler:
            def transform(self, values):
                return values

        class FakeRegressor:
            def predict(self, values):
                return np.asarray([[np.log(2.0), 0.0, -1.0, -1.0]])

        correction = TitleResidualCorrectionBundle(
            pca=FakePca(),
            scaler=FakeScaler(),
            regressor=FakeRegressor(),
            embedding_dimensions=2,
            correction_strength=1.0,
            minimum_increment_views=1.0,
        )
        adjusted = correction.apply(
            np.asarray([[100.0, 150.0, 200.0, 250.0]]),
            np.asarray([[0.1, 0.2]]),
        )

        np.testing.assert_allclose(adjusted, [[201.0, 202.0, 203.0, 204.0]])
        self.assertTrue((np.diff(adjusted, axis=1) > 0).all())

    def test_viral_scenario_keeps_one_primary_forecast_and_higher_upside(self) -> None:
        class FakeTrajectory:
            def __init__(self, values):
                self.values = np.asarray([values], dtype=float)

            def predict_views(self, raw_features):
                return self.values

        class FakeClassifier:
            def predict_probability(self, raw_features):
                return np.asarray([0.14])

        frame = pd.DataFrame(
            {
                "prior_d7_view_count": [10],
                "prior_d7_median_views": [1_400],
                "ch_avg_views_per_video_at_publish": [2_000],
            }
        )
        bundle = ViralScenarioTrajectoryModelBundle(
            normal_trajectory=FakeTrajectory([1_400, 1_500, 1_600, 1_700]),
            breakout_classifier=FakeClassifier(),  # type: ignore[arg-type]
            viral_trajectory=FakeTrajectory([8_000, 8_000, 8_000, 8_000]),
        )
        scenario = bundle.predict_scenario_frame(frame)
        self.assertEqual(scenario.loc[0, "normal_day_7_views"], 1_400)
        self.assertEqual(scenario.loc[0, "breakout_probability"], 0.14)
        self.assertEqual(scenario.loc[0, "viral_upside_day_7_views"], 10_000)
        viral = scenario.filter(like="viral_upside").to_numpy()
        self.assertTrue((np.diff(viral, axis=1) > 0).all())

    def test_breakout_threshold_uses_established_history(self) -> None:
        frame = pd.DataFrame(
            {
                "prior_d7_view_count": [10, 1],
                "prior_d7_median_views": [3_000, 3_000],
                "ch_avg_views_per_video_at_publish": [50_000, 4_000],
            }
        )
        np.testing.assert_array_equal(
            breakout_threshold_views(frame), [15_000, 20_000]
        )


if __name__ == "__main__":
    unittest.main()
