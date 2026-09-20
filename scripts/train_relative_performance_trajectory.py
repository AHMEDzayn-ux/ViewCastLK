"""Train and gate a channel-relative independent-horizon trajectory.

The model predicts each horizon's log performance relative to a channel
reference available before publication. Absolute views are reconstructed only
at inference, and a one-view reconciliation constraint guarantees a strictly
increasing cumulative trajectory without feeding earlier predictions into
later horizon models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_checkpoint5_models import CATEGORY_SMOOTHING  # noqa: E402
from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    inner_validation_positions,
    load_all_horizons,
    load_transition,
    sha256_file,
)
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    ReconciledIndependentTrajectoryModelBundle,
    RelativePerformanceHorizonModelBundle,
    ViralScenarioTrajectoryModelBundle,
    build_xgb_regressor,
    channel_baseline_views,
    regression_metrics,
    views_from_log_predictions,
)


FEATURE_OPTIONS: dict[str, Any] = {
    "include_enhanced_features": True,
    "collapse_rare_categories": True,
    "rare_category_min_count": 100,
    "drop_features": (
        "publish_dow_sin",
        "publish_dow_cos",
        "title_has_question",
        "topic_technology",
    ),
}
BASELINE_STRATEGIES = (
    "d7_median",
    "interpolated_d7_d30",
    "channel_average",
)
MINIMUM_HISTORY_COUNT = 5
FALLBACK_VIEWS = 1_000.0
MINIMUM_BASELINE_VIEWS = 100.0
MINIMUM_INCREMENT_VIEWS = 1.0


def relative_log_target(
    X: pd.DataFrame,
    actual_views: np.ndarray,
    *,
    horizon_days: int,
    strategy: str,
) -> np.ndarray:
    actual = np.asarray(actual_views, dtype=float)
    baseline = channel_baseline_views(
        X,
        horizon_days=horizon_days,
        strategy=strategy,
        minimum_history_count=MINIMUM_HISTORY_COUNT,
        fallback_views=FALLBACK_VIEWS,
        minimum_baseline_views=MINIMUM_BASELINE_VIEWS,
    )
    if len(actual) != len(baseline) or not np.isfinite(actual).all():
        raise ValueError("Actual views and channel baseline must be finite and aligned")
    if (actual < 0).any():
        raise ValueError("Actual views cannot be negative")
    return np.log1p(actual) - np.log1p(baseline)


def reconstruct_views(
    X: pd.DataFrame,
    relative_log_prediction: np.ndarray,
    *,
    horizon_days: int,
    strategy: str,
) -> np.ndarray:
    baseline = channel_baseline_views(
        X,
        horizon_days=horizon_days,
        strategy=strategy,
        minimum_history_count=MINIMUM_HISTORY_COUNT,
        fallback_views=FALLBACK_VIEWS,
        minimum_baseline_views=MINIMUM_BASELINE_VIEWS,
    )
    return views_from_log_predictions(
        np.log1p(baseline) + np.asarray(relative_log_prediction, dtype=float)
    )


def strict_reconcile(predictions: np.ndarray) -> np.ndarray:
    result = np.maximum(0.0, np.asarray(predictions, dtype=float)).copy()
    if result.ndim != 2 or result.shape[1] != len(HORIZONS):
        raise ValueError("Trajectory predictions must have four horizon columns")
    for position in range(1, result.shape[1]):
        result[:, position] = np.maximum(
            result[:, position],
            result[:, position - 1] + MINIMUM_INCREMENT_VIEWS,
        )
    return result


def fit_fold_model(
    *,
    X: pd.DataFrame,
    target: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    horizon_days: int,
    strategy: str,
    max_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, int, int]:
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed_training = preprocessor.fit_transform(
        X.iloc[training], target[training]
    )
    transformed_validation = preprocessor.transform(X.iloc[validation])
    model = build_xgb_regressor(
        n_estimators=max_estimators,
        early_stopping_rounds=30,
        n_jobs=n_jobs,
    )
    model.fit(
        transformed_training,
        target[training],
        eval_set=[(transformed_validation, target[validation])],
        verbose=False,
    )
    relative_prediction = model.predict(transformed_validation)
    absolute_prediction = reconstruct_views(
        X.iloc[validation],
        relative_prediction,
        horizon_days=horizon_days,
        strategy=strategy,
    )
    return (
        absolute_prediction,
        int(model.best_iteration) + 1,
        transformed_training.shape[1],
    )


def cross_validate_strategies(
    *,
    X: pd.DataFrame,
    targets: dict[int, np.ndarray],
    channels: pd.Series,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    predictions = {
        strategy: np.full((len(X), len(HORIZONS)), np.nan, dtype=float)
        for strategy in BASELINE_STRATEGIES
    }
    fold_rows: list[dict[str, Any]] = []
    split_positions = list(GroupKFold(n_splits=folds).split(X, groups=channels))
    for fold, (training, validation) in enumerate(split_positions, start=1):
        details = []
        for strategy in BASELINE_STRATEGIES:
            for column, horizon in enumerate(HORIZONS):
                target = relative_log_target(
                    X,
                    targets[horizon],
                    horizon_days=horizon,
                    strategy=strategy,
                )
                predicted, best_iteration, feature_count = fit_fold_model(
                    X=X,
                    target=target,
                    training=training,
                    validation=validation,
                    horizon_days=horizon,
                    strategy=strategy,
                    max_estimators=max_estimators,
                    n_jobs=n_jobs,
                )
                predictions[strategy][validation, column] = predicted
                metrics = regression_metrics(targets[horizon][validation], predicted)
                fold_rows.append(
                    {
                        "fold": fold,
                        "strategy": strategy,
                        "horizon_days": horizon,
                        "training_rows": len(training),
                        "validation_rows": len(validation),
                        "features": feature_count,
                        "best_iteration": best_iteration,
                        **metrics,
                    }
                )
                details.append(f"{strategy}/D{horizon}={best_iteration}")
        print(f"Relative target fold {fold}: " + ", ".join(details), flush=True)

    metric_rows: list[dict[str, Any]] = []
    reconciled: dict[str, np.ndarray] = {}
    for strategy, raw in predictions.items():
        if not np.isfinite(raw).all():
            raise AssertionError(f"Missing OOF predictions for {strategy}")
        corrected = strict_reconcile(raw)
        reconciled[strategy] = corrected
        for method, values in (("raw_independent", raw), ("strict_reconciled", corrected)):
            for column, horizon in enumerate(HORIZONS):
                metric_rows.append(
                    {
                        "strategy": strategy,
                        "method": method,
                        "horizon_days": horizon,
                        **regression_metrics(targets[horizon], values[:, column]),
                    }
                )
    return reconciled, pd.DataFrame(fold_rows), pd.DataFrame(metric_rows)


def average_rmsle(metrics: pd.DataFrame, *, strategy: str, method: str) -> float:
    selected = metrics[metrics["strategy"].eq(strategy) & metrics["method"].eq(method)]
    return float(selected["rmsle"].mean())


def fit_relative_horizon_model(
    *,
    X: pd.DataFrame,
    actual_views: np.ndarray,
    channels: pd.Series,
    horizon_days: int,
    strategy: str,
    max_estimators: int,
    n_jobs: int,
) -> RelativePerformanceHorizonModelBundle:
    target = relative_log_target(
        X,
        actual_views,
        horizon_days=horizon_days,
        strategy=strategy,
    )
    training, validation = inner_validation_positions(
        X, channels.reset_index(drop=True)
    )
    selection_preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed_training = selection_preprocessor.fit_transform(
        X.iloc[training], target[training]
    )
    transformed_validation = selection_preprocessor.transform(X.iloc[validation])
    selection_model = build_xgb_regressor(
        n_estimators=max_estimators,
        early_stopping_rounds=30,
        n_jobs=n_jobs,
    )
    selection_model.fit(
        transformed_training,
        target[training],
        eval_set=[(transformed_validation, target[validation])],
        verbose=False,
    )
    selected_estimators = max(1, int(selection_model.best_iteration) + 1)

    final_preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed = final_preprocessor.fit_transform(X, target)
    final_model = build_xgb_regressor(
        n_estimators=selected_estimators,
        early_stopping_rounds=None,
        n_jobs=n_jobs,
    )
    final_model.fit(transformed, target, verbose=False)
    return RelativePerformanceHorizonModelBundle(
        horizon_days=horizon_days,
        preprocessor=final_preprocessor,
        regressor=final_model,
        baseline_strategy=strategy,
        minimum_history_count=MINIMUM_HISTORY_COUNT,
        fallback_views=FALLBACK_VIEWS,
        minimum_baseline_views=MINIMUM_BASELINE_VIEWS,
        training_metadata={
            "target": "log1p(actual_views) - log1p(channel_baseline_views)",
            "baseline_strategy": strategy,
            "training_rows": len(X),
            "training_channels": int(channels.nunique()),
            "selected_n_estimators": selected_estimators,
        },
    )


def trajectory_metric_rows(
    method: str,
    targets: dict[int, np.ndarray],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "horizon_days": horizon,
            "rows": len(predictions),
            **regression_metrics(targets[horizon], predictions[:, column]),
        }
        for column, horizon in enumerate(HORIZONS)
    ]


def run_training(
    *,
    project_root: Path,
    output_dir: Path,
    artifact_version: str,
    baseline_scenario_checkpoint: Path,
    baseline_normal_checkpoint: Path,
    folds: int,
    cv_max_estimators: int,
    max_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    loaded = load_all_horizons(project_root)
    transitions = {pair: load_transition(loaded, *pair) for pair in TRANSITIONS}
    channel_frames = [loaded[7].assignments["channel_id"]] + [
        transitions[pair].metadata["channel_id"] for pair in TRANSITIONS
    ]
    test_channels, split_seed, test_ratios = choose_common_test_channels(
        channel_frames
    )
    X, targets, metadata = complete_subset(loaded, HORIZONS)
    actual_matrix = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual_matrix, axis=1) >= 0).all(axis=1)
    testing = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~testing & naturally_monotone
    evaluation = testing & naturally_monotone
    development_X = X.loc[development].reset_index(drop=True)
    development_targets = {
        horizon: targets[horizon][development] for horizon in HORIZONS
    }
    development_channels = metadata.loc[
        development, "channel_id"
    ].astype(str).reset_index(drop=True)

    print("Cross-validating relative-performance target strategies", flush=True)
    oof, fold_metrics, development_metrics = cross_validate_strategies(
        X=development_X,
        targets=development_targets,
        channels=development_channels,
        folds=folds,
        max_estimators=cv_max_estimators,
        n_jobs=n_jobs,
    )
    strategy_scores = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "mean_horizon_rmsle": average_rmsle(
                    development_metrics,
                    strategy=strategy,
                    method="strict_reconciled",
                ),
            }
            for strategy in BASELINE_STRATEGIES
        ]
    ).sort_values(["mean_horizon_rmsle", "strategy"])
    selected_strategy = str(strategy_scores.iloc[0]["strategy"])

    print(f"Training selected strategy: {selected_strategy}", flush=True)
    horizon_models = [
        fit_relative_horizon_model(
            X=development_X,
            actual_views=development_targets[horizon],
            channels=development_channels,
            horizon_days=horizon,
            strategy=selected_strategy,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        for horizon in HORIZONS
    ]
    trajectory = ReconciledIndependentTrajectoryModelBundle(
        horizon_models=horizon_models,
        minimum_increment_views=MINIMUM_INCREMENT_VIEWS,
        training_metadata={
            "construction": (
                "four independent channel-relative targets plus strict reconciliation"
            ),
            "baseline_strategy": selected_strategy,
            "components_consume_prior_predictions": False,
        },
    )
    normal_model_path = models_dir / "relative_performance_trajectory.joblib"
    joblib.dump(trajectory, normal_model_path)
    saved_trajectory = joblib.load(normal_model_path)

    baseline_scenario_manifest = json.loads(
        (baseline_scenario_checkpoint / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_scenario_path = (
        baseline_scenario_checkpoint / baseline_scenario_manifest["model_path"]
    )
    if sha256_file(baseline_scenario_path) != baseline_scenario_manifest["model_sha256"]:
        raise RuntimeError("Baseline scenario checksum mismatch")
    baseline_scenario = joblib.load(baseline_scenario_path)

    baseline_normal_manifest = json.loads(
        (baseline_normal_checkpoint / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_normal_path = (
        baseline_normal_checkpoint / baseline_normal_manifest["model_path"]
    )
    if sha256_file(baseline_normal_path) != baseline_normal_manifest["model_sha256"]:
        raise RuntimeError("Baseline normal trajectory checksum mismatch")
    baseline_normal = joblib.load(baseline_normal_path)

    test_X = X.loc[evaluation].reset_index(drop=True)
    test_targets = {horizon: targets[horizon][evaluation] for horizon in HORIZONS}
    relative_raw = saved_trajectory.predict_independent_views(test_X)
    relative_test = saved_trajectory.predict_views(test_X)
    baseline_test = baseline_normal.predict_views(test_X)
    scenario_baseline_test = baseline_scenario.predict_views(test_X)
    if not np.allclose(baseline_test, scenario_baseline_test):
        raise AssertionError("Scenario and normal checkpoint baselines disagree")
    test_metrics = pd.DataFrame(
        trajectory_metric_rows("existing_strict_growth", test_targets, baseline_test)
        + trajectory_metric_rows(
            "relative_performance_raw", test_targets, relative_raw
        )
        + trajectory_metric_rows(
            "relative_performance_strict", test_targets, relative_test
        )
    )

    baseline_cv_metrics = pd.read_csv(
        baseline_normal_checkpoint / "development_cv_oof_metrics.csv"
    )
    baseline_cv_rmsle = float(baseline_cv_metrics["rmsle"].mean())
    relative_cv_rmsle = float(strategy_scores.iloc[0]["mean_horizon_rmsle"])
    baseline_test_rmsle = float(
        test_metrics.loc[
            test_metrics["method"].eq("existing_strict_growth"), "rmsle"
        ].mean()
    )
    relative_test_rmsle = float(
        test_metrics.loc[
            test_metrics["method"].eq("relative_performance_strict"), "rmsle"
        ].mean()
    )
    release_gate = bool(
        relative_cv_rmsle < baseline_cv_rmsle
        and relative_test_rmsle < baseline_test_rmsle
    )

    scenario_path: Path | None = None
    if release_gate:
        promoted = ViralScenarioTrajectoryModelBundle(
            normal_trajectory=saved_trajectory,
            breakout_classifier=baseline_scenario.breakout_classifier,
            viral_trajectory=baseline_scenario.viral_trajectory,
            minimum_viral_uplift_views=baseline_scenario.minimum_viral_uplift_views,
            breakout_minimum_views=baseline_scenario.breakout_minimum_views,
            breakout_baseline_multiplier=baseline_scenario.breakout_baseline_multiplier,
            breakout_minimum_history_count=(
                baseline_scenario.breakout_minimum_history_count
            ),
            training_metadata={
                **baseline_scenario.training_metadata,
                "normal_trajectory": artifact_version,
            },
        )
        scenario_path = models_dir / "viral_scenario_trajectory.joblib"
        joblib.dump(promoted, scenario_path)

    sample = metadata.loc[
        evaluation, ["source_row_index", "video_id", "channel_id"]
    ].reset_index(drop=True)
    for column, horizon in enumerate(HORIZONS):
        sample[f"actual_day_{horizon}_views"] = test_targets[horizon]
        sample[f"baseline_day_{horizon}_views"] = baseline_test[:, column]
        sample[f"relative_day_{horizon}_views"] = relative_test[:, column]
    sample.head(100).to_csv(output_dir / "sample_test_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "development_cv_fold_metrics.csv", index=False)
    development_metrics.to_csv(
        output_dir / "development_cv_strategy_metrics.csv", index=False
    )
    strategy_scores.to_csv(output_dir / "strategy_selection.csv", index=False)
    test_metrics.to_csv(output_dir / "reserved_test_metrics.csv", index=False)
    pd.DataFrame(
        {
            "fold_row": np.arange(len(development_X)),
            **{
                f"{strategy}_day_{horizon}_views": oof[strategy][:, column]
                for strategy in BASELINE_STRATEGIES
                for column, horizon in enumerate(HORIZONS)
            },
        }
    ).to_csv(output_dir / "development_oof_predictions.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "test": "development and reserved channels disjoint",
                "status": (
                    "PASS"
                    if set(development_channels).isdisjoint(test_channels)
                    else "FAIL"
                ),
            },
            {
                "test": "relative predictions finite and nonnegative",
                "status": (
                    "PASS"
                    if np.isfinite(relative_test).all() and (relative_test >= 0).all()
                    else "FAIL"
                ),
            },
            {
                "test": "relative trajectories strictly increasing",
                "status": (
                    "PASS" if (np.diff(relative_test, axis=1) > 0).all() else "FAIL"
                ),
            },
            {"test": "saved relative model reload", "status": "PASS"},
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    if validation["status"].ne("PASS").any():
        raise AssertionError("Relative-performance validation failed")

    model_record = (
        scenario_path.relative_to(output_dir).as_posix()
        if scenario_path is not None
        else None
    )
    manifest = {
        "artifact_version": artifact_version,
        "status": (
            "relative_performance_normal_promoted"
            if release_gate
            else "relative_performance_benchmarked_not_promoted"
        ),
        "model_path": model_record,
        "model_sha256": sha256_file(scenario_path) if scenario_path else None,
        "normal_candidate_path": normal_model_path.relative_to(output_dir).as_posix(),
        "normal_candidate_sha256": sha256_file(normal_model_path),
        "horizons": list(HORIZONS),
        "target": "log1p(actual_views) - log1p(channel_baseline_views)",
        "candidate_baseline_strategies": list(BASELINE_STRATEGIES),
        "selected_baseline_strategy": selected_strategy,
        "minimum_history_count": MINIMUM_HISTORY_COUNT,
        "fallback_views": FALLBACK_VIEWS,
        "minimum_baseline_views": MINIMUM_BASELINE_VIEWS,
        "minimum_increment_views": MINIMUM_INCREMENT_VIEWS,
        "selection_data": "five-fold channel-grouped development OOF only",
        "development_rows": len(development_X),
        "reserved_test_rows": int(evaluation.sum()),
        "common_split": {
            "type": "channel_grouped_shared_across_all_horizons",
            "selection_seed": split_seed,
            "test_channels": len(test_channels),
            "source_test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "comparison": {
            "development_baseline_mean_rmsle": baseline_cv_rmsle,
            "development_relative_mean_rmsle": relative_cv_rmsle,
            "development_improvement_pct": (
                (baseline_cv_rmsle - relative_cv_rmsle) / baseline_cv_rmsle * 100
            ),
            "reserved_baseline_mean_rmsle": baseline_test_rmsle,
            "reserved_relative_mean_rmsle": relative_test_rmsle,
            "reserved_improvement_pct": (
                (baseline_test_rmsle - relative_test_rmsle)
                / baseline_test_rmsle
                * 100
            ),
            "release_gate_passed": release_gate,
        },
        "release_gate": (
            "mean horizon RMSLE must improve on development OOF and reserved channels"
        ),
        "output_contract": baseline_scenario_manifest["output_contract"],
        "breakout_definition": baseline_scenario_manifest["breakout_definition"],
        "classifier_reserved_test": baseline_scenario_manifest[
            "classifier_reserved_test"
        ],
        "feature_contract": horizon_models[0].preprocessor.feature_contract(),
        "limitations": [
            *baseline_scenario_manifest["limitations"],
            "Relative targets depend on the quality of pre-publication channel history.",
        ],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nStrategy selection", flush=True)
    print(strategy_scores.to_string(index=False), flush=True)
    print("\nReserved test metrics", flush=True)
    print(test_metrics.to_string(index=False), flush=True)
    print(f"\nRelease gate: {'PASS' if release_gate else 'FAIL'}", flush=True)
    return {
        "manifest": manifest,
        "development_metrics": development_metrics,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint23_relative_performance",
    )
    parser.add_argument(
        "--artifact-version", default="checkpoint23_relative_performance"
    )
    parser.add_argument(
        "--baseline-scenario-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint22_breakout_ensemble_20260915"
        ),
    )
    parser.add_argument(
        "--baseline-normal-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint18_strict_growth_trajectory_20260914"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-max-estimators", type=int, default=700)
    parser.add_argument("--max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        baseline_scenario_checkpoint=args.baseline_scenario_checkpoint,
        baseline_normal_checkpoint=args.baseline_normal_checkpoint,
        folds=args.folds,
        cv_max_estimators=args.cv_max_estimators,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
