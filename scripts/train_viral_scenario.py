"""Train a calibrated breakout gate and conditional viral-upside trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_checkpoint5_models import CATEGORY_SMOOTHING  # noqa: E402
from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    fit_horizon_model,
    fit_log_target_components,
    load_all_horizons,
    load_transition,
    sha256_file,
)
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    CalibratedBreakoutClassifierBundle,
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    ViralScenarioTrajectoryModelBundle,
    breakout_threshold_views,
    regression_metrics,
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
BREAKOUT_MINIMUM_VIEWS = 10_000.0
BREAKOUT_BASELINE_MULTIPLIER = 5.0
BREAKOUT_MINIMUM_HISTORY_COUNT = 5
MINIMUM_INCREMENT_VIEWS = 1.0


def breakout_labels(X: pd.DataFrame, day_7_views: np.ndarray) -> np.ndarray:
    thresholds = breakout_threshold_views(
        X,
        minimum_views=BREAKOUT_MINIMUM_VIEWS,
        baseline_multiplier=BREAKOUT_BASELINE_MULTIPLIER,
        minimum_history_count=BREAKOUT_MINIMUM_HISTORY_COUNT,
    )
    return np.asarray(day_7_views, dtype=float) >= thresholds


def build_classifier(
    *,
    n_estimators: int,
    n_jobs: int,
    early_stopping_rounds: int | None,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=n_jobs,
    )


def log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def classification_metrics(
    actual: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(actual, dtype=bool)
    predicted = np.asarray(probability, dtype=float)
    return {
        "rows": len(labels),
        "breakout_rows": int(labels.sum()),
        "breakout_rate_pct": float(labels.mean() * 100),
        "roc_auc": float(roc_auc_score(labels, predicted)),
        "average_precision": float(average_precision_score(labels, predicted)),
        "brier_score": float(brier_score_loss(labels, predicted)),
        "log_loss": float(log_loss(labels, predicted)),
    }


def fit_breakout_classifier(
    *,
    X: pd.DataFrame,
    labels: np.ndarray,
    channels: pd.Series,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[CalibratedBreakoutClassifierBundle, pd.DataFrame, dict[str, Any]]:
    oof_probability = np.full(len(X), np.nan, dtype=float)
    fold_rows = []
    selected_estimators = []
    for fold, (training, validation) in enumerate(
        GroupKFold(n_splits=folds).split(X, groups=channels), start=1
    ):
        preprocessor = HorizonDatasetPreprocessor(
            category_smoothing=CATEGORY_SMOOTHING,
            **FEATURE_OPTIONS,
        )
        transformed_training = preprocessor.fit_transform(
            X.iloc[training], labels[training].astype(float)
        )
        transformed_validation = preprocessor.transform(X.iloc[validation])
        classifier = build_classifier(
            n_estimators=max_estimators,
            n_jobs=n_jobs,
            early_stopping_rounds=30,
        )
        classifier.fit(
            transformed_training,
            labels[training],
            eval_set=[(transformed_validation, labels[validation])],
            verbose=False,
        )
        probability = classifier.predict_proba(transformed_validation)[:, 1]
        oof_probability[validation] = probability
        selected = int(classifier.best_iteration) + 1
        selected_estimators.append(selected)
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "training_breakout_rate_pct": float(
                    labels[training].mean() * 100
                ),
                "validation_breakout_rate_pct": float(
                    labels[validation].mean() * 100
                ),
                "selected_n_estimators": selected,
                **classification_metrics(labels[validation], probability),
            }
        )
        print(
            f"Classifier fold {fold}: estimators={selected}, "
            f"AUC={fold_rows[-1]['roc_auc']:.3f}, "
            f"AP={fold_rows[-1]['average_precision']:.3f}",
            flush=True,
        )
    if not np.isfinite(oof_probability).all():
        raise AssertionError("Incomplete classifier OOF probabilities")

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(log_odds(oof_probability), labels)
    calibrated_oof = calibrator.predict_proba(log_odds(oof_probability))[:, 1]
    final_estimators = max(1, int(np.median(selected_estimators)))
    final_preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed = final_preprocessor.fit_transform(X, labels.astype(float))
    final_classifier = build_classifier(
        n_estimators=final_estimators,
        n_jobs=n_jobs,
        early_stopping_rounds=None,
    )
    final_classifier.fit(transformed, labels, verbose=False)
    metadata = {
        "training_rows": len(X),
        "training_breakout_rows": int(labels.sum()),
        "training_breakout_rate_pct": float(labels.mean() * 100),
        "selected_n_estimators": final_estimators,
        "cross_fitted_uncalibrated": classification_metrics(
            labels, oof_probability
        ),
        "cross_fitted_calibrated": classification_metrics(
            labels, calibrated_oof
        ),
    }
    bundle = CalibratedBreakoutClassifierBundle(
        preprocessor=final_preprocessor,
        classifier=final_classifier,
        calibrator=calibrator,
        training_metadata=metadata,
    )
    return bundle, pd.DataFrame(fold_rows), metadata


def fit_viral_trajectory(
    *,
    X: pd.DataFrame,
    targets: dict[int, np.ndarray],
    channels: pd.Series,
    max_estimators: int,
    n_jobs: int,
) -> MonotonicTrajectoryModelBundle:
    base = fit_horizon_model(
        horizon=7,
        X=X,
        y=targets[7],
        channels=channels,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
        preprocessor_options=FEATURE_OPTIONS,
    )
    increments = []
    for from_day, to_day in TRANSITIONS:
        growth = np.maximum(0.0, targets[to_day] - targets[from_day])
        preprocessor, regressor, selected = fit_log_target_components(
            X=X,
            target_views=growth,
            channels=channels,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
            preprocessor_options=FEATURE_OPTIONS,
        )
        increments.append(
            NonnegativeIncrementModelBundle(
                from_horizon_days=from_day,
                to_horizon_days=to_day,
                preprocessor=preprocessor,
                regressor=regressor,
                training_metadata={
                    "target": f"viral_day_{from_day}_to_{to_day}_increment",
                    "training_rows": len(X),
                    "selected_n_estimators": selected,
                },
            )
        )
    return MonotonicTrajectoryModelBundle(
        base_model=base,
        increment_models=increments,
        minimum_increment_views=MINIMUM_INCREMENT_VIEWS,
        training_metadata={
            "conditional_population": "day-7 breakout videos",
            "training_rows": len(X),
            "training_channels": int(channels.nunique()),
        },
    )


def calibration_bins(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"actual": np.asarray(labels, dtype=int), "probability": probability}
    )
    frame["bin"] = pd.qcut(
        frame["probability"], q=bins, duplicates="drop"
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            rows=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            actual_breakout_rate=("actual", "mean"),
        )
        .reset_index()
        .assign(bin=lambda values: values["bin"].astype(str))
    )


def run_training(
    *,
    project_root: Path,
    normal_checkpoint: Path,
    output_dir: Path,
    artifact_version: str,
    folds: int,
    classifier_max_estimators: int,
    viral_max_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    normal_manifest = json.loads(
        (normal_checkpoint / "training_manifest.json").read_text(encoding="utf-8")
    )
    normal_model_path = normal_checkpoint / normal_manifest["model_path"]
    if sha256_file(normal_model_path) != normal_manifest["model_sha256"]:
        raise RuntimeError("Normal trajectory checksum mismatch")
    normal_trajectory = joblib.load(normal_model_path)

    loaded = load_all_horizons(project_root)
    transitions = {
        pair: load_transition(loaded, *pair) for pair in TRANSITIONS
    }
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
    development_labels = breakout_labels(
        development_X, development_targets[7]
    )

    classifier, classifier_folds, classifier_metadata = fit_breakout_classifier(
        X=development_X,
        labels=development_labels,
        channels=development_channels,
        folds=folds,
        max_estimators=classifier_max_estimators,
        n_jobs=n_jobs,
    )
    classifier_folds.to_csv(
        output_dir / "classifier_cv_fold_metrics.csv", index=False
    )

    viral_X = development_X.loc[development_labels].reset_index(drop=True)
    viral_targets = {
        horizon: values[development_labels] for horizon, values in development_targets.items()
    }
    viral_channels = development_channels.loc[development_labels].reset_index(
        drop=True
    )
    print(
        f"Training viral trajectory on {len(viral_X):,} breakout rows",
        flush=True,
    )
    viral_trajectory = fit_viral_trajectory(
        X=viral_X,
        targets=viral_targets,
        channels=viral_channels,
        max_estimators=viral_max_estimators,
        n_jobs=n_jobs,
    )
    scenario_bundle = ViralScenarioTrajectoryModelBundle(
        normal_trajectory=normal_trajectory,
        breakout_classifier=classifier,
        viral_trajectory=viral_trajectory,
        minimum_viral_uplift_views=1.0,
        breakout_minimum_views=BREAKOUT_MINIMUM_VIEWS,
        breakout_baseline_multiplier=BREAKOUT_BASELINE_MULTIPLIER,
        breakout_minimum_history_count=BREAKOUT_MINIMUM_HISTORY_COUNT,
        training_metadata={
            "normal_forecast": normal_manifest["artifact_version"],
            "user_contract": (
                "normal forecast; breakout probability; conditional viral-upside forecast"
            ),
        },
    )
    model_path = models_dir / "viral_scenario_trajectory.joblib"
    joblib.dump(scenario_bundle, model_path)
    saved = joblib.load(model_path)

    test_X = X.loc[evaluation].reset_index(drop=True)
    test_targets = {
        horizon: targets[horizon][evaluation] for horizon in HORIZONS
    }
    test_labels = breakout_labels(test_X, test_targets[7])
    probability = saved.predict_breakout_probability(test_X)
    normal = saved.predict_views(test_X)
    viral = saved.predict_viral_upside_views(test_X)
    classifier_test = classification_metrics(test_labels, probability)
    pd.DataFrame([classifier_test]).to_csv(
        output_dir / "classifier_test_metrics.csv", index=False
    )
    calibration_bins(test_labels, probability).to_csv(
        output_dir / "classifier_test_calibration.csv", index=False
    )

    scenario_rows = []
    for column, horizon in enumerate(HORIZONS):
        for scenario_name, predictions in (
            ("normal", normal[:, column]),
            ("conditional_viral_upside", viral[:, column]),
        ):
            metrics = regression_metrics(
                test_targets[horizon][test_labels], predictions[test_labels]
            )
            scenario_rows.append(
                {
                    "population": "actual_breakouts",
                    "scenario": scenario_name,
                    "horizon_days": horizon,
                    "coverage_actual_at_or_below_pct": float(
                        (
                            test_targets[horizon][test_labels]
                            <= predictions[test_labels]
                        ).mean()
                        * 100
                    ),
                    "rmsle": metrics["rmsle"],
                    "wape_pct": metrics["wape_pct"],
                    "median_absolute_error_views": metrics[
                        "median_absolute_error_views"
                    ],
                }
            )
    scenario_metrics = pd.DataFrame(scenario_rows)
    scenario_metrics.to_csv(output_dir / "viral_scenario_test_metrics.csv", index=False)

    scenarios = saved.predict_scenario_frame(test_X)
    samples = metadata.loc[evaluation, ["source_row_index", "video_id", "channel_id"]].reset_index(drop=True)
    samples = pd.concat([samples, scenarios.reset_index(drop=True)], axis=1)
    for horizon in HORIZONS:
        samples[f"actual_day_{horizon}_views"] = test_targets[horizon]
    samples["actual_breakout"] = test_labels
    samples.sort_values(
        ["actual_breakout", "breakout_probability"], ascending=[False, False]
    ).head(50).to_csv(output_dir / "sample_scenario_predictions.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "test": "probabilities finite and bounded",
                "status": "PASS"
                if np.isfinite(probability).all()
                and ((probability >= 0) & (probability <= 1)).all()
                else "FAIL",
            },
            {
                "test": "normal trajectories strictly increasing",
                "status": "PASS"
                if (np.diff(normal, axis=1) > 0).all()
                else "FAIL",
            },
            {
                "test": "viral upside above normal and strictly increasing",
                "status": "PASS"
                if (viral > normal).all() and (np.diff(viral, axis=1) > 0).all()
                else "FAIL",
            },
            {
                "test": "development and test channels disjoint",
                "status": "PASS"
                if set(development_channels).isdisjoint(test_channels)
                else "FAIL",
            },
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    if validation["status"].ne("PASS").any():
        raise AssertionError("Viral scenario validation failed")

    manifest = {
        "artifact_version": artifact_version,
        "status": "two_scenario_breakout_model_test_evaluated",
        "model_path": model_path.relative_to(output_dir).as_posix(),
        "model_sha256": sha256_file(model_path),
        "horizons": list(HORIZONS),
        "output_contract": {
            "primary": "normal_day_{horizon}_views",
            "conditional_upside": "viral_upside_day_{horizon}_views",
            "condition_probability": "breakout_probability",
            "wording": (
                "Normally expected around N views. There is a P% breakout "
                "chance; if that happens, around V views."
            ),
            "combined_weighted_prediction_exposed": False,
        },
        "breakout_definition": {
            "horizon_days": 7,
            "minimum_views": BREAKOUT_MINIMUM_VIEWS,
            "baseline_multiplier": BREAKOUT_BASELINE_MULTIPLIER,
            "minimum_history_count": BREAKOUT_MINIMUM_HISTORY_COUNT,
            "baseline": (
                "prior day-7 channel median when at least five observations "
                "exist; otherwise point-in-time channel average"
            ),
        },
        "classifier": classifier_metadata,
        "classifier_reserved_test": classifier_test,
        "normal_trajectory_source": {
            "artifact_version": normal_manifest["artifact_version"],
            "model_sha256": normal_manifest["model_sha256"],
        },
        "viral_training_rows": len(viral_X),
        "viral_training_channels": int(viral_channels.nunique()),
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
        "feature_contract": classifier.preprocessor.feature_contract(),
        "source_training_table_sha256": sha256_file(
            project_root / "Dataset" / "viewcastlk_training_table.csv"
        ),
        "limitations": [
            "The breakout probability is pre-publication and cannot know future audience reactions.",
            "The current structured title features do not encode the full title meaning or thumbnail content.",
            "The viral-upside value is conditional on breakout, not a second equally likely point prediction.",
        ],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nClassifier reserved-test metrics:", flush=True)
    print(pd.DataFrame([classifier_test]).to_string(index=False), flush=True)
    print("\nConditional viral scenario metrics:", flush=True)
    print(scenario_metrics.to_string(index=False), flush=True)
    return {
        "manifest": manifest,
        "classifier_test": classifier_test,
        "scenario_metrics": scenario_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normal-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "checkpoint18_strict_growth_trajectory_20260914",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint19_viral_scenario",
    )
    parser.add_argument("--artifact-version", default="checkpoint19_viral_scenario")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--classifier-max-estimators", type=int, default=700)
    parser.add_argument("--viral-max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        project_root=PROJECT_ROOT,
        normal_checkpoint=args.normal_checkpoint,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        folds=args.folds,
        classifier_max_estimators=args.classifier_max_estimators,
        viral_max_estimators=args.viral_max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
