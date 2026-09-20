"""Train a strictly increasing trajectory from a day-7 base and learned growth.

Each component sees only the original pre-publication feature row. The growth
models never consume an earlier model prediction, so training is not
autoregressive. The final arithmetic representation guarantees strictly
increasing day-7/14/21/30 cumulative predictions.
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
    component_feature_importance,
    fit_horizon_model,
    fit_log_target_components,
    load_all_horizons,
    load_transition,
    metric_row,
    sample_triple_predictions,
    sha256_file,
)
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    EXCLUDED_MODEL_COLUMNS,
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    build_xgb_regressor,
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
MINIMUM_INCREMENT_VIEWS = 1.0


def fit_fold_component(
    *,
    X: pd.DataFrame,
    target_views: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    max_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, int, int]:
    target_log = np.log1p(np.asarray(target_views, dtype=float))
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed_training = preprocessor.fit_transform(
        X.iloc[training], target_log[training]
    )
    transformed_validation = preprocessor.transform(X.iloc[validation])
    model = build_xgb_regressor(
        n_estimators=max_estimators,
        early_stopping_rounds=30,
        n_jobs=n_jobs,
    )
    model.fit(
        transformed_training,
        target_log[training],
        eval_set=[(transformed_validation, target_log[validation])],
        verbose=False,
    )
    predictions = views_from_log_predictions(
        model.predict(transformed_validation)
    )
    return (
        predictions,
        transformed_training.shape[1],
        int(model.best_iteration) + 1,
    )


def cross_validate_growth_architecture(
    *,
    X: pd.DataFrame,
    targets: dict[int, np.ndarray],
    channels: pd.Series,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = list(GroupKFold(n_splits=folds).split(X, groups=channels))
    oof = np.full((len(X), len(HORIZONS)), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (training, validation) in enumerate(positions, start=1):
        base, features, best_iteration = fit_fold_component(
            X=X,
            target_views=targets[7],
            training=training,
            validation=validation,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        fold_predictions = [base]
        component_details = [("day_7_base", features, best_iteration)]
        current = base
        for from_day, to_day in TRANSITIONS:
            actual_increment = np.maximum(
                0.0, targets[to_day] - targets[from_day]
            )
            increment, features, best_iteration = fit_fold_component(
                X=X,
                target_views=actual_increment,
                training=training,
                validation=validation,
                max_estimators=max_estimators,
                n_jobs=n_jobs,
            )
            current = current + np.maximum(MINIMUM_INCREMENT_VIEWS, increment)
            fold_predictions.append(current)
            component_details.append(
                (f"day_{from_day}_to_{to_day}_growth", features, best_iteration)
            )
        predicted = np.column_stack(fold_predictions)
        oof[validation] = predicted
        for column, horizon in enumerate(HORIZONS):
            metrics = regression_metrics(targets[horizon][validation], predicted[:, column])
            fold_rows.append(
                {
                    "fold": fold_number,
                    "horizon_days": horizon,
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    "rmsle": metrics["rmsle"],
                    "log_r2": metrics["log_r2"],
                    "wape_pct": metrics["wape_pct"],
                    "total_view_capture_pct": metrics[
                        "total_view_capture_pct"
                    ],
                    "median_absolute_error_views": metrics[
                        "median_absolute_error_views"
                    ],
                }
            )
        detail = ", ".join(
            f"{name}={iteration}" for name, _, iteration in component_details
        )
        print(f"Fold {fold_number}: {detail}", flush=True)
    if not np.isfinite(oof).all() or not (np.diff(oof, axis=1) > 0).all():
        raise AssertionError("Growth cross-validation produced invalid trajectories")

    oof_rows = []
    for column, horizon in enumerate(HORIZONS):
        metrics = regression_metrics(targets[horizon], oof[:, column])
        oof_rows.append(
            {
                "horizon_days": horizon,
                "rows": len(X),
                "rmsle": metrics["rmsle"],
                "log_r2": metrics["log_r2"],
                "wape_pct": metrics["wape_pct"],
                "total_view_capture_pct": metrics[
                    "total_view_capture_pct"
                ],
                "median_absolute_error_views": metrics[
                    "median_absolute_error_views"
                ],
                "flat_trajectory_rows": 0,
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def run_training(
    *,
    project_root: Path,
    output_dir: Path,
    artifact_version: str,
    cv_folds: int,
    cv_max_estimators: int,
    max_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

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

    print("Running grouped development cross-validation", flush=True)
    cv_folds_frame, cv_oof = cross_validate_growth_architecture(
        X=development_X,
        targets=development_targets,
        channels=development_channels,
        folds=cv_folds,
        max_estimators=cv_max_estimators,
        n_jobs=n_jobs,
    )
    cv_folds_frame.to_csv(output_dir / "development_cv_fold_metrics.csv", index=False)
    cv_oof.to_csv(output_dir / "development_cv_oof_metrics.csv", index=False)

    print("Training final day-7 base", flush=True)
    base_model = fit_horizon_model(
        horizon=7,
        X=development_X,
        y=development_targets[7],
        channels=development_channels,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
        preprocessor_options=FEATURE_OPTIONS,
    )
    increment_models: list[NonnegativeIncrementModelBundle] = []
    component_rows = [
        {"component": "day_7_base", **base_model.training_metadata}
    ]
    importance_rows = component_feature_importance("day_7_base", base_model)
    for from_day, to_day in TRANSITIONS:
        print(f"Training learned growth day {from_day}->{to_day}", flush=True)
        growth = np.maximum(
            0.0,
            development_targets[to_day] - development_targets[from_day],
        )
        preprocessor, regressor, selected = fit_log_target_components(
            X=development_X,
            target_views=growth,
            channels=development_channels,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
            preprocessor_options=FEATURE_OPTIONS,
        )
        increment_model = NonnegativeIncrementModelBundle(
            from_horizon_days=from_day,
            to_horizon_days=to_day,
            preprocessor=preprocessor,
            regressor=regressor,
            training_metadata={
                "target": f"day_{from_day}_to_day_{to_day}_view_increment",
                "target_scale": "log1p",
                "training_rows": len(development_X),
                "training_channels": int(development_channels.nunique()),
                "selected_n_estimators": selected,
            },
        )
        increment_models.append(increment_model)
        component = f"day_{from_day}_to_{to_day}_growth"
        component_rows.append(
            {"component": component, **increment_model.training_metadata}
        )
        importance_rows.extend(
            component_feature_importance(component, increment_model)
        )

    trajectory = MonotonicTrajectoryModelBundle(
        base_model=base_model,
        increment_models=increment_models,
        minimum_increment_views=MINIMUM_INCREMENT_VIEWS,
        training_metadata={
            "construction": "day-7 base plus independently learned positive increments",
            "components_consume_prior_predictions": False,
            "complete_development_rows": len(development_X),
        },
    )
    model_path = models_dir / "strict_growth_trajectory.joblib"
    joblib.dump(trajectory, model_path)
    saved = joblib.load(model_path)

    test_X = X.loc[evaluation].reset_index(drop=True)
    predictions = saved.predict_views(test_X)
    metric_rows = []
    for column, horizon in enumerate(HORIZONS):
        metric_rows.append(
            metric_row(
                f"complete_day_7_14_21_30_day_{horizon}",
                "aligned_learned_growth",
                targets[horizon][evaluation],
                predictions[:, column],
            )
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "complete_trajectory_test_metrics.csv", index=False)
    metrics.to_csv(output_dir / "transition_test_metrics.csv", index=False)
    metrics[metrics["scope"].str.contains(r"day_(?:7|14|21)$", regex=True)].to_csv(
        output_dir / "triple_horizon_test_metrics.csv", index=False
    )
    sample_triple_predictions(
        metadata.loc[evaluation].reset_index(drop=True),
        {horizon: targets[horizon][evaluation] for horizon in HORIZONS},
        predictions,
    ).to_csv(output_dir / "sample_predictions.csv", index=False)
    pd.DataFrame(component_rows).to_csv(
        output_dir / "trained_components.csv", index=False
    )
    pd.DataFrame(importance_rows).sort_values(
        ["component", "importance"], ascending=[True, False]
    ).to_csv(output_dir / "feature_importance.csv", index=False)

    increment_rows = []
    for model in increment_models:
        predicted_increment = model.predict_increment_views(test_X)
        actual_increment = (
            targets[model.to_horizon_days][evaluation]
            - targets[model.from_horizon_days][evaluation]
        )
        increment_rows.append(
            {
                "transition": (
                    f"day_{model.from_horizon_days}_to_{model.to_horizon_days}"
                ),
                "actual_median_increment": float(np.median(actual_increment)),
                "predicted_median_increment": float(
                    np.median(predicted_increment)
                ),
                "predicted_floor_applications": int(
                    (predicted_increment < MINIMUM_INCREMENT_VIEWS).sum()
                ),
            }
        )
    pd.DataFrame(increment_rows).to_csv(
        output_dir / "increment_diagnostics.csv", index=False
    )
    validation = pd.DataFrame(
        [
            {
                "test": "saved bundle reload",
                "status": "PASS",
            },
            {
                "test": "trajectories strictly increasing",
                "status": "PASS"
                if (np.diff(predictions, axis=1) > 0).all()
                else "FAIL",
            },
            {
                "test": "predictions finite and nonnegative",
                "status": "PASS"
                if np.isfinite(predictions).all() and (predictions >= 0).all()
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
        raise AssertionError("Strict growth validation failed")

    manifest = {
        "artifact_version": artifact_version,
        "status": "strictly_increasing_aligned_growth_test_evaluated",
        "source_training_table": {
            "path": "Dataset/viewcastlk_training_table.csv",
            "sha256": sha256_file(
                project_root / "Dataset" / "viewcastlk_training_table.csv"
            ),
        },
        "model_path": model_path.relative_to(output_dir).as_posix(),
        "model_sha256": sha256_file(model_path),
        "horizons": list(HORIZONS),
        "construction": (
            "day-7 base plus three growth models trained directly from the "
            "same pre-publication features; no component consumes another "
            "component's prediction"
        ),
        "minimum_increment_views": MINIMUM_INCREMENT_VIEWS,
        "flat_prediction_rows": int(
            (np.diff(predictions, axis=1) <= 0).any(axis=1).sum()
        ),
        "common_split": {
            "type": "channel_grouped_shared_across_all_horizons",
            "selection_seed": split_seed,
            "test_size_target": 0.20,
            "test_channels": len(test_channels),
            "source_test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "components": component_rows,
        "feature_contract": base_model.preprocessor.feature_contract(),
        "preprocessor_options": FEATURE_OPTIONS,
        "complete_four_horizon_rows": int(len(X)),
        "complete_four_horizon_development_rows": int(development.sum()),
        "complete_four_horizon_test_rows": int(testing.sum()),
        "complete_monotone_four_horizon_test_rows": int(evaluation.sum()),
        "end_to_end_day_30_testable": True,
        "development_cross_validation": {
            "folds": cv_folds,
            "reserved_test_used": False,
            "metrics_path": "development_cv_oof_metrics.csv",
        },
        "limitations": [
            "The one-view minimum increment is a presentation constraint and can exceed truly flat observed growth.",
            "Point predictions remain conservative for unrecognized breakout videos.",
            "The common channel-grouped test set has been evaluated and must not be used for further tuning.",
        ],
        "explicitly_excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "labels": "day_7_14_21_30",
                "rows": len(X),
                "channels": metadata["channel_id"].nunique(),
                "negative_growth_rows": int((~naturally_monotone).sum()),
            }
        ]
    ).to_csv(output_dir / "overlap_summary.csv", index=False)
    print("\nDevelopment OOF metrics:", flush=True)
    print(cv_oof.to_string(index=False), flush=True)
    print("\nReserved-channel metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)
    return {"manifest": manifest, "cv_oof": cv_oof, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint18_strict_growth_trajectory",
    )
    parser.add_argument(
        "--artifact-version", default="checkpoint18_strict_growth_trajectory"
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-max-estimators", type=int, default=500)
    parser.add_argument("--max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        cv_folds=args.cv_folds,
        cv_max_estimators=args.cv_max_estimators,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
