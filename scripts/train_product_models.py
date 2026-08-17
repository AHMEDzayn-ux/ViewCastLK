"""Train and save four independent ViewCastLK horizon model bundles.

The final test partitions are assigned and recorded but deliberately not scored
here. This command performs grouped cross-validation on development rows, then
fits one candidate product bundle per horizon on all development rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewcastlk_ml.modeling import (  # noqa: E402
    MVP_XGB_PARAMS,
    CategoryMedianBaseline,
    CategoryTierMedianBaseline,
    GlobalMedianBaseline,
    HorizonModelBundle,
    build_xgb_regressor,
    log_target_inlier_mask,
    regression_metrics,
    views_from_log_predictions,
)
from viewcastlk_ml.preprocessing import (  # noqa: E402
    RAW_INPUT_COLUMNS,
    USER_REMOVED_FEATURES,
    HorizonPreprocessor,
)


VALID_HORIZONS = (7, 14, 21, 30)
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_HOLDOUT_CANDIDATES = 200
N_CV_FOLDS = 5
CATEGORY_SMOOTHING = 10.0
OUTLIER_SIGMA = 3.0


def is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().eq("true")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolate_horizon(frame: pd.DataFrame, horizon: int):
    target_column = f"d{horizon}_views"
    usable_column = f"d{horizon}_usable"
    keep = (
        is_true(frame["eligible"])
        & is_true(frame[usable_column])
        & frame[target_column].notna()
    )
    selected = frame.loc[keep]
    X = selected.loc[:, RAW_INPUT_COLUMNS].copy()
    y = selected[target_column].astype(float).rename("target_views")
    metadata = selected[["video_id", "channel_id", "published_at"]].copy()
    return X, y, metadata


def choose_group_holdout(X: pd.DataFrame, groups: pd.Series):
    candidates = GroupShuffleSplit(
        n_splits=N_HOLDOUT_CANDIDATES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    ).split(X, groups=groups)
    development_positions, test_positions = min(
        candidates,
        key=lambda positions: abs(len(positions[1]) / len(X) - TEST_SIZE),
    )
    return X.index[development_positions], X.index[test_positions]


def grouped_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    splitter = GroupKFold(n_splits=N_CV_FOLDS)
    return [
        (X.index[train_positions], X.index[validation_positions])
        for train_positions, validation_positions in splitter.split(X, y, groups)
    ]


def fit_baselines(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    train_channels: pd.Series,
):
    return {
        "global_median": GlobalMedianBaseline().fit(X_train, y_train_log),
        "category_median": CategoryMedianBaseline().fit(X_train, y_train_log),
        "category_tier_median": CategoryTierMedianBaseline().fit(
            X_train,
            y_train_log,
            channel_ids=train_channels.to_numpy(),
        ),
    }


def metrics_row(
    *,
    horizon: int,
    fold: int,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    train_rows_before_filter: int,
    train_rows_after_filter: int,
    lower_log_bound: float,
    upper_log_bound: float,
    best_iteration: int | None = None,
) -> dict[str, Any]:
    return {
        "horizon_days": horizon,
        "fold": fold,
        "model": model_name,
        "train_rows_before_filter": train_rows_before_filter,
        "train_rows_after_filter": train_rows_after_filter,
        "outliers_removed": train_rows_before_filter - train_rows_after_filter,
        "lower_log_bound": lower_log_bound,
        "upper_log_bound": upper_log_bound,
        "best_iteration": best_iteration,
        **regression_metrics(y_true, y_pred),
    }


def train_horizon_cv(
    *,
    horizon: int,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    metadata_development: pd.DataFrame,
    folds,
    n_jobs: int,
    n_estimators: int,
    model_overrides: dict[str, Any] | None = None,
):
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        X_train_all = X_development.loc[train_index]
        y_train_log_all = np.log1p(y_development.loc[train_index].astype(float))
        inlier_mask, lower, upper = log_target_inlier_mask(
            y_train_log_all, sigma=OUTLIER_SIGMA
        )
        inlier_index = train_index[inlier_mask]
        X_train = X_development.loc[inlier_index]
        y_train_log = np.log1p(y_development.loc[inlier_index].astype(float))
        X_validation = X_development.loc[validation_index]
        y_validation = y_development.loc[validation_index].astype(float)
        y_validation_log = np.log1p(y_validation)

        baselines = fit_baselines(
            X_train,
            y_train_log,
            metadata_development.loc[inlier_index, "channel_id"],
        )
        fold_predictions: dict[str, np.ndarray] = {}
        for model_name, baseline in baselines.items():
            predicted_views = views_from_log_predictions(baseline.predict(X_validation))
            fold_predictions[model_name] = predicted_views
            fold_metric_rows.append(
                metrics_row(
                    horizon=horizon,
                    fold=fold_number,
                    model_name=model_name,
                    y_true=y_validation,
                    y_pred=predicted_views,
                    train_rows_before_filter=len(train_index),
                    train_rows_after_filter=len(inlier_index),
                    lower_log_bound=lower,
                    upper_log_bound=upper,
                )
            )

        preprocessor = HorizonPreprocessor(
            category_smoothing=CATEGORY_SMOOTHING
        )
        X_train_transformed = preprocessor.fit_transform(X_train, y_train_log)
        X_validation_transformed = preprocessor.transform(X_validation)
        regressor = build_xgb_regressor(
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            **(model_overrides or {}),
        )
        regressor.fit(
            X_train_transformed,
            y_train_log,
            eval_set=[(X_validation_transformed, y_validation_log)],
            verbose=False,
        )
        predicted_views = views_from_log_predictions(
            regressor.predict(X_validation_transformed)
        )
        fold_predictions["xgboost_mvp"] = predicted_views
        best_iteration = int(regressor.best_iteration)
        best_iterations.append(best_iteration + 1)
        fold_metric_rows.append(
            metrics_row(
                horizon=horizon,
                fold=fold_number,
                model_name="xgboost_mvp",
                y_true=y_validation,
                y_pred=predicted_views,
                train_rows_before_filter=len(train_index),
                train_rows_after_filter=len(inlier_index),
                lower_log_bound=lower,
                upper_log_bound=upper,
                best_iteration=best_iteration,
            )
        )

        prediction_frame = metadata_development.loc[validation_index].copy()
        prediction_frame["horizon_days"] = horizon
        prediction_frame["fold"] = fold_number
        prediction_frame["actual_views"] = y_validation
        for model_name, values in fold_predictions.items():
            prediction_frame[f"predicted_{model_name}"] = values
        prediction_rows.append(prediction_frame.reset_index(names="source_row_index"))
        print(
            f"day {horizon} fold {fold_number}/{N_CV_FOLDS}: "
            f"xgboost MAPE={fold_metric_rows[-1]['mape_nonzero_pct']:.2f}% "
            f"best_trees={best_iteration + 1}"
        )

    return (
        pd.DataFrame(fold_metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        best_iterations,
    )


def summarise_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prediction_columns = {
        column.removeprefix("predicted_"): column
        for column in predictions.columns
        if column.startswith("predicted_")
    }
    for horizon, horizon_predictions in predictions.groupby("horizon_days"):
        for model_name, prediction_column in prediction_columns.items():
            rows.append(
                {
                    "horizon_days": int(horizon),
                    "model": model_name,
                    **regression_metrics(
                        horizon_predictions["actual_views"],
                        horizon_predictions[prediction_column],
                    ),
                }
            )
    for model_name, prediction_column in prediction_columns.items():
        rows.append(
            {
                "horizon_days": "combined",
                "model": model_name,
                **regression_metrics(
                    predictions["actual_views"], predictions[prediction_column]
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_development_bundle(
    *,
    horizon: int,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    selected_n_estimators: int,
    model_overrides: dict[str, Any] | None = None,
) -> HorizonModelBundle:
    y_log_all = np.log1p(y_development.astype(float))
    inlier_mask, lower, upper = log_target_inlier_mask(
        y_log_all, sigma=OUTLIER_SIGMA
    )
    training_index = X_development.index[inlier_mask]
    X_train = X_development.loc[training_index]
    y_train_log = np.log1p(y_development.loc[training_index].astype(float))

    preprocessor = HorizonPreprocessor(category_smoothing=CATEGORY_SMOOTHING)
    X_transformed = preprocessor.fit_transform(X_train, y_train_log)
    regressor = build_xgb_regressor(
        n_estimators=selected_n_estimators,
        early_stopping_rounds=None,
        **(model_overrides or {}),
    )
    regressor.fit(X_transformed, y_train_log, verbose=False)
    return HorizonModelBundle(
        horizon_days=horizon,
        preprocessor=preprocessor,
        regressor=regressor,
        training_metadata={
            "status": "candidate_untouched_test_not_evaluated",
            "training_rows_before_outlier_filter": int(len(X_development)),
            "training_rows_after_outlier_filter": int(len(X_train)),
            "outliers_removed": int(len(X_development) - len(X_train)),
            "outlier_sigma": OUTLIER_SIGMA,
            "lower_log_bound": lower,
            "upper_log_bound": upper,
            "selected_n_estimators": int(selected_n_estimators),
            "category_smoothing": CATEGORY_SMOOTHING,
            "random_state": RANDOM_STATE,
            "xgboost_parameters": {
                **MVP_XGB_PARAMS,
                **(model_overrides or {}),
                "n_estimators": int(selected_n_estimators),
                "early_stopping_rounds": None,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "viewcastlk_training_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "mvp_v1",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--objective", default="reg:squarederror")
    parser.add_argument("--eval-metric", default="rmse")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--artifact-version", default="mvp_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, low_memory=False)
    model_overrides = {
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "learning_rate": args.learning_rate,
    }
    all_fold_metrics = []
    all_predictions = []
    holdout_rows = []
    feature_importance_rows = []
    artifact_records = []

    for horizon in VALID_HORIZONS:
        print(f"\nTraining independent day-{horizon} model")
        X, y, metadata = isolate_horizon(frame, horizon)
        development_index, test_index = choose_group_holdout(
            X, metadata["channel_id"]
        )
        X_development = X.loc[development_index]
        y_development = y.loc[development_index]
        metadata_development = metadata.loc[development_index]
        folds = grouped_folds(
            X_development,
            y_development,
            metadata_development["channel_id"],
        )

        for partition, indexes in (
            ("development", development_index),
            ("test_untouched", test_index),
        ):
            assignment = metadata.loc[indexes].copy()
            assignment["horizon_days"] = horizon
            assignment["partition"] = partition
            holdout_rows.append(assignment.reset_index(names="source_row_index"))

        fold_metrics, predictions, best_iterations = train_horizon_cv(
            horizon=horizon,
            X_development=X_development,
            y_development=y_development,
            metadata_development=metadata_development,
            folds=folds,
            n_jobs=args.n_jobs,
            n_estimators=args.n_estimators,
            model_overrides=model_overrides,
        )
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)

        selected_n_estimators = max(1, int(np.median(best_iterations)))
        bundle = fit_development_bundle(
            horizon=horizon,
            X_development=X_development,
            y_development=y_development,
            selected_n_estimators=selected_n_estimators,
            model_overrides=model_overrides,
        )
        artifact_path = models_dir / f"day_{horizon}_model.joblib"
        joblib.dump(bundle, artifact_path)

        category_state_path = models_dir / f"day_{horizon}_category_encoding.json"
        category_state_path.write_text(
            json.dumps(bundle.preprocessor.category_encoding_state(), indent=2),
            encoding="utf-8",
        )
        feature_order_path = models_dir / f"day_{horizon}_feature_order.json"
        feature_order_path.write_text(
            json.dumps(bundle.feature_names, indent=2), encoding="utf-8"
        )

        for feature, gain in bundle.regressor.get_booster().get_score(
            importance_type="gain"
        ).items():
            feature_importance_rows.append(
                {"horizon_days": horizon, "feature": feature, "gain": float(gain)}
            )
        artifact_records.append(
            {
                "horizon_days": horizon,
                "model_path": str(artifact_path.relative_to(args.output_dir)),
                "model_sha256": sha256_file(artifact_path),
                "category_encoding_path": str(
                    category_state_path.relative_to(args.output_dir)
                ),
                "feature_order_path": str(
                    feature_order_path.relative_to(args.output_dir)
                ),
                "selected_n_estimators": selected_n_estimators,
                "output_feature_count": len(bundle.feature_names),
                **bundle.training_metadata,
            }
        )

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary_metrics = summarise_oof_predictions(predictions)
    holdout_assignments = pd.concat(holdout_rows, ignore_index=True)
    feature_importance = pd.DataFrame(feature_importance_rows)

    fold_metrics.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "cv_predictions.csv", index=False)
    summary_metrics.to_csv(args.output_dir / "cv_summary_metrics.csv", index=False)
    holdout_assignments.to_csv(
        args.output_dir / "holdout_assignments.csv", index=False
    )
    feature_importance.to_csv(
        args.output_dir / "feature_importance_gain.csv", index=False
    )

    manifest = {
        "artifact_version": args.artifact_version,
        "status": "candidate_untouched_test_not_evaluated",
        "source_dataset": str(args.input),
        "source_dataset_sha256": sha256_file(args.input),
        "horizons": list(VALID_HORIZONS),
        "split": {
            "type": "channel_grouped",
            "test_size": TEST_SIZE,
            "cv_folds": N_CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
        "target": "horizon-specific log1p views",
        "user_removed_features": list(USER_REMOVED_FEATURES),
        "model_configuration": model_overrides,
        "metric_note": "MAPE excludes zero targets because percentage error is undefined at zero; zero counts are reported separately.",
        "models": artifact_records,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nCross-validated summary (final test remains untouched):")
    print(summary_metrics.to_string(index=False))
    print(f"\nSaved candidate artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
