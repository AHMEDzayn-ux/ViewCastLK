"""Train the first XGBoost product candidates from finalized horizon datasets.

This checkpoint consumes the exact CSVs from notebook 01 and the exact
channel-grouped assignments from notebook 04. It never recreates either.
Cross-validation is restricted to development rows. The reserved test rows are
not transformed, predicted, or scored.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    EXCLUDED_MODEL_COLUMNS,
    LLM_SCORE_COLUMNS,
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    MVP_XGB_PARAMS,
    CategoryMedianBaseline,
    GlobalMedianBaseline,
    HorizonModelBundle,
    build_xgb_regressor,
    log_target_inlier_mask,
    regression_metrics,
    views_from_log_predictions,
)


HORIZONS = (7, 14, 21, 30)
CV_FOLDS = 5
CATEGORY_SMOOTHING = 10.0
OUTLIER_SIGMA = 3.0
MODEL_NAME = "xgboost_checkpoint5"


class FixedSubscriberTierMedianBaseline:
    """Training-only category and fixed subscriber-tier median baseline."""

    def fit(self, X: pd.DataFrame, y):
        categories = (
            X["category_name"]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .reset_index(drop=True)
        )
        tiers = (
            X["subscriber_tier"]
            .astype("string")
            .fillna("missing")
            .astype(str)
            .reset_index(drop=True)
        )
        target = pd.Series(
            np.asarray(y, dtype=float).reshape(-1)
        ).reset_index(drop=True)
        if not (len(categories) == len(tiers) == len(target)):
            raise ValueError("X and y must contain the same number of rows")
        training = pd.DataFrame(
            {
                "category": categories,
                "subscriber_tier": tiers,
                "target": target,
            }
        )
        self.global_median_ = float(target.median())
        self.category_mapping_ = (
            training.groupby("category")["target"]
            .median()
            .astype(float)
            .to_dict()
        )
        self.category_tier_mapping_ = (
            training.groupby(["category", "subscriber_tier"])["target"]
            .median()
            .astype(float)
            .to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        categories = (
            X["category_name"]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
        tiers = (
            X["subscriber_tier"]
            .astype("string")
            .fillna("missing")
            .astype(str)
        )
        return np.asarray(
            [
                self.category_tier_mapping_.get(
                    (category, tier),
                    self.category_mapping_.get(
                        category, self.global_median_
                    ),
                )
                for category, tier in zip(categories, tiers)
            ],
            dtype=float,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_horizon_checkpoint(
    project_root: Path, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Path, Path]:
    data_path = (
        project_root
        / "Dataset"
        / "model_horizon_datasets"
        / f"viewcastlk_day_{horizon}.csv"
    )
    assignment_path = (
        project_root
        / "Dataset"
        / "model_split_metadata"
        / f"viewcastlk_day_{horizon}_split_assignments.csv"
    )
    frame = pd.read_csv(data_path, low_memory=False)
    assignments = pd.read_csv(assignment_path, low_memory=False)
    target = f"d{horizon}_views"

    if target not in frame:
        raise ValueError(f"Day {horizon}: missing target {target}")
    if len(frame) != len(assignments):
        raise ValueError(
            f"Day {horizon}: horizon data and split metadata row counts differ"
        )
    expected_positions = np.arange(len(frame))
    actual_positions = assignments["horizon_row_position"].to_numpy()
    if not np.array_equal(actual_positions, expected_positions):
        raise ValueError(
            f"Day {horizon}: split metadata is not aligned to horizon row order"
        )
    if {"video_id", "channel_id"} & set(frame.columns):
        raise ValueError(
            f"Day {horizon}: identifier leaked into the model horizon CSV"
        )
    other_targets = {
        f"d{other}_views" for other in HORIZONS if other != horizon
    }
    leaked_targets = other_targets & set(frame.columns)
    if leaked_targets:
        raise ValueError(
            f"Day {horizon}: other horizon targets leaked: {sorted(leaked_targets)}"
        )

    X = frame.drop(columns=[target]).copy()
    y = pd.to_numeric(frame[target], errors="raise").astype(float)
    if y.isna().any() or (y < 0).any():
        raise ValueError(f"Day {horizon}: target must be non-negative and complete")
    return X, y, assignments, data_path, assignment_path


def fit_baselines(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    channel_ids: pd.Series,
) -> dict[str, Any]:
    return {
        "global_median": GlobalMedianBaseline().fit(X_train, y_train_log),
        "category_median": CategoryMedianBaseline().fit(X_train, y_train_log),
        "category_tier_median": FixedSubscriberTierMedianBaseline().fit(
            X_train, y_train_log
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
    best_trees: int | None = None,
) -> dict[str, Any]:
    return {
        "horizon_days": horizon,
        "fold": fold,
        "model": model_name,
        "train_rows_before_filter": train_rows_before_filter,
        "train_rows_after_filter": train_rows_after_filter,
        "outliers_removed": (
            train_rows_before_filter - train_rows_after_filter
        ),
        "lower_log_bound": lower_log_bound,
        "upper_log_bound": upper_log_bound,
        "best_trees": best_trees,
        **regression_metrics(y_true, y_pred),
    }


def train_horizon_cross_validation(
    *,
    horizon: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    n_estimators: int,
    n_jobs: int,
    include_llm_scores: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    best_tree_counts: list[int] = []

    development = assignments["partition"].eq("development")
    for fold in range(1, CV_FOLDS + 1):
        validation_mask = (
            development
            & assignments["cv_validation_fold"].eq(fold)
        )
        training_mask = development & ~validation_mask
        train_positions_all = assignments.loc[
            training_mask, "horizon_row_position"
        ].astype(int).to_numpy()
        validation_positions = assignments.loc[
            validation_mask, "horizon_row_position"
        ].astype(int).to_numpy()

        train_channels = set(
            assignments.loc[training_mask, "channel_id"].astype(str)
        )
        validation_channels = set(
            assignments.loc[validation_mask, "channel_id"].astype(str)
        )
        if not train_channels.isdisjoint(validation_channels):
            raise AssertionError(
                f"Day {horizon} fold {fold}: channel leakage detected"
            )

        y_train_log_all = np.log1p(y.iloc[train_positions_all])
        inlier_mask, lower, upper = log_target_inlier_mask(
            y_train_log_all, sigma=OUTLIER_SIGMA
        )
        train_positions = train_positions_all[inlier_mask]
        X_train = X.iloc[train_positions]
        y_train_log = np.log1p(y.iloc[train_positions])
        X_validation = X.iloc[validation_positions]
        y_validation = y.iloc[validation_positions]
        y_validation_log = np.log1p(y_validation)

        baselines = fit_baselines(
            X_train,
            y_train_log,
            assignments.iloc[train_positions]["channel_id"],
        )
        fold_predictions: dict[str, np.ndarray] = {}
        for baseline_name, baseline in baselines.items():
            predicted_views = views_from_log_predictions(
                baseline.predict(X_validation)
            )
            fold_predictions[baseline_name] = predicted_views
            metric_rows.append(
                metrics_row(
                    horizon=horizon,
                    fold=fold,
                    model_name=baseline_name,
                    y_true=y_validation,
                    y_pred=predicted_views,
                    train_rows_before_filter=len(train_positions_all),
                    train_rows_after_filter=len(train_positions),
                    lower_log_bound=lower,
                    upper_log_bound=upper,
                )
            )

        preprocessor = HorizonDatasetPreprocessor(
            category_smoothing=CATEGORY_SMOOTHING,
            include_llm_scores=include_llm_scores,
        )
        X_train_transformed = preprocessor.fit_transform(
            X_train, y_train_log
        )
        X_validation_transformed = preprocessor.transform(X_validation)
        regressor = build_xgb_regressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
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
        best_trees = int(regressor.best_iteration) + 1
        best_tree_counts.append(best_trees)
        fold_predictions[MODEL_NAME] = predicted_views
        metric_rows.append(
            metrics_row(
                horizon=horizon,
                fold=fold,
                model_name=MODEL_NAME,
                y_true=y_validation,
                y_pred=predicted_views,
                train_rows_before_filter=len(train_positions_all),
                train_rows_after_filter=len(train_positions),
                lower_log_bound=lower,
                upper_log_bound=upper,
                best_trees=best_trees,
            )
        )

        prediction_frame = assignments.iloc[validation_positions].copy()
        prediction_frame["actual_views"] = y_validation.to_numpy()
        for model_name, values in fold_predictions.items():
            prediction_frame[f"predicted_{model_name}"] = values
        prediction_frames.append(prediction_frame)
        xgb_metrics = metric_rows[-1]
        print(
            f"day {horizon} fold {fold}/{CV_FOLDS}: "
            f"WAPE={xgb_metrics['wape_pct']:.2f}%, "
            f"RMSLE={xgb_metrics['rmsle']:.4f}, "
            f"best trees={best_trees}"
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        best_tree_counts,
    )


def summarise_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_columns = {
        column.removeprefix("predicted_"): column
        for column in predictions
        if column.startswith("predicted_")
    }
    for horizon, group in predictions.groupby("horizon_days", sort=True):
        for model_name, column in prediction_columns.items():
            rows.append(
                {
                    "horizon_days": str(int(horizon)),
                    "model": model_name,
                    **regression_metrics(
                        group["actual_views"], group[column]
                    ),
                }
            )
    for model_name, column in prediction_columns.items():
        rows.append(
            {
                "horizon_days": "combined",
                "model": model_name,
                **regression_metrics(
                    predictions["actual_views"], predictions[column]
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_development_bundle(
    *,
    horizon: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    selected_n_estimators: int,
    n_jobs: int,
    include_llm_scores: bool,
) -> HorizonModelBundle:
    development_positions_all = assignments.loc[
        assignments["partition"].eq("development"),
        "horizon_row_position",
    ].astype(int).to_numpy()
    y_log_all = np.log1p(y.iloc[development_positions_all])
    inlier_mask, lower, upper = log_target_inlier_mask(
        y_log_all, sigma=OUTLIER_SIGMA
    )
    training_positions = development_positions_all[inlier_mask]
    X_train = X.iloc[training_positions]
    y_train_log = np.log1p(y.iloc[training_positions])

    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        include_llm_scores=include_llm_scores,
    )
    transformed = preprocessor.fit_transform(X_train, y_train_log)
    regressor = build_xgb_regressor(
        n_estimators=selected_n_estimators,
        early_stopping_rounds=None,
        n_jobs=n_jobs,
    )
    regressor.fit(transformed, y_train_log, verbose=False)
    return HorizonModelBundle(
        horizon_days=horizon,
        preprocessor=preprocessor,
        regressor=regressor,
        training_metadata={
            "status": "candidate_reserved_test_not_evaluated",
            "development_rows_before_outlier_filter": int(
                len(development_positions_all)
            ),
            "training_rows_after_outlier_filter": int(
                len(training_positions)
            ),
            "outliers_removed": int(
                len(development_positions_all) - len(training_positions)
            ),
            "outlier_sigma": OUTLIER_SIGMA,
            "lower_log_bound": float(lower),
            "upper_log_bound": float(upper),
            "selected_n_estimators": int(selected_n_estimators),
            "category_smoothing": CATEGORY_SMOOTHING,
            "llm_scores_enabled": bool(include_llm_scores),
        },
    )


def validate_training_outputs(
    *,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    artifact_records: list[dict[str, Any]],
    loaded_assignments: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, condition: bool, detail: str = "") -> None:
        checks.append(
            {
                "test": name,
                "status": "PASS" if bool(condition) else "FAIL",
                "detail": detail,
            }
        )

    add(
        "four horizon bundles saved",
        {record["horizon_days"] for record in artifact_records}
        == set(HORIZONS),
    )
    add(
        "all predictions finite",
        np.isfinite(
            predictions.filter(like="predicted_").to_numpy(dtype=float)
        ).all(),
    )
    add(
        "all predictions non-negative",
        (
            predictions.filter(like="predicted_").to_numpy(dtype=float) >= 0
        ).all(),
    )
    add(
        "summary includes all baselines and XGBoost",
        {
            "global_median",
            "category_median",
            "category_tier_median",
            MODEL_NAME,
        }.issubset(summary["model"]),
    )

    for horizon in HORIZONS:
        assignments = loaded_assignments[horizon]
        development_positions = set(
            assignments.loc[
                assignments["partition"].eq("development"),
                "horizon_row_position",
            ].astype(int)
        )
        test_positions = set(
            assignments.loc[
                assignments["partition"].eq("test_reserved"),
                "horizon_row_position",
            ].astype(int)
        )
        prediction_positions = set(
            predictions.loc[
                predictions["horizon_days"].eq(horizon),
                "horizon_row_position",
            ].astype(int)
        )
        add(
            f"day {horizon} every development row predicted once",
            prediction_positions == development_positions,
        )
        add(
            f"day {horizon} reserved test never predicted",
            prediction_positions.isdisjoint(test_positions),
        )

    results = pd.DataFrame(checks)
    failures = results[results["status"].eq("FAIL")]
    if not failures.empty:
        raise AssertionError(
            "Training output checks failed:\n"
            + failures.to_string(index=False)
        )
    return results


def train_all_horizons(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    n_estimators: int = 800,
    n_jobs: int = 4,
    include_llm_scores: bool = False,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = project_root / "artifacts" / "checkpoint5_xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    dataset_records: list[dict[str, Any]] = []
    loaded_assignments: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        print(f"\nTraining independent day-{horizon} model")
        X, y, assignments, data_path, assignment_path = (
            load_horizon_checkpoint(project_root, horizon)
        )
        loaded_assignments[horizon] = assignments
        fold_metrics, predictions, best_tree_counts = (
            train_horizon_cross_validation(
                horizon=horizon,
                X=X,
                y=y,
                assignments=assignments,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                include_llm_scores=include_llm_scores,
            )
        )
        all_metrics.append(fold_metrics)
        all_predictions.append(predictions)

        selected_n_estimators = max(
            1, int(np.median(best_tree_counts))
        )
        bundle = fit_development_bundle(
            horizon=horizon,
            X=X,
            y=y,
            assignments=assignments,
            selected_n_estimators=selected_n_estimators,
            n_jobs=n_jobs,
            include_llm_scores=include_llm_scores,
        )

        model_path = models_dir / f"day_{horizon}_model.joblib"
        encoding_path = (
            models_dir / f"day_{horizon}_category_encoding.json"
        )
        feature_path = models_dir / f"day_{horizon}_feature_order.json"
        joblib.dump(bundle, model_path)
        encoding_path.write_text(
            json.dumps(
                bundle.preprocessor.category_encoding_state(), indent=2
            ),
            encoding="utf-8",
        )
        feature_path.write_text(
            json.dumps(bundle.feature_names, indent=2),
            encoding="utf-8",
        )

        gains = bundle.regressor.get_booster().get_score(
            importance_type="gain"
        )
        for feature in bundle.feature_names:
            importance_rows.append(
                {
                    "horizon_days": horizon,
                    "feature": feature,
                    "gain": float(gains.get(feature, 0.0)),
                }
            )

        artifact_records.append(
            {
                "horizon_days": horizon,
                "model_path": str(model_path.relative_to(output_dir)),
                "model_sha256": sha256_file(model_path),
                "category_encoding_path": str(
                    encoding_path.relative_to(output_dir)
                ),
                "feature_order_path": str(
                    feature_path.relative_to(output_dir)
                ),
                "input_feature_count": int(X.shape[1]),
                "output_feature_count": len(bundle.feature_names),
                **bundle.training_metadata,
            }
        )
        dataset_records.append(
            {
                "horizon_days": horizon,
                "horizon_dataset": str(data_path.relative_to(project_root)),
                "horizon_dataset_sha256": sha256_file(data_path),
                "split_assignments": str(
                    assignment_path.relative_to(project_root)
                ),
                "split_assignments_sha256": sha256_file(assignment_path),
            }
        )

    fold_metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarise_oof_predictions(predictions)
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance["gain_share_within_horizon"] = (
        feature_importance["gain"]
        / feature_importance.groupby("horizon_days")["gain"].transform("sum")
    ).fillna(0.0)

    fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    summary.to_csv(output_dir / "cv_summary_metrics.csv", index=False)
    feature_importance.to_csv(
        output_dir / "feature_importance_gain.csv", index=False
    )

    first_bundle = joblib.load(
        output_dir / artifact_records[0]["model_path"]
    )
    (output_dir / "feature_contract.json").write_text(
        json.dumps(first_bundle.preprocessor.feature_contract(), indent=2),
        encoding="utf-8",
    )

    validation_results = validate_training_outputs(
        summary=summary,
        predictions=predictions,
        artifact_records=artifact_records,
        loaded_assignments=loaded_assignments,
    )
    validation_results.to_csv(
        output_dir / "training_validation_tests.csv", index=False
    )

    manifest = {
        "artifact_version": "checkpoint5_xgboost",
        "status": "candidate_reserved_test_not_evaluated",
        "horizons_are_independent": True,
        "target": "horizon-specific log1p views",
        "primary_metric": "wape_pct",
        "legacy_metric_note": "MAPE is retained only for historical CSV compatibility and is not used for model selection or headline reporting.",
        "split_source": "notebook 04 saved channel-grouped assignments",
        "cross_validation": {
            "folds": CV_FOLDS,
            "group": "channel_id",
            "reserved_test_used": False,
        },
        "outlier_filter": {
            "scale": "log1p target",
            "sigma": OUTLIER_SIGMA,
            "fit_on": "training fold only",
        },
        "category_target_encoding": {
            "scale": "mean log1p target",
            "smoothing": CATEGORY_SMOOTHING,
            "fit_on": "training fold only",
            "unknown_category": "training global mean",
        },
        "llm_scores_enabled": bool(include_llm_scores),
        "llm_score_columns_reserved": list(LLM_SCORE_COLUMNS),
        "explicitly_excluded_model_columns": list(
            EXCLUDED_MODEL_COLUMNS
        ),
        "xgboost_configuration": {
            **MVP_XGB_PARAMS,
            "n_estimators_upper_limit_for_cv": int(n_estimators),
            "n_jobs": int(n_jobs),
        },
        "datasets": dataset_records,
        "models": artifact_records,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nCross-validated summary; reserved test remains untouched:")
    print(summary.to_string(index=False))
    print(
        f"\nPASS: {len(validation_results)} training-output checks. "
        f"Saved artifacts to {output_dir}"
    )
    return {
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "summary": summary,
        "feature_importance": feature_importance,
        "validation_results": validation_results,
        "manifest": manifest,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint5_xgboost",
    )
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--include-llm-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_all_horizons(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        include_llm_scores=args.include_llm_scores,
    )


if __name__ == "__main__":
    main()
