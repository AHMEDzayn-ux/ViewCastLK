"""Compare high-view-sensitive objectives using the frozen development folds.

The reserved test partition is never transformed or predicted. Candidate
selection is based on out-of-fold WAPE separately for each independent horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_checkpoint5_models import (  # noqa: E402
    CATEGORY_SMOOTHING,
    CV_FOLDS,
    HORIZONS,
    load_horizon_checkpoint,
)
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    EXCLUDED_MODEL_COLUMNS,
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    ScaleAwareHorizonModelBundle,
    build_xgb_regressor,
    log_target_inlier_mask,
    regression_metrics,
    views_from_log_predictions,
)


RANDOM_STATE = 42
OUTLIER_SIGMA = 3.0


@dataclass(frozen=True)
class ObjectiveCandidate:
    name: str
    target_scale: str
    objective: str
    eval_metric: str
    sample_weight: str
    remove_log_outliers: bool


CANDIDATES = (
    ObjectiveCandidate(
        name="log_squared_filtered",
        target_scale="log1p",
        objective="reg:squarederror",
        eval_metric="rmse",
        sample_weight="none",
        remove_log_outliers=True,
    ),
    ObjectiveCandidate(
        name="log_squared_all_rows",
        target_scale="log1p",
        objective="reg:squarederror",
        eval_metric="rmse",
        sample_weight="none",
        remove_log_outliers=False,
    ),
    ObjectiveCandidate(
        name="log_weighted_sqrt",
        target_scale="log1p",
        objective="reg:squarederror",
        eval_metric="rmse",
        sample_weight="sqrt_views",
        remove_log_outliers=False,
    ),
    ObjectiveCandidate(
        name="log_weighted_linear_capped",
        target_scale="log1p",
        objective="reg:squarederror",
        eval_metric="rmse",
        sample_weight="linear_views_capped",
        remove_log_outliers=False,
    ),
    ObjectiveCandidate(
        name="raw_absolute",
        target_scale="views",
        objective="reg:absoluteerror",
        eval_metric="mae",
        sample_weight="none",
        remove_log_outliers=False,
    ),
    ObjectiveCandidate(
        name="raw_squared",
        target_scale="views",
        objective="reg:squarederror",
        eval_metric="rmse",
        sample_weight="none",
        remove_log_outliers=False,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_weights(
    y_views: pd.Series,
    strategy: str,
    *,
    reference_median: float | None = None,
) -> tuple[np.ndarray | None, float | None]:
    values = np.asarray(y_views, dtype=float)
    if strategy == "none":
        return None, reference_median
    if reference_median is None:
        positive = values[values > 0]
        reference_median = float(np.median(positive)) if len(positive) else 1.0
    ratio = (values + 1.0) / (reference_median + 1.0)
    if strategy == "sqrt_views":
        weights = np.clip(np.sqrt(ratio), 0.25, 8.0)
    elif strategy == "linear_views_capped":
        weights = np.clip(ratio, 0.10, 20.0)
    else:
        raise ValueError(f"Unknown sample-weight strategy: {strategy}")
    weights = weights / weights.mean()
    return weights.astype(float), reference_median


def model_target(y_views: pd.Series, target_scale: str) -> pd.Series:
    if target_scale == "log1p":
        return np.log1p(y_views.astype(float))
    if target_scale == "views":
        return y_views.astype(float)
    raise ValueError(f"Unknown target scale: {target_scale}")


def views_from_candidate_predictions(
    predictions: np.ndarray, target_scale: str
) -> np.ndarray:
    if target_scale == "log1p":
        return views_from_log_predictions(predictions)
    if target_scale == "views":
        return np.maximum(0.0, np.asarray(predictions, dtype=float))
    raise ValueError(f"Unknown target scale: {target_scale}")


def candidate_fold(
    *,
    candidate: ObjectiveCandidate,
    horizon: int,
    fold: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    n_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    development = assignments["partition"].eq("development")
    validation_mask = (
        development & assignments["cv_validation_fold"].eq(fold)
    )
    training_mask = development & ~validation_mask
    training_positions_all = assignments.loc[
        training_mask, "horizon_row_position"
    ].astype(int).to_numpy()
    validation_positions = assignments.loc[
        validation_mask, "horizon_row_position"
    ].astype(int).to_numpy()

    training_channels = set(
        assignments.loc[training_mask, "channel_id"].astype(str)
    )
    validation_channels = set(
        assignments.loc[validation_mask, "channel_id"].astype(str)
    )
    if not training_channels.isdisjoint(validation_channels):
        raise AssertionError(
            f"Day {horizon} fold {fold}: channel leakage"
        )

    if candidate.remove_log_outliers:
        mask, lower, upper = log_target_inlier_mask(
            np.log1p(y.iloc[training_positions_all]),
            sigma=OUTLIER_SIGMA,
        )
        training_positions = training_positions_all[mask]
    else:
        training_positions = training_positions_all
        lower = np.nan
        upper = np.nan

    X_train = X.iloc[training_positions]
    y_train_views = y.iloc[training_positions]
    X_validation = X.iloc[validation_positions]
    y_validation_views = y.iloc[validation_positions]

    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed_train = preprocessor.fit_transform(
        X_train, np.log1p(y_train_views)
    )
    transformed_validation = preprocessor.transform(X_validation)
    train_target = model_target(
        y_train_views, candidate.target_scale
    )
    validation_target = model_target(
        y_validation_views, candidate.target_scale
    )
    train_weights, reference_median = sample_weights(
        y_train_views, candidate.sample_weight
    )
    validation_weights, _ = sample_weights(
        y_validation_views,
        candidate.sample_weight,
        reference_median=reference_median,
    )

    model = build_xgb_regressor(
        objective=candidate.objective,
        eval_metric=candidate.eval_metric,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    fit_kwargs: dict[str, Any] = {
        "eval_set": [
            (transformed_validation, validation_target)
        ],
        "verbose": False,
    }
    if train_weights is not None:
        fit_kwargs["sample_weight"] = train_weights
        fit_kwargs["sample_weight_eval_set"] = [
            validation_weights
        ]
    model.fit(transformed_train, train_target, **fit_kwargs)
    best_trees = int(model.best_iteration) + 1
    predicted_views = views_from_candidate_predictions(
        model.predict(transformed_validation),
        candidate.target_scale,
    )

    prediction_frame = assignments.iloc[validation_positions][
        [
            "horizon_days",
            "horizon_row_position",
            "source_row_index",
            "video_id",
            "channel_id",
            "cv_validation_fold",
        ]
    ].copy()
    prediction_frame["candidate"] = candidate.name
    prediction_frame["actual_views"] = (
        y_validation_views.to_numpy()
    )
    prediction_frame["predicted_views"] = predicted_views

    metrics = {
        "horizon_days": horizon,
        "fold": fold,
        "candidate": candidate.name,
        "best_trees": best_trees,
        "training_rows_before_filter": len(
            training_positions_all
        ),
        "training_rows_after_filter": len(training_positions),
        "outliers_removed": len(training_positions_all)
        - len(training_positions),
        "lower_log_bound": lower,
        "upper_log_bound": upper,
        **regression_metrics(
            y_validation_views, predicted_views
        ),
    }
    return prediction_frame, metrics, best_trees


def summarise_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (horizon, candidate), group in predictions.groupby(
        ["horizon_days", "candidate"], sort=True
    ):
        rows.append(
            {
                "horizon_days": str(int(horizon)),
                "candidate": candidate,
                **regression_metrics(
                    group["actual_views"],
                    group["predicted_views"],
                ),
            }
        )
    for candidate, group in predictions.groupby("candidate"):
        rows.append(
            {
                "horizon_days": "combined",
                "candidate": candidate,
                **regression_metrics(
                    group["actual_views"],
                    group["predicted_views"],
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_selected_bundle(
    *,
    horizon: int,
    candidate: ObjectiveCandidate,
    selected_trees: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    n_jobs: int,
) -> ScaleAwareHorizonModelBundle:
    positions_all = assignments.loc[
        assignments["partition"].eq("development"),
        "horizon_row_position",
    ].astype(int).to_numpy()
    if candidate.remove_log_outliers:
        mask, lower, upper = log_target_inlier_mask(
            np.log1p(y.iloc[positions_all]), sigma=OUTLIER_SIGMA
        )
        positions = positions_all[mask]
    else:
        positions = positions_all
        lower = np.nan
        upper = np.nan

    X_train = X.iloc[positions]
    y_train_views = y.iloc[positions]
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed = preprocessor.fit_transform(
        X_train, np.log1p(y_train_views)
    )
    target = model_target(
        y_train_views, candidate.target_scale
    )
    weights, reference_median = sample_weights(
        y_train_views, candidate.sample_weight
    )
    model = build_xgb_regressor(
        objective=candidate.objective,
        eval_metric=candidate.eval_metric,
        n_estimators=selected_trees,
        early_stopping_rounds=None,
        n_jobs=n_jobs,
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights
    model.fit(transformed, target, **fit_kwargs)
    return ScaleAwareHorizonModelBundle(
        horizon_days=horizon,
        preprocessor=preprocessor,
        regressor=model,
        prediction_scale=candidate.target_scale,
        training_metadata={
            "status": "candidate_reserved_test_not_evaluated",
            "primary_metric": "wape_pct",
            "candidate": asdict(candidate),
            "selected_n_estimators": int(selected_trees),
            "training_rows_before_filter": int(
                len(positions_all)
            ),
            "training_rows_after_filter": int(len(positions)),
            "outliers_removed": int(
                len(positions_all) - len(positions)
            ),
            "lower_log_bound": (
                None if np.isnan(lower) else float(lower)
            ),
            "upper_log_bound": (
                None if np.isnan(upper) else float(upper)
            ),
            "weight_reference_median_views": reference_median,
        },
    )


def run_comparison(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    n_estimators: int = 800,
    n_jobs: int = 4,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = (
            project_root / "artifacts" / "checkpoint10_wape"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    best_tree_counts: dict[tuple[int, str], list[int]] = {}
    loaded: dict[
        int, tuple[pd.DataFrame, pd.Series, pd.DataFrame]
    ] = {}

    for horizon in HORIZONS:
        X, y, assignments, _, _ = load_horizon_checkpoint(
            project_root, horizon
        )
        loaded[horizon] = (X, y, assignments)
        for candidate in CANDIDATES:
            key = (horizon, candidate.name)
            best_tree_counts[key] = []
            print(f"\nDay {horizon}: {candidate.name}")
            for fold in range(1, CV_FOLDS + 1):
                prediction, metrics, best_trees = candidate_fold(
                    candidate=candidate,
                    horizon=horizon,
                    fold=fold,
                    X=X,
                    y=y,
                    assignments=assignments,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                )
                all_predictions.append(prediction)
                fold_metric_rows.append(metrics)
                best_tree_counts[key].append(best_trees)
                print(
                    f"fold {fold}: "
                    f"WAPE={metrics['wape_pct']:.2f}%, "
                    f"capture={metrics['total_view_capture_pct']:.2f}%, "
                    f"RMSLE={metrics['rmsle']:.3f}, "
                    f"trees={best_trees}"
                )

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    summary = summarise_predictions(predictions)
    summary.to_csv(
        output_dir / "objective_summary.csv", index=False
    )
    fold_metrics.to_csv(
        output_dir / "objective_fold_metrics.csv", index=False
    )
    predictions.to_csv(
        output_dir / "objective_oof_predictions.csv", index=False
    )

    per_horizon = summary[
        summary["horizon_days"].isin(
            [str(horizon) for horizon in HORIZONS]
        )
    ].copy()
    winner_rows = (
        per_horizon.sort_values(
            ["horizon_days", "wape_pct", "rmsle"]
        )
        .groupby("horizon_days", as_index=False)
        .head(1)
    )
    candidate_lookup = {
        candidate.name: candidate for candidate in CANDIDATES
    }
    selected_records = []
    selected_prediction_frames = []

    for row in winner_rows.itertuples(index=False):
        horizon = int(row.horizon_days)
        candidate = candidate_lookup[row.candidate]
        selected_trees = max(
            1,
            int(
                np.median(
                    best_tree_counts[(horizon, candidate.name)]
                )
            ),
        )
        X, y, assignments = loaded[horizon]
        bundle = fit_selected_bundle(
            horizon=horizon,
            candidate=candidate,
            selected_trees=selected_trees,
            X=X,
            y=y,
            assignments=assignments,
            n_jobs=n_jobs,
        )
        model_path = models_dir / f"day_{horizon}_model.joblib"
        feature_path = (
            models_dir / f"day_{horizon}_feature_order.json"
        )
        encoding_path = (
            models_dir
            / f"day_{horizon}_category_encoding.json"
        )
        joblib.dump(bundle, model_path)
        feature_path.write_text(
            json.dumps(bundle.feature_names, indent=2),
            encoding="utf-8",
        )
        encoding_path.write_text(
            json.dumps(
                bundle.preprocessor.category_encoding_state(),
                indent=2,
            ),
            encoding="utf-8",
        )
        selected_records.append(
            {
                "horizon_days": horizon,
                "selected_candidate": candidate.name,
                "selected_n_estimators": selected_trees,
                "selection_oof_wape_pct": float(row.wape_pct),
                "selection_oof_rmsle": float(row.rmsle),
                "model_path": str(
                    model_path.relative_to(output_dir)
                ),
                "model_sha256": sha256_file(model_path),
                "feature_order_path": str(
                    feature_path.relative_to(output_dir)
                ),
                "category_encoding_path": str(
                    encoding_path.relative_to(output_dir)
                ),
                **bundle.training_metadata,
            }
        )
        selected_prediction_frames.append(
            predictions[
                predictions["horizon_days"].eq(horizon)
                & predictions["candidate"].eq(candidate.name)
            ].copy()
        )

    selected_predictions = pd.concat(
        selected_prediction_frames, ignore_index=True
    )
    selected_metrics = regression_metrics(
        selected_predictions["actual_views"],
        selected_predictions["predicted_views"],
    )
    selected_predictions.to_csv(
        output_dir / "selected_oof_predictions.csv", index=False
    )
    pd.DataFrame(selected_records).to_csv(
        output_dir / "selected_models.csv", index=False
    )

    test_rows = []
    for horizon in HORIZONS:
        X, y, assignments = loaded[horizon]
        predicted_positions = set(
            selected_predictions.loc[
                selected_predictions["horizon_days"].eq(horizon),
                "horizon_row_position",
            ].astype(int)
        )
        development_positions = set(
            assignments.loc[
                assignments["partition"].eq("development"),
                "horizon_row_position",
            ].astype(int)
        )
        reserved_positions = set(
            assignments.loc[
                assignments["partition"].eq("test_reserved"),
                "horizon_row_position",
            ].astype(int)
        )
        test_rows.extend(
            [
                {
                    "test": f"day {horizon} development OOF coverage",
                    "status": (
                        "PASS"
                        if predicted_positions
                        == development_positions
                        else "FAIL"
                    ),
                },
                {
                    "test": f"day {horizon} reserved test untouched",
                    "status": (
                        "PASS"
                        if predicted_positions.isdisjoint(
                            reserved_positions
                        )
                        else "FAIL"
                    ),
                },
            ]
        )
    validation = pd.DataFrame(test_rows)
    if not validation["status"].eq("PASS").all():
        raise AssertionError(
            validation[validation["status"].eq("FAIL")].to_string(
                index=False
            )
        )
    validation.to_csv(
        output_dir / "validation_tests.csv", index=False
    )

    manifest = {
        "artifact_version": "checkpoint10_wape",
        "status": "candidate_reserved_test_not_evaluated",
        "primary_metric": "wape_pct",
        "mape_status": "removed_from_model_selection_and_headline_reporting",
        "split": "saved channel-grouped development folds",
        "reserved_test_used": False,
        "candidate_definitions": [
            asdict(candidate) for candidate in CANDIDATES
        ],
        "selected_models": selected_records,
        "selected_oof_combined_metrics": selected_metrics,
        "selection_note": "Per-horizon winners are selected on development OOF WAPE. Final unbiased performance requires the still-reserved test.",
        "explicitly_excluded_model_columns": list(
            EXCLUDED_MODEL_COLUMNS
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPer-horizon WAPE winners")
    print(
        winner_rows[
            [
                "horizon_days",
                "candidate",
                "wape_pct",
                "total_view_capture_pct",
                "top_decile_view_capture_pct",
                "rmsle",
            ]
        ].to_string(index=False)
    )
    print("\nSelected combined OOF metrics")
    print(json.dumps(selected_metrics, indent=2))
    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "winner_rows": winner_rows,
        "selected_predictions": selected_predictions,
        "selected_metrics": selected_metrics,
        "validation": validation,
        "manifest": manifest,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "checkpoint10_wape",
    )
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_comparison(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
