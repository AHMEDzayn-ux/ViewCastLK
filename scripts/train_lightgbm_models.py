"""Train and evaluate LightGBM candidates on the frozen grouped folds.

The four view horizons remain independent. Candidate and tree-count selection
uses only channel-grouped development out-of-fold predictions; the reserved
test partition is never transformed or predicted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import lightgbm as lgb
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
    build_lgbm_regressor,
    log_target_inlier_mask,
    regression_metrics,
    views_from_log_predictions,
)


RANDOM_STATE = 42
OUTLIER_SIGMA = 3.0
EARLY_STOPPING_ROUNDS = 75


@dataclass(frozen=True)
class LightGBMCandidate:
    name: str
    target_scale: str
    objective: str
    remove_log_outliers: bool
    extra_params: dict[str, Any] = field(default_factory=dict)


CANDIDATES = (
    LightGBMCandidate(
        name="log_l2_filtered",
        target_scale="log1p",
        objective="regression",
        remove_log_outliers=True,
    ),
    LightGBMCandidate(
        name="log_l2_all_rows",
        target_scale="log1p",
        objective="regression",
        remove_log_outliers=False,
    ),
    LightGBMCandidate(
        name="log_l1_all_rows",
        target_scale="log1p",
        objective="regression_l1",
        remove_log_outliers=False,
    ),
    LightGBMCandidate(
        name="raw_l1",
        target_scale="views",
        objective="regression_l1",
        remove_log_outliers=False,
    ),
    LightGBMCandidate(
        name="raw_huber",
        target_scale="views",
        objective="huber",
        remove_log_outliers=False,
        extra_params={"alpha": 0.9},
    ),
    LightGBMCandidate(
        name="raw_tweedie",
        target_scale="views",
        objective="tweedie",
        remove_log_outliers=False,
        extra_params={"tweedie_variance_power": 1.2},
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_target(y_views: pd.Series, target_scale: str) -> pd.Series:
    if target_scale == "log1p":
        return np.log1p(y_views.astype(float))
    if target_scale == "views":
        return y_views.astype(float)
    raise ValueError(f"Unknown target scale: {target_scale}")


def predictions_in_views(predictions: np.ndarray, target_scale: str) -> np.ndarray:
    if target_scale == "log1p":
        return views_from_log_predictions(predictions)
    if target_scale == "views":
        return np.maximum(0.0, np.asarray(predictions, dtype=float))
    raise ValueError(f"Unknown target scale: {target_scale}")


def wape_eval(target_scale: str) -> Callable[..., tuple[str, float, bool]]:
    """Return a LightGBM callback metric that evaluates view-scale WAPE."""

    def metric(y_true, y_pred, weight=None):
        actual = predictions_in_views(np.asarray(y_true, dtype=float), target_scale)
        predicted = predictions_in_views(np.asarray(y_pred, dtype=float), target_scale)
        absolute_error = np.abs(actual - predicted)
        if weight is not None:
            weights = np.asarray(weight, dtype=float)
            absolute_error = absolute_error * weights
            actual = actual * weights
        denominator = float(actual.sum())
        value = float(absolute_error.sum() / denominator) if denominator > 0 else np.inf
        return "view_wape", value, False

    return metric


def training_positions(
    *,
    y: pd.Series,
    positions_all: np.ndarray,
    candidate: LightGBMCandidate,
) -> tuple[np.ndarray, float | None, float | None]:
    if not candidate.remove_log_outliers:
        return positions_all, None, None
    mask, lower, upper = log_target_inlier_mask(
        np.log1p(y.iloc[positions_all]), sigma=OUTLIER_SIGMA
    )
    return positions_all[mask], float(lower), float(upper)


def fit_fold(
    *,
    candidate: LightGBMCandidate,
    horizon: int,
    fold: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    n_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    development = assignments["partition"].eq("development")
    validation_mask = development & assignments["cv_validation_fold"].eq(fold)
    train_mask = development & ~validation_mask
    train_positions_all = assignments.loc[
        train_mask, "horizon_row_position"
    ].astype(int).to_numpy()
    validation_positions = assignments.loc[
        validation_mask, "horizon_row_position"
    ].astype(int).to_numpy()

    train_channels = set(assignments.loc[train_mask, "channel_id"].astype(str))
    validation_channels = set(
        assignments.loc[validation_mask, "channel_id"].astype(str)
    )
    if not train_channels.isdisjoint(validation_channels):
        raise AssertionError(f"Day {horizon} fold {fold}: channel leakage")

    train_positions, lower, upper = training_positions(
        y=y,
        positions_all=train_positions_all,
        candidate=candidate,
    )
    X_train = X.iloc[train_positions]
    X_validation = X.iloc[validation_positions]
    y_train_views = y.iloc[train_positions]
    y_validation_views = y.iloc[validation_positions]

    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed_train = preprocessor.fit_transform(
        X_train, np.log1p(y_train_views)
    )
    transformed_validation = preprocessor.transform(X_validation)
    train_target = model_target(y_train_views, candidate.target_scale)
    validation_target = model_target(y_validation_views, candidate.target_scale)

    model = build_lgbm_regressor(
        objective=candidate.objective,
        metric="None",
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        **candidate.extra_params,
    )
    model.fit(
        transformed_train,
        train_target,
        eval_X=transformed_validation,
        eval_y=validation_target,
        eval_metric=wape_eval(candidate.target_scale),
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_trees = int(model.best_iteration_)
    predicted_views = predictions_in_views(
        model.predict(transformed_validation, num_iteration=best_trees),
        candidate.target_scale,
    )

    predictions = assignments.iloc[validation_positions][
        [
            "horizon_days",
            "horizon_row_position",
            "source_row_index",
            "video_id",
            "channel_id",
            "cv_validation_fold",
        ]
    ].copy()
    predictions["candidate"] = candidate.name
    predictions["actual_views"] = y_validation_views.to_numpy()
    predictions["predicted_views"] = predicted_views

    metrics = {
        "horizon_days": horizon,
        "fold": fold,
        "candidate": candidate.name,
        "best_trees": best_trees,
        "training_rows_before_filter": int(len(train_positions_all)),
        "training_rows_after_filter": int(len(train_positions)),
        "outliers_removed": int(len(train_positions_all) - len(train_positions)),
        "lower_log_bound": lower,
        "upper_log_bound": upper,
        **regression_metrics(y_validation_views, predicted_views),
    }
    return predictions, metrics


def summarise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (horizon, candidate), group in predictions.groupby(
        ["horizon_days", "candidate"], sort=True
    ):
        rows.append(
            {
                "horizon_days": str(int(horizon)),
                "candidate": candidate,
                **regression_metrics(group["actual_views"], group["predicted_views"]),
            }
        )
    for candidate, group in predictions.groupby("candidate", sort=True):
        rows.append(
            {
                "horizon_days": "combined",
                "candidate": candidate,
                **regression_metrics(group["actual_views"], group["predicted_views"]),
            }
        )
    return pd.DataFrame(rows)


def select_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        horizon_rows = summary[
            summary["horizon_days"].astype(str).eq(str(horizon))
        ].copy()
        winner = horizon_rows.sort_values(["wape_pct", "rmsle"]).iloc[0]
        rows.append(winner)
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_development_bundle(
    *,
    horizon: int,
    candidate: LightGBMCandidate,
    selected_trees: int,
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    n_jobs: int,
) -> ScaleAwareHorizonModelBundle:
    positions_all = assignments.loc[
        assignments["partition"].eq("development"), "horizon_row_position"
    ].astype(int).to_numpy()
    positions, lower, upper = training_positions(
        y=y,
        positions_all=positions_all,
        candidate=candidate,
    )
    X_train = X.iloc[positions]
    y_train_views = y.iloc[positions]
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed = preprocessor.fit_transform(X_train, np.log1p(y_train_views))
    target = model_target(y_train_views, candidate.target_scale)
    model = build_lgbm_regressor(
        objective=candidate.objective,
        metric="None",
        n_estimators=selected_trees,
        n_jobs=n_jobs,
        **candidate.extra_params,
    )
    model.fit(transformed, target)
    return ScaleAwareHorizonModelBundle(
        horizon_days=horizon,
        preprocessor=preprocessor,
        regressor=model,
        prediction_scale=candidate.target_scale,
        training_metadata={
            "algorithm": "LightGBM",
            "lightgbm_version": lgb.__version__,
            "status": "candidate_reserved_test_not_evaluated",
            "primary_metric": "wape_pct",
            "candidate": asdict(candidate),
            "selected_n_estimators": selected_trees,
            "training_rows_before_filter": int(len(positions_all)),
            "training_rows_after_filter": int(len(positions)),
            "outliers_removed": int(len(positions_all) - len(positions)),
            "lower_log_bound": lower,
            "upper_log_bound": upper,
        },
    )


def feature_importance_frame(
    horizon: int, bundle: ScaleAwareHorizonModelBundle
) -> pd.DataFrame:
    gain = bundle.regressor.booster_.feature_importance(importance_type="gain")
    split = bundle.regressor.booster_.feature_importance(importance_type="split")
    total_gain = float(gain.sum())
    return pd.DataFrame(
        {
            "horizon_days": horizon,
            "feature": bundle.feature_names,
            "gain": gain,
            "gain_share": gain / total_gain if total_gain > 0 else 0.0,
            "split_count": split,
        }
    ).sort_values("gain", ascending=False)


def sample_predictions(selected_predictions: pd.DataFrame) -> pd.DataFrame:
    samples = []
    for horizon, group in selected_predictions.groupby("horizon_days", sort=True):
        ordered = group.sort_values("actual_views").reset_index(drop=True)
        positions = np.unique(
            np.rint(np.linspace(0, len(ordered) - 1, 8)).astype(int)
        )
        sample = ordered.iloc[positions][
            ["horizon_days", "video_id", "actual_views", "predicted_views"]
        ].copy()
        sample["absolute_error_views"] = np.abs(
            sample["actual_views"] - sample["predicted_views"]
        )
        samples.append(sample)
    return pd.concat(samples, ignore_index=True)


def run_lightgbm_training(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    n_estimators: int = 2000,
    n_jobs: int = 4,
) -> dict[str, Any]:
    output_dir = output_dir or project_root / "artifacts" / "checkpoint11_lightgbm"
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    loaded: dict[int, tuple[pd.DataFrame, pd.Series, pd.DataFrame, Path, Path]] = {}
    for horizon in HORIZONS:
        checkpoint = load_horizon_checkpoint(project_root, horizon)
        loaded[horizon] = checkpoint
        X, y, assignments, _, _ = checkpoint
        for candidate in CANDIDATES:
            print(f"\nDay {horizon}: {candidate.name}")
            for fold in range(1, CV_FOLDS + 1):
                fold_predictions, fold_metrics = fit_fold(
                    candidate=candidate,
                    horizon=horizon,
                    fold=fold,
                    X=X,
                    y=y,
                    assignments=assignments,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                )
                prediction_frames.append(fold_predictions)
                metric_rows.append(fold_metrics)
                print(
                    f"fold {fold}: WAPE={fold_metrics['wape_pct']:.2f}%, "
                    f"capture={fold_metrics['total_view_capture_pct']:.2f}%, "
                    f"RMSLE={fold_metrics['rmsle']:.3f}, "
                    f"trees={fold_metrics['best_trees']}"
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    summary = summarise_predictions(predictions)
    selected = select_candidates(summary)
    candidate_lookup = {candidate.name: candidate for candidate in CANDIDATES}

    selected_prediction_frames = []
    model_records = []
    importance_frames = []
    validation_rows: list[dict[str, str]] = []
    for horizon in HORIZONS:
        selected_name = str(
            selected.loc[selected["horizon_days"].astype(str).eq(str(horizon)), "candidate"].iloc[0]
        )
        selected_predictions = predictions[
            predictions["horizon_days"].eq(horizon)
            & predictions["candidate"].eq(selected_name)
        ].copy()
        selected_prediction_frames.append(selected_predictions)
        horizon_fold_metrics = fold_metrics[
            fold_metrics["horizon_days"].eq(horizon)
            & fold_metrics["candidate"].eq(selected_name)
        ]
        selected_trees = max(1, int(np.median(horizon_fold_metrics["best_trees"])))
        X, y, assignments, data_path, assignment_path = loaded[horizon]
        bundle = fit_development_bundle(
            horizon=horizon,
            candidate=candidate_lookup[selected_name],
            selected_trees=selected_trees,
            X=X,
            y=y,
            assignments=assignments,
            n_jobs=n_jobs,
        )
        model_path = models_dir / f"day_{horizon}_lightgbm.joblib"
        joblib.dump(bundle, model_path)
        loaded_bundle = joblib.load(model_path)
        development_positions = assignments.loc[
            assignments["partition"].eq("development"), "horizon_row_position"
        ].astype(int).to_numpy()
        smoke = loaded_bundle.predict_views(X.iloc[development_positions[:5]])
        model_type_ok = loaded_bundle.regressor.__class__.__module__.startswith("lightgbm")
        smoke_ok = len(smoke) == 5 and np.isfinite(smoke).all() and (smoke >= 0).all()
        development_set = set(development_positions)
        predicted_set = set(selected_predictions["horizon_row_position"].astype(int))
        reserved_positions = set(
            assignments.loc[
                assignments["partition"].eq("reserved_test"),
                "horizon_row_position",
            ].astype(int)
        )
        coverage_ok = predicted_set == development_set
        reserved_ok = predicted_set.isdisjoint(reserved_positions)
        for test_name, passed in (
            ("development OOF coverage", coverage_ok),
            ("reserved test untouched", reserved_ok),
            ("saved LightGBM smoke prediction", smoke_ok and model_type_ok),
        ):
            validation_rows.append(
                {
                    "test": f"day {horizon} {test_name}",
                    "status": "PASS" if passed else "FAIL",
                }
            )
        importance_frames.append(feature_importance_frame(horizon, loaded_bundle))
        selected_row = selected[
            selected["horizon_days"].astype(str).eq(str(horizon))
        ].iloc[0]
        model_records.append(
            {
                "horizon_days": horizon,
                "candidate": selected_name,
                "selected_n_estimators": selected_trees,
                "model_path": model_path.relative_to(output_dir).as_posix(),
                "model_sha256": sha256_file(model_path),
                "development_oof_wape_pct": float(selected_row["wape_pct"]),
                "development_oof_rmsle": float(selected_row["rmsle"]),
                "development_oof_total_view_capture_pct": float(
                    selected_row["total_view_capture_pct"]
                ),
                "development_oof_top_decile_view_capture_pct": float(
                    selected_row["top_decile_view_capture_pct"]
                ),
                "dataset_sha256": sha256_file(data_path),
                "split_assignments_sha256": sha256_file(assignment_path),
            }
        )

    validation = pd.DataFrame(validation_rows)
    if validation["status"].ne("PASS").any():
        raise AssertionError("One or more LightGBM artifact checks failed")
    selected_predictions = pd.concat(selected_prediction_frames, ignore_index=True)
    combined_metrics = regression_metrics(
        selected_predictions["actual_views"],
        selected_predictions["predicted_views"],
    )

    predictions.to_csv(output_dir / "candidate_oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "candidate_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    selected.to_csv(output_dir / "selected_models.csv", index=False)
    selected_predictions.to_csv(
        output_dir / "selected_oof_predictions.csv", index=False
    )
    sample_predictions(selected_predictions).to_csv(
        output_dir / "sample_predictions.csv", index=False
    )
    pd.concat(importance_frames, ignore_index=True).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)

    manifest = {
        "artifact_version": "checkpoint11_lightgbm",
        "algorithm": "LightGBM",
        "lightgbm_version": lgb.__version__,
        "status": "candidate_reserved_test_not_evaluated",
        "primary_metric": "wape_pct",
        "validation_design": "five saved channel-grouped development folds",
        "reserved_test_used": False,
        "early_stopping_metric": "view-scale WAPE",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "maximum_estimators": n_estimators,
        "model_parameters": build_lgbm_regressor(n_jobs=n_jobs).get_params(),
        "candidate_definitions": [asdict(candidate) for candidate in CANDIDATES],
        "selected_oof_combined_metrics": combined_metrics,
        "models": model_records,
        "explicitly_excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    selected_display = selected[
        [
            "horizon_days",
            "candidate",
            "wape_pct",
            "rmsle",
            "total_view_capture_pct",
            "top_decile_view_capture_pct",
        ]
    ]
    print("\nSelected LightGBM models")
    print(selected_display.to_string(index=False))
    print("\nCombined selected development OOF metrics")
    print(
        json.dumps(
            {
                key: combined_metrics[key]
                for key in (
                    "wape_pct",
                    "total_view_capture_pct",
                    "top_decile_wape_pct",
                    "top_decile_view_capture_pct",
                    "median_absolute_error_views",
                    "mae_views",
                    "rmsle",
                    "log_r2",
                )
            },
            indent=2,
        )
    )
    print("\nValidation")
    print(validation.to_string(index=False))
    print(f"\nSaved artifacts to {output_dir}")
    return {
        "summary": summary,
        "selected": selected,
        "combined_metrics": combined_metrics,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint11_lightgbm",
    )
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_lightgbm_training(
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
