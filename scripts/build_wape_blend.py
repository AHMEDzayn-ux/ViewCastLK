"""Build the WAPE-selected horizon ensembles without touching reserved test rows.

The component models were compared with the exact saved channel-grouped folds.
This script uses cross-fitted blend weights: the weight applied to one fold is
chosen using only the other four folds. The final deployable weight is the
median of those five independently fitted weights.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_wape_objectives import (  # noqa: E402
    CANDIDATES,
    fit_selected_bundle,
)
from scripts.train_checkpoint5_models import (  # noqa: E402
    CV_FOLDS,
    HORIZONS,
    load_horizon_checkpoint,
)
from viewcastlk_ml.horizon_preprocessing import EXCLUDED_MODEL_COLUMNS  # noqa: E402
from viewcastlk_ml.modeling import (  # noqa: E402
    EnsembleHorizonModelBundle,
    regression_metrics,
)


WEIGHT_GRID = np.linspace(0.0, 1.0, 101)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(actual.sum())
    if denominator <= 0:
        return np.inf
    return float(np.abs(actual - predicted).sum() / denominator * 100.0)


def pivot_horizon_predictions(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    horizon_rows = predictions[predictions["horizon_days"].eq(horizon)].copy()
    index_columns = [
        "horizon_days",
        "horizon_row_position",
        "source_row_index",
        "video_id",
        "channel_id",
        "cv_validation_fold",
        "actual_views",
    ]
    pivoted = horizon_rows.pivot(
        index=index_columns,
        columns="candidate",
        values="predicted_views",
    ).reset_index()
    if pivoted[list(candidate.name for candidate in CANDIDATES)].isna().any().any():
        raise AssertionError(f"Day {horizon}: missing candidate OOF prediction")
    return pivoted


def cross_fitted_pair(
    frame: pd.DataFrame, candidate_a: str, candidate_b: str
) -> tuple[np.ndarray, list[float]]:
    blended = np.empty(len(frame), dtype=float)
    fold_weights: list[float] = []
    actual = frame["actual_views"].to_numpy(dtype=float)
    prediction_a = frame[candidate_a].to_numpy(dtype=float)
    prediction_b = frame[candidate_b].to_numpy(dtype=float)

    for fold in range(1, CV_FOLDS + 1):
        validation = frame["cv_validation_fold"].eq(fold).to_numpy()
        fitting = ~validation
        fitting_actual = actual[fitting]
        losses = [
            wape(
                fitting_actual,
                alpha * prediction_a[fitting] + (1.0 - alpha) * prediction_b[fitting],
            )
            for alpha in WEIGHT_GRID
        ]
        alpha = float(WEIGHT_GRID[int(np.argmin(losses))])
        fold_weights.append(alpha)
        blended[validation] = (
            alpha * prediction_a[validation]
            + (1.0 - alpha) * prediction_b[validation]
        )
    return blended, fold_weights


def select_blends(predictions: pd.DataFrame) -> dict[str, Any]:
    candidate_names = [candidate.name for candidate in CANDIDATES]
    pair_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.DataFrame] = []
    selected_specs: dict[int, dict[str, Any]] = {}

    for horizon in HORIZONS:
        frame = pivot_horizon_predictions(predictions, horizon)
        horizon_candidates: list[tuple[dict[str, Any], np.ndarray]] = []
        for candidate_a, candidate_b in itertools.combinations_with_replacement(
            candidate_names, 2
        ):
            if candidate_a == candidate_b:
                blended = frame[candidate_a].to_numpy(dtype=float)
                fold_weights = [1.0] * CV_FOLDS
            else:
                blended, fold_weights = cross_fitted_pair(
                    frame, candidate_a, candidate_b
                )
            metrics = regression_metrics(frame["actual_views"], blended)
            record = {
                "horizon_days": horizon,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "fold_weights_on_candidate_a": "|".join(
                    f"{weight:.2f}" for weight in fold_weights
                ),
                "final_median_weight_on_candidate_a": float(np.median(fold_weights)),
                **{
                    key: metrics[key]
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
            }
            pair_rows.append(record)
            horizon_candidates.append((record, blended))

        winner, winner_predictions = min(
            horizon_candidates,
            key=lambda item: (item[0]["wape_pct"], item[0]["rmsle"]),
        )
        selected_specs[horizon] = winner
        selected = frame[
            [
                "horizon_days",
                "horizon_row_position",
                "source_row_index",
                "video_id",
                "channel_id",
                "cv_validation_fold",
                "actual_views",
            ]
        ].copy()
        selected["predicted_views"] = winner_predictions
        selected["candidate_a"] = winner["candidate_a"]
        selected["candidate_b"] = winner["candidate_b"]
        selected_rows.append(selected)

    selected_predictions = pd.concat(selected_rows, ignore_index=True)
    selected_metrics = regression_metrics(
        selected_predictions["actual_views"],
        selected_predictions["predicted_views"],
    )
    return {
        "pair_summary": pd.DataFrame(pair_rows),
        "selected_specs": selected_specs,
        "selected_predictions": selected_predictions,
        "selected_metrics": selected_metrics,
    }


def build_blended_models(
    *,
    project_root: Path = PROJECT_ROOT,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    n_jobs: int = 4,
) -> dict[str, Any]:
    source_dir = source_dir or project_root / "artifacts" / "checkpoint10_wape"
    output_dir = output_dir or project_root / "artifacts" / "checkpoint10_wape_blended"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(source_dir / "objective_oof_predictions.csv")
    fold_metrics = pd.read_csv(source_dir / "objective_fold_metrics.csv")
    result = select_blends(predictions)
    pair_summary = result["pair_summary"]
    selected_specs = result["selected_specs"]
    selected_predictions = result["selected_predictions"]

    pair_summary.to_csv(output_dir / "cross_fitted_blend_summary.csv", index=False)
    selected_predictions.to_csv(
        output_dir / "cross_fitted_blend_predictions.csv", index=False
    )

    candidate_lookup = {candidate.name: candidate for candidate in CANDIDATES}
    model_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, str]] = []

    for horizon in HORIZONS:
        X, y, assignments, _, _ = load_horizon_checkpoint(project_root, horizon)
        spec = selected_specs[horizon]
        candidate_names = [spec["candidate_a"], spec["candidate_b"]]
        alpha = float(spec["final_median_weight_on_candidate_a"])
        weights = [alpha, 1.0 - alpha]
        if candidate_names[0] == candidate_names[1] or np.isclose(alpha, 1.0):
            candidate_names = [candidate_names[0]]
            weights = [1.0]
        elif np.isclose(alpha, 0.0):
            candidate_names = [candidate_names[1]]
            weights = [1.0]

        components = []
        tree_counts = []
        for candidate_name in candidate_names:
            tree_values = fold_metrics.loc[
                fold_metrics["horizon_days"].eq(horizon)
                & fold_metrics["candidate"].eq(candidate_name),
                "best_trees",
            ]
            if len(tree_values) != CV_FOLDS:
                raise AssertionError(
                    f"Day {horizon} {candidate_name}: expected {CV_FOLDS} tree counts"
                )
            selected_trees = max(1, int(np.median(tree_values)))
            tree_counts.append(selected_trees)
            components.append(
                fit_selected_bundle(
                    horizon=horizon,
                    candidate=candidate_lookup[candidate_name],
                    selected_trees=selected_trees,
                    X=X,
                    y=y,
                    assignments=assignments,
                    n_jobs=n_jobs,
                )
            )

        bundle = EnsembleHorizonModelBundle(
            horizon_days=horizon,
            components=components,
            weights=weights,
            training_metadata={
                "status": "candidate_reserved_test_not_evaluated",
                "primary_metric": "wape_pct",
                "component_candidates": candidate_names,
                "weights": weights,
                "selected_n_estimators": tree_counts,
                "weight_method": "median_of_five_cross_fitted_fold_weights",
                "selection_oof_wape_pct": float(spec["wape_pct"]),
                "selection_oof_rmsle": float(spec["rmsle"]),
            },
        )
        model_path = models_dir / f"day_{horizon}_ensemble.joblib"
        joblib.dump(bundle, model_path)
        model_rows.append(
            {
                "horizon_days": horizon,
                "components": "|".join(candidate_names),
                "weights": "|".join(f"{weight:.2f}" for weight in weights),
                "selected_n_estimators": "|".join(map(str, tree_counts)),
                "cross_fitted_oof_wape_pct": float(spec["wape_pct"]),
                "cross_fitted_oof_total_view_capture_pct": float(
                    spec["total_view_capture_pct"]
                ),
                "cross_fitted_oof_top_decile_view_capture_pct": float(
                    spec["top_decile_view_capture_pct"]
                ),
                "cross_fitted_oof_rmsle": float(spec["rmsle"]),
                "model_path": str(model_path.relative_to(output_dir)),
                "model_sha256": sha256_file(model_path),
            }
        )

        horizon_predictions = selected_predictions[
            selected_predictions["horizon_days"].eq(horizon)
        ]
        predicted_positions = set(
            horizon_predictions["horizon_row_position"].astype(int)
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
        checks = {
            f"day {horizon} development OOF coverage": (
                predicted_positions == development_positions
            ),
            f"day {horizon} reserved test untouched": predicted_positions.isdisjoint(
                reserved_positions
            ),
            f"day {horizon} saved ensemble smoke prediction": bool(
                np.isfinite(bundle.predict_views(X.iloc[:5])).all()
            ),
        }
        validation_rows.extend(
            {"test": name, "status": "PASS" if passed else "FAIL"}
            for name, passed in checks.items()
        )

    model_summary = pd.DataFrame(model_rows)
    validation = pd.DataFrame(validation_rows)
    if not validation["status"].eq("PASS").all():
        raise AssertionError(validation.to_string(index=False))
    model_summary.to_csv(output_dir / "selected_ensembles.csv", index=False)
    validation.to_csv(output_dir / "validation_tests.csv", index=False)

    visible_metrics = {
        key: result["selected_metrics"][key]
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
    }
    manifest = {
        "artifact_version": "checkpoint10_wape_blended",
        "status": "candidate_reserved_test_not_evaluated",
        "primary_metric": "wape_pct",
        "mape_status": "not_reported_or_used",
        "validation_design": "five saved channel-grouped development folds",
        "channel_ids_used_as_features": False,
        "reserved_test_used": False,
        "selected_oof_combined_metrics": visible_metrics,
        "selected_ensembles": model_rows,
        "candidate_definitions": [asdict(candidate) for candidate in CANDIDATES],
        "explicitly_excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
        "selection_note": (
            "Component pairs and blend weights use development OOF predictions only. "
            "Final unbiased performance requires the still-reserved test."
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Selected WAPE ensembles")
    print(
        model_summary[
            [
                "horizon_days",
                "components",
                "weights",
                "cross_fitted_oof_wape_pct",
                "cross_fitted_oof_total_view_capture_pct",
                "cross_fitted_oof_top_decile_view_capture_pct",
                "cross_fitted_oof_rmsle",
            ]
        ].to_string(index=False)
    )
    print("\nCombined development OOF metrics (MAPE intentionally omitted)")
    print(json.dumps(visible_metrics, indent=2))
    print("\nValidation")
    print(validation.to_string(index=False))
    return {
        "pair_summary": pair_summary,
        "selected_specs": selected_specs,
        "selected_predictions": selected_predictions,
        "selected_metrics": visible_metrics,
        "model_summary": model_summary,
        "validation": validation,
        "manifest": manifest,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint10_wape",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint10_wape_blended",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_blended_models(
        project_root=PROJECT_ROOT,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
