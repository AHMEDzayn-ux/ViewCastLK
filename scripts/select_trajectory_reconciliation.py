"""Select and release a monotonic reconciliation rule with grouped OOF data.

The four horizon regressors are trained on every clean label available for
their horizon. Their raw predictions are then reconciled into a cumulative
trajectory. Reconciliation is selected only from channel-grouped development
OOF predictions; reserved channels are opened once as a release gate.
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
from sklearn.isotonic import isotonic_regression
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    fit_horizon_model,
    load_all_horizons,
    load_transition,
    metric_row,
    sha256_file,
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
MONOTONIC_METHODS = ("forward_max", "backward_min", "isotonic_log")


def reconcile_predictions(predictions: np.ndarray, method: str) -> np.ndarray:
    values = np.maximum(0.0, np.asarray(predictions, dtype=float))
    if values.ndim != 2 or values.shape[1] != len(HORIZONS):
        raise ValueError("Expected one prediction column per forecast horizon")
    if method == "raw":
        return values.copy()
    if method == "forward_max":
        return np.maximum.accumulate(values, axis=1)
    if method == "backward_min":
        return np.minimum.accumulate(values[:, ::-1], axis=1)[:, ::-1]
    if method == "isotonic_log":
        logged = np.log1p(values)
        return np.expm1(
            np.vstack(
                [isotonic_regression(row, increasing=True) for row in logged]
            )
        )
    raise ValueError(f"Unknown reconciliation method: {method}")


def evaluate_methods(
    actual: np.ndarray,
    raw_predictions: np.ndarray,
    *,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in ("raw", *MONOTONIC_METHODS):
        predictions = reconcile_predictions(raw_predictions, method)
        horizon_rmsle: list[float] = []
        for column, horizon in enumerate(HORIZONS):
            metrics = metric_row(
                f"day_{horizon}", method, actual[:, column], predictions[:, column]
            )
            rows.append({"partition": partition, **metrics})
            horizon_rmsle.append(float(metrics["rmsle"]))
        summaries.append(
            {
                "partition": partition,
                "method": method,
                "mean_rmsle": float(np.mean(horizon_rmsle)),
                "monotonic": bool(
                    (np.diff(predictions, axis=1) >= -1e-9).all()
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def run_selection(
    *,
    project_root: Path,
    full_checkpoint: Path,
    aligned_checkpoint: Path,
    output_dir: Path,
    artifact_version: str,
    cv_folds: int = 5,
    cv_max_estimators: int = 500,
    n_jobs: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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
    actual = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual, axis=1) >= 0).all(axis=1)
    reserved = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~reserved & naturally_monotone
    evaluation = reserved & naturally_monotone
    development_positions = np.flatnonzero(development)
    development_channels = metadata.loc[
        development, "channel_id"
    ].astype(str).reset_index(drop=True)

    oof_raw = np.full((len(X), len(HORIZONS)), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=cv_folds)
    for fold, (_, validation_relative) in enumerate(
        splitter.split(development_positions, groups=development_channels),
        start=1,
    ):
        validation_positions = development_positions[validation_relative]
        validation_channels = set(
            metadata.iloc[validation_positions]["channel_id"].astype(str)
        )
        print(
            f"OOF fold {fold}/{cv_folds}: "
            f"{len(validation_positions):,} complete validation rows",
            flush=True,
        )
        fold_raw = np.zeros((len(validation_positions), len(HORIZONS)))
        for column, horizon in enumerate(HORIZONS):
            horizon_data = loaded[horizon]
            channel_ids = horizon_data.assignments["channel_id"].astype(str)
            training = ~channel_ids.isin(test_channels | validation_channels)
            positions = horizon_data.assignments.loc[
                training, "horizon_row_position"
            ].to_numpy(dtype=int)
            training_channels = horizon_data.assignments.loc[
                training, "channel_id"
            ].reset_index(drop=True)
            model = fit_horizon_model(
                horizon=horizon,
                X=horizon_data.X.iloc[positions].reset_index(drop=True),
                y=horizon_data.y.iloc[positions].to_numpy(dtype=float),
                channels=training_channels,
                max_estimators=cv_max_estimators,
                n_jobs=n_jobs,
                preprocessor_options=FEATURE_OPTIONS,
            )
            fold_raw[:, column] = model.predict_views(
                X.iloc[validation_positions].reset_index(drop=True)
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "horizon_days": horizon,
                    "training_rows": len(positions),
                    "training_channels": int(training_channels.nunique()),
                    "validation_rows": len(validation_positions),
                    "validation_channels": len(validation_channels),
                    "selected_n_estimators": model.training_metadata[
                        "selected_n_estimators"
                    ],
                }
            )
        oof_raw[validation_positions] = fold_raw

    if np.isnan(oof_raw[development]).any():
        raise AssertionError("Development OOF predictions are incomplete")

    oof_metrics, oof_summary = evaluate_methods(
        actual[development], oof_raw[development], partition="development_oof"
    )
    eligible = oof_summary[
        oof_summary["method"].isin(MONOTONIC_METHODS)
        & oof_summary["monotonic"]
    ]
    selected_method = str(
        eligible.sort_values(["mean_rmsle", "method"]).iloc[0]["method"]
    )

    final_path = (
        full_checkpoint / "models" / "independent_monotonic_trajectory.joblib"
    )
    full_manifest = json.loads(
        (full_checkpoint / "training_manifest.json").read_text(encoding="utf-8")
    )
    final_model = joblib.load(final_path)
    final_model.reconciliation = selected_method
    final_model.training_metadata = {
        **dict(final_model.training_metadata),
        "reconciliation": selected_method,
        "reconciliation_selection": "channel_grouped_development_oof",
    }
    promoted_path = models_dir / "reconciled_independent_trajectory.joblib"
    joblib.dump(final_model, promoted_path)
    promoted_model = joblib.load(promoted_path)

    evaluation_X = X.loc[evaluation].reset_index(drop=True)
    final_raw = promoted_model.predict_independent_views(evaluation_X)
    test_metrics, test_summary = evaluate_methods(
        actual[evaluation], final_raw, partition="reserved_channels"
    )

    aligned_path = (
        aligned_checkpoint / "models" / "independent_monotonic_trajectory.joblib"
    )
    aligned_model = joblib.load(aligned_path)
    aligned_predictions = aligned_model.predict_views(evaluation_X)
    aligned_rmsle = []
    aligned_rows = []
    for column, horizon in enumerate(HORIZONS):
        metrics = metric_row(
            f"day_{horizon}",
            "aligned_complete_cohort_forward_max",
            actual[evaluation, column],
            aligned_predictions[:, column],
        )
        aligned_rows.append({"partition": "reserved_channels", **metrics})
        aligned_rmsle.append(float(metrics["rmsle"]))
    aligned_summary = {
        "partition": "reserved_channels",
        "method": "aligned_complete_cohort_forward_max",
        "mean_rmsle": float(np.mean(aligned_rmsle)),
        "monotonic": bool(
            (np.diff(aligned_predictions, axis=1) >= -1e-9).all()
        ),
    }
    test_metrics = pd.concat(
        [test_metrics, pd.DataFrame(aligned_rows)], ignore_index=True
    )
    test_summary = pd.concat(
        [test_summary, pd.DataFrame([aligned_summary])], ignore_index=True
    )

    selected_oof = float(
        oof_summary.loc[
            oof_summary["method"] == selected_method, "mean_rmsle"
        ].iloc[0]
    )
    forward_oof = float(
        oof_summary.loc[
            oof_summary["method"] == "forward_max", "mean_rmsle"
        ].iloc[0]
    )
    selected_test = float(
        test_summary.loc[
            test_summary["method"] == selected_method, "mean_rmsle"
        ].iloc[0]
    )
    release_gate = bool(
        selected_oof < forward_oof
        and selected_test < aligned_summary["mean_rmsle"]
        and (np.diff(promoted_model.predict_views(evaluation_X), axis=1) >= -1e-9)
        .all()
    )

    promoted_predictions = promoted_model.predict_views(evaluation_X)
    selected_test_metrics = test_metrics[
        test_metrics["method"] == selected_method
    ].reset_index(drop=True)
    selected_test_metrics.to_csv(
        output_dir / "complete_trajectory_test_metrics.csv", index=False
    )
    selected_test_metrics[
        selected_test_metrics["scope"].isin(["day_7", "day_14", "day_21"])
    ].to_csv(output_dir / "triple_horizon_test_metrics.csv", index=False)
    selected_test_metrics.to_csv(
        output_dir / "transition_test_metrics.csv", index=False
    )
    pd.DataFrame(full_manifest["components"]).to_csv(
        output_dir / "trained_components.csv", index=False
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
    validation = pd.DataFrame(
        [
            {"test": "saved bundle reload", "status": "PASS"},
            {
                "test": "predictions finite and nonnegative",
                "status": "PASS"
                if np.isfinite(promoted_predictions).all()
                and (promoted_predictions >= 0).all()
                else "FAIL",
            },
            {
                "test": "trajectories nondecreasing",
                "status": "PASS"
                if (np.diff(promoted_predictions, axis=1) >= -1e-9).all()
                else "FAIL",
            },
            {
                "test": "development and reserved channels disjoint",
                "status": "PASS"
                if set(metadata.loc[development, "channel_id"].astype(str)).isdisjoint(
                    set(metadata.loc[evaluation, "channel_id"].astype(str))
                )
                else "FAIL",
            },
            {
                "test": "release gate",
                "status": "PASS" if release_gate else "FAIL",
            },
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    samples = metadata.loc[evaluation].reset_index(drop=True).copy()
    for column, horizon in enumerate(HORIZONS):
        samples[f"predicted_day_{horizon}_views"] = promoted_predictions[:, column]
        samples[f"actual_day_{horizon}_views"] = actual[evaluation, column]
    samples = samples.sort_values(
        ["actual_day_30_views", "video_id"]
    ).reset_index(drop=True)
    sample_positions = np.unique(
        np.rint(np.linspace(0, len(samples) - 1, 8)).astype(int)
    )
    samples.iloc[sample_positions].to_csv(
        output_dir / "sample_predictions.csv", index=False
    )

    pd.DataFrame(fold_rows).to_csv(
        output_dir / "development_cv_folds.csv", index=False
    )
    oof_metrics.to_csv(output_dir / "development_oof_metrics.csv", index=False)
    oof_summary.to_csv(output_dir / "development_oof_summary.csv", index=False)
    test_metrics.to_csv(output_dir / "reserved_test_metrics.csv", index=False)
    test_summary.to_csv(output_dir / "reserved_test_summary.csv", index=False)

    manifest = {
        "artifact_version": artifact_version,
        "status": (
            "reconciliation_promoted" if release_gate else "release_gate_failed"
        ),
        "source_training_table": {
            "path": "Dataset/viewcastlk_training_table.csv",
            "sha256": sha256_file(
                project_root / "Dataset" / "viewcastlk_training_table.csv"
            ),
        },
        "model_path": promoted_path.relative_to(output_dir).as_posix(),
        "model_sha256": sha256_file(promoted_path),
        "horizons": list(HORIZONS),
        "construction": (
            "four horizon-specific regressors trained on every clean label, "
            "followed by log-space isotonic monotonic reconciliation"
        ),
        "components": full_manifest["components"],
        "selected_reconciliation": selected_method,
        "selection_data": "five-fold channel-grouped development OOF",
        "full_horizon_checkpoint": str(full_checkpoint),
        "aligned_baseline_checkpoint": str(aligned_checkpoint),
        "feature_options": FEATURE_OPTIONS,
        "common_split": {
            "selection_seed": split_seed,
            "test_channels": len(test_channels),
            "source_test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "development_rows": int(development.sum()),
        "reserved_test_rows": int(evaluation.sum()),
        "complete_four_horizon_rows": int(len(X)),
        "complete_four_horizon_test_rows": int(reserved.sum()),
        "complete_monotone_four_horizon_test_rows": int(evaluation.sum()),
        "end_to_end_day_30_testable": True,
        "comparison": {
            "development_forward_max_mean_rmsle": forward_oof,
            "development_selected_mean_rmsle": selected_oof,
            "reserved_aligned_baseline_mean_rmsle": aligned_summary[
                "mean_rmsle"
            ],
            "reserved_selected_mean_rmsle": selected_test,
            "reserved_improvement_pct": float(
                (
                    aligned_summary["mean_rmsle"] - selected_test
                )
                / aligned_summary["mean_rmsle"]
                * 100
            ),
            "release_gate_passed": release_gate,
        },
        "release_gate": (
            "selected monotonic rule must beat forward-max on development OOF "
            "and the aligned clean baseline on reserved channels"
        ),
        "limitations": [
            "Evaluation excludes rows whose observed cumulative views decrease between horizons.",
            "Point forecasts remain conservative for unrecognized breakout videos.",
            "The reserved channel split has been evaluated and must not be used for further tuning.",
        ],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nDevelopment reconciliation summary:", flush=True)
    print(oof_summary.to_string(index=False), flush=True)
    print("\nReserved-channel reconciliation summary:", flush=True)
    print(test_summary.to_string(index=False), flush=True)
    print(f"\nSelected reconciliation: {selected_method}", flush=True)
    print(f"Release gate passed: {release_gate}", flush=True)
    return {
        "manifest": manifest,
        "oof_summary": oof_summary,
        "test_summary": test_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint26_latest_clean_full_20260918",
    )
    parser.add_argument(
        "--aligned-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint25_latest_clean_independent_20260918"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint27_reconciled_latest_clean_20260918"
        ),
    )
    parser.add_argument(
        "--artifact-version",
        default="checkpoint27_reconciled_latest_clean_20260918",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-max-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_selection(
        project_root=PROJECT_ROOT,
        full_checkpoint=args.full_checkpoint,
        aligned_checkpoint=args.aligned_checkpoint,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        cv_folds=args.cv_folds,
        cv_max_estimators=args.cv_max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
