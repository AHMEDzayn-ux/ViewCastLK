"""Train four aligned independent horizons with monotonic reconciliation."""

from __future__ import annotations

import argparse
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

from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    component_feature_importance,
    fit_horizon_model,
    load_all_horizons,
    load_transition,
    metric_row,
    sample_triple_predictions,
    sha256_file,
)
from viewcastlk_ml.horizon_preprocessing import EXCLUDED_MODEL_COLUMNS  # noqa: E402
from viewcastlk_ml.modeling import (  # noqa: E402
    ReconciledIndependentTrajectoryModelBundle,
)


def run_training(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path,
    artifact_version: str,
    max_estimators: int = 1_000,
    n_jobs: int = 4,
    preprocessor_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    actual_matrix = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual_matrix, axis=1) >= 0).all(axis=1)
    testing = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~testing & naturally_monotone
    evaluation = testing & naturally_monotone
    development_X = X.loc[development].reset_index(drop=True)
    development_channels = metadata.loc[
        development, "channel_id"
    ].reset_index(drop=True)

    models = []
    component_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        print(
            f"Training aligned independent day-{horizon} "
            f"({len(development_X):,} complete rows)"
        )
        model = fit_horizon_model(
            horizon=horizon,
            X=development_X,
            y=targets[horizon][development],
            channels=development_channels,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
            preprocessor_options=preprocessor_options,
        )
        models.append(model)
        component = f"complete_cohort_day_{horizon}_independent"
        component_rows.append({"component": component, **model.training_metadata})
        importance_rows.extend(component_feature_importance(component, model))

    trajectory = ReconciledIndependentTrajectoryModelBundle(
        horizon_models=models,
        training_metadata={
            "construction": "aligned independent horizons plus cumulative maximum",
            "complete_development_rows": int(development.sum()),
            "common_channel_holdout_seed": split_seed,
        },
    )
    model_path = models_dir / "independent_monotonic_trajectory.joblib"
    joblib.dump(trajectory, model_path)
    saved = joblib.load(model_path)

    test_X = X.loc[evaluation].reset_index(drop=True)
    raw_predictions = saved.predict_independent_views(test_X)
    predictions = saved.predict_views(test_X)
    violation_rows = int((np.diff(raw_predictions, axis=1) < 0).any(axis=1).sum())
    metric_rows = []
    for column, horizon in enumerate(HORIZONS):
        actual = targets[horizon][evaluation]
        metric_rows.append(
            metric_row(
                f"complete_day_7_14_21_30_day_{horizon}",
                "aligned_independent_raw",
                actual,
                raw_predictions[:, column],
            )
        )
        metric_rows.append(
            metric_row(
                f"complete_day_7_14_21_30_day_{horizon}",
                "aligned_independent_cumulative_max",
                actual,
                predictions[:, column],
            )
        )
    metrics = pd.DataFrame(metric_rows)
    triple_metrics = metrics[
        metrics["scope"].str.contains(r"day_(?:7|14|21)$", regex=True)
    ].copy()
    validation = pd.DataFrame(
        [
            {"test": "saved bundle reload", "status": "PASS"},
            {
                "test": "reconciled trajectories nondecreasing",
                "status": "PASS"
                if (np.diff(predictions, axis=1) >= -1e-12).all()
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
                if set(metadata.loc[development, "channel_id"].astype(str)).isdisjoint(
                    set(metadata.loc[evaluation, "channel_id"].astype(str))
                )
                else "FAIL",
            },
        ]
    )
    if validation["status"].ne("PASS").any():
        raise AssertionError("Independent trajectory validation failed")

    samples = sample_triple_predictions(
        metadata.loc[evaluation].reset_index(drop=True),
        {horizon: targets[horizon][evaluation] for horizon in HORIZONS},
        predictions,
    )
    metrics.to_csv(output_dir / "complete_trajectory_test_metrics.csv", index=False)
    metrics.to_csv(output_dir / "transition_test_metrics.csv", index=False)
    triple_metrics.to_csv(output_dir / "triple_horizon_test_metrics.csv", index=False)
    samples.to_csv(output_dir / "sample_predictions.csv", index=False)
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    pd.DataFrame(component_rows).to_csv(
        output_dir / "trained_components.csv", index=False
    )
    pd.DataFrame(importance_rows).sort_values(
        ["component", "importance"], ascending=[True, False]
    ).to_csv(output_dir / "feature_importance.csv", index=False)
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

    manifest = {
        "artifact_version": artifact_version,
        "status": "aligned_independent_complete_trajectory_test_evaluated",
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
            "four independent models trained on one aligned complete cohort, "
            "followed by cumulative-maximum monotonic reconciliation"
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
        "feature_contract": models[0].preprocessor.feature_contract(),
        "preprocessor_options": dict(preprocessor_options or {}),
        "complete_four_horizon_rows": int(len(X)),
        "complete_four_horizon_development_rows": int(development.sum()),
        "complete_four_horizon_test_rows": int(testing.sum()),
        "complete_monotone_four_horizon_test_rows": int(evaluation.sum()),
        "raw_independent_order_violation_rows": violation_rows,
        "end_to_end_day_30_testable": True,
        "limitations": [
            "Uses fewer rows because every training video must have all four labels.",
            "The common comparison holdout has been evaluated and must not be used for tuning.",
            "Cumulative maximum can raise later predictions when independent horizons disagree.",
        ],
        "explicitly_excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nAligned independent complete-trajectory metrics:")
    print(metrics.to_string(index=False))
    print(f"\nRaw ordering violations: {violation_rows:,}/{int(evaluation.sum()):,}")
    print(f"Saved aligned independent artifact to {output_dir}")
    return {"manifest": manifest, "metrics": metrics, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint16_aligned_independent",
    )
    parser.add_argument(
        "--artifact-version", default="checkpoint16_aligned_independent"
    )
    parser.add_argument("--max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--enhanced-features", action="store_true")
    parser.add_argument("--compact-categories", action="store_true")
    parser.add_argument("--rare-category-min-count", type=int, default=100)
    parser.add_argument("--drop-feature", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
        preprocessor_options={
            "include_enhanced_features": args.enhanced_features,
            "collapse_rare_categories": args.compact_categories,
            "rare_category_min_count": args.rare_category_min_count,
            "drop_features": tuple(args.drop_feature),
        },
    )


if __name__ == "__main__":
    main()
