"""Compare v4 and v5 feature profiles without touching the locked test set.

Every candidate is evaluated with the same five channel-grouped folds on the
complete, naturally monotone development cohort used by the independent
trajectory model. The reserved common-channel holdout is excluded completely.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    load_all_horizons,
    load_transition,
    sha256_file,
)
from scripts.train_checkpoint5_models import CATEGORY_SMOOTHING  # noqa: E402
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    build_xgb_regressor,
    regression_metrics,
    views_from_log_predictions,
)


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    include_enhanced_features: bool = False
    collapse_rare_categories: bool = False
    rare_category_min_count: int = 100
    drop_features: tuple[str, ...] = ()

    def preprocessor_options(self) -> dict[str, Any]:
        return {
            "include_enhanced_features": self.include_enhanced_features,
            "collapse_rare_categories": self.collapse_rare_categories,
            "rare_category_min_count": self.rare_category_min_count,
            "drop_features": self.drop_features,
        }


PROFILES = (
    FeatureProfile(name="current_v4"),
    FeatureProfile(name="compact_v4", collapse_rare_categories=True),
    FeatureProfile(
        name="enhanced_v5",
        include_enhanced_features=True,
        collapse_rare_categories=True,
    ),
    FeatureProfile(
        name="enhanced_pruned_v5",
        include_enhanced_features=True,
        collapse_rare_categories=True,
        drop_features=(
            "publish_dow_sin",
            "publish_dow_cos",
            "title_has_question",
            "topic_technology",
        ),
    ),
)


def evaluate(
    *,
    project_root: Path,
    output_dir: Path,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_all_horizons(project_root)
    transitions = {
        pair: load_transition(loaded, *pair) for pair in TRANSITIONS
    }
    channel_frames = [loaded[7].assignments["channel_id"]] + [
        transitions[pair].metadata["channel_id"] for pair in TRANSITIONS
    ]
    test_channels, split_seed, _ = choose_common_test_channels(channel_frames)

    X, targets, metadata = complete_subset(loaded, HORIZONS)
    actual_matrix = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual_matrix, axis=1) >= 0).all(axis=1)
    locked_test = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~locked_test & naturally_monotone
    development_X = X.loc[development].reset_index(drop=True)
    development_targets = {
        horizon: targets[horizon][development] for horizon in HORIZONS
    }
    channels = metadata.loc[development, "channel_id"].astype(str).reset_index(
        drop=True
    )
    split_positions = list(
        GroupKFold(n_splits=folds).split(development_X, groups=channels)
    )

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        print(f"\nProfile: {profile.name}", flush=True)
        for horizon in HORIZONS:
            actual = np.asarray(development_targets[horizon], dtype=float)
            target_log = np.log1p(actual)
            oof = np.full(len(actual), np.nan, dtype=float)
            for fold_number, (training, validation) in enumerate(
                split_positions, start=1
            ):
                preprocessor = HorizonDatasetPreprocessor(
                    category_smoothing=CATEGORY_SMOOTHING,
                    **profile.preprocessor_options(),
                )
                transformed_training = preprocessor.fit_transform(
                    development_X.iloc[training], target_log[training]
                )
                transformed_validation = preprocessor.transform(
                    development_X.iloc[validation]
                )
                model = build_xgb_regressor(
                    n_estimators=max_estimators,
                    early_stopping_rounds=30,
                    n_jobs=n_jobs,
                )
                model.fit(
                    transformed_training,
                    target_log[training],
                    eval_set=[
                        (transformed_validation, target_log[validation])
                    ],
                    verbose=False,
                )
                predictions = views_from_log_predictions(
                    model.predict(transformed_validation)
                )
                oof[validation] = predictions
                metrics = regression_metrics(actual[validation], predictions)
                fold_rows.append(
                    {
                        "profile": profile.name,
                        "horizon_days": horizon,
                        "fold": fold_number,
                        "training_rows": len(training),
                        "validation_rows": len(validation),
                        "features": transformed_training.shape[1],
                        "best_iteration": int(model.best_iteration) + 1,
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
                print(
                    f"  day {horizon:>2}, fold {fold_number}: "
                    f"RMSLE={metrics['rmsle']:.4f}, "
                    f"features={transformed_training.shape[1]}",
                    flush=True,
                )
                if fold_number == 1:
                    feature_rows.extend(
                        {
                            "profile": profile.name,
                            "horizon_days": horizon,
                            "feature": feature,
                        }
                        for feature in preprocessor.get_feature_names_out()
                    )
            if not np.isfinite(oof).all():
                raise AssertionError(
                    f"{profile.name} day {horizon}: incomplete OOF predictions"
                )
            metrics = regression_metrics(actual, oof)
            oof_rows.append(
                {
                    "profile": profile.name,
                    "horizon_days": horizon,
                    "rows": len(actual),
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

    fold_metrics = pd.DataFrame(fold_rows)
    oof_metrics = pd.DataFrame(oof_rows)
    fold_summary = (
        fold_metrics.groupby(["profile", "horizon_days"], as_index=False)
        .agg(
            mean_fold_rmsle=("rmsle", "mean"),
            std_fold_rmsle=("rmsle", "std"),
            min_features=("features", "min"),
            max_features=("features", "max"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .merge(oof_metrics, on=["profile", "horizon_days"], validate="one_to_one")
    )
    rankings = (
        oof_metrics.groupby("profile", as_index=False)
        .agg(
            mean_rmsle=("rmsle", "mean"),
            mean_log_r2=("log_r2", "mean"),
            mean_wape_pct=("wape_pct", "mean"),
            mean_total_view_capture_pct=("total_view_capture_pct", "mean"),
        )
        .sort_values(["mean_rmsle", "mean_log_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    baseline_rmsle = float(
        rankings.loc[rankings["profile"] == "current_v4", "mean_rmsle"].iloc[0]
    )
    rankings["rmsle_change_vs_v4_pct"] = (
        (rankings["mean_rmsle"] / baseline_rmsle) - 1.0
    ) * 100.0

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof_metrics.to_csv(output_dir / "oof_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "profile_horizon_summary.csv", index=False)
    rankings.to_csv(output_dir / "profile_rankings.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(
        output_dir / "profile_feature_inventory.csv", index=False
    )
    manifest = {
        "status": "development_only_feature_ablation_complete",
        "selection_metric": "mean channel-grouped out-of-fold RMSLE",
        "profiles": [asdict(profile) for profile in PROFILES],
        "folds": folds,
        "development_rows": int(len(development_X)),
        "development_channels": int(channels.nunique()),
        "complete_rows": int(len(X)),
        "naturally_monotone_rows": int(naturally_monotone.sum()),
        "locked_test_rows_excluded": int((locked_test & naturally_monotone).sum()),
        "locked_test_channels_excluded": len(test_channels),
        "locked_test_selection_seed": split_seed,
        "locked_test_evaluated_during_ablation": False,
        "source_training_table_sha256": sha256_file(
            project_root / "Dataset" / "viewcastlk_training_table.csv"
        ),
        "selected_profile": str(rankings.iloc[0]["profile"]),
    }
    (output_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nProfile rankings:", flush=True)
    print(rankings.to_string(index=False), flush=True)
    return {
        "manifest": manifest,
        "rankings": rankings,
        "horizon_summary": fold_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "feature_ablation_v5_20260914",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        folds=args.folds,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
