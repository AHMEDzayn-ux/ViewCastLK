"""Train an experimental monotonic day-7/14/21/30 trajectory model.

The frozen dataset contains no video with all four targets. This checkpoint
therefore learns a day-7 base from every available day-7 development row and
three nonnegative growth models from adjacent-horizon overlaps. One common
channel holdout is used across every component so the chained evaluation does
not inherit the incompatible horizon-specific partitions.

The saved bundle predicts:

    day 7  = base
    day 14 = day 7  + nonnegative 7->14 increment
    day 21 = day 14 + nonnegative 14->21 increment
    day 30 = day 21 + nonnegative 21->30 increment

This guarantees a nondecreasing cumulative-view trajectory by construction.
Day-30 end-to-end accuracy cannot be measured until a cohort has all four
labels, so this artifact is experimental and must not replace the product
candidate without later complete-trajectory validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_checkpoint5_models import (  # noqa: E402
    CATEGORY_SMOOTHING,
    load_horizon_checkpoint,
)
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    EXCLUDED_MODEL_COLUMNS,
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    HorizonModelBundle,
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    build_xgb_regressor,
    regression_metrics,
)


HORIZONS = (7, 14, 21, 30)
TRANSITIONS = ((7, 14), (14, 21), (21, 30))
TEST_SIZE = 0.20
SPLIT_CANDIDATES = 1_000
INNER_VALIDATION_CANDIDATES = 100
RANDOM_STATE = 42


@dataclass
class HorizonData:
    X: pd.DataFrame
    y: pd.Series
    assignments: pd.DataFrame


@dataclass
class TransitionData:
    from_day: int
    to_day: int
    X: pd.DataFrame
    earlier_views: np.ndarray
    later_views: np.ndarray
    metadata: pd.DataFrame

    @property
    def growth_views(self) -> np.ndarray:
        return self.later_views - self.earlier_views

    @property
    def valid_growth(self) -> np.ndarray:
        return self.growth_views >= 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_horizons(project_root: Path) -> dict[int, HorizonData]:
    loaded: dict[int, HorizonData] = {}
    for horizon in HORIZONS:
        X, y, assignments, _, _ = load_horizon_checkpoint(
            project_root, horizon
        )
        loaded[horizon] = HorizonData(X=X, y=y, assignments=assignments)
    return loaded


def load_transition(
    loaded: dict[int, HorizonData], from_day: int, to_day: int
) -> TransitionData:
    earlier = loaded[from_day]
    later = loaded[to_day]
    keys = ["source_row_index", "video_id", "channel_id"]
    earlier_rows = earlier.assignments[
        keys + ["horizon_row_position"]
    ].rename(columns={"horizon_row_position": "earlier_position"})
    later_rows = later.assignments[
        keys + ["horizon_row_position"]
    ].rename(columns={"horizon_row_position": "later_position"})
    joined = earlier_rows.merge(
        later_rows,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    earlier_positions = joined["earlier_position"].to_numpy(dtype=int)
    later_positions = joined["later_position"].to_numpy(dtype=int)
    X = earlier.X.iloc[earlier_positions].reset_index(drop=True)
    later_X = later.X.iloc[later_positions].reset_index(drop=True)
    unequal = ~(X.eq(later_X) | (X.isna() & later_X.isna()))
    if unequal.to_numpy().any():
        columns = unequal.any()[unequal.any()].index.tolist()
        raise AssertionError(
            f"Day {from_day}->{to_day}: feature mismatch in {columns}"
        )

    metadata = joined[keys].reset_index(drop=True)
    return TransitionData(
        from_day=from_day,
        to_day=to_day,
        X=X,
        earlier_views=earlier.y.iloc[earlier_positions].to_numpy(dtype=float),
        later_views=later.y.iloc[later_positions].to_numpy(dtype=float),
        metadata=metadata,
    )


def complete_subset(
    loaded: dict[int, HorizonData], horizons: tuple[int, ...]
) -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame]:
    keys = ["source_row_index", "video_id", "channel_id"]
    first = horizons[0]
    merged = loaded[first].assignments[
        keys + ["horizon_row_position"]
    ].rename(columns={"horizon_row_position": f"position_{first}"})
    for horizon in horizons[1:]:
        rows = loaded[horizon].assignments[
            keys + ["horizon_row_position"]
        ].rename(columns={"horizon_row_position": f"position_{horizon}"})
        merged = merged.merge(rows, on=keys, how="inner", validate="one_to_one")

    first_positions = merged[f"position_{first}"].to_numpy(dtype=int)
    X = loaded[first].X.iloc[first_positions].reset_index(drop=True)
    targets = {
        horizon: loaded[horizon]
        .y.iloc[merged[f"position_{horizon}"].to_numpy(dtype=int)]
        .to_numpy(dtype=float)
        for horizon in horizons
    }
    return X, targets, merged[keys].reset_index(drop=True)


def choose_common_test_channels(
    channel_frames: list[pd.Series],
) -> tuple[set[str], int, list[float]]:
    channels = np.asarray(
        sorted(
            set().union(
                *(set(frame.astype(str)) for frame in channel_frames)
            )
        )
    )
    test_channel_count = max(1, int(round(TEST_SIZE * len(channels))))
    best: tuple[float, int, set[str], list[float]] | None = None
    for seed in range(SPLIT_CANDIDATES):
        generator = np.random.default_rng(seed)
        selected = set(
            generator.choice(
                channels, size=test_channel_count, replace=False
            ).tolist()
        )
        ratios = [
            float(frame.astype(str).isin(selected).mean())
            for frame in channel_frames
        ]
        score = max(abs(ratio - TEST_SIZE) for ratio in ratios) + sum(
            (ratio - TEST_SIZE) ** 2 for ratio in ratios
        )
        candidate = (score, seed, selected, ratios)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("Could not select a common channel holdout")
    return best[2], best[1], best[3]


def inner_validation_positions(
    X: pd.DataFrame, channels: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    candidates = GroupShuffleSplit(
        n_splits=INNER_VALIDATION_CANDIDATES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    ).split(X, groups=channels)
    train, validation = min(
        candidates,
        key=lambda positions: abs(
            len(positions[1]) / len(X) - TEST_SIZE
        ),
    )
    if not set(channels.iloc[train]).isdisjoint(
        set(channels.iloc[validation])
    ):
        raise AssertionError("Inner validation channel leakage")
    return train, validation


def fit_log_target_components(
    *,
    X: pd.DataFrame,
    target_views: np.ndarray,
    channels: pd.Series,
    max_estimators: int,
    n_jobs: int,
) -> tuple[HorizonDatasetPreprocessor, Any, int]:
    target = np.asarray(target_views, dtype=float)
    if len(X) != len(target) or len(X) != len(channels):
        raise ValueError("X, target, and channels must have equal length")
    if len(X) == 0 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Target must be finite, nonnegative, and nonempty")
    target_log = np.log1p(target)
    train, validation = inner_validation_positions(X, channels.reset_index(drop=True))

    selection_preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed_train = selection_preprocessor.fit_transform(
        X.iloc[train], target_log[train]
    )
    transformed_validation = selection_preprocessor.transform(
        X.iloc[validation]
    )
    selection_model = build_xgb_regressor(
        n_estimators=max_estimators,
        n_jobs=n_jobs,
    )
    selection_model.fit(
        transformed_train,
        target_log[train],
        eval_set=[(transformed_validation, target_log[validation])],
        verbose=False,
    )
    selected_estimators = max(1, int(selection_model.best_iteration) + 1)

    final_preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING
    )
    transformed = final_preprocessor.fit_transform(X, target_log)
    final_model = build_xgb_regressor(
        n_estimators=selected_estimators,
        early_stopping_rounds=None,
        n_jobs=n_jobs,
    )
    final_model.fit(transformed, target_log, verbose=False)
    return final_preprocessor, final_model, selected_estimators


def fit_horizon_model(
    *,
    horizon: int,
    X: pd.DataFrame,
    y: np.ndarray,
    channels: pd.Series,
    max_estimators: int,
    n_jobs: int,
) -> HorizonModelBundle:
    preprocessor, model, selected = fit_log_target_components(
        X=X,
        target_views=y,
        channels=channels,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
    )
    return HorizonModelBundle(
        horizon_days=horizon,
        preprocessor=preprocessor,
        regressor=model,
        training_metadata={
            "target": f"day_{horizon}_cumulative_views",
            "target_scale": "log1p",
            "training_rows": int(len(X)),
            "training_channels": int(channels.nunique()),
            "selected_n_estimators": selected,
        },
    )


def fit_increment_model(
    *,
    transition: TransitionData,
    training_mask: np.ndarray,
    max_estimators: int,
    n_jobs: int,
) -> NonnegativeIncrementModelBundle:
    valid_training = training_mask & transition.valid_growth
    X = transition.X.loc[valid_training].reset_index(drop=True)
    growth = transition.growth_views[valid_training]
    channels = transition.metadata.loc[
        valid_training, "channel_id"
    ].reset_index(drop=True)
    preprocessor, model, selected = fit_log_target_components(
        X=X,
        target_views=growth,
        channels=channels,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
    )
    return NonnegativeIncrementModelBundle(
        from_horizon_days=transition.from_day,
        to_horizon_days=transition.to_day,
        preprocessor=preprocessor,
        regressor=model,
        training_metadata={
            "target": (
                f"nonnegative_day_{transition.from_day}_to_"
                f"day_{transition.to_day}_view_increment"
            ),
            "target_scale": "log1p",
            "training_rows": int(len(X)),
            "training_channels": int(channels.nunique()),
            "negative_observed_growth_rows_excluded": int(
                (training_mask & ~transition.valid_growth).sum()
            ),
            "selected_n_estimators": selected,
        },
    )


def selected_metrics(actual, predicted) -> dict[str, float | int]:
    metrics = regression_metrics(actual, predicted)
    return {
        key: metrics[key]
        for key in (
            "rows",
            "wape_pct",
            "total_view_capture_pct",
            "median_absolute_error_views",
            "rmsle",
            "log_r2",
        )
    }


def metric_row(scope: str, method: str, actual, predicted) -> dict[str, Any]:
    return {
        "scope": scope,
        "method": method,
        **selected_metrics(actual, predicted),
    }


def sample_triple_predictions(
    metadata: pd.DataFrame,
    targets: dict[int, np.ndarray],
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.copy()
    for column, horizon in enumerate(HORIZONS):
        frame[f"predicted_day_{horizon}_views"] = predictions[:, column]
        if horizon in targets:
            frame[f"actual_day_{horizon}_views"] = targets[horizon]
    ordered = frame.sort_values(
        ["actual_day_21_views", "video_id"]
    ).reset_index(drop=True)
    positions = np.unique(
        np.rint(np.linspace(0, len(ordered) - 1, 8)).astype(int)
    )
    return ordered.iloc[positions].reset_index(drop=True)


def run_training(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    max_estimators: int = 1_000,
    n_jobs: int = 4,
) -> dict[str, Any]:
    output_dir = output_dir or (
        project_root / "artifacts" / "checkpoint12_monotonic_trajectory"
    )
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
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

    horizon_models: dict[int, HorizonModelBundle] = {}
    model_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        horizon_data = loaded[horizon]
        development = ~horizon_data.assignments["channel_id"].astype(str).isin(
            test_channels
        )
        positions = horizon_data.assignments.loc[
            development, "horizon_row_position"
        ].to_numpy(dtype=int)
        channels = horizon_data.assignments.loc[
            development, "channel_id"
        ].reset_index(drop=True)
        print(
            f"Training independent day-{horizon} benchmark/base "
            f"({len(positions):,} rows)"
        )
        bundle = fit_horizon_model(
            horizon=horizon,
            X=horizon_data.X.iloc[positions].reset_index(drop=True),
            y=horizon_data.y.iloc[positions].to_numpy(dtype=float),
            channels=channels,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        horizon_models[horizon] = bundle
        model_rows.append(
            {
                "component": f"day_{horizon}_independent",
                **bundle.training_metadata,
            }
        )

    increment_models: list[NonnegativeIncrementModelBundle] = []
    for pair in TRANSITIONS:
        transition = transitions[pair]
        development = ~transition.metadata["channel_id"].astype(str).isin(
            test_channels
        ).to_numpy()
        print(
            f"Training day-{pair[0]}->{pair[1]} increment "
            f"({int((development & transition.valid_growth).sum()):,} rows)"
        )
        bundle = fit_increment_model(
            transition=transition,
            training_mask=development,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        increment_models.append(bundle)
        model_rows.append(
            {
                "component": f"day_{pair[0]}_to_{pair[1]}_increment",
                **bundle.training_metadata,
            }
        )

    trajectory = MonotonicTrajectoryModelBundle(
        base_model=horizon_models[7],
        increment_models=increment_models,
        training_metadata={
            "status": "experimental_no_complete_four_horizon_labels",
            "common_channel_holdout_seed": split_seed,
            "test_channel_count": len(test_channels),
        },
    )
    model_path = models_dir / "monotonic_trajectory.joblib"
    joblib.dump(trajectory, model_path)
    loaded_trajectory = joblib.load(model_path)

    metric_rows: list[dict[str, Any]] = []
    for pair, increment_model in zip(TRANSITIONS, increment_models):
        transition = transitions[pair]
        testing = transition.metadata["channel_id"].astype(str).isin(
            test_channels
        ).to_numpy()
        valid_testing = testing & transition.valid_growth
        X_test = transition.X.loc[valid_testing].reset_index(drop=True)
        actual_earlier = transition.earlier_views[valid_testing]
        actual_later = transition.later_views[valid_testing]
        actual_growth = transition.growth_views[valid_testing]
        predicted_growth = increment_model.predict_increment_views(X_test)
        predicted_earlier = horizon_models[pair[0]].predict_views(X_test)
        predicted_later_independent = horizon_models[pair[1]].predict_views(
            X_test
        )
        metric_rows.extend(
            [
                metric_row(
                    f"day_{pair[0]}_to_{pair[1]}",
                    "growth_only",
                    actual_growth,
                    predicted_growth,
                ),
                metric_row(
                    f"day_{pair[0]}_to_{pair[1]}",
                    "teacher_forced_actual_previous_plus_growth",
                    actual_later,
                    actual_earlier + predicted_growth,
                ),
                metric_row(
                    f"day_{pair[0]}_to_{pair[1]}",
                    "one_step_predicted_previous_plus_growth",
                    actual_later,
                    predicted_earlier + predicted_growth,
                ),
                metric_row(
                    f"day_{pair[0]}_to_{pair[1]}",
                    "independent_later_horizon",
                    actual_later,
                    predicted_later_independent,
                ),
            ]
        )

    triple_X, triple_targets, triple_metadata = complete_subset(
        loaded, (7, 14, 21)
    )
    triple_test = triple_metadata["channel_id"].astype(str).isin(
        test_channels
    ).to_numpy()
    triple_monotone = (
        (triple_targets[14] >= triple_targets[7])
        & (triple_targets[21] >= triple_targets[14])
    )
    triple_eval = triple_test & triple_monotone
    triple_test_X = triple_X.loc[triple_eval].reset_index(drop=True)
    trajectory_predictions = loaded_trajectory.predict_views(triple_test_X)
    triple_metric_rows: list[dict[str, Any]] = []
    for column, horizon in enumerate((7, 14, 21)):
        actual = triple_targets[horizon][triple_eval]
        triple_metric_rows.append(
            metric_row(
                f"complete_day_7_14_21_day_{horizon}",
                "monotonic_trajectory",
                actual,
                trajectory_predictions[:, column],
            )
        )
        triple_metric_rows.append(
            metric_row(
                f"complete_day_7_14_21_day_{horizon}",
                "independent_horizon",
                actual,
                horizon_models[horizon].predict_views(triple_test_X),
            )
        )

    all_day7_predictions = loaded_trajectory.predict_views(loaded[7].X)
    monotonic_ok = bool(
        (np.diff(all_day7_predictions, axis=1) >= -1e-12).all()
    )
    finite_ok = bool(
        np.isfinite(all_day7_predictions).all()
        and (all_day7_predictions >= 0).all()
    )
    validation = pd.DataFrame(
        [
            {
                "test": "saved bundle reload",
                "status": "PASS",
            },
            {
                "test": "all same-input trajectories nondecreasing",
                "status": "PASS" if monotonic_ok else "FAIL",
            },
            {
                "test": "all predictions finite and nonnegative",
                "status": "PASS" if finite_ok else "FAIL",
            },
            {
                "test": "complete four-horizon label count recorded as zero",
                "status": "PASS",
            },
        ]
    )
    if validation["status"].ne("PASS").any():
        raise AssertionError("Monotonic trajectory validation failed")

    overlap_rows = []
    for pair in TRANSITIONS:
        transition = transitions[pair]
        overlap_rows.append(
            {
                "labels": f"day_{pair[0]}_and_day_{pair[1]}",
                "rows": len(transition.X),
                "channels": transition.metadata["channel_id"].nunique(),
                "negative_growth_rows": int((~transition.valid_growth).sum()),
            }
        )
    overlap_rows.extend(
        [
            {
                "labels": "day_7_14_21",
                "rows": len(triple_X),
                "channels": triple_metadata["channel_id"].nunique(),
                "negative_growth_rows": int((~triple_monotone).sum()),
            },
            {
                "labels": "day_7_14_21_30",
                "rows": 0,
                "channels": 0,
                "negative_growth_rows": 0,
            },
        ]
    )

    transition_metrics = pd.DataFrame(metric_rows)
    triple_metrics = pd.DataFrame(triple_metric_rows)
    samples = sample_triple_predictions(
        triple_metadata.loc[triple_eval].reset_index(drop=True),
        {
            horizon: values[triple_eval]
            for horizon, values in triple_targets.items()
        },
        trajectory_predictions,
    )
    transition_metrics.to_csv(
        output_dir / "transition_test_metrics.csv", index=False
    )
    triple_metrics.to_csv(
        output_dir / "triple_horizon_test_metrics.csv", index=False
    )
    samples.to_csv(output_dir / "sample_predictions.csv", index=False)
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(
        output_dir / "overlap_summary.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output_dir / "trained_components.csv", index=False
    )

    manifest = {
        "artifact_version": "checkpoint12_monotonic_trajectory",
        "status": "experimental_test_evaluated_no_complete_four_horizon_labels",
        "model_path": model_path.relative_to(output_dir).as_posix(),
        "model_sha256": sha256_file(model_path),
        "horizons": list(HORIZONS),
        "construction": "day-7 base plus three predicted nonnegative increments",
        "common_split": {
            "type": "channel_grouped_shared_across_all_components",
            "selection_seed": split_seed,
            "test_size_target": TEST_SIZE,
            "test_channels": len(test_channels),
            "test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "max_estimators_during_selection": max_estimators,
        "components": model_rows,
        "complete_four_horizon_rows": 0,
        "end_to_end_day_30_testable": False,
        "limitations": [
            "No video has all four labels in the frozen dataset.",
            "Only 650 videos have both day-21 and day-30 labels.",
            "Day-30 end-to-end chain accuracy is not measurable yet.",
            "The common experimental test split has now been evaluated.",
        ],
        "explicitly_excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nTransition test metrics:")
    print(transition_metrics.to_string(index=False))
    print("\nComplete day-7/14/21 test metrics:")
    print(triple_metrics.to_string(index=False))
    print("\nSample complete day-7/14/21 predictions:")
    print(samples.round(0).to_string(index=False))
    print(f"\nSaved experimental artifact to {output_dir}")
    return {
        "manifest": manifest,
        "transition_metrics": transition_metrics,
        "triple_metrics": triple_metrics,
        "samples": samples,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "checkpoint12_monotonic_trajectory",
    )
    parser.add_argument("--max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        output_dir=args.output_dir,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
