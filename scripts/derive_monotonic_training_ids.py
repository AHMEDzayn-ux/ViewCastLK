"""Reproduce the exact current trajectory-model training-video set.

The deployed artifact was trained with the common channel holdout algorithm in
the historical ``train_monotonic_trajectory.py`` checkpoint. This script
reconstructs that split from the committed horizon assignments, verifies every
recorded component count and split statistic, and only then writes the union of
video IDs used by any fitted component.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewcastlk_ml.training_ids import write_training_video_ids  # noqa: E402


HORIZONS = (7, 14, 21, 30)
TRANSITIONS = ((7, 14), (14, 21), (21, 30))
TEST_SIZE = 0.20
SPLIT_CANDIDATES = 1_000


def load_assignments(project_root: Path) -> dict[int, pd.DataFrame]:
    source = project_root / "Dataset" / "model_split_metadata" / "all_horizon_split_assignments.csv"
    combined = pd.read_csv(source, low_memory=False)
    loaded: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        frame = combined.loc[combined["horizon_days"].eq(horizon)].copy()
        frame["horizon_row_position"] = frame["horizon_row_position"].astype(int)
        if frame[["source_row_index", "video_id", "channel_id"]].duplicated().any():
            raise ValueError(f"Day {horizon} assignments are not unique")
        loaded[horizon] = frame.reset_index(drop=True)
    return loaded


def transition_metadata(
    assignments: dict[int, pd.DataFrame], from_day: int, to_day: int
) -> pd.DataFrame:
    keys = ["source_row_index", "video_id", "channel_id"]
    earlier = assignments[from_day][keys + ["horizon_row_position"]].rename(
        columns={"horizon_row_position": "earlier_position"}
    )
    later = assignments[to_day][keys + ["horizon_row_position"]].rename(
        columns={"horizon_row_position": "later_position"}
    )
    return earlier.merge(later, on=keys, how="inner", validate="one_to_one")


def choose_common_test_channels(
    channel_frames: list[pd.Series],
) -> tuple[set[str], int, list[float]]:
    channels = np.asarray(
        sorted(set().union(*(set(frame.astype(str)) for frame in channel_frames)))
    )
    test_channel_count = max(1, int(round(TEST_SIZE * len(channels))))
    best: tuple[float, int, set[str], list[float]] | None = None
    for seed in range(SPLIT_CANDIDATES):
        selected = set(
            np.random.default_rng(seed)
            .choice(channels, size=test_channel_count, replace=False)
            .tolist()
        )
        ratios = [float(frame.astype(str).isin(selected).mean()) for frame in channel_frames]
        score = max(abs(ratio - TEST_SIZE) for ratio in ratios) + sum(
            (ratio - TEST_SIZE) ** 2 for ratio in ratios
        )
        candidate = (score, seed, selected, ratios)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("Could not reconstruct the common channel holdout")
    return best[2], best[1], best[3]


def derive_training_ids(project_root: Path) -> set[str]:
    assignments = load_assignments(project_root)
    transitions = {
        pair: transition_metadata(assignments, *pair) for pair in TRANSITIONS
    }
    test_channels, seed, ratios = choose_common_test_channels(
        [assignments[7]["channel_id"]]
        + [transitions[pair]["channel_id"] for pair in TRANSITIONS]
    )

    manifest_path = (
        project_root
        / "prediction_api"
        / "model_artifacts"
        / "viewcastlk_monotonic_trajectory_experimental_v1"
        / "evaluation"
        / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_split = manifest["common_split"]
    if seed != expected_split["selection_seed"]:
        raise AssertionError(f"Split seed mismatch: reconstructed {seed}")
    if len(test_channels) != expected_split["test_channels"]:
        raise AssertionError("Test-channel count does not match the deployed artifact")
    expected_ratios = list(expected_split["test_row_fractions"].values())
    if not np.allclose(ratios, expected_ratios, rtol=0, atol=1e-15):
        raise AssertionError("Common split row fractions do not match the artifact")

    components = {item["component"]: item for item in manifest["components"]}
    training_ids: set[str] = set()
    for horizon in HORIZONS:
        rows = assignments[horizon]
        development = ~rows["channel_id"].astype(str).isin(test_channels)
        component = components[f"day_{horizon}_independent"]
        if int(development.sum()) != component["training_rows"]:
            raise AssertionError(f"Day {horizon} training-row count mismatch")
        training_ids.update(rows.loc[development, "video_id"].astype(str))

    for from_day, to_day in TRANSITIONS:
        metadata = transitions[(from_day, to_day)]
        earlier = pd.read_csv(
            project_root / "Dataset" / "model_horizon_datasets" / f"viewcastlk_day_{from_day}.csv",
            usecols=[f"d{from_day}_views"],
        )[f"d{from_day}_views"].to_numpy(dtype=float)
        later = pd.read_csv(
            project_root / "Dataset" / "model_horizon_datasets" / f"viewcastlk_day_{to_day}.csv",
            usecols=[f"d{to_day}_views"],
        )[f"d{to_day}_views"].to_numpy(dtype=float)
        growth = later[metadata["later_position"].to_numpy(dtype=int)] - earlier[
            metadata["earlier_position"].to_numpy(dtype=int)
        ]
        development = ~metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
        valid = growth >= 0
        component = components[f"day_{from_day}_to_{to_day}_increment"]
        if int((development & valid).sum()) != component["training_rows"]:
            raise AssertionError(f"Day {from_day}->{to_day} training-row count mismatch")
        if int((development & ~valid).sum()) != component["negative_observed_growth_rows_excluded"]:
            raise AssertionError(f"Day {from_day}->{to_day} exclusion count mismatch")

    if training_ids & set(
        pd.concat(assignments.values())
        .loc[lambda frame: frame["channel_id"].astype(str).isin(test_channels), "video_id"]
        .astype(str)
    ):
        raise AssertionError("A held-out channel video entered the reconstructed training set")
    return training_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "prediction_api"
            / "model_artifacts"
            / "viewcastlk_monotonic_trajectory_experimental_v1"
            / "training_video_ids.txt"
        ),
    )
    args = parser.parse_args()
    training_ids = derive_training_ids(PROJECT_ROOT)
    count = write_training_video_ids(training_ids, args.output)
    print(f"Verified and wrote {count:,} exact training video IDs to {args.output}")


if __name__ == "__main__":
    main()
