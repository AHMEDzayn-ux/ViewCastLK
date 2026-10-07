"""Evaluate a Day-7 forecaster on the newest matured videos by publication date."""

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

from scripts.train_monotonic_trajectory import (  # noqa: E402
    fit_horizon_model,
    load_all_horizons,
)


FEATURE_OPTIONS = {
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


def factor_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_log = np.log1p(np.maximum(0.0, np.asarray(actual, dtype=float)))
    predicted_log = np.log1p(np.maximum(0.0, np.asarray(predicted, dtype=float)))
    absolute_log_error = np.abs(predicted_log - actual_log)
    predicted_rank = pd.Series(predicted_log).rank()
    actual_rank = pd.Series(actual_log).rank()
    ranking = (
        predicted_rank.corr(actual_rank)
        if predicted_rank.nunique() > 1 and actual_rank.nunique() > 1
        else np.nan
    )
    return {
        "within_factor_two_pct": float((absolute_log_error <= np.log(2.0)).mean() * 100),
        "typical_error_multiple": float(np.exp(np.median(absolute_log_error))),
        "spearman_ranking": float(ranking),
    }


def evaluate(
    *,
    project_root: Path,
    output_dir: Path,
    test_fraction: float,
    max_estimators: int,
    n_jobs: int,
) -> None:
    day7 = load_all_horizons(project_root)[7]
    source = pd.read_csv(
        project_root / "Dataset" / "viewcastlk_training_table.csv",
        usecols=["published_at"],
        low_memory=False,
    )
    published = pd.to_datetime(
        source.iloc[day7.assignments["source_row_index"].to_numpy(dtype=int)][
            "published_at"
        ].reset_index(drop=True),
        utc=True,
        errors="raise",
    )
    order = np.argsort(published.to_numpy(), kind="stable")
    split_position = int(np.floor((1.0 - test_fraction) * len(order)))
    training_positions = order[:split_position]
    testing_positions = order[split_position:]

    model = fit_horizon_model(
        horizon=7,
        X=day7.X.iloc[training_positions].reset_index(drop=True),
        y=day7.y.iloc[training_positions].to_numpy(dtype=float),
        channels=day7.assignments.iloc[training_positions]["channel_id"].reset_index(
            drop=True
        ),
        max_estimators=max_estimators,
        n_jobs=n_jobs,
        preprocessor_options=FEATURE_OPTIONS,
    )

    test_X = day7.X.iloc[testing_positions].reset_index(drop=True)
    actual = day7.y.iloc[testing_positions].to_numpy(dtype=float)
    prediction = model.predict_views(test_X)
    training_median = float(np.median(day7.y.iloc[training_positions]))
    channel_baseline = pd.to_numeric(
        test_X["prior_d7_median_views"], errors="coerce"
    ).fillna(training_median).clip(lower=0).to_numpy(dtype=float)
    known_channel = (
        pd.to_numeric(test_X["prior_d7_view_count"], errors="coerce")
        .fillna(0)
        .gt(0)
        .to_numpy()
    )

    rows: list[dict[str, object]] = []
    for population, mask in (
        ("all", np.ones(len(actual), dtype=bool)),
        ("channels_with_prior_history", known_channel),
        ("channels_without_prior_history", ~known_channel),
    ):
        for method, values in (
            ("v10_day7_model", prediction),
            ("channel_prior_day7_median", channel_baseline),
        ):
            metrics = factor_metrics(actual[mask], values[mask])
            rows.append(
                {
                    "population": population,
                    "method": method,
                    "rows": int(mask.sum()),
                    **metrics,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "date_split_day7_metrics.csv", index=False)
    manifest = {
        "method": "newest matured Day-7 labels by publication time",
        "test_fraction": test_fraction,
        "training_rows": int(len(training_positions)),
        "test_rows": int(len(testing_positions)),
        "training_end_utc": published.iloc[training_positions].max().isoformat(),
        "test_start_utc": published.iloc[testing_positions].min().isoformat(),
        "test_end_utc": published.iloc[testing_positions].max().isoformat(),
        "selected_n_estimators": model.training_metadata["selected_n_estimators"],
        "feature_options": FEATURE_OPTIONS,
    }
    (output_dir / "date_split_day7_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--max-estimators", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be between zero and one")
    evaluate(
        project_root=args.project_root,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
