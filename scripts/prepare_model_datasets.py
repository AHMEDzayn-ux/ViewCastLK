"""Build leakage-safe horizon datasets and channel-grouped split metadata.

This is the script equivalent of notebooks 01 and 04.  It makes retraining
from a newer master training-table snapshot reproducible from the command line.
"""

from __future__ import annotations

import argparse
import heapq
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (7, 14, 21, 30)
RANDOM_STATE = 42
TEST_SIZE = 0.20
HOLDOUT_CANDIDATES = 200
CV_FOLDS = 5

SUBSCRIBER_TIER_ORDER = (
    "under_1k", "1k_to_10k", "10k_to_100k", "100k_to_250k",
    "250k_to_500k", "500k_to_1m", "1m_plus", "missing",
)
TOPIC_CANONICAL_GROUPS = {
    "Music": {
        "Christian music", "Classical music", "Electronic music",
        "Hip hop music", "Music", "Music of Asia", "Music of Latin America",
        "Pop music", "Rock music", "Soul music",
    },
    "Gaming": {
        "Action game", "Action-adventure game", "Casual game",
        "Puzzle video game", "Racing video game", "Role-playing video game",
        "Simulation video game", "Sports game", "Strategy video game",
        "Video game culture",
    },
    "Sports": {
        "Association football", "Boxing", "Cricket", "Motorsport", "Sport",
        "Volleyball",
    },
    "Entertainment": {"Entertainment", "Film", "Performing arts", "Television program"},
    "Health": {"Health", "Physical fitness"},
    "Lifestyle": {"Lifestyle (sociology)"},
    "Politics": {"Politics", "Military"},
    "Knowledge": {"Knowledge", "Business"},
}
TOPIC_CANONICAL_LOOKUP = {
    original: canonical
    for canonical, originals in TOPIC_CANONICAL_GROUPS.items()
    for original in originals
}
TOPIC_PARENT_CHILDREN = {
    "Society": {"Politics", "Religion"},
    "Lifestyle": {
        "Tourism", "Vehicle", "Hobby", "Food", "Fashion", "Pet",
        "Technology", "Health",
    },
    "Entertainment": {"Humour"},
}
MODEL_TOPIC_LABELS = (
    "Entertainment", "Fashion", "Food", "Gaming", "Health", "Hobby",
    "Humour", "Knowledge", "Lifestyle", "Music", "Pet", "Politics",
    "Religion", "Society", "Sports", "Technology", "Tourism", "Vehicle",
)

HORIZON_COLUMNS = {
    f"d{horizon}_{suffix}"
    for horizon in HORIZONS
    for suffix in ("views", "likes", "comments", "hours_off", "usable")
}
POST_PUBLICATION_AUDIT_COLUMNS = {
    "ch_stats_as_of", "title_changed", "description_changed"
}
EXCLUDED_MODEL_COLUMNS = {
    "video_id", "channel_id", "category_id", "title", "published_at",
    "thumbnail_url", "default_audio_language", "eligible",
    "is_live_broadcast", "channel_stats_backfilled", "channel_country",
    "ch_views_at_publish", "tags", "definition", "caption", "made_for_kids",
    "topic_categories",
}


def is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def extract_topic_labels(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    labels: list[str] = []
    for item in str(value).split("|"):
        item = item.strip()
        if not item:
            continue
        label = unquote(urlparse(item).path.rsplit("/", 1)[-1]).replace("_", " ").strip()
        label = TOPIC_CANONICAL_LOOKUP.get(label, label)
        if label and label not in labels:
            labels.append(label)
    label_set = set(labels)
    for parent, children in TOPIC_PARENT_CHILDREN.items():
        if parent in label_set and label_set.intersection(children):
            label_set.remove(parent)
    return "|".join(sorted(label_set, key=str.casefold)) if label_set else pd.NA


class RunningViewStats:
    """Streaming view statistics available strictly before a prediction time."""

    def __init__(self) -> None:
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.count = 0
        self.log_mean = 0.0
        self.log_m2 = 0.0
        self.last_views = np.nan
        self.recent: deque[float] = deque(maxlen=5)

    def add(self, views: float) -> None:
        value = float(views)
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

        self.count += 1
        logged = float(np.log1p(value))
        delta = logged - self.log_mean
        self.log_mean += delta / self.count
        self.log_m2 += delta * (logged - self.log_mean)
        self.last_views = value
        self.recent.append(value)

    @property
    def median(self) -> float:
        if not self.count:
            return np.nan
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2.0
        return -self.lower[0]

    @property
    def log_std(self) -> float:
        return np.sqrt(self.log_m2 / (self.count - 1)) if self.count > 1 else np.nan

    @property
    def recent_median(self) -> float:
        return float(np.median(self.recent)) if self.recent else np.nan

    @property
    def recent_log_trend(self) -> float:
        """Return the per-video slope across the five latest log view counts."""
        if len(self.recent) < 2:
            return np.nan
        values = np.log1p(np.asarray(self.recent, dtype=float))
        positions = np.arange(len(values), dtype=float)
        return float(np.polyfit(positions, values, 1)[0])


def _label_events(
    group: pd.DataFrame,
    horizon: int,
) -> list[tuple[pd.Timestamp, str, bool, float]]:
    target = pd.to_numeric(group[f"d{horizon}_views"], errors="coerce")
    hours_off = pd.to_numeric(group[f"d{horizon}_hours_off"], errors="coerce")
    valid = is_true(group[f"d{horizon}_usable"]) & target.notna() & (target >= 0)
    is_short = (
        is_true(group["is_short"])
        if "is_short" in group
        else pd.Series(False, index=group.index)
    )
    available_at = (
        group["_published_utc"]
        + pd.to_timedelta(horizon, unit="D")
        + pd.to_timedelta(hours_off, unit="h")
    )
    events = [
        (
            available_at.loc[index],
            str(group.at[index, "category_name"]),
            bool(is_short.loc[index]),
            float(target.loc[index]),
        )
        for index in group.index[valid & available_at.notna()]
    ]
    return sorted(events, key=lambda item: item[0])


def add_channel_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only history that was observable before each video's publication.

    A prior horizon target becomes eligible at its actual observation timestamp,
    reconstructed from the video's publication time, horizon, and hours offset.
    This prevents a future label from leaking backward into an earlier forecast.
    """

    result = frame.copy()
    result["_published_utc"] = pd.to_datetime(
        result["published_at"], errors="coerce", utc=True
    )
    history_columns = (
        "prior_channel_video_count",
        "uploads_previous_7d",
        "uploads_previous_30d",
        "days_since_previous_upload",
        "prior_d7_view_count",
        "prior_d7_median_views",
        "prior_d7_mean_log_views",
        "prior_d7_std_log_views",
        "prior_d7_last_views",
        "prior_d7_recent5_median_views",
        "prior_d7_recent_log_trend",
        "prior_same_category_d7_count",
        "prior_same_category_d7_median_views",
        "prior_same_format_d7_count",
        "prior_same_format_d7_median_views",
        "prior_d30_view_count",
        "prior_d30_median_views",
        "prior_d30_mean_log_views",
    )
    feature_values = {
        column: np.full(len(result), np.nan, dtype=float)
        for column in history_columns
    }
    row_positions = pd.Series(np.arange(len(result)), index=result.index)

    valid_rows = result[result["_published_utc"].notna()]
    for _, group in valid_rows.groupby("channel_id", sort=False, dropna=False):
        ordered = group.sort_values(["_published_utc", "video_id"], kind="stable")
        events = {horizon: _label_events(ordered, horizon) for horizon in (7, 30)}
        event_positions = {7: 0, 30: 0}
        stats = {7: RunningViewStats(), 30: RunningViewStats()}
        category_d7: dict[str, RunningViewStats] = {}
        format_d7: dict[bool, RunningViewStats] = {}
        prior_uploads_7d: deque[pd.Timestamp] = deque()
        prior_uploads_30d: deque[pd.Timestamp] = deque()
        previous_upload: pd.Timestamp | None = None
        prior_video_count = 0

        for published_at, same_time in ordered.groupby("_published_utc", sort=True):
            for horizon in (7, 30):
                horizon_events = events[horizon]
                while (
                    event_positions[horizon] < len(horizon_events)
                    and horizon_events[event_positions[horizon]][0] <= published_at
                ):
                    _, category, was_short, views = horizon_events[
                        event_positions[horizon]
                    ]
                    stats[horizon].add(views)
                    if horizon == 7:
                        category_d7.setdefault(category, RunningViewStats()).add(views)
                        format_d7.setdefault(was_short, RunningViewStats()).add(views)
                    event_positions[horizon] += 1

            while (
                prior_uploads_7d
                and prior_uploads_7d[0] < published_at - pd.Timedelta(days=7)
            ):
                prior_uploads_7d.popleft()
            while (
                prior_uploads_30d
                and prior_uploads_30d[0] < published_at - pd.Timedelta(days=30)
            ):
                prior_uploads_30d.popleft()
            days_since = (
                (published_at - previous_upload).total_seconds() / 86_400
                if previous_upload is not None
                else np.nan
            )

            positions = row_positions.loc[same_time.index].to_numpy(dtype=int)
            common_values = {
                "prior_channel_video_count": prior_video_count,
                "uploads_previous_7d": len(prior_uploads_7d),
                "uploads_previous_30d": len(prior_uploads_30d),
                "days_since_previous_upload": days_since,
                "prior_d7_view_count": stats[7].count,
                "prior_d7_median_views": stats[7].median,
                "prior_d7_mean_log_views": stats[7].log_mean if stats[7].count else np.nan,
                "prior_d7_std_log_views": stats[7].log_std,
                "prior_d7_last_views": stats[7].last_views,
                "prior_d7_recent5_median_views": stats[7].recent_median,
                "prior_d7_recent_log_trend": stats[7].recent_log_trend,
                "prior_d30_view_count": stats[30].count,
                "prior_d30_median_views": stats[30].median,
                "prior_d30_mean_log_views": stats[30].log_mean if stats[30].count else np.nan,
            }
            for column, value in common_values.items():
                feature_values[column][positions] = value
            categories = same_time["category_name"].astype(str)
            feature_values["prior_same_category_d7_count"][positions] = [
                category_d7[category].count if category in category_d7 else 0
                for category in categories
            ]
            feature_values["prior_same_category_d7_median_views"][positions] = [
                category_d7[category].median if category in category_d7 else np.nan
                for category in categories
            ]
            current_formats = (
                is_true(same_time["is_short"])
                if "is_short" in same_time
                else pd.Series(False, index=same_time.index)
            )
            feature_values["prior_same_format_d7_count"][positions] = [
                format_d7[bool(value)].count if bool(value) in format_d7 else 0
                for value in current_formats
            ]
            feature_values["prior_same_format_d7_median_views"][positions] = [
                format_d7[bool(value)].median
                if bool(value) in format_d7
                else np.nan
                for value in current_formats
            ]

            prior_video_count += len(same_time)
            prior_uploads_7d.extend([published_at] * len(same_time))
            prior_uploads_30d.extend([published_at] * len(same_time))
            previous_upload = published_at

    for column, values in feature_values.items():
        result[column] = values
    return result.drop(columns="_published_utc")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_channel_history_features(frame)
    published_slt = pd.to_datetime(
        frame["published_at"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Colombo")
    frame["publish_time_bucket"] = pd.cut(
        published_slt.dt.hour,
        bins=[-0.001, 5.999, 14.999, 20.999, 23.999],
        labels=["early_morning", "morning_afternoon", "evening", "late_night"],
        include_lowest=True,
    ).astype("string").fillna("unknown")

    subscribers = pd.to_numeric(frame["ch_subs_at_publish"], errors="coerce")
    frame["subscriber_tier"] = pd.cut(
        subscribers,
        bins=[-np.inf, 999, 9_999, 99_999, 249_999, 499_999, 999_999, np.inf],
        labels=SUBSCRIBER_TIER_ORDER[:-1],
        include_lowest=True,
    ).astype("string").fillna("missing")

    channel_views = pd.to_numeric(frame["ch_views_at_publish"], errors="coerce")
    channel_videos = pd.to_numeric(
        frame["ch_videos_at_publish"], errors="coerce"
    ).where(lambda values: values > 0)
    frame["ch_avg_views_per_video_at_publish"] = (
        channel_views / channel_videos
    ).replace([np.inf, -np.inf], np.nan).mask(is_true(frame["channel_stats_backfilled"]))

    frame["topic_categories"] = frame["topic_categories"].map(extract_topic_labels)
    topic_sets = frame["topic_categories"].fillna("").map(
        lambda value: {topic for topic in value.split("|") if topic}
    )
    for topic in MODEL_TOPIC_LABELS:
        feature = "topic_" + topic.casefold().replace(" ", "_")
        frame[feature] = topic_sets.map(lambda values, topic=topic: topic in values)
    frame["topic_missing"] = frame["topic_categories"].isna()
    return frame


def horizon_mask(frame: pd.DataFrame, horizon: int) -> pd.Series:
    target = f"d{horizon}_views"
    return (
        is_true(frame["eligible"])
        & ~is_true(frame["is_live_broadcast"])
        & ~is_true(frame["title_changed"])
        & ~is_true(frame["channel_stats_backfilled"])
        & is_true(frame[f"d{horizon}_usable"])
        & pd.to_numeric(frame[target], errors="coerce").notna()
    )


def isolate_horizon(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = f"d{horizon}_views"
    excluded = (
        (HORIZON_COLUMNS - {target})
        | POST_PUBLICATION_AUDIT_COLUMNS
        | EXCLUDED_MODEL_COLUMNS
    )
    columns = [column for column in frame.columns if column not in excluded]
    return frame.loc[horizon_mask(frame, horizon), columns].copy()


def choose_channel_holdout(groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(groups))
    candidates = GroupShuffleSplit(
        n_splits=HOLDOUT_CANDIDATES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    ).split(positions, groups=groups)
    return min(candidates, key=lambda pair: abs(len(pair[1]) / len(groups) - TEST_SIZE))


def validation_fold_numbers(groups: pd.Series) -> np.ndarray:
    folds = np.zeros(len(groups), dtype=np.int64)
    positions = np.arange(len(groups))
    for fold, (_, validation) in enumerate(
        GroupKFold(n_splits=CV_FOLDS).split(positions, groups=groups), start=1
    ):
        folds[validation] = fold
    return folds


def prepare(master_path: Path, dataset_root: Path, results_root: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, low_memory=False)
    engineered = engineer_features(master)
    model_dir = dataset_root / "model_horizon_datasets"
    split_dir = dataset_root / "model_split_metadata"
    model_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    all_assignments: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        mask = horizon_mask(master, horizon)
        model_frame = isolate_horizon(engineered, horizon).reset_index(drop=True)
        model_path = model_dir / f"viewcastlk_day_{horizon}.csv"
        model_frame.to_csv(model_path, index=False)

        source_rows = master.loc[mask, ["video_id", "channel_id"]].copy()
        source_rows.insert(0, "source_row_index", source_rows.index)
        source_rows = source_rows.reset_index(drop=True)
        if len(source_rows) != len(model_frame):
            raise AssertionError(f"Day {horizon}: source/model row count mismatch")
        groups = source_rows["channel_id"].astype(str)
        development_positions, test_positions = choose_channel_holdout(groups)
        assignments = source_rows.copy()
        assignments.insert(0, "horizon_row_position", np.arange(len(assignments)))
        assignments.insert(0, "horizon_days", horizon)
        assignments["partition"] = "development"
        assignments.loc[test_positions, "partition"] = "test_reserved"
        assignments["cv_validation_fold"] = pd.Series(
            pd.NA, index=assignments.index, dtype="Int64"
        )
        development_groups = assignments.loc[
            development_positions, "channel_id"
        ].astype(str).reset_index(drop=True)
        assignments.loc[development_positions, "cv_validation_fold"] = (
            validation_fold_numbers(development_groups)
        )
        assignments.to_csv(
            split_dir / f"viewcastlk_day_{horizon}_split_assignments.csv",
            index=False,
        )
        all_assignments.append(assignments)

        development = assignments[assignments["partition"] == "development"]
        testing = assignments[assignments["partition"] == "test_reserved"]
        development_channels = set(development["channel_id"].astype(str))
        test_channels = set(testing["channel_id"].astype(str))
        if not development_channels.isdisjoint(test_channels):
            raise AssertionError(f"Day {horizon}: channel leakage into holdout")
        summaries.append({
            "horizon_days": horizon,
            "total_rows": len(assignments),
            "development_rows": len(development),
            "test_rows": len(testing),
            "test_row_percent": 100 * len(testing) / len(assignments),
            "development_channels": len(development_channels),
            "test_channels": len(test_channels),
            "channel_overlap": 0,
        })
        for fold in range(1, CV_FOLDS + 1):
            validation = development[development["cv_validation_fold"] == fold]
            training = development[development["cv_validation_fold"] != fold]
            train_channels = set(training["channel_id"].astype(str))
            validation_channels = set(validation["channel_id"].astype(str))
            if not train_channels.isdisjoint(validation_channels):
                raise AssertionError(f"Day {horizon} fold {fold}: channel leakage")
            fold_summaries.append({
                "horizon_days": horizon,
                "fold": fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "training_channels": len(train_channels),
                "validation_channels": len(validation_channels),
                "channel_overlap": 0,
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(results_root / "split_summary.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(results_root / "cv_fold_summary.csv", index=False)
    pd.concat(all_assignments, ignore_index=True).to_csv(
        split_dir / "all_horizon_split_assignments.csv", index=False
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "Dataset" / "viewcastlk_training_table.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "model_splits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare(args.input, args.dataset_root, args.results_root)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nPASS: horizon datasets and channel-grouped splits regenerated.")


if __name__ == "__main__":
    main()
