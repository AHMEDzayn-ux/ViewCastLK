"""Current fold-safe feature contract for the four independent horizon models.

The horizon CSV files are already row-filtered and contain only pre-publication
predictors plus one target. This module performs the fitted transformations
that must be learned from training rows only:

* smoothed target encoding for category_name
* one-hot vocabularies for default_language and publish_time_bucket

All other transformations are deterministic. Numeric missing values remain
missing so XGBoost can learn their default tree directions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .preprocessing import MISSING_CATEGORY, SmoothedTargetEncoder


LLM_SCORE_COLUMNS = (
    "title_urgency",
    "title_emotional_appeal",
    "title_seriousness",
    "title_curiosity_gap",
)

SUPPORTED_RAW_NUMERIC_COLUMNS = (
    "duration_seconds",
    "ch_subs_at_publish",
    "ch_avg_views_per_video_at_publish",
    "ch_videos_at_publish",
    "channel_age_days_at_publish",
)

RAW_NUMERIC_COLUMNS = SUPPORTED_RAW_NUMERIC_COLUMNS

ENGINEERED_NUMERIC_COLUMNS = (
    "ch_videos_per_day",
)

TOPIC_COLUMNS = (
    "topic_entertainment",
    "topic_fashion",
    "topic_food",
    "topic_gaming",
    "topic_health",
    "topic_hobby",
    "topic_humour",
    "topic_knowledge",
    "topic_lifestyle",
    "topic_music",
    "topic_pet",
    "topic_politics",
    "topic_religion",
    "topic_society",
    "topic_sports",
    "topic_technology",
    "topic_tourism",
    "topic_vehicle",
    "topic_missing",
)

BOOLEAN_COLUMNS = (
    "is_short",
    "publish_is_weekend",
) + TOPIC_COLUMNS

CATEGORICAL_COLUMNS_LEGACY = (
    "default_language",
    "publish_time_bucket",
)

CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS_LEGACY + (
    "subscriber_tier",
)

SUBSCRIBER_TIER_ORDER = (
    "under_1k",
    "1k_to_10k",
    "10k_to_100k",
    "100k_to_250k",
    "250k_to_500k",
    "500k_to_1m",
    "1m_plus",
    "missing",
)

TARGET_ENCODED_COLUMN = "category_name"

# These can remain in an analysis CSV, but this preprocessor never passes them
# into the model. The list records decisions already made in earlier checkpoints.
EXCLUDED_MODEL_COLUMNS = (
    "definition",
    "caption",
    "made_for_kids",
    "publish_hour_slt",
    "publish_dow_slt",
    "tag_count",
    "description_length",
    "title_length",
    "title_word_count",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
    "title_upper_ratio",
    "title_script",
    "ch_views_at_publish",
    "video_id",
    "channel_id",
    "published_at",
)

REQUIRED_TRAINING_COLUMNS = (
    (TARGET_ENCODED_COLUMN,)
    + RAW_NUMERIC_COLUMNS
    + BOOLEAN_COLUMNS
    + CATEGORICAL_COLUMNS
)


def _binary_as_float(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )
    return mapped.fillna(pd.to_numeric(series, errors="coerce")).astype(float)


def _normalise_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY).astype(str)


def _safe_upload_rate(frame: pd.DataFrame) -> pd.Series:
    videos = pd.to_numeric(frame["ch_videos_at_publish"], errors="coerce")
    age = pd.to_numeric(
        frame["channel_age_days_at_publish"], errors="coerce"
    ).where(lambda values: values > 0)
    return (videos / age).replace([np.inf, -np.inf], np.nan)


def subscriber_tier_from_count(series: pd.Series) -> pd.Series:
    subscriber_count = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        subscriber_count,
        bins=[
            -np.inf,
            999,
            9_999,
            99_999,
            249_999,
            499_999,
            999_999,
            np.inf,
        ],
        labels=SUBSCRIBER_TIER_ORDER[:-1],
        include_lowest=True,
    ).astype("string").fillna("missing")


class HorizonDatasetPreprocessor(BaseEstimator, TransformerMixin):
    """Transform the finalized horizon feature table into numeric model input.

    Setting include_llm_scores to True is intentionally strict during fitting:
    all four score columns must exist and be complete. Transformation remains
    tolerant of missing scores so a live API failure can still return a
    degraded prediction after a score-enabled model has been trained.
    """

    def __init__(
        self,
        category_smoothing: float = 10.0,
        include_llm_scores: bool = False,
    ):
        self.category_smoothing = category_smoothing
        self.include_llm_scores = include_llm_scores

    def _validate_training_frame(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected X to be a pandas DataFrame")
        missing = sorted(set(REQUIRED_TRAINING_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                "Training data is missing required feature columns: "
                + ", ".join(missing)
            )
        if self.include_llm_scores:
            missing_llm = sorted(set(LLM_SCORE_COLUMNS) - set(frame.columns))
            if missing_llm:
                raise ValueError(
                    "LLM scoring is enabled but columns are missing: "
                    + ", ".join(missing_llm)
                )
            incomplete = [
                column
                for column in LLM_SCORE_COLUMNS
                if pd.to_numeric(frame[column], errors="coerce").isna().any()
            ]
            if incomplete:
                raise ValueError(
                    "LLM scoring can be enabled only after a complete consistent "
                    "backfill. Incomplete columns: "
                    + ", ".join(incomplete)
                )

    def _categorical_frame(
        self,
        frame: pd.DataFrame,
        categorical_columns: Iterable[str],
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in categorical_columns:
            if column == "subscriber_tier":
                tier_scheme = getattr(
                    self, "subscriber_tier_scheme_", "legacy_broad_v1"
                )
                if (
                    tier_scheme == "legacy_broad_v1"
                    and "ch_subs_at_publish" in frame
                ):
                    subscriber_count = pd.to_numeric(
                        frame["ch_subs_at_publish"], errors="coerce"
                    )
                    result[column] = pd.cut(
                        subscriber_count,
                        bins=[-np.inf, 999, 9_999, 99_999, 999_999, np.inf],
                        labels=(
                            "under_1k",
                            "1k_to_10k",
                            "10k_to_100k",
                            "100k_to_1m",
                            "1m_plus",
                        ),
                        include_lowest=True,
                    ).astype("string").fillna("missing")
                elif column in frame:
                    result[column] = _normalise_category(frame[column])
                elif "ch_subs_at_publish" in frame:
                    result[column] = subscriber_tier_from_count(
                        frame["ch_subs_at_publish"]
                    )
                else:
                    result[column] = "missing"
            elif column in frame:
                result[column] = _normalise_category(frame[column])
            else:
                result[column] = MISSING_CATEGORY
        return result

    def _deterministic_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in SUPPORTED_RAW_NUMERIC_COLUMNS:
            source = (
                frame[column]
                if column in frame
                else pd.Series(np.nan, index=frame.index)
            )
            result[column] = pd.to_numeric(source, errors="coerce")

        if {
            "ch_videos_at_publish",
            "channel_age_days_at_publish",
        }.issubset(frame.columns):
            result["ch_videos_per_day"] = _safe_upload_rate(frame)
        else:
            result["ch_videos_per_day"] = np.nan

        for column in BOOLEAN_COLUMNS:
            source = (
                frame[column]
                if column in frame
                else pd.Series(np.nan, index=frame.index)
            )
            result[column] = _binary_as_float(source)

        if self.include_llm_scores:
            for column in LLM_SCORE_COLUMNS:
                source = (
                    frame[column]
                    if column in frame
                    else pd.Series(np.nan, index=frame.index)
                )
                result[column] = pd.to_numeric(source, errors="coerce")
        return result

    def fit(self, X: pd.DataFrame, y):
        self._validate_training_frame(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(target):
            raise ValueError("X and y must contain the same number of rows")
        if not np.isfinite(target).all():
            raise ValueError("The log-scale training target must be finite")

        self.category_encoder_ = SmoothedTargetEncoder(
            smoothing=self.category_smoothing,
            output_name="category_encoded_log",
        ).fit(X[TARGET_ENCODED_COLUMN], target)

        self.subscriber_tier_scheme_ = "refined_v2"
        self.one_hot_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        ).fit(self._categorical_frame(X, CATEGORICAL_COLUMNS))
        self.categorical_columns_ = list(CATEGORICAL_COLUMNS)

        deterministic_columns = (
            list(RAW_NUMERIC_COLUMNS)
            + list(ENGINEERED_NUMERIC_COLUMNS)
            + list(BOOLEAN_COLUMNS)
        )
        if self.include_llm_scores:
            deterministic_columns += list(LLM_SCORE_COLUMNS)

        one_hot_names = self.one_hot_encoder_.get_feature_names_out(
            self.categorical_columns_
        ).tolist()
        self.deterministic_columns_ = deterministic_columns
        self.output_feature_names_ = (
            deterministic_columns
            + ["category_encoded_log"]
            + one_hot_names
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            (
                "category_encoder_",
                "one_hot_encoder_",
                "deterministic_columns_",
                "output_feature_names_",
            ),
        )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected X to be a pandas DataFrame")

        deterministic = self._deterministic_frame(X).loc[
            :, self.deterministic_columns_
        ]
        category_source = (
            X[TARGET_ENCODED_COLUMN]
            if TARGET_ENCODED_COLUMN in X
            else pd.Series(pd.NA, index=X.index)
        )
        category_encoded = self.category_encoder_.transform(category_source)
        category_encoded.index = X.index

        # Models saved before subscriber tiers were added used two categorical
        # columns. Preserve their inference compatibility.
        categorical_columns = getattr(
            self, "categorical_columns_", list(CATEGORICAL_COLUMNS_LEGACY)
        )
        categorical_frame = self._categorical_frame(
            X, categorical_columns
        )
        one_hot_values = self.one_hot_encoder_.transform(categorical_frame)
        one_hot = pd.DataFrame(
            one_hot_values,
            index=X.index,
            columns=self.one_hot_encoder_.get_feature_names_out(
                categorical_columns
            ),
        )
        transformed = pd.concat(
            [deterministic, category_encoded, one_hot], axis=1
        )
        return transformed.loc[:, self.output_feature_names_].astype(float)

    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "output_feature_names_")
        return np.asarray(self.output_feature_names_, dtype=object)

    def category_encoding_state(self) -> dict[str, object]:
        check_is_fitted(self, "category_encoder_")
        return {
            "smoothing": float(self.category_encoder_.smoothing),
            "global_mean_log_views": float(
                self.category_encoder_.global_mean_
            ),
            "mapping": dict(self.category_encoder_.mapping_),
            "counts": dict(self.category_encoder_.category_counts_),
        }

    def feature_contract(self) -> dict[str, object]:
        return {
            "target_encoded_source": TARGET_ENCODED_COLUMN,
            "raw_numeric": list(RAW_NUMERIC_COLUMNS),
            "engineered_numeric": list(ENGINEERED_NUMERIC_COLUMNS),
            "boolean": list(BOOLEAN_COLUMNS),
            "one_hot_categorical": list(
                getattr(
                    self,
                    "categorical_columns_",
                    CATEGORICAL_COLUMNS_LEGACY,
                )
            ),
            "llm_scores_available_for_later": list(LLM_SCORE_COLUMNS),
            "llm_scores_enabled": bool(self.include_llm_scores),
            "explicitly_excluded": list(EXCLUDED_MODEL_COLUMNS),
        }
