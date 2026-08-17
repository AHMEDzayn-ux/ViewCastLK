"""Fold-safe preprocessing shared by ViewCastLK training and inference.

The module deliberately contains no model fitting. It converts the raw
pre-publication feature contract into a numeric matrix that a horizon-specific
regressor can consume. Fitted state (category target encoding and one-hot
vocabularies) always comes from training rows only.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


MISSING_CATEGORY = "__MISSING__"

LLM_SCORE_COLUMNS = (
    "title_urgency",
    "title_emotional_appeal",
    "title_seriousness",
    "title_curiosity_gap",
)

USER_REMOVED_FEATURES = (
    "title_length",
    "title_word_count",
    "title_upper_ratio",
    "tag_count",
    "description_length",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
)

# The stable input contract. LLM scores are intentionally absent until the
# complete one-model backfill is available.
RAW_INPUT_COLUMNS = (
    "category_name",
    "publish_hour_slt",
    "duration_seconds",
    "is_short",
    "made_for_kids",
    "default_language",
    "publish_is_weekend",
    "title_length",
    "title_word_count",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
    "title_upper_ratio",
    "title_script",
    "tag_count",
    "description_length",
    "ch_subs_at_publish",
    "ch_views_at_publish",
    "ch_videos_at_publish",
    "channel_age_days_at_publish",
)

BOOLEAN_COLUMNS = (
    "is_short",
    "made_for_kids",
    "publish_is_weekend",
)

SUPPORTED_NUMERIC_COLUMNS = (
    "duration_seconds",
    "title_length",
    "title_word_count",
    "title_upper_ratio",
    "tag_count",
    "description_length",
    "ch_subs_at_publish",
    "ch_views_at_publish",
    "ch_videos_at_publish",
    "channel_age_days_at_publish",
    "ch_videos_per_day",
    "ch_views_per_video",
    "ch_views_per_sub",
)

ACTIVE_NUMERIC_COLUMNS = tuple(
    column for column in SUPPORTED_NUMERIC_COLUMNS
    if column not in USER_REMOVED_FEATURES
)

SUPPORTED_BOOLEAN_COLUMNS = (
    "is_short",
    "made_for_kids",
    "publish_is_weekend",
    "title_has_number",
    "title_has_question",
    "title_has_exclaim",
)

REQUIRED_TRAINING_INPUT_COLUMNS = tuple(
    column for column in RAW_INPUT_COLUMNS
    if column not in USER_REMOVED_FEATURES
)

# category_name is handled separately by supervised target encoding.
ACTIVE_CATEGORICAL_COLUMNS = (
    "default_language",
    "title_script",
    "publish_time_bucket",
)


def _binary_as_float(series: pd.Series) -> pd.Series:
    text_values = series.astype("string").str.strip().str.lower()
    mapped = text_values.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce").where(
        lambda values: values > 0
    )
    return (numerator_numeric / denominator_numeric).replace(
        [np.inf, -np.inf], np.nan
    )


def _with_input_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the raw contract in order, adding omitted inference fields as NaN."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected X to be a pandas DataFrame")
    return frame.reindex(columns=RAW_INPUT_COLUMNS).copy()


def engineer_prepublication_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering without fitting or using targets."""
    result = _with_input_contract(frame)

    # Normalise every supported boolean so older saved artifact versions remain usable.
    for column in SUPPORTED_BOOLEAN_COLUMNS:
        result[column] = _binary_as_float(result[column])

    hour = pd.to_numeric(result["publish_hour_slt"], errors="coerce")
    result["publish_time_bucket"] = pd.cut(
        hour,
        bins=[-0.001, 5.999, 14.999, 20.999, 23.999],
        labels=(
            "early_morning",
            "morning_afternoon",
            "evening",
            "late_night",
        ),
        include_lowest=True,
    )

    result["ch_videos_per_day"] = _safe_ratio(
        result["ch_videos_at_publish"], result["channel_age_days_at_publish"]
    )
    result["ch_views_per_video"] = _safe_ratio(
        result["ch_views_at_publish"], result["ch_videos_at_publish"]
    )
    result["ch_views_per_sub"] = _safe_ratio(
        result["ch_views_at_publish"], result["ch_subs_at_publish"]
    )
    return result


def _normalise_categories(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY).astype(str)


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Encode one categorical column from a training-only log target.

    Unknown categories fall back to the fitted global mean. The caller must
    supply ``log1p(views)`` as ``y``; the raw target is never accepted silently.
    """

    def __init__(self, smoothing: float = 10.0, output_name: str = "category_encoded"):
        self.smoothing = smoothing
        self.output_name = output_name

    @staticmethod
    def _as_series(values: pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
        if isinstance(values, pd.DataFrame):
            if values.shape[1] != 1:
                raise ValueError("SmoothedTargetEncoder expects exactly one column")
            return values.iloc[:, 0]
        if isinstance(values, pd.Series):
            return values
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("SmoothedTargetEncoder expects one-dimensional values")
        return pd.Series(array)

    def fit(self, X, y):
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")

        categories = _normalise_categories(self._as_series(X)).reset_index(drop=True)
        target = pd.Series(np.asarray(y, dtype=float).reshape(-1)).reset_index(drop=True)
        if len(categories) != len(target):
            raise ValueError("X and y must contain the same number of rows")
        if target.isna().any() or not np.isfinite(target).all():
            raise ValueError("Target encoding requires a finite log-scale target")

        self.global_mean_ = float(target.mean())
        grouped = pd.DataFrame({"category": categories, "target": target}).groupby(
            "category", dropna=False
        )["target"].agg(["count", "mean"])
        encoded = (
            grouped["count"] * grouped["mean"]
            + self.smoothing * self.global_mean_
        ) / (grouped["count"] + self.smoothing)
        self.mapping_ = {str(key): float(value) for key, value in encoded.items()}
        self.category_counts_ = {
            str(key): int(value) for key, value in grouped["count"].items()
        }
        self.n_features_in_ = 1
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, ("mapping_", "global_mean_"))
        source = self._as_series(X)
        categories = _normalise_categories(source)
        encoded = categories.map(self.mapping_).fillna(self.global_mean_).astype(float)
        return pd.DataFrame(
            {self.output_name: encoded.to_numpy()}, index=source.index
        )

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        check_is_fitted(self, "mapping_")
        return np.asarray([self.output_name], dtype=object)


class HorizonPreprocessor(BaseEstimator, TransformerMixin):
    """Convert one horizon's raw input rows into a model-ready numeric matrix.

    ``fit`` must receive that horizon's training-fold ``log1p`` target. Numeric
    missing values remain NaN for XGBoost's native missing-value handling.
    """

    def __init__(self, category_smoothing: float = 10.0):
        self.category_smoothing = category_smoothing

    @staticmethod
    def _categorical_frame(engineered: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=engineered.index)
        for column in ACTIVE_CATEGORICAL_COLUMNS:
            result[column] = _normalise_categories(engineered[column])
        return result

    def fit(self, X: pd.DataFrame, y):
        missing_columns = sorted(set(REQUIRED_TRAINING_INPUT_COLUMNS) - set(X.columns))
        if missing_columns:
            raise ValueError(
                "Training data is missing raw contract columns: "
                + ", ".join(missing_columns)
            )

        engineered = engineer_prepublication_features(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(engineered) != len(target):
            raise ValueError("X and y must contain the same number of rows")

        self.category_encoder_ = SmoothedTargetEncoder(
            smoothing=self.category_smoothing,
            output_name="category_encoded",
        ).fit(engineered["category_name"], target)

        self.one_hot_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        ).fit(self._categorical_frame(engineered))

        one_hot_names = self.one_hot_encoder_.get_feature_names_out(
            ACTIVE_CATEGORICAL_COLUMNS
        ).tolist()
        self.output_feature_names_ = list(
            ACTIVE_NUMERIC_COLUMNS + BOOLEAN_COLUMNS
        ) + ["category_encoded"] + one_hot_names
        self.numeric_columns_ = list(ACTIVE_NUMERIC_COLUMNS)
        self.boolean_columns_ = list(BOOLEAN_COLUMNS)
        self.feature_names_in_ = np.asarray(RAW_INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(RAW_INPUT_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            ("category_encoder_", "one_hot_encoder_", "output_feature_names_"),
        )
        engineered = engineer_prepublication_features(X)

        # Fallbacks keep the previously trained v1 bundles loadable after a
        # later active-feature decision changes these module-level constants.
        numeric_columns = getattr(
            self,
            "numeric_columns_",
            [column for column in SUPPORTED_NUMERIC_COLUMNS if column in self.output_feature_names_],
        )
        boolean_columns = getattr(
            self,
            "boolean_columns_",
            [column for column in SUPPORTED_BOOLEAN_COLUMNS if column in self.output_feature_names_],
        )
        numeric = engineered[list(numeric_columns + boolean_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        category_encoded = self.category_encoder_.transform(
            engineered["category_name"]
        )
        one_hot_values = self.one_hot_encoder_.transform(
            self._categorical_frame(engineered)
        )
        one_hot = pd.DataFrame(
            one_hot_values,
            index=engineered.index,
            columns=self.one_hot_encoder_.get_feature_names_out(
                ACTIVE_CATEGORICAL_COLUMNS
            ),
        )

        transformed = pd.concat([numeric, category_encoded, one_hot], axis=1)
        return transformed.loc[:, self.output_feature_names_].astype(float)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        check_is_fitted(self, "output_feature_names_")
        return np.asarray(self.output_feature_names_, dtype=object)

    def category_encoding_state(self) -> dict[str, object]:
        """Return the serialisable category state needed for audit/deployment."""
        check_is_fitted(self, "category_encoder_")
        return {
            "smoothing": float(self.category_encoder_.smoothing),
            "global_mean_log_views": float(self.category_encoder_.global_mean_),
            "mapping": dict(self.category_encoder_.mapping_),
            "counts": dict(self.category_encoder_.category_counts_),
        }
