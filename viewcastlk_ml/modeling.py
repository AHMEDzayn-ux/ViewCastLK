"""Model, baseline, evaluation, and artifact components for ViewCastLK."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from .preprocessing import HorizonPreprocessor


MISSING_CATEGORY = "__MISSING__"

MVP_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": 4,
}

MVP_LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "l2",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": -1,
}


def log_target_inlier_mask(
    y_log: pd.Series | np.ndarray,
    *,
    sigma: float = 3.0,
) -> tuple[np.ndarray, float, float]:
    """Fit a log-target sigma rule and return its training-row mask and bounds."""
    values = np.asarray(y_log, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("The log target must contain only finite values")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    lower = mean - sigma * standard_deviation
    upper = mean + sigma * standard_deviation
    return (values >= lower) & (values <= upper), lower, upper


def views_from_log_predictions(log_predictions) -> np.ndarray:
    """Invert log1p predictions and prevent impossible negative view counts."""
    return np.maximum(0.0, np.expm1(np.asarray(log_predictions, dtype=float)))


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Return transparent view-scale metrics, including zero-target handling."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values")

    nonzero = actual > 0
    absolute_error = np.abs(actual - predicted)
    percentage_error = absolute_error[nonzero] / actual[nonzero]
    symmetric_denominator = np.abs(actual) + np.abs(predicted)
    symmetric_terms = np.divide(
        2.0 * absolute_error,
        symmetric_denominator,
        out=np.zeros_like(absolute_error),
        where=symmetric_denominator > 0,
    )
    actual_log = np.log1p(np.maximum(actual, 0.0))
    predicted_log = np.log1p(np.maximum(predicted, 0.0))
    total_actual_views = float(actual.sum())
    wape = (
        float(absolute_error.sum() / total_actual_views * 100)
        if total_actual_views > 0
        else np.nan
    )
    total_view_capture = (
        float(predicted.sum() / total_actual_views * 100)
        if total_actual_views > 0
        else np.nan
    )
    top_decile_cutoff = float(np.quantile(actual, 0.90))
    top_decile = actual >= top_decile_cutoff
    top_decile_actual = float(actual[top_decile].sum())
    top_decile_wape = (
        float(absolute_error[top_decile].sum() / top_decile_actual * 100)
        if top_decile_actual > 0
        else np.nan
    )
    top_decile_capture = (
        float(predicted[top_decile].sum() / top_decile_actual * 100)
        if top_decile_actual > 0
        else np.nan
    )
    return {
        "rows": int(len(actual)),
        "zero_target_rows": int((~nonzero).sum()),
        "wape_pct": wape,
        "total_view_capture_pct": total_view_capture,
        "top_decile_cutoff_views": top_decile_cutoff,
        "top_decile_wape_pct": top_decile_wape,
        "top_decile_view_capture_pct": top_decile_capture,
        "median_absolute_error_views": float(np.median(absolute_error)),
        # Legacy diagnostic only. New model selection uses WAPE.
        "mape_nonzero_pct": float(percentage_error.mean() * 100)
        if nonzero.any()
        else np.nan,
        "median_ape_nonzero_pct": float(np.median(percentage_error) * 100)
        if nonzero.any()
        else np.nan,
        "smape_pct": float(symmetric_terms.mean() * 100),
        "mae_views": float(mean_absolute_error(actual, predicted)),
        "rmse_views": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "rmsle": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
        "log_mae": float(mean_absolute_error(actual_log, predicted_log)),
        "log_r2": float(r2_score(actual_log, predicted_log)),
    }


def _categories(frame: pd.DataFrame) -> pd.Series:
    return frame["category_name"].astype("string").fillna(MISSING_CATEGORY).astype(str)


class GlobalMedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training-set median log target for every row."""

    def fit(self, X: pd.DataFrame, y):
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(target):
            raise ValueError("X and y must contain the same number of rows")
        self.global_median_ = float(np.median(target))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "global_median_")
        return np.full(len(X), self.global_median_, dtype=float)


class CategoryMedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training median log target for a category."""

    def fit(self, X: pd.DataFrame, y):
        categories = _categories(X).reset_index(drop=True)
        target = pd.Series(np.asarray(y, dtype=float).reshape(-1))
        if len(categories) != len(target):
            raise ValueError("X and y must contain the same number of rows")
        self.global_median_ = float(target.median())
        self.mapping_ = (
            pd.DataFrame({"category": categories, "target": target})
            .groupby("category")["target"]
            .median()
            .astype(float)
            .to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ("global_median_", "mapping_"))
        return (
            _categories(X)
            .map(self.mapping_)
            .fillna(self.global_median_)
            .to_numpy(dtype=float)
        )


class CategoryTierMedianBaseline(BaseEstimator, RegressorMixin):
    """Category median baseline refined by a training-fitted channel-size tier.

    The exported training table does not contain the ingest-time size tier. This
    implementation reconstructs it without leakage from category-specific
    subscriber terciles fitted on unique training channels only.
    """

    TIER_NAMES = ("small", "mid", "mega")

    def fit(self, X: pd.DataFrame, y, *, channel_ids):
        categories = _categories(X).reset_index(drop=True)
        subscribers = pd.to_numeric(
            X["ch_subs_at_publish"], errors="coerce"
        ).reset_index(drop=True)
        channels = pd.Series(channel_ids).astype("string").reset_index(drop=True)
        target = pd.Series(np.asarray(y, dtype=float).reshape(-1))
        if not (len(categories) == len(subscribers) == len(channels) == len(target)):
            raise ValueError("X, y, and channel_ids must contain the same number of rows")

        channel_table = pd.DataFrame(
            {
                "category": categories,
                "channel_id": channels,
                "subscribers": subscribers,
            }
        ).groupby(["category", "channel_id"], as_index=False)["subscribers"].median()

        self.tier_boundaries_ = {}
        for category, group in channel_table.groupby("category"):
            valid = group["subscribers"].dropna()
            if valid.empty:
                self.tier_boundaries_[str(category)] = (np.nan, np.nan)
            else:
                lower, upper = valid.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
                self.tier_boundaries_[str(category)] = (float(lower), float(upper))

        tiers = self._assign_tiers(categories, subscribers)
        training = pd.DataFrame(
            {"category": categories, "tier": tiers, "target": target}
        )
        self.global_median_ = float(target.median())
        self.category_mapping_ = training.groupby("category")["target"].median().astype(float).to_dict()
        self.category_tier_mapping_ = (
            training.groupby(["category", "tier"])["target"]
            .median()
            .astype(float)
            .to_dict()
        )
        return self

    def _assign_tiers(
        self, categories: pd.Series, subscribers: pd.Series
    ) -> pd.Series:
        tiers = []
        for category, subscriber_count in zip(categories, subscribers):
            lower, upper = self.tier_boundaries_.get(str(category), (np.nan, np.nan))
            if pd.isna(subscriber_count) or np.isnan(lower) or np.isnan(upper):
                tiers.append("mid")
            elif subscriber_count <= lower:
                tiers.append("small")
            elif subscriber_count <= upper:
                tiers.append("mid")
            else:
                tiers.append("mega")
        return pd.Series(tiers, index=categories.index, dtype="string")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            (
                "tier_boundaries_",
                "global_median_",
                "category_mapping_",
                "category_tier_mapping_",
            ),
        )
        categories = _categories(X)
        subscribers = pd.to_numeric(X["ch_subs_at_publish"], errors="coerce")
        tiers = self._assign_tiers(categories, subscribers)
        predictions = []
        for category, tier in zip(categories, tiers):
            predictions.append(
                self.category_tier_mapping_.get(
                    (str(category), str(tier)),
                    self.category_mapping_.get(str(category), self.global_median_),
                )
            )
        return np.asarray(predictions, dtype=float)


def build_xgb_regressor(**overrides) -> XGBRegressor:
    parameters = dict(MVP_XGB_PARAMS)
    parameters.update(overrides)
    return XGBRegressor(**parameters)


def build_lgbm_regressor(**overrides) -> LGBMRegressor:
    """Build the reproducible LightGBM regressor used by checkpoint 11."""
    parameters = dict(MVP_LIGHTGBM_PARAMS)
    parameters.update(overrides)
    return LGBMRegressor(**parameters)


@dataclass
class HorizonModelBundle:
    """Serializable unit used by a future prediction API for one horizon."""

    horizon_days: int
    preprocessor: HorizonPreprocessor
    regressor: XGBRegressor
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def predict_log_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_features)
        return np.asarray(self.regressor.predict(transformed), dtype=float)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        return views_from_log_predictions(self.predict_log_views(raw_features))

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.get_feature_names_out())


@dataclass
class ScaleAwareHorizonModelBundle:
    """Serializable horizon model supporting log-scale or raw-view training."""

    horizon_days: int
    preprocessor: Any
    regressor: Any
    prediction_scale: str
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_features)
        prediction = np.asarray(
            self.regressor.predict(transformed), dtype=float
        )
        if self.prediction_scale == "log1p":
            return views_from_log_predictions(prediction)
        if self.prediction_scale == "views":
            return np.maximum(0.0, prediction)
        raise ValueError(
            f"Unknown prediction scale: {self.prediction_scale}"
        )

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.get_feature_names_out())


def channel_baseline_views(
    raw_features: pd.DataFrame,
    *,
    horizon_days: int,
    strategy: str = "d7_median",
    minimum_history_count: int = 5,
    fallback_views: float = 1_000.0,
    minimum_baseline_views: float = 100.0,
) -> np.ndarray:
    """Build a leakage-safe channel reference available before publication."""

    if horizon_days not in {7, 14, 21, 30}:
        raise ValueError("horizon_days must be one of 7, 14, 21, or 30")
    if strategy not in {"d7_median", "interpolated_d7_d30", "channel_average"}:
        raise ValueError(f"Unknown channel baseline strategy: {strategy}")
    if minimum_history_count < 1:
        raise ValueError("minimum_history_count must be positive")
    if fallback_views < 0 or minimum_baseline_views < 0:
        raise ValueError("Channel baseline fallbacks cannot be negative")

    def numeric(column: str) -> pd.Series:
        source = (
            raw_features[column]
            if column in raw_features
            else pd.Series(np.nan, index=raw_features.index)
        )
        return pd.to_numeric(source, errors="coerce")

    channel_average = numeric("ch_avg_views_per_video_at_publish")
    channel_average = channel_average.where(channel_average >= 0)
    d7_count = numeric("prior_d7_view_count")
    d7_median = numeric("prior_d7_median_views")
    valid_d7 = (d7_count >= minimum_history_count) & (d7_median >= 0)
    stable_d7 = d7_median.where(valid_d7, channel_average)
    stable_d7 = stable_d7.fillna(float(fallback_views)).clip(
        lower=float(minimum_baseline_views)
    )

    if strategy == "d7_median":
        return stable_d7.to_numpy(dtype=float)
    if strategy == "channel_average":
        baseline = channel_average.fillna(stable_d7).clip(
            lower=float(minimum_baseline_views)
        )
        return baseline.to_numpy(dtype=float)

    d30_count = numeric("prior_d30_view_count")
    d30_median = numeric("prior_d30_median_views")
    valid_d30 = (d30_count >= minimum_history_count) & (d30_median >= 0)
    stable_d30 = d30_median.where(valid_d30, stable_d7).fillna(stable_d7)
    d7_values = stable_d7.to_numpy(dtype=float)
    d30_values = np.maximum(d7_values, stable_d30.to_numpy(dtype=float))
    position = (float(horizon_days) - 7.0) / (30.0 - 7.0)
    baseline_log = (
        (1.0 - position) * np.log1p(d7_values)
        + position * np.log1p(d30_values)
    )
    return np.maximum(float(minimum_baseline_views), np.expm1(baseline_log))


@dataclass
class RelativePerformanceHorizonModelBundle:
    """Predict log performance relative to a pre-publication channel baseline."""

    horizon_days: int
    preprocessor: Any
    regressor: Any
    baseline_strategy: str = "d7_median"
    minimum_history_count: int = 5
    fallback_views: float = 1_000.0
    minimum_baseline_views: float = 100.0
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def predict_relative_log_performance(
        self, raw_features: pd.DataFrame
    ) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_features)
        return np.asarray(self.regressor.predict(transformed), dtype=float)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        baseline = channel_baseline_views(
            raw_features,
            horizon_days=self.horizon_days,
            strategy=self.baseline_strategy,
            minimum_history_count=self.minimum_history_count,
            fallback_views=self.fallback_views,
            minimum_baseline_views=self.minimum_baseline_views,
        )
        predicted_log_views = np.log1p(baseline) + self.predict_relative_log_performance(
            raw_features
        )
        return views_from_log_predictions(predicted_log_views)

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.get_feature_names_out())


@dataclass
class EnsembleHorizonModelBundle:
    """Serializable weighted ensemble for one independent horizon."""

    horizon_days: int
    components: list[ScaleAwareHorizonModelBundle]
    weights: list[float]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    blend_method: str = "arithmetic"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("An ensemble requires at least one component")
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have equal length")
        if any(component.horizon_days != self.horizon_days for component in self.components):
            raise ValueError("Every component must match the ensemble horizon")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("Ensemble weights cannot be negative")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Ensemble weights must sum to 1")

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        component_predictions = [
            component.predict_views(raw_features) for component in self.components
        ]
        stacked = np.vstack(component_predictions)
        if self.blend_method == "geometric":
            combined = np.expm1(
                np.average(
                    np.log1p(stacked),
                    axis=0,
                    weights=np.asarray(self.weights, dtype=float),
                )
            )
        elif self.blend_method == "median":
            combined = np.median(stacked, axis=0)
        else:
            combined = np.average(
                stacked,
                axis=0,
                weights=np.asarray(self.weights, dtype=float),
            )
        return np.maximum(0.0, combined)

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for component in self.components:
            names.extend(component.feature_names)
        return list(dict.fromkeys(names))


@dataclass
class NonnegativeIncrementModelBundle:
    """Serializable model for nonnegative growth between two horizons."""

    from_horizon_days: int
    to_horizon_days: int
    preprocessor: Any
    regressor: Any
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.from_horizon_days >= self.to_horizon_days:
            raise ValueError("Increment horizons must be strictly increasing")

    def predict_increment_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_features)
        predicted_log_increment = np.asarray(
            self.regressor.predict(transformed), dtype=float
        )
        return views_from_log_predictions(predicted_log_increment)

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.get_feature_names_out())


@dataclass
class EnsembleIncrementModelBundle:
    """Blend independently fitted nonnegative increment models."""

    from_horizon_days: int
    to_horizon_days: int
    components: list[NonnegativeIncrementModelBundle]
    weights: list[float]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    blend_method: str = "arithmetic"

    def __post_init__(self) -> None:
        if self.from_horizon_days >= self.to_horizon_days:
            raise ValueError("Increment horizons must be strictly increasing")
        if not self.components:
            raise ValueError("An increment ensemble requires at least one component")
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have equal length")
        expected = (self.from_horizon_days, self.to_horizon_days)
        if any(
            (component.from_horizon_days, component.to_horizon_days) != expected
            for component in self.components
        ):
            raise ValueError("Every component must match the ensemble transition")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("Ensemble weights cannot be negative")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Ensemble weights must sum to 1")
        if self.blend_method not in {"arithmetic", "geometric", "median"}:
            raise ValueError("Unknown ensemble blend method")
        if self.blend_method not in {"arithmetic", "geometric", "median"}:
            raise ValueError("Unknown ensemble blend method")

    def predict_increment_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        component_predictions = [
            component.predict_increment_views(raw_features)
            for component in self.components
        ]
        stacked = np.vstack(component_predictions)
        if self.blend_method == "geometric":
            combined = np.expm1(
                np.average(
                    np.log1p(stacked),
                    axis=0,
                    weights=np.asarray(self.weights, dtype=float),
                )
            )
        elif self.blend_method == "median":
            combined = np.median(stacked, axis=0)
        else:
            combined = np.average(
                stacked,
                axis=0,
                weights=np.asarray(self.weights, dtype=float),
            )
        return np.maximum(0.0, combined)

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for component in self.components:
            names.extend(component.feature_names)
        return list(dict.fromkeys(names))


@dataclass
class MonotonicTrajectoryModelBundle:
    """Compose a day-7 base with positive increments into one trajectory.

    The representation makes decreasing cumulative-view predictions
    impossible by construction rather than repairing them after inference.
    """

    base_model: Any
    increment_models: list[NonnegativeIncrementModelBundle]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    minimum_increment_views: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_increment_views < 0:
            raise ValueError("minimum_increment_views cannot be negative")
        base_horizon = getattr(self.base_model, "horizon_days", None)
        if base_horizon != 7:
            raise ValueError("The trajectory base model must predict day 7")
        expected_transitions = ((7, 14), (14, 21), (21, 30))
        actual_transitions = tuple(
            (model.from_horizon_days, model.to_horizon_days)
            for model in self.increment_models
        )
        if actual_transitions != expected_transitions:
            raise ValueError(
                "Increment models must form the chain 7->14->21->30"
            )

    @property
    def horizons(self) -> tuple[int, int, int, int]:
        return (7, 14, 21, 30)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        current = np.maximum(
            0.0,
            np.asarray(self.base_model.predict_views(raw_features), dtype=float),
        )
        predictions = [current]
        for increment_model in self.increment_models:
            increment = increment_model.predict_increment_views(raw_features)
            minimum_increment = float(
                getattr(self, "minimum_increment_views", 0.0)
            )
            current = current + np.maximum(minimum_increment, increment)
            predictions.append(current)
        return np.column_stack(predictions)

    def predict_frame(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        values = self.predict_views(raw_features)
        return pd.DataFrame(
            values,
            index=raw_features.index,
            columns=[f"day_{horizon}_views" for horizon in self.horizons],
        )


@dataclass
class ReconciledIndependentTrajectoryModelBundle:
    """Independently predict each horizon, then enforce cumulative ordering.

    Cumulative-maximum reconciliation prevents an early-horizon error from
    becoming an input to later predictions while still guaranteeing
    day_7 <= day_14 <= day_21 <= day_30.
    """

    horizon_models: list[Any]
    training_metadata: dict[str, Any] = field(default_factory=dict)
    minimum_increment_views: float = 0.0
    reconciliation: str = "forward_max"

    def __post_init__(self) -> None:
        if self.minimum_increment_views < 0:
            raise ValueError("minimum_increment_views cannot be negative")
        if self.reconciliation not in {
            "forward_max",
            "backward_min",
            "isotonic_log",
        }:
            raise ValueError("Unknown monotonic reconciliation method")
        horizons = tuple(
            getattr(model, "horizon_days", None) for model in self.horizon_models
        )
        if horizons != (7, 14, 21, 30):
            raise ValueError(
                "Independent models must be ordered day 7, 14, 21, and 30"
            )

    @property
    def horizons(self) -> tuple[int, int, int, int]:
        return (7, 14, 21, 30)

    def predict_independent_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        predictions = np.column_stack(
            [model.predict_views(raw_features) for model in self.horizon_models]
        )
        return np.maximum(0.0, np.asarray(predictions, dtype=float))

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        predictions = self.predict_independent_views(raw_features)
        method = getattr(self, "reconciliation", "forward_max")
        minimum_increment = float(self.minimum_increment_views)
        if method == "forward_max":
            result = predictions.copy()
            for position in range(1, result.shape[1]):
                result[:, position] = np.maximum(
                    result[:, position],
                    result[:, position - 1] + minimum_increment,
                )
            return result
        if minimum_increment:
            offsets = np.arange(predictions.shape[1]) * minimum_increment
            adjusted = predictions - offsets
        else:
            offsets = np.zeros(predictions.shape[1], dtype=float)
            adjusted = predictions
        if method == "backward_min":
            reconciled = np.minimum.accumulate(adjusted[:, ::-1], axis=1)[:, ::-1]
        elif method == "isotonic_log":
            logged = np.log1p(np.maximum(adjusted, 0.0))
            reconciled = np.expm1(
                np.vstack(
                    [
                        isotonic_regression(row, increasing=True)
                        for row in logged
                    ]
                )
            )
        else:
            raise ValueError("Unknown monotonic reconciliation method")
        return np.maximum(0.0, reconciled + offsets)

    def predict_frame(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        values = self.predict_views(raw_features)
        return pd.DataFrame(
            values,
            index=raw_features.index,
            columns=[f"day_{horizon}_views" for horizon in self.horizons],
        )


@dataclass
class TitleResidualCorrectionBundle:
    """Apply a regularized embedding correction to a base log-view trajectory."""

    pca: Any
    scaler: Any
    regressor: Any
    embedding_dimensions: int
    correction_strength: float
    correction_clip: float = 1.5
    minimum_increment_views: float = 1.0
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive")
        if self.correction_strength < 0:
            raise ValueError("correction_strength cannot be negative")
        if self.correction_clip <= 0:
            raise ValueError("correction_clip must be positive")
        if self.minimum_increment_views < 0:
            raise ValueError("minimum_increment_views cannot be negative")

    def predict_log_correction(self, title_embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(title_embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("title_embeddings must be a two-dimensional matrix")
        projected = np.asarray(self.pca.transform(embeddings), dtype=float)
        projected = projected[:, : self.embedding_dimensions]
        standardized = self.scaler.transform(projected)
        correction = np.asarray(self.regressor.predict(standardized), dtype=float)
        if correction.ndim == 1:
            correction = correction[:, None]
        return np.clip(correction, -self.correction_clip, self.correction_clip)

    def apply(
        self,
        base_trajectory_views: np.ndarray,
        title_embeddings: np.ndarray,
    ) -> np.ndarray:
        base = np.maximum(0.0, np.asarray(base_trajectory_views, dtype=float))
        if base.ndim != 2 or base.shape[1] != 4:
            raise ValueError("base_trajectory_views must have four horizon columns")
        correction = self.predict_log_correction(title_embeddings)
        if correction.shape != base.shape:
            raise ValueError("Title corrections must match the base trajectory shape")
        corrected = views_from_log_predictions(
            np.log1p(base) + self.correction_strength * correction
        )
        for position in range(1, corrected.shape[1]):
            corrected[:, position] = np.maximum(
                corrected[:, position],
                corrected[:, position - 1] + self.minimum_increment_views,
            )
        return corrected


def normalize_title_for_embedding(value: Any) -> str:
    """Match the deterministic title normalization used during training."""

    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TitleCorrectedTrajectoryModelBundle:
    """Encode raw titles and apply a residual correction to a base trajectory."""

    base_trajectory: Any
    correction: TitleResidualCorrectionBundle
    embedding_model_source: str
    title_column: str = "title"
    batch_size: int = 128
    device: str = "cpu"
    local_files_only: bool = True
    training_metadata: dict[str, Any] = field(default_factory=dict)
    _encoder: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def horizons(self) -> tuple[int, int, int, int]:
        return (7, 14, 21, 30)

    def _resolved_model_source(self) -> str:
        marker = "artifact://"
        if self.embedding_model_source.startswith(marker):
            relative = self.embedding_model_source[len(marker) :]
            return str(Path(__file__).resolve().parents[1] / relative)
        return self.embedding_model_source

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self._resolved_model_source(),
                device=self.device,
                local_files_only=self.local_files_only,
            )
        return self._encoder

    def predict_title_embeddings(self, raw_features: pd.DataFrame) -> np.ndarray:
        if self.title_column not in raw_features:
            raise ValueError(
                f"Raw features must include the {self.title_column!r} column"
            )
        titles = [
            normalize_title_for_embedding(value)
            for value in raw_features[self.title_column]
        ]
        embeddings = self._load_encoder().encode(
            titles,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def predict_views_from_embeddings(
        self,
        raw_features: pd.DataFrame,
        title_embeddings: np.ndarray,
    ) -> np.ndarray:
        base = self.base_trajectory.predict_views(raw_features)
        return self.correction.apply(base, title_embeddings)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        embeddings = self.predict_title_embeddings(raw_features)
        return self.predict_views_from_embeddings(raw_features, embeddings)

    def predict_frame(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        values = self.predict_views(raw_features)
        return pd.DataFrame(
            values,
            index=raw_features.index,
            columns=[f"day_{horizon}_views" for horizon in self.horizons],
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state


def breakout_threshold_views(
    raw_features: pd.DataFrame,
    *,
    minimum_views: float = 10_000.0,
    baseline_multiplier: float = 5.0,
    minimum_history_count: int = 5,
) -> np.ndarray:
    """Return a channel-relative, pre-publication breakout threshold."""

    def numeric(column: str) -> pd.Series:
        source = (
            raw_features[column]
            if column in raw_features
            else pd.Series(np.nan, index=raw_features.index)
        )
        return pd.to_numeric(source, errors="coerce")

    history_count = numeric("prior_d7_view_count")
    history_median = numeric("prior_d7_median_views")
    channel_average = numeric("ch_avg_views_per_video_at_publish")
    baseline = history_median.where(
        history_count >= minimum_history_count,
        channel_average,
    ).fillna(1_000.0).clip(lower=100.0)
    return np.maximum(
        float(minimum_views),
        float(baseline_multiplier) * baseline.to_numpy(dtype=float),
    )


@dataclass
class CalibratedBreakoutClassifierBundle:
    """Predict a calibrated probability of a channel-relative breakout."""

    preprocessor: Any
    classifier: Any
    calibrator: Any | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def predict_probability(self, raw_features: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_features)
        probability = np.asarray(
            self.classifier.predict_proba(transformed)[:, 1], dtype=float
        )
        if self.calibrator is not None:
            clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
            log_odds = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            probability = np.asarray(
                self.calibrator.predict_proba(log_odds)[:, 1], dtype=float
            )
        return np.clip(probability, 0.0, 1.0)


@dataclass
class ProbabilityCalibratorBundle:
    """Apply a fitted calibration estimator to raw binary probabilities."""

    method: str
    estimator: Any | None = None

    def predict_probability(self, raw_probability: np.ndarray) -> np.ndarray:
        probability = np.clip(
            np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6
        )
        if self.method == "identity":
            calibrated = probability
        elif self.method == "platt":
            log_odds = np.log(probability / (1.0 - probability)).reshape(-1, 1)
            calibrated = self.estimator.predict_proba(log_odds)[:, 1]
        elif self.method == "beta":
            features = np.column_stack(
                [np.log(probability), np.log1p(-probability)]
            )
            calibrated = self.estimator.predict_proba(features)[:, 1]
        elif self.method == "isotonic":
            calibrated = self.estimator.predict(probability)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


@dataclass
class EnsembleBreakoutClassifierBundle:
    """Blend raw breakout classifiers, then calibrate the combined probability."""

    components: list[CalibratedBreakoutClassifierBundle]
    weights: list[float]
    blend_method: str = "arithmetic"
    calibrator: ProbabilityCalibratorBundle | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("A classifier ensemble requires at least one component")
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have equal length")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("Ensemble weights cannot be negative")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Ensemble weights must sum to 1")
        if self.blend_method not in {"arithmetic", "logit"}:
            raise ValueError("Unknown classifier blend method")

    def predict_probability(self, raw_features: pd.DataFrame) -> np.ndarray:
        probabilities = np.column_stack(
            [component.predict_probability(raw_features) for component in self.components]
        )
        weights = np.asarray(self.weights, dtype=float)
        if self.blend_method == "logit":
            clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            raw = 1.0 / (1.0 + np.exp(-(logits @ weights)))
        else:
            raw = probabilities @ weights
        if self.calibrator is not None:
            raw = self.calibrator.predict_probability(raw)
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


@dataclass
class ViralScenarioTrajectoryModelBundle:
    """Expose a primary normal forecast plus a conditional viral scenario."""

    normal_trajectory: Any
    breakout_classifier: Any
    viral_trajectory: Any
    minimum_viral_uplift_views: float = 1.0
    breakout_minimum_views: float = 10_000.0
    breakout_baseline_multiplier: float = 5.0
    breakout_minimum_history_count: int = 5
    training_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizons(self) -> tuple[int, int, int, int]:
        return (7, 14, 21, 30)

    def predict_views(self, raw_features: pd.DataFrame) -> np.ndarray:
        """Return the normal trajectory used as the primary point forecast."""
        return np.asarray(
            self.normal_trajectory.predict_views(raw_features), dtype=float
        )

    def predict_breakout_probability(
        self, raw_features: pd.DataFrame
    ) -> np.ndarray:
        return self.breakout_classifier.predict_probability(raw_features)

    def predict_viral_upside_views(
        self, raw_features: pd.DataFrame
    ) -> np.ndarray:
        normal = self.predict_views(raw_features)
        return self._viral_upside_above_normal(raw_features, normal)

    def _viral_upside_above_normal(
        self,
        raw_features: pd.DataFrame,
        normal: np.ndarray,
    ) -> np.ndarray:
        viral = np.asarray(
            self.viral_trajectory.predict_views(raw_features), dtype=float
        ).copy()
        threshold = breakout_threshold_views(
            raw_features,
            minimum_views=self.breakout_minimum_views,
            baseline_multiplier=self.breakout_baseline_multiplier,
            minimum_history_count=self.breakout_minimum_history_count,
        )
        uplift = float(self.minimum_viral_uplift_views)
        viral[:, 0] = np.maximum.reduce(
            [viral[:, 0], threshold, normal[:, 0] + uplift]
        )
        for position in range(1, len(self.horizons)):
            viral[:, position] = np.maximum.reduce(
                [
                    viral[:, position],
                    viral[:, position - 1] + uplift,
                    normal[:, position] + uplift,
                ]
            )
        return viral

    def predict_scenario_frame(
        self, raw_features: pd.DataFrame
    ) -> pd.DataFrame:
        normal = self.predict_views(raw_features)
        viral = self._viral_upside_above_normal(raw_features, normal)
        probability = self.predict_breakout_probability(raw_features)
        values: dict[str, np.ndarray] = {}
        for position, horizon in enumerate(self.horizons):
            values[f"normal_day_{horizon}_views"] = normal[:, position]
        values["breakout_probability"] = probability
        for position, horizon in enumerate(self.horizons):
            values[f"viral_upside_day_{horizon}_views"] = viral[:, position]
        return pd.DataFrame(values, index=raw_features.index)
