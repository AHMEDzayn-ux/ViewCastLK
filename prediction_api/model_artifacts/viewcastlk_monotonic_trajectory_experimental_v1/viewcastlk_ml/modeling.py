"""Model, baseline, evaluation, and artifact components for ViewCastLK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
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


@dataclass
class EnsembleHorizonModelBundle:
    """Serializable weighted ensemble for one independent horizon."""

    horizon_days: int
    components: list[ScaleAwareHorizonModelBundle]
    weights: list[float]
    training_metadata: dict[str, Any] = field(default_factory=dict)

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
        return np.maximum(
            0.0,
            np.average(
                np.vstack(component_predictions),
                axis=0,
                weights=np.asarray(self.weights, dtype=float),
            ),
        )

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
class MonotonicTrajectoryModelBundle:
    """Compose a day-7 base with positive increments into one trajectory.

    The representation makes decreasing cumulative-view predictions
    impossible by construction rather than repairing them after inference.
    """

    base_model: Any
    increment_models: list[NonnegativeIncrementModelBundle]
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            current = current + np.maximum(0.0, increment)
            predictions.append(current)
        return np.column_stack(predictions)

    def predict_frame(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        values = self.predict_views(raw_features)
        return pd.DataFrame(
            values,
            index=raw_features.index,
            columns=[f"day_{horizon}_views" for horizon in self.horizons],
        )
