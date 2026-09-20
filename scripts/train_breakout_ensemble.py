"""Benchmark, calibrate, and optionally promote a breakout classifier ensemble.

All candidate and blend selection is based on channel-grouped development OOF
predictions. An outer fold estimates selection plus calibration without letting
the held fold influence its blend or calibration method. The existing reserved
channel set is used only as a final release gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_checkpoint5_models import CATEGORY_SMOOTHING  # noqa: E402
from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    load_all_horizons,
    load_transition,
    sha256_file,
)
from scripts.train_viral_scenario import breakout_labels  # noqa: E402
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    CalibratedBreakoutClassifierBundle,
    EnsembleBreakoutClassifierBundle,
    ProbabilityCalibratorBundle,
    ViralScenarioTrajectoryModelBundle,
)


FEATURE_OPTIONS: dict[str, Any] = {
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
CALIBRATION_METHODS = ("identity", "platt", "beta", "isotonic")


@dataclass(frozen=True)
class ClassifierCandidate:
    name: str
    algorithm: str
    balanced: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


CANDIDATES = (
    ClassifierCandidate("xgb_current", "xgboost"),
    ClassifierCandidate(
        "xgb_shallow",
        "xgboost",
        overrides={"max_depth": 3, "min_child_weight": 20, "reg_lambda": 3.0},
    ),
    ClassifierCandidate("xgb_balanced", "xgboost", balanced=True),
    ClassifierCandidate("lgb_standard", "lightgbm"),
    ClassifierCandidate("lgb_balanced", "lightgbm", balanced=True),
)


@dataclass(frozen=True)
class ProbabilityBlendRecipe:
    name: str
    weights: tuple[float, ...]
    blend_method: str

    @property
    def component_count(self) -> int:
        return sum(weight > 0 for weight in self.weights)


def probability_blend_recipes(
    candidates: tuple[ClassifierCandidate, ...] = CANDIDATES,
) -> list[ProbabilityBlendRecipe]:
    count = len(candidates)
    recipes: list[ProbabilityBlendRecipe] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()

    def append(weights: list[float], method: str = "arithmetic") -> None:
        key = tuple(round(value, 10) for value in weights)
        identity = (method, key)
        if identity in seen:
            return
        seen.add(identity)
        members = [
            f"{candidate.name}:{weight:.2f}"
            for candidate, weight in zip(candidates, key)
            if weight > 0
        ]
        recipes.append(
            ProbabilityBlendRecipe(
                f"{method}[{'+'.join(members)}]", key, method
            )
        )

    for position in range(count):
        weights = [0.0] * count
        weights[position] = 1.0
        append(weights)
    for left, right in combinations(range(count), 2):
        for step in range(1, 10):
            weights = [0.0] * count
            weights[left] = step / 10
            weights[right] = 1.0 - weights[left]
            append(weights)
            append(weights, "logit")
    for positions in combinations(range(count), 3):
        weights = [0.0] * count
        for position in positions:
            weights[position] = 1.0 / 3.0
        append(weights)
        append(weights, "logit")
    all_weights = [1.0 / count] * count
    append(all_weights)
    append(all_weights, "logit")
    return recipes


def classification_metrics(
    labels: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    actual = np.asarray(labels, dtype=bool)
    predicted = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return {
        "rows": len(actual),
        "breakout_rows": int(actual.sum()),
        "breakout_rate_pct": float(actual.mean() * 100),
        "roc_auc": float(roc_auc_score(actual, predicted)),
        "average_precision": float(average_precision_score(actual, predicted)),
        "brier_score": float(brier_score_loss(actual, predicted)),
        "log_loss": float(log_loss(actual, predicted)),
    }


def positive_class_weight(labels: np.ndarray) -> float:
    positives = int(np.asarray(labels, dtype=bool).sum())
    negatives = len(labels) - positives
    return float(negatives / max(positives, 1))


def build_classifier(
    candidate: ClassifierCandidate,
    *,
    n_estimators: int,
    n_jobs: int,
    class_weight: float,
    early_stopping: bool,
):
    weight = class_weight if candidate.balanced else 1.0
    if candidate.algorithm == "xgboost":
        parameters = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
            "tree_method": "hist",
            "early_stopping_rounds": 30 if early_stopping else None,
            "random_state": 42,
            "n_jobs": n_jobs,
            "scale_pos_weight": weight,
        }
        parameters.update(candidate.overrides)
        return XGBClassifier(**parameters)
    if candidate.algorithm == "lightgbm":
        parameters = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbosity": -1,
            "scale_pos_weight": weight,
        }
        parameters.update(candidate.overrides)
        return LGBMClassifier(**parameters)
    raise ValueError(f"Unknown classifier algorithm: {candidate.algorithm}")


def fit_classifier_fold(
    candidate: ClassifierCandidate,
    *,
    transformed_training: pd.DataFrame,
    transformed_validation: pd.DataFrame,
    labels_training: np.ndarray,
    labels_validation: np.ndarray,
    max_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, int]:
    classifier = build_classifier(
        candidate,
        n_estimators=max_estimators,
        n_jobs=n_jobs,
        class_weight=positive_class_weight(labels_training),
        early_stopping=True,
    )
    if candidate.algorithm == "xgboost":
        classifier.fit(
            transformed_training,
            labels_training,
            eval_set=[(transformed_validation, labels_validation)],
            verbose=False,
        )
        best_iteration = int(classifier.best_iteration) + 1
    else:
        classifier.fit(
            transformed_training,
            labels_training,
            eval_X=transformed_validation,
            eval_y=labels_validation,
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(30, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(classifier.best_iteration_)
    probability = classifier.predict_proba(transformed_validation)[:, 1]
    return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0), best_iteration


def generate_candidate_oof(
    *,
    X: pd.DataFrame,
    labels: np.ndarray,
    channels: pd.Series,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    predictions = {
        candidate.name: np.full(len(X), np.nan, dtype=float)
        for candidate in CANDIDATES
    }
    fold_ids = np.zeros(len(X), dtype=int)
    rows: list[dict[str, Any]] = []
    for fold, (training, validation) in enumerate(
        GroupKFold(n_splits=folds).split(X, groups=channels), start=1
    ):
        fold_ids[validation] = fold
        preprocessor = HorizonDatasetPreprocessor(
            category_smoothing=CATEGORY_SMOOTHING,
            **FEATURE_OPTIONS,
        )
        transformed_training = preprocessor.fit_transform(
            X.iloc[training], labels[training].astype(float)
        )
        transformed_validation = preprocessor.transform(X.iloc[validation])
        details = []
        for candidate in CANDIDATES:
            probability, best_iteration = fit_classifier_fold(
                candidate,
                transformed_training=transformed_training,
                transformed_validation=transformed_validation,
                labels_training=labels[training],
                labels_validation=labels[validation],
                max_estimators=max_estimators,
                n_jobs=n_jobs,
            )
            predictions[candidate.name][validation] = probability
            rows.append(
                {
                    "fold": fold,
                    "candidate": candidate.name,
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    "features": transformed_training.shape[1],
                    "best_iteration": best_iteration,
                    **classification_metrics(labels[validation], probability),
                }
            )
            details.append(f"{candidate.name}={best_iteration}")
        print(f"Classifier fold {fold}: " + ", ".join(details), flush=True)
    if (fold_ids == 0).any() or not all(
        np.isfinite(probability).all() for probability in predictions.values()
    ):
        raise AssertionError("Incomplete classifier OOF predictions")
    return predictions, fold_ids, pd.DataFrame(rows)


def recipe_prediction_matrix(
    candidate_probabilities: np.ndarray,
    recipes: list[ProbabilityBlendRecipe],
) -> np.ndarray:
    probabilities = np.clip(
        np.asarray(candidate_probabilities, dtype=float), 1e-6, 1 - 1e-6
    )
    logits = np.log(probabilities / (1.0 - probabilities))
    combined = []
    for recipe in recipes:
        weights = np.asarray(recipe.weights, dtype=float)
        if recipe.blend_method == "logit":
            prediction = 1.0 / (1.0 + np.exp(-(logits @ weights)))
        else:
            prediction = probabilities @ weights
        combined.append(prediction)
    return np.column_stack(combined)


def select_recipe_index(
    labels: np.ndarray,
    recipe_probabilities: np.ndarray,
    recipes: list[ProbabilityBlendRecipe],
) -> int:
    actual = np.asarray(labels, dtype=float)[:, None]
    probability = np.clip(recipe_probabilities, 1e-9, 1 - 1e-9)
    losses = -np.mean(
        actual * np.log(probability)
        + (1.0 - actual) * np.log1p(-probability),
        axis=0,
    )
    return min(
        range(len(recipes)),
        key=lambda index: (
            float(losses[index]),
            recipes[index].component_count,
            recipes[index].name,
        ),
    )


def fit_calibrator(
    method: str,
    raw_probability: np.ndarray,
    labels: np.ndarray,
) -> ProbabilityCalibratorBundle:
    probability = np.clip(
        np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6
    )
    actual = np.asarray(labels, dtype=bool)
    if method == "identity":
        return ProbabilityCalibratorBundle(method="identity")
    if method == "platt":
        features = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    elif method == "beta":
        features = np.column_stack(
            [np.log(probability), np.log1p(-probability)]
        )
        estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip")
        estimator.fit(probability, actual.astype(float))
        return ProbabilityCalibratorBundle(method="isotonic", estimator=estimator)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    estimator.fit(features, actual)
    return ProbabilityCalibratorBundle(method=method, estimator=estimator)


def cross_calibrated_probability(
    *,
    method: str,
    raw_probability: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    if method == "identity":
        return np.asarray(raw_probability[positions], dtype=float)
    output = np.full(len(positions), np.nan, dtype=float)
    position_folds = fold_ids[positions]
    for fold in np.unique(position_folds):
        training = positions[position_folds != fold]
        validation_local = np.flatnonzero(position_folds == fold)
        validation = positions[validation_local]
        calibrator = fit_calibrator(
            method, raw_probability[training], labels[training]
        )
        output[validation_local] = calibrator.predict_probability(
            raw_probability[validation]
        )
    if not np.isfinite(output).all():
        raise AssertionError("Cross-calibration produced missing probabilities")
    return output


def select_calibration_method(
    *,
    raw_probability: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    positions: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    rows = []
    for method in CALIBRATION_METHODS:
        probability = cross_calibrated_probability(
            method=method,
            raw_probability=raw_probability,
            labels=labels,
            fold_ids=fold_ids,
            positions=positions,
        )
        rows.append(
            {
                "calibration_method": method,
                **classification_metrics(labels[positions], probability),
            }
        )
    metrics = pd.DataFrame(rows)
    selected = metrics.sort_values(
        ["log_loss", "brier_score", "calibration_method"]
    ).iloc[0]
    return str(selected["calibration_method"]), metrics


def fit_outer_candidate_probabilities(
    *,
    training_X: pd.DataFrame,
    validation_X: pd.DataFrame,
    training_labels: np.ndarray,
    candidates: set[str],
    inner_fold_metrics: pd.DataFrame,
    max_estimators: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed_training = preprocessor.fit_transform(
        training_X, training_labels.astype(float)
    )
    transformed_validation = preprocessor.transform(validation_X)
    probabilities = {}
    for candidate in CANDIDATES:
        if candidate.name not in candidates:
            continue
        iterations = inner_fold_metrics.loc[
            inner_fold_metrics["candidate"].eq(candidate.name), "best_iteration"
        ]
        selected_iterations = max(1, int(np.median(iterations)))
        classifier = build_classifier(
            candidate,
            n_estimators=min(max_estimators, selected_iterations),
            n_jobs=n_jobs,
            class_weight=positive_class_weight(training_labels),
            early_stopping=False,
        )
        classifier.fit(transformed_training, training_labels)
        probabilities[candidate.name] = np.clip(
            np.asarray(
                classifier.predict_proba(transformed_validation)[:, 1], dtype=float
            ),
            0.0,
            1.0,
        )
    return probabilities


def blend_probability_mapping(
    probabilities: dict[str, np.ndarray], recipe: ProbabilityBlendRecipe
) -> np.ndarray:
    active = [
        (candidate.name, float(weight))
        for candidate, weight in zip(CANDIDATES, recipe.weights)
        if weight > 0
    ]
    matrix = np.column_stack([probabilities[name] for name, _ in active])
    weights = np.asarray([weight for _, weight in active], dtype=float)
    clipped = np.clip(matrix, 1e-6, 1.0 - 1e-6)
    if recipe.blend_method == "logit":
        logits = np.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + np.exp(-(logits @ weights)))
    return clipped @ weights


def nested_cross_selected_probabilities(
    *,
    X: pd.DataFrame,
    labels: np.ndarray,
    channels: pd.Series,
    recipes: list[ProbabilityBlendRecipe],
    outer_folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate selection and calibration with genuinely unseen outer channels."""

    ensemble_output = np.full(len(labels), np.nan, dtype=float)
    baseline_output = np.full(len(labels), np.nan, dtype=float)
    rows = []
    outer_splitter = GroupKFold(n_splits=outer_folds)
    for outer_fold, (training, validation) in enumerate(
        outer_splitter.split(X, groups=channels), start=1
    ):
        inner_X = X.iloc[training].reset_index(drop=True)
        inner_labels = labels[training]
        inner_channels = channels.iloc[training].reset_index(drop=True)
        inner_folds = min(outer_folds - 1, int(inner_channels.nunique()))
        if inner_folds < 2:
            raise ValueError("Nested classifier selection requires two channel folds")
        print(
            f"Outer fold {outer_fold}: generating {inner_folds}-fold inner OOF",
            flush=True,
        )
        inner_oof, inner_fold_ids, inner_metrics = generate_candidate_oof(
            X=inner_X,
            labels=inner_labels,
            channels=inner_channels,
            folds=inner_folds,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        inner_matrix = np.column_stack(
            [inner_oof[candidate.name] for candidate in CANDIDATES]
        )
        inner_recipe_probabilities = recipe_prediction_matrix(inner_matrix, recipes)
        recipe_index = select_recipe_index(
            inner_labels, inner_recipe_probabilities, recipes
        )
        recipe = recipes[recipe_index]
        raw_inner = inner_recipe_probabilities[:, recipe_index]
        calibration_method, _ = select_calibration_method(
            raw_probability=raw_inner,
            labels=inner_labels,
            fold_ids=inner_fold_ids,
            positions=np.arange(len(inner_labels)),
        )
        ensemble_calibrator = fit_calibrator(
            calibration_method, raw_inner, inner_labels
        )
        baseline_calibrator = fit_calibrator(
            "platt", inner_oof["xgb_current"], inner_labels
        )

        active_candidates = {
            candidate.name
            for candidate, weight in zip(CANDIDATES, recipe.weights)
            if weight > 0
        }
        active_candidates.add("xgb_current")
        outer_probabilities = fit_outer_candidate_probabilities(
            training_X=X.iloc[training],
            validation_X=X.iloc[validation],
            training_labels=inner_labels,
            candidates=active_candidates,
            inner_fold_metrics=inner_metrics,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
        ensemble_raw = blend_probability_mapping(outer_probabilities, recipe)
        ensemble_output[validation] = ensemble_calibrator.predict_probability(
            ensemble_raw
        )
        baseline_output[validation] = baseline_calibrator.predict_probability(
            outer_probabilities["xgb_current"]
        )
        rows.append(
            {
                "held_out_fold": outer_fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "recipe": recipe.name,
                "blend_method": recipe.blend_method,
                "component_count": recipe.component_count,
                "calibration_method": calibration_method,
                **{
                    f"weight_{candidate.name}": weight
                    for candidate, weight in zip(CANDIDATES, recipe.weights)
                },
            }
        )
    if not np.isfinite(ensemble_output).all() or not np.isfinite(
        baseline_output
    ).all():
        raise AssertionError("Nested classifier predictions are incomplete")
    return ensemble_output, baseline_output, pd.DataFrame(rows)


def calibration_bins(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"actual": np.asarray(labels, dtype=bool), "probability": probability}
    )
    frame["bin"] = pd.qcut(
        frame["probability"], q=bins, duplicates="drop"
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            rows=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            actual_breakout_rate=("actual", "mean"),
        )
        .reset_index()
    )


def train_selected_classifier(
    *,
    X: pd.DataFrame,
    labels: np.ndarray,
    channels: pd.Series,
    recipe: ProbabilityBlendRecipe,
    calibration_method: str,
    raw_oof_probability: np.ndarray,
    fold_metrics: pd.DataFrame,
    max_estimators: int,
    n_jobs: int,
) -> EnsembleBreakoutClassifierBundle:
    preprocessor = HorizonDatasetPreprocessor(
        category_smoothing=CATEGORY_SMOOTHING,
        **FEATURE_OPTIONS,
    )
    transformed = preprocessor.fit_transform(X, labels.astype(float))
    components = []
    active_weights = []
    for candidate, weight in zip(CANDIDATES, recipe.weights):
        if weight <= 0:
            continue
        iterations = fold_metrics.loc[
            fold_metrics["candidate"].eq(candidate.name), "best_iteration"
        ]
        selected_iterations = max(1, int(np.median(iterations)))
        classifier = build_classifier(
            candidate,
            n_estimators=min(max_estimators, selected_iterations),
            n_jobs=n_jobs,
            class_weight=positive_class_weight(labels),
            early_stopping=False,
        )
        classifier.fit(transformed, labels)
        components.append(
            CalibratedBreakoutClassifierBundle(
                preprocessor=preprocessor,
                classifier=classifier,
                calibrator=None,
                training_metadata={
                    "candidate": candidate.name,
                    "algorithm": candidate.algorithm,
                    "balanced": candidate.balanced,
                    "selected_n_estimators": selected_iterations,
                },
            )
        )
        active_weights.append(float(weight))
    calibrator = fit_calibrator(calibration_method, raw_oof_probability, labels)
    return EnsembleBreakoutClassifierBundle(
        components=components,
        weights=active_weights,
        blend_method=recipe.blend_method,
        calibrator=calibrator,
        training_metadata={
            "recipe": recipe.name,
            "calibration_method": calibration_method,
            "training_rows": len(X),
            "training_channels": int(channels.nunique()),
        },
    )


def run_training(
    *,
    project_root: Path,
    output_dir: Path,
    artifact_version: str,
    baseline_checkpoint: Path,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    loaded = load_all_horizons(project_root)
    transitions = {pair: load_transition(loaded, *pair) for pair in TRANSITIONS}
    channel_frames = [loaded[7].assignments["channel_id"]] + [
        transitions[pair].metadata["channel_id"] for pair in TRANSITIONS
    ]
    test_channels, split_seed, _ = choose_common_test_channels(channel_frames)
    X, targets, metadata = complete_subset(loaded, HORIZONS)
    actual_matrix = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual_matrix, axis=1) >= 0).all(axis=1)
    testing = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~testing & naturally_monotone
    evaluation = testing & naturally_monotone
    development_X = X.loc[development].reset_index(drop=True)
    development_labels = breakout_labels(development_X, targets[7][development])
    development_channels = metadata.loc[
        development, "channel_id"
    ].astype(str).reset_index(drop=True)

    print("Generating classifier candidate OOF probabilities", flush=True)
    candidate_oof, fold_ids, fold_metrics = generate_candidate_oof(
        X=development_X,
        labels=development_labels,
        channels=development_channels,
        folds=folds,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
    )
    candidate_matrix = np.column_stack(
        [candidate_oof[candidate.name] for candidate in CANDIDATES]
    )
    recipes = probability_blend_recipes()
    recipe_probabilities = recipe_prediction_matrix(candidate_matrix, recipes)
    print(f"Evaluating {len(recipes)} classifier blend recipes", flush=True)
    print("Running nested channel-grouped selection estimate", flush=True)
    nested_ensemble, nested_baseline, fold_selections = (
        nested_cross_selected_probabilities(
            X=development_X,
            labels=development_labels,
            channels=development_channels,
            recipes=recipes,
            outer_folds=folds,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
    )
    all_positions = np.arange(len(development_labels))
    selected_recipe_index = select_recipe_index(
        development_labels, recipe_probabilities, recipes
    )
    selected_recipe = recipes[selected_recipe_index]
    selected_raw_oof = recipe_probabilities[:, selected_recipe_index]
    selected_calibration, calibration_comparison = select_calibration_method(
        raw_probability=selected_raw_oof,
        labels=development_labels,
        fold_ids=fold_ids,
        positions=all_positions,
    )
    selected_cross_calibrated = cross_calibrated_probability(
        method=selected_calibration,
        raw_probability=selected_raw_oof,
        labels=development_labels,
        fold_ids=fold_ids,
        positions=all_positions,
    )
    baseline_cross_calibrated = cross_calibrated_probability(
        method="platt",
        raw_probability=candidate_oof["xgb_current"],
        labels=development_labels,
        fold_ids=fold_ids,
        positions=all_positions,
    )

    development_rows = []
    for candidate in CANDIDATES:
        development_rows.append(
            {
                "method": f"raw_{candidate.name}",
                **classification_metrics(
                    development_labels, candidate_oof[candidate.name]
                ),
            }
        )
    development_rows.extend(
        [
            {
                "method": "baseline_xgb_current_nested_platt",
                **classification_metrics(
                    development_labels, nested_baseline
                ),
            },
            {
                "method": "ensemble_nested_cross_selected",
                **classification_metrics(development_labels, nested_ensemble),
            },
            {
                "method": "baseline_xgb_current_cross_platt_diagnostic",
                **classification_metrics(
                    development_labels, baseline_cross_calibrated
                ),
            },
            {
                "method": "selected_recipe_cross_calibrated_diagnostic",
                **classification_metrics(
                    development_labels, selected_cross_calibrated
                ),
            },
        ]
    )
    development_metrics = pd.DataFrame(development_rows)

    ensemble_classifier = train_selected_classifier(
        X=development_X,
        labels=development_labels,
        channels=development_channels,
        recipe=selected_recipe,
        calibration_method=selected_calibration,
        raw_oof_probability=selected_raw_oof,
        fold_metrics=fold_metrics,
        max_estimators=max_estimators,
        n_jobs=n_jobs,
    )
    classifier_path = models_dir / "breakout_classifier_ensemble.joblib"
    joblib.dump(ensemble_classifier, classifier_path)
    saved_classifier = joblib.load(classifier_path)

    baseline_manifest = json.loads(
        (baseline_checkpoint / "training_manifest.json").read_text(encoding="utf-8")
    )
    baseline_model_path = baseline_checkpoint / baseline_manifest["model_path"]
    if sha256_file(baseline_model_path) != baseline_manifest["model_sha256"]:
        raise RuntimeError("Baseline viral scenario checksum mismatch")
    baseline_scenario = joblib.load(baseline_model_path)
    test_X = X.loc[evaluation].reset_index(drop=True)
    test_labels = breakout_labels(test_X, targets[7][evaluation])
    baseline_probability = baseline_scenario.predict_breakout_probability(test_X)
    ensemble_probability = saved_classifier.predict_probability(test_X)
    test_metrics = pd.DataFrame(
        [
            {
                "method": "existing_single_xgb_platt",
                **classification_metrics(test_labels, baseline_probability),
            },
            {
                "method": "selected_classifier_ensemble",
                **classification_metrics(test_labels, ensemble_probability),
            },
        ]
    )
    baseline_test = test_metrics.iloc[0]
    ensemble_test = test_metrics.iloc[1]
    baseline_development = development_metrics.loc[
        development_metrics["method"].eq("baseline_xgb_current_nested_platt")
    ].iloc[0]
    ensemble_development = development_metrics.loc[
        development_metrics["method"].eq("ensemble_nested_cross_selected")
    ].iloc[0]
    release_gate = bool(
        ensemble_development["log_loss"] < baseline_development["log_loss"]
        and ensemble_development["brier_score"] < baseline_development["brier_score"]
        and ensemble_test["log_loss"] <= baseline_test["log_loss"]
        and ensemble_test["brier_score"] <= baseline_test["brier_score"]
        and ensemble_test["roc_auc"] >= baseline_test["roc_auc"] - 0.002
        and ensemble_test["average_precision"]
        >= baseline_test["average_precision"] - 0.002
    )

    scenario_path = None
    if release_gate:
        promoted = ViralScenarioTrajectoryModelBundle(
            normal_trajectory=baseline_scenario.normal_trajectory,
            breakout_classifier=saved_classifier,
            viral_trajectory=baseline_scenario.viral_trajectory,
            minimum_viral_uplift_views=baseline_scenario.minimum_viral_uplift_views,
            breakout_minimum_views=baseline_scenario.breakout_minimum_views,
            breakout_baseline_multiplier=baseline_scenario.breakout_baseline_multiplier,
            breakout_minimum_history_count=baseline_scenario.breakout_minimum_history_count,
            training_metadata={
                **baseline_scenario.training_metadata,
                "breakout_classifier": artifact_version,
            },
        )
        scenario_path = models_dir / "viral_scenario_trajectory.joblib"
        joblib.dump(promoted, scenario_path)

    selected_frame = pd.DataFrame(
        [
            {
                "recipe": selected_recipe.name,
                "blend_method": selected_recipe.blend_method,
                "component_count": selected_recipe.component_count,
                "calibration_method": selected_calibration,
                **{
                    f"weight_{candidate.name}": weight
                    for candidate, weight in zip(CANDIDATES, selected_recipe.weights)
                },
            }
        ]
    )
    selected_frame.to_csv(output_dir / "selected_classifier_recipe.csv", index=False)
    fold_metrics.to_csv(output_dir / "candidate_fold_metrics.csv", index=False)
    fold_selections.to_csv(output_dir / "cross_selected_fold_recipes.csv", index=False)
    pd.DataFrame(
        {
            "fold": fold_ids,
            "actual_breakout": development_labels,
            **candidate_oof,
        }
    ).to_csv(output_dir / "candidate_oof_probabilities.csv", index=False)
    development_metrics.to_csv(output_dir / "development_metrics.csv", index=False)
    calibration_comparison.to_csv(
        output_dir / "development_calibration_comparison.csv", index=False
    )
    test_metrics.to_csv(output_dir / "reserved_test_metrics.csv", index=False)
    calibration_bins(test_labels, baseline_probability).assign(
        method="existing_single_xgb_platt"
    ).to_csv(output_dir / "baseline_test_calibration.csv", index=False)
    calibration_bins(test_labels, ensemble_probability).assign(
        method="selected_classifier_ensemble"
    ).to_csv(output_dir / "ensemble_test_calibration.csv", index=False)
    pd.DataFrame(
        {
            "video_id": metadata.loc[evaluation, "video_id"].reset_index(drop=True),
            "actual_breakout": test_labels,
            "baseline_probability": baseline_probability,
            "ensemble_probability": ensemble_probability,
        }
    ).head(100).to_csv(output_dir / "sample_test_probabilities.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "test": "development and reserved channels disjoint",
                "status": "PASS"
                if set(development_channels).isdisjoint(test_channels)
                else "FAIL",
            },
            {
                "test": "ensemble probabilities finite and bounded",
                "status": "PASS"
                if np.isfinite(ensemble_probability).all()
                and ((ensemble_probability >= 0) & (ensemble_probability <= 1)).all()
                else "FAIL",
            },
            {"test": "saved classifier reload", "status": "PASS"},
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    if validation["status"].ne("PASS").any():
        raise AssertionError("Classifier ensemble validation failed")

    manifest = {
        "artifact_version": artifact_version,
        "status": (
            "classifier_ensemble_promoted"
            if release_gate
            else "classifier_ensemble_benchmarked_not_promoted"
        ),
        "selection_data": (
            "nested channel-grouped development OOF only; reserved channels used "
            "once for release"
        ),
        "selection_metric": "cross-calibrated log loss with Brier tie-break",
        "candidate_definitions": [asdict(candidate) for candidate in CANDIDATES],
        "blend_recipe_count": len(recipes),
        "calibration_methods": list(CALIBRATION_METHODS),
        "selected_recipe": selected_frame.iloc[0].to_dict(),
        "classifier_path": classifier_path.relative_to(output_dir).as_posix(),
        "classifier_sha256": sha256_file(classifier_path),
        "model_path": (
            scenario_path.relative_to(output_dir).as_posix()
            if scenario_path is not None
            else None
        ),
        "model_sha256": (
            sha256_file(scenario_path) if scenario_path is not None else None
        ),
        "scenario_model_path": (
            scenario_path.relative_to(output_dir).as_posix()
            if scenario_path is not None
            else None
        ),
        "scenario_model_sha256": (
            sha256_file(scenario_path) if scenario_path is not None else None
        ),
        "development_rows": len(development_X),
        "development_breakout_rows": int(development_labels.sum()),
        "reserved_test_rows": int(evaluation.sum()),
        "reserved_breakout_rows": int(test_labels.sum()),
        "split_seed": split_seed,
        "horizons": list(HORIZONS),
        "output_contract": baseline_manifest["output_contract"],
        "breakout_definition": baseline_manifest["breakout_definition"],
        "classifier": ensemble_classifier.training_metadata,
        "classifier_reserved_test": {
            key: value
            for key, value in ensemble_test.to_dict().items()
            if key != "method"
        },
        "normal_trajectory_source": baseline_manifest[
            "normal_trajectory_source"
        ],
        "viral_training_rows": baseline_manifest["viral_training_rows"],
        "viral_training_channels": baseline_manifest["viral_training_channels"],
        "common_split": baseline_manifest["common_split"],
        "feature_contract": ensemble_classifier.components[
            0
        ].preprocessor.feature_contract(),
        "source_training_table_sha256": baseline_manifest[
            "source_training_table_sha256"
        ],
        "limitations": baseline_manifest["limitations"],
        "comparison": {
            "development_baseline": baseline_development.to_dict(),
            "development_ensemble": ensemble_development.to_dict(),
            "reserved_baseline": baseline_test.to_dict(),
            "reserved_ensemble": ensemble_test.to_dict(),
            "release_gate_passed": release_gate,
        },
        "release_gate": (
            "development and reserved log loss plus Brier must not regress; "
            "reserved AUC and AP tolerance is 0.002"
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSelected classifier recipe", flush=True)
    print(selected_frame.to_string(index=False), flush=True)
    print("\nDevelopment metrics", flush=True)
    print(development_metrics.to_string(index=False), flush=True)
    print("\nReserved test metrics", flush=True)
    print(test_metrics.to_string(index=False), flush=True)
    print(f"\nRelease gate: {'PASS' if release_gate else 'FAIL'}", flush=True)
    return {
        "manifest": manifest,
        "development_metrics": development_metrics,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint22_breakout_ensemble",
    )
    parser.add_argument(
        "--artifact-version", default="checkpoint22_breakout_ensemble"
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoint19_viral_scenario_20260915",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        baseline_checkpoint=args.baseline_checkpoint,
        folds=args.folds,
        max_estimators=args.max_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
