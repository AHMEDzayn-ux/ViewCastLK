"""Select and train a strictly increasing multi-algorithm trajectory ensemble.

Candidate models and blend weights are selected only from channel-grouped
development out-of-fold predictions. An outer cross-selection pass estimates
the benefit of weight selection without letting a fold choose its own weights.
The reserved channel partition is evaluated only after the final development
recipe has been fixed.
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
from sklearn.model_selection import GroupKFold


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
from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    HorizonDatasetPreprocessor,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    EnsembleHorizonModelBundle,
    EnsembleIncrementModelBundle,
    MonotonicTrajectoryModelBundle,
    NonnegativeIncrementModelBundle,
    ScaleAwareHorizonModelBundle,
    build_lgbm_regressor,
    build_xgb_regressor,
    regression_metrics,
    views_from_log_predictions,
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
MINIMUM_INCREMENT_VIEWS = 1.0


@dataclass(frozen=True)
class Candidate:
    name: str
    algorithm: str
    objective: str
    eval_metric: str
    overrides: dict[str, Any] = field(default_factory=dict)


CANDIDATES = (
    Candidate(
        name="xgb_l2_current",
        algorithm="xgboost",
        objective="reg:squarederror",
        eval_metric="rmse",
    ),
    Candidate(
        name="xgb_l1_robust",
        algorithm="xgboost",
        objective="reg:absoluteerror",
        eval_metric="mae",
        overrides={"learning_rate": 0.04},
    ),
    Candidate(
        name="xgb_l2_shallow",
        algorithm="xgboost",
        objective="reg:squarederror",
        eval_metric="rmse",
        overrides={
            "max_depth": 4,
            "min_child_weight": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    ),
    Candidate(
        name="lgb_l2",
        algorithm="lightgbm",
        objective="regression",
        eval_metric="l2",
    ),
    Candidate(
        name="lgb_l1_robust",
        algorithm="lightgbm",
        objective="regression_l1",
        eval_metric="l1",
    ),
)


@dataclass(frozen=True)
class BlendRecipe:
    name: str
    weights: tuple[float, ...]
    blend_method: str

    @property
    def component_count(self) -> int:
        return sum(weight > 0 for weight in self.weights)


def blend_recipes(candidates: tuple[Candidate, ...] = CANDIDATES) -> list[BlendRecipe]:
    """Return deterministic single, pair-grid, and uniform multi-model blends."""

    count = len(candidates)
    recipes: list[BlendRecipe] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()

    def append(weights: list[float], blend_method: str = "arithmetic") -> None:
        key = tuple(round(value, 10) for value in weights)
        identity = (blend_method, key)
        if identity in seen:
            return
        seen.add(identity)
        members = [
            f"{candidate.name}:{weight:.2f}"
            for candidate, weight in zip(candidates, key)
            if weight > 0
        ]
        recipes.append(
            BlendRecipe(f"{blend_method}[{'+'.join(members)}]", key, blend_method)
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
            append(weights, "geometric")
    for positions in combinations(range(count), 3):
        weights = [0.0] * count
        for position in positions:
            weights[position] = 1.0 / 3.0
        append(weights)
        append(weights, "geometric")
        append(weights, "median")
    all_weights = [1.0 / count] * count
    append(all_weights)
    append(all_weights, "geometric")
    append(all_weights, "median")
    return recipes


def component_targets(targets: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    values = {"day_7_base": np.asarray(targets[7], dtype=float)}
    for from_day, to_day in TRANSITIONS:
        values[f"day_{from_day}_to_{to_day}_growth"] = np.maximum(
            0.0,
            np.asarray(targets[to_day], dtype=float)
            - np.asarray(targets[from_day], dtype=float),
        )
    return values


def component_sequence() -> list[tuple[str, int]]:
    return [("day_7_base", 7)] + [
        (f"day_{from_day}_to_{to_day}_growth", to_day)
        for from_day, to_day in TRANSITIONS
    ]


def build_candidate_model(
    candidate: Candidate,
    *,
    n_estimators: int,
    n_jobs: int,
    early_stopping: bool,
):
    if candidate.algorithm == "xgboost":
        return build_xgb_regressor(
            objective=candidate.objective,
            eval_metric=candidate.eval_metric,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            early_stopping_rounds=30 if early_stopping else None,
            **candidate.overrides,
        )
    if candidate.algorithm == "lightgbm":
        return build_lgbm_regressor(
            objective=candidate.objective,
            metric=candidate.eval_metric,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            **candidate.overrides,
        )
    raise ValueError(f"Unknown candidate algorithm: {candidate.algorithm}")


def fit_candidate(
    candidate: Candidate,
    *,
    transformed_training: pd.DataFrame,
    transformed_validation: pd.DataFrame,
    target_training: np.ndarray,
    target_validation: np.ndarray,
    max_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, int]:
    model = build_candidate_model(
        candidate,
        n_estimators=max_estimators,
        n_jobs=n_jobs,
        early_stopping=True,
    )
    if candidate.algorithm == "xgboost":
        model.fit(
            transformed_training,
            target_training,
            eval_set=[(transformed_validation, target_validation)],
            verbose=False,
        )
        best_iteration = int(model.best_iteration) + 1
        raw_prediction = model.predict(transformed_validation)
    else:
        model.fit(
            transformed_training,
            target_training,
            eval_X=transformed_validation,
            eval_y=target_validation,
            eval_metric=candidate.eval_metric,
            callbacks=[
                lgb.early_stopping(30, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(model.best_iteration_)
        raw_prediction = model.predict(
            transformed_validation, num_iteration=best_iteration
        )
    return views_from_log_predictions(raw_prediction), best_iteration


def generate_candidate_oof(
    *,
    X: pd.DataFrame,
    targets: dict[int, np.ndarray],
    channels: pd.Series,
    folds: int,
    max_estimators: int,
    n_jobs: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
    pd.DataFrame,
]:
    components = component_targets(targets)
    predictions = {
        component: {
            candidate.name: np.full(len(X), np.nan, dtype=float)
            for candidate in CANDIDATES
        }
        for component in components
    }
    fold_ids = np.zeros(len(X), dtype=int)
    metric_rows: list[dict[str, Any]] = []
    positions = list(GroupKFold(n_splits=folds).split(X, groups=channels))
    for fold, (training, validation) in enumerate(positions, start=1):
        fold_ids[validation] = fold
        for component, target_views in components.items():
            target_log = np.log1p(target_views)
            preprocessor = HorizonDatasetPreprocessor(
                category_smoothing=CATEGORY_SMOOTHING,
                **FEATURE_OPTIONS,
            )
            transformed_training = preprocessor.fit_transform(
                X.iloc[training], target_log[training]
            )
            transformed_validation = preprocessor.transform(X.iloc[validation])
            details = []
            for candidate in CANDIDATES:
                predicted, best_iteration = fit_candidate(
                    candidate,
                    transformed_training=transformed_training,
                    transformed_validation=transformed_validation,
                    target_training=target_log[training],
                    target_validation=target_log[validation],
                    max_estimators=max_estimators,
                    n_jobs=n_jobs,
                )
                predictions[component][candidate.name][validation] = predicted
                metrics = regression_metrics(target_views[validation], predicted)
                metric_rows.append(
                    {
                        "fold": fold,
                        "component": component,
                        "candidate": candidate.name,
                        "training_rows": len(training),
                        "validation_rows": len(validation),
                        "features": transformed_training.shape[1],
                        "best_iteration": best_iteration,
                        **metrics,
                    }
                )
                details.append(f"{candidate.name}={best_iteration}")
            print(
                f"Fold {fold} {component}: " + ", ".join(details),
                flush=True,
            )
    if (fold_ids == 0).any():
        raise AssertionError("Every development row must receive one fold")
    for component_predictions in predictions.values():
        for predicted in component_predictions.values():
            if not np.isfinite(predicted).all() or (predicted < 0).any():
                raise AssertionError("Candidate OOF predictions are invalid")
    return predictions, fold_ids, pd.DataFrame(metric_rows)


def load_candidate_oof(
    source_dir: Path,
    *,
    expected_rows: int,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Reload previously generated candidate OOF predictions for reselection."""

    frame = pd.read_csv(source_dir / "candidate_oof_predictions.csv")
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} candidate rows, found {len(frame)}"
        )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for component, _ in component_sequence():
        predictions[component] = {}
        for candidate in CANDIDATES:
            column = f"{component}__{candidate.name}"
            predictions[component][candidate.name] = frame[column].to_numpy(
                dtype=float
            )
    fold_ids = frame["fold"].to_numpy(dtype=int)
    fold_metrics = pd.read_csv(source_dir / "candidate_fold_metrics.csv")
    if not np.isfinite(
        np.column_stack(
            [
                values
                for component in predictions.values()
                for values in component.values()
            ]
        )
    ).all():
        raise ValueError("Reloaded candidate predictions are not finite")
    return predictions, fold_ids, fold_metrics


def prediction_matrix(
    predictions: dict[str, np.ndarray],
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> np.ndarray:
    return np.column_stack([predictions[candidate.name] for candidate in candidates])


def recipe_prediction_matrix(
    candidate_predictions: np.ndarray,
    recipes: list[BlendRecipe],
) -> np.ndarray:
    candidates = np.asarray(candidate_predictions, dtype=float)
    combined = []
    for recipe in recipes:
        weights = np.asarray(recipe.weights, dtype=float)
        active = weights > 0
        if recipe.blend_method == "geometric":
            prediction = np.expm1(
                np.log1p(candidates[:, active]) @ weights[active]
            )
        elif recipe.blend_method == "median":
            prediction = np.median(candidates[:, active], axis=1)
        else:
            prediction = candidates @ weights
        combined.append(prediction)
    return np.column_stack(combined)


def select_recipe_index(
    *,
    actual_views: np.ndarray,
    current_views: np.ndarray | None,
    recipe_predictions: np.ndarray,
    recipes: list[BlendRecipe],
) -> int:
    candidate_views = recipe_predictions
    if current_views is not None:
        candidate_views = current_views[:, None] + np.maximum(
            MINIMUM_INCREMENT_VIEWS, recipe_predictions
        )
    error = (
        np.log1p(np.asarray(actual_views, dtype=float))[:, None]
        - np.log1p(candidate_views)
    )
    mean_squared_error = np.mean(np.square(error), axis=0)
    return min(
        range(len(recipes)),
        key=lambda index: (
            float(mean_squared_error[index]),
            recipes[index].component_count,
            recipes[index].name,
        ),
    )


def select_sequence(
    *,
    predictions: dict[str, dict[str, np.ndarray]],
    targets: dict[int, np.ndarray],
    selection_positions: np.ndarray,
    application_positions: np.ndarray,
    recipes: list[BlendRecipe],
) -> tuple[np.ndarray, list[tuple[str, int, BlendRecipe]]]:
    selected_current: np.ndarray | None = None
    applied_current: np.ndarray | None = None
    applied_trajectory: list[np.ndarray] = []
    selected: list[tuple[str, int, BlendRecipe]] = []
    for component, horizon in component_sequence():
        matrix = prediction_matrix(predictions[component])
        recipe_matrix = recipe_prediction_matrix(matrix, recipes)
        recipe_index = select_recipe_index(
            actual_views=np.asarray(targets[horizon])[selection_positions],
            current_views=selected_current,
            recipe_predictions=recipe_matrix[selection_positions],
            recipes=recipes,
        )
        recipe = recipes[recipe_index]
        selected_component = recipe_matrix[selection_positions, recipe_index]
        applied_component = recipe_matrix[application_positions, recipe_index]
        if selected_current is None:
            selected_current = selected_component
            applied_current = applied_component
        else:
            selected_current = selected_current + np.maximum(
                MINIMUM_INCREMENT_VIEWS, selected_component
            )
            applied_current = applied_current + np.maximum(  # type: ignore[operator]
                MINIMUM_INCREMENT_VIEWS, applied_component
            )
        applied_trajectory.append(np.asarray(applied_current, dtype=float))
        selected.append((component, horizon, recipe))
    return np.column_stack(applied_trajectory), selected


def cross_selected_ensemble(
    *,
    predictions: dict[str, dict[str, np.ndarray]],
    targets: dict[int, np.ndarray],
    fold_ids: np.ndarray,
    recipes: list[BlendRecipe],
) -> tuple[np.ndarray, pd.DataFrame]:
    trajectory = np.full((len(fold_ids), len(HORIZONS)), np.nan, dtype=float)
    selection_rows: list[dict[str, Any]] = []
    all_positions = np.arange(len(fold_ids))
    for fold in sorted(np.unique(fold_ids)):
        selection_positions = all_positions[fold_ids != fold]
        application_positions = all_positions[fold_ids == fold]
        fold_prediction, selected = select_sequence(
            predictions=predictions,
            targets=targets,
            selection_positions=selection_positions,
            application_positions=application_positions,
            recipes=recipes,
        )
        trajectory[application_positions] = fold_prediction
        for component, horizon, recipe in selected:
            selection_rows.append(
                {
                    "held_out_fold": int(fold),
                    "component": component,
                    "horizon_days": horizon,
                    "recipe": recipe.name,
                    "blend_method": recipe.blend_method,
                    "component_count": recipe.component_count,
                    **{
                        f"weight_{candidate.name}": weight
                        for candidate, weight in zip(CANDIDATES, recipe.weights)
                    },
                }
            )
    if not np.isfinite(trajectory).all() or not (
        np.diff(trajectory, axis=1) > 0
    ).all():
        raise AssertionError("Cross-selected ensemble trajectories are invalid")
    return trajectory, pd.DataFrame(selection_rows)


def single_candidate_trajectory(
    predictions: dict[str, dict[str, np.ndarray]],
    candidate_name: str,
) -> np.ndarray:
    current = predictions["day_7_base"][candidate_name]
    trajectory = [current]
    for from_day, to_day in TRANSITIONS:
        increment = predictions[f"day_{from_day}_to_{to_day}_growth"][candidate_name]
        current = current + np.maximum(MINIMUM_INCREMENT_VIEWS, increment)
        trajectory.append(current)
    return np.column_stack(trajectory)


def trajectory_metrics(
    method: str,
    targets: dict[int, np.ndarray],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for column, horizon in enumerate(HORIZONS):
        rows.append(
            {
                "method": method,
                "horizon_days": horizon,
                "rows": len(predictions),
                **regression_metrics(targets[horizon], predictions[:, column]),
            }
        )
    return rows


def fit_full_model(
    candidate: Candidate,
    *,
    transformed: pd.DataFrame,
    target_log: np.ndarray,
    n_estimators: int,
    n_jobs: int,
):
    model = build_candidate_model(
        candidate,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        early_stopping=False,
    )
    model.fit(transformed, target_log)
    return model


def train_selected_bundle(
    *,
    X: pd.DataFrame,
    targets: dict[int, np.ndarray],
    channels: pd.Series,
    selected: list[tuple[str, int, BlendRecipe]],
    fold_metrics: pd.DataFrame,
    n_jobs: int,
) -> MonotonicTrajectoryModelBundle:
    candidate_lookup = {candidate.name: candidate for candidate in CANDIDATES}
    built: dict[str, Any] = {}
    component_target_values = component_targets(targets)
    for component, horizon, recipe in selected:
        target_views = component_target_values[component]
        target_log = np.log1p(target_views)
        preprocessor = HorizonDatasetPreprocessor(
            category_smoothing=CATEGORY_SMOOTHING,
            **FEATURE_OPTIONS,
        )
        transformed = preprocessor.fit_transform(X, target_log)
        component_models = []
        active_weights = []
        for candidate, weight in zip(CANDIDATES, recipe.weights):
            if weight <= 0:
                continue
            selected_iterations = fold_metrics.loc[
                fold_metrics["component"].eq(component)
                & fold_metrics["candidate"].eq(candidate.name),
                "best_iteration",
            ]
            n_estimators = max(1, int(np.median(selected_iterations)))
            model = fit_full_model(
                candidate,
                transformed=transformed,
                target_log=target_log,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
            )
            metadata = {
                "candidate": candidate.name,
                "algorithm": candidate.algorithm,
                "objective": candidate.objective,
                "selected_n_estimators": n_estimators,
                "training_rows": len(X),
                "training_channels": int(channels.nunique()),
            }
            if component == "day_7_base":
                component_models.append(
                    ScaleAwareHorizonModelBundle(
                        horizon_days=7,
                        preprocessor=preprocessor,
                        regressor=model,
                        prediction_scale="log1p",
                        training_metadata=metadata,
                    )
                )
            else:
                _, from_day, _, to_day, _ = component.split("_")
                component_models.append(
                    NonnegativeIncrementModelBundle(
                        from_horizon_days=int(from_day),
                        to_horizon_days=int(to_day),
                        preprocessor=preprocessor,
                        regressor=model,
                        training_metadata=metadata,
                    )
                )
            active_weights.append(float(weight))
        if component == "day_7_base":
            built[component] = EnsembleHorizonModelBundle(
                horizon_days=7,
                components=component_models,
                weights=active_weights,
                training_metadata={"recipe": recipe.name},
                blend_method=recipe.blend_method,
            )
        else:
            first = component_models[0]
            built[component] = EnsembleIncrementModelBundle(
                from_horizon_days=first.from_horizon_days,
                to_horizon_days=first.to_horizon_days,
                components=component_models,
                weights=active_weights,
                training_metadata={"recipe": recipe.name},
                blend_method=recipe.blend_method,
            )
    return MonotonicTrajectoryModelBundle(
        base_model=built["day_7_base"],
        increment_models=[
            built[f"day_{from_day}_to_{to_day}_growth"]
            for from_day, to_day in TRANSITIONS
        ],
        minimum_increment_views=MINIMUM_INCREMENT_VIEWS,
        training_metadata={
            "construction": "development-selected multi-algorithm ensemble",
            "components_consume_prior_predictions": False,
        },
    )


def average_rmsle(metrics: pd.DataFrame, method: str) -> float:
    return float(metrics.loc[metrics["method"].eq(method), "rmsle"].mean())


def run_training(
    *,
    project_root: Path,
    output_dir: Path,
    artifact_version: str,
    baseline_checkpoint: Path,
    folds: int,
    max_estimators: int,
    n_jobs: int,
    reuse_candidates_from: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    loaded = load_all_horizons(project_root)
    transitions = {pair: load_transition(loaded, *pair) for pair in TRANSITIONS}
    channel_frames = [loaded[7].assignments["channel_id"]] + [
        transitions[pair].metadata["channel_id"] for pair in TRANSITIONS
    ]
    test_channels, split_seed, test_ratios = choose_common_test_channels(
        channel_frames
    )
    X, targets, metadata = complete_subset(loaded, HORIZONS)
    actual_matrix = np.column_stack([targets[horizon] for horizon in HORIZONS])
    naturally_monotone = (np.diff(actual_matrix, axis=1) >= 0).all(axis=1)
    testing = metadata["channel_id"].astype(str).isin(test_channels).to_numpy()
    development = ~testing & naturally_monotone
    evaluation = testing & naturally_monotone
    development_X = X.loc[development].reset_index(drop=True)
    development_targets = {
        horizon: targets[horizon][development] for horizon in HORIZONS
    }
    development_channels = metadata.loc[
        development, "channel_id"
    ].astype(str).reset_index(drop=True)

    if reuse_candidates_from is None:
        print("Generating grouped candidate OOF predictions", flush=True)
        candidate_oof, fold_ids, fold_metrics = generate_candidate_oof(
            X=development_X,
            targets=development_targets,
            channels=development_channels,
            folds=folds,
            max_estimators=max_estimators,
            n_jobs=n_jobs,
        )
    else:
        print(f"Reusing candidate OOF predictions from {reuse_candidates_from}")
        candidate_oof, fold_ids, fold_metrics = load_candidate_oof(
            reuse_candidates_from,
            expected_rows=len(development_X),
        )
    recipes = blend_recipes()
    print(f"Evaluating {len(recipes)} blend recipes per component", flush=True)
    cross_selected, fold_selections = cross_selected_ensemble(
        predictions=candidate_oof,
        targets=development_targets,
        fold_ids=fold_ids,
        recipes=recipes,
    )
    all_positions = np.arange(len(development_X))
    full_selected_oof, selected = select_sequence(
        predictions=candidate_oof,
        targets=development_targets,
        selection_positions=all_positions,
        application_positions=all_positions,
        recipes=recipes,
    )

    development_metric_rows = []
    for candidate in CANDIDATES:
        development_metric_rows.extend(
            trajectory_metrics(
                f"single_{candidate.name}",
                development_targets,
                single_candidate_trajectory(candidate_oof, candidate.name),
            )
        )
    development_metric_rows.extend(
        trajectory_metrics(
            "ensemble_cross_selected",
            development_targets,
            cross_selected,
        )
    )
    development_metric_rows.extend(
        trajectory_metrics(
            "ensemble_full_oof_selection_diagnostic",
            development_targets,
            full_selected_oof,
        )
    )
    development_metrics = pd.DataFrame(development_metric_rows)

    selection_frame = pd.DataFrame(
        [
            {
                "component": component,
                "horizon_days": horizon,
                "recipe": recipe.name,
                "blend_method": recipe.blend_method,
                "component_count": recipe.component_count,
                **{
                    f"weight_{candidate.name}": weight
                    for candidate, weight in zip(CANDIDATES, recipe.weights)
                },
            }
            for component, horizon, recipe in selected
        ]
    )
    print("Training selected ensemble on all development rows", flush=True)
    ensemble = train_selected_bundle(
        X=development_X,
        targets=development_targets,
        channels=development_channels,
        selected=selected,
        fold_metrics=fold_metrics,
        n_jobs=n_jobs,
    )
    model_path = models_dir / "strict_growth_ensemble.joblib"
    joblib.dump(ensemble, model_path)
    saved = joblib.load(model_path)

    baseline_manifest = json.loads(
        (baseline_checkpoint / "training_manifest.json").read_text(encoding="utf-8")
    )
    baseline_path = baseline_checkpoint / baseline_manifest["model_path"]
    if sha256_file(baseline_path) != baseline_manifest["model_sha256"]:
        raise RuntimeError("Baseline trajectory checksum mismatch")
    baseline = joblib.load(baseline_path)
    test_X = X.loc[evaluation].reset_index(drop=True)
    test_targets = {horizon: targets[horizon][evaluation] for horizon in HORIZONS}
    ensemble_test = saved.predict_views(test_X)
    baseline_test = baseline.predict_views(test_X)
    test_metrics = pd.DataFrame(
        trajectory_metrics("existing_strict_growth", test_targets, baseline_test)
        + trajectory_metrics("selected_ensemble", test_targets, ensemble_test)
    )

    baseline_oof_method = "single_xgb_l2_current"
    ensemble_cv_rmsle = average_rmsle(
        development_metrics, "ensemble_cross_selected"
    )
    baseline_cv_rmsle = average_rmsle(development_metrics, baseline_oof_method)
    ensemble_test_rmsle = average_rmsle(test_metrics, "selected_ensemble")
    baseline_test_rmsle = average_rmsle(test_metrics, "existing_strict_growth")
    release_gate = (
        ensemble_cv_rmsle < baseline_cv_rmsle
        and ensemble_test_rmsle <= baseline_test_rmsle
    )

    prediction_output = metadata.loc[
        evaluation, ["source_row_index", "video_id", "channel_id"]
    ].reset_index(drop=True)
    for column, horizon in enumerate(HORIZONS):
        prediction_output[f"ensemble_day_{horizon}_views"] = ensemble_test[:, column]
        prediction_output[f"baseline_day_{horizon}_views"] = baseline_test[:, column]
        prediction_output[f"actual_day_{horizon}_views"] = test_targets[horizon]
    prediction_output.head(100).to_csv(
        output_dir / "sample_test_predictions.csv", index=False
    )

    wide_oof = pd.DataFrame({"fold": fold_ids})
    for horizon in HORIZONS:
        wide_oof[f"actual_day_{horizon}_views"] = development_targets[horizon]
    for component, component_predictions in candidate_oof.items():
        for candidate_name, prediction in component_predictions.items():
            wide_oof[f"{component}__{candidate_name}"] = prediction
    wide_oof.to_csv(output_dir / "candidate_oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "candidate_fold_metrics.csv", index=False)
    fold_selections.to_csv(output_dir / "cross_selected_fold_recipes.csv", index=False)
    selection_frame.to_csv(output_dir / "selected_ensemble_recipe.csv", index=False)
    development_metrics.to_csv(
        output_dir / "development_trajectory_metrics.csv", index=False
    )
    test_metrics.to_csv(output_dir / "reserved_test_metrics.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "test": "development and reserved channels disjoint",
                "status": "PASS"
                if set(development_channels).isdisjoint(test_channels)
                else "FAIL",
            },
            {
                "test": "candidate OOF predictions finite",
                "status": "PASS"
                if all(
                    np.isfinite(values).all()
                    for component in candidate_oof.values()
                    for values in component.values()
                )
                else "FAIL",
            },
            {
                "test": "ensemble test trajectories strictly increasing",
                "status": "PASS"
                if (np.diff(ensemble_test, axis=1) > 0).all()
                else "FAIL",
            },
            {
                "test": "saved ensemble reload",
                "status": "PASS",
            },
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    if validation["status"].ne("PASS").any():
        raise AssertionError("Ensemble validation failed")

    manifest = {
        "artifact_version": artifact_version,
        "status": (
            "ensemble_release_gate_passed"
            if release_gate
            else "ensemble_benchmarked_not_promoted"
        ),
        "selection_metric": "mean horizon RMSLE",
        "selection_data": "channel-grouped development OOF only",
        "weight_evaluation": "outer-fold cross-selection",
        "candidate_definitions": [asdict(candidate) for candidate in CANDIDATES],
        "blend_recipe_count_per_component": len(recipes),
        "selected_recipe": selection_frame.to_dict(orient="records"),
        "model_path": model_path.relative_to(output_dir).as_posix(),
        "model_sha256": sha256_file(model_path),
        "development_rows": len(development_X),
        "development_channels": int(development_channels.nunique()),
        "reserved_test_rows": int(evaluation.sum()),
        "reserved_test_channels": len(test_channels),
        "common_split": {
            "selection_seed": split_seed,
            "source_test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "comparison": {
            "development_baseline_mean_rmsle": baseline_cv_rmsle,
            "development_cross_selected_ensemble_mean_rmsle": ensemble_cv_rmsle,
            "development_rmsle_improvement_pct": (
                (baseline_cv_rmsle - ensemble_cv_rmsle)
                / baseline_cv_rmsle
                * 100
            ),
            "reserved_baseline_mean_rmsle": baseline_test_rmsle,
            "reserved_ensemble_mean_rmsle": ensemble_test_rmsle,
            "reserved_rmsle_improvement_pct": (
                (baseline_test_rmsle - ensemble_test_rmsle)
                / baseline_test_rmsle
                * 100
            ),
            "release_gate_passed": release_gate,
        },
        "release_gate": (
            "development cross-selected mean RMSLE must improve and reserved "
            "mean RMSLE must not regress"
        ),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nSelected recipes", flush=True)
    print(selection_frame.to_string(index=False), flush=True)
    print("\nDevelopment trajectory metrics", flush=True)
    print(development_metrics.to_string(index=False), flush=True)
    print("\nReserved test metrics", flush=True)
    print(test_metrics.to_string(index=False), flush=True)
    print(
        f"\nRelease gate: {'PASS' if release_gate else 'FAIL'}; "
        f"artifact={output_dir}",
        flush=True,
    )
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
        default=PROJECT_ROOT / "artifacts" / "checkpoint20_growth_ensemble",
    )
    parser.add_argument(
        "--artifact-version", default="checkpoint20_growth_ensemble"
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "checkpoint18_strict_growth_trajectory_20260914",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--reuse-candidates-from", type=Path)
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
        reuse_candidates_from=args.reuse_candidates_from,
    )


if __name__ == "__main__":
    main()
