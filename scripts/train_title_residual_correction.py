"""Test a multilingual title-embedding correction for normal view forecasts.

The text model predicts only the log residual left by the existing structured
trajectory. Base residuals come from channel-grouped out-of-fold predictions,
and the reserved channel set is opened only after PCA size, Ridge penalty, and
correction strength have been selected on development rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_monotonic_trajectory import (  # noqa: E402
    HORIZONS,
    TRANSITIONS,
    choose_common_test_channels,
    complete_subset,
    load_all_horizons,
    load_transition,
    sha256_file,
)
from viewcastlk_ml.modeling import (  # noqa: E402
    TitleCorrectedTrajectoryModelBundle,
    TitleResidualCorrectionBundle,
    ViralScenarioTrajectoryModelBundle,
    regression_metrics,
    views_from_log_predictions,
)


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PCA_DIMENSIONS = (32, 64, 128)
RIDGE_ALPHAS = (10.0, 100.0, 1_000.0, 10_000.0)
CORRECTION_STRENGTHS = (0.25, 0.5, 0.75, 1.0)
CORRECTION_CLIP = 1.5
MINIMUM_INCREMENT_VIEWS = 1.0


def normalize_title(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def title_digest(titles: pd.Series) -> str:
    digest = hashlib.sha256()
    for title in titles:
        encoded = str(title).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def aligned_titles(project_root: Path, metadata: pd.DataFrame) -> pd.Series:
    source = pd.read_csv(
        project_root / "Dataset" / "viewcastlk_training_table.csv",
        usecols=["video_id", "title"],
        low_memory=False,
    )
    positions = metadata["source_row_index"].to_numpy(dtype=int)
    aligned = source.iloc[positions].reset_index(drop=True)
    if not (
        aligned["video_id"].astype(str).to_numpy()
        == metadata["video_id"].astype(str).to_numpy()
    ).all():
        raise AssertionError("Source titles do not align with complete-cohort videos")
    return aligned["title"].map(normalize_title)


def load_or_create_embeddings(
    *,
    titles: pd.Series,
    cache_path: Path,
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_digest = title_digest(titles)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_digest = str(cached["title_sha256"].item())
            cached_model = str(cached["model_name"].item())
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        if cached_digest != expected_digest or cached_model != model_name:
            raise ValueError("Cached title embeddings do not match this cohort/model")
        if len(embeddings) != len(titles):
            raise ValueError("Cached title embedding row count is incorrect")
        print(f"Reusing {len(embeddings):,} cached title embeddings", flush=True)
    else:
        from sentence_transformers import SentenceTransformer

        codes, unique_titles = pd.factorize(titles, sort=False)
        if (codes < 0).any():
            raise AssertionError("Normalized titles must not contain missing values")
        print(
            f"Encoding {len(unique_titles):,} unique titles with {model_name}",
            flush=True,
        )
        encoder = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )
        unique_embeddings = encoder.encode(
            unique_titles.tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        embeddings = unique_embeddings[codes]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            title_sha256=np.asarray(expected_digest),
            model_name=np.asarray(model_name),
        )
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Title embeddings must be a finite matrix")
    metadata = {
        "model_name": model_name,
        "rows": len(embeddings),
        "dimensions": embeddings.shape[1],
        "unique_titles": int(titles.nunique()),
        "title_sha256": expected_digest,
        "cache_path": cache_path.as_posix(),
        "cache_sha256": sha256_file(cache_path),
    }
    return embeddings, metadata


def baseline_oof_trajectory(
    source_dir: Path,
    development_targets: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(source_dir / "candidate_oof_predictions.csv")
    if len(frame) != len(development_targets[7]):
        raise ValueError("Base OOF prediction row count does not match development")
    for horizon in HORIZONS:
        if not np.allclose(
            frame[f"actual_day_{horizon}_views"].to_numpy(dtype=float),
            development_targets[horizon],
        ):
            raise AssertionError(f"Base OOF Day-{horizon} labels are misaligned")
    current = frame["day_7_base__xgb_l2_current"].to_numpy(dtype=float)
    trajectory = [current.copy()]
    for from_day, to_day in TRANSITIONS:
        increment = frame[
            f"day_{from_day}_to_{to_day}_growth__xgb_l2_current"
        ].to_numpy(dtype=float)
        current = current + np.maximum(MINIMUM_INCREMENT_VIEWS, increment)
        trajectory.append(current.copy())
    result = np.column_stack(trajectory)
    if not np.isfinite(result).all() or not (np.diff(result, axis=1) > 0).all():
        raise AssertionError("Base OOF trajectory is invalid")
    return result, frame["fold"].to_numpy(dtype=int)


def residual_targets(
    targets: dict[int, np.ndarray], base_predictions: np.ndarray
) -> np.ndarray:
    actual = np.column_stack([targets[horizon] for horizon in HORIZONS])
    return np.log1p(actual) - np.log1p(base_predictions)


def cross_fit_text_models(
    *,
    embeddings: np.ndarray,
    residuals: np.ndarray,
    fold_ids: np.ndarray,
    dimensions: tuple[int, ...],
    ridge_alphas: tuple[float, ...],
) -> tuple[dict[tuple[int, float], np.ndarray], np.ndarray, pd.DataFrame]:
    predictions = {
        (dimension, alpha): np.full_like(residuals, np.nan, dtype=float)
        for dimension in dimensions
        for alpha in ridge_alphas
    }
    constant = np.full_like(residuals, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids)):
        training = fold_ids != fold
        validation = fold_ids == fold
        component_count = min(
            max(dimensions), int(training.sum()) - 1, embeddings.shape[1]
        )
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            iterated_power=3,
            random_state=42,
        )
        projected_training = pca.fit_transform(embeddings[training])
        projected_validation = pca.transform(embeddings[validation])
        constant[validation] = residuals[training].mean(axis=0)
        for dimension in dimensions:
            scaler = StandardScaler()
            training_values = scaler.fit_transform(
                projected_training[:, :dimension]
            )
            validation_values = scaler.transform(
                projected_validation[:, :dimension]
            )
            for alpha in ridge_alphas:
                regressor = Ridge(alpha=alpha)
                regressor.fit(training_values, residuals[training])
                predicted = regressor.predict(validation_values)
                predictions[(dimension, alpha)][validation] = predicted
                rows.append(
                    {
                        "fold": int(fold),
                        "pca_dimensions": dimension,
                        "ridge_alpha": alpha,
                        "training_rows": int(training.sum()),
                        "validation_rows": int(validation.sum()),
                        "pca_explained_variance_pct": float(
                            pca.explained_variance_ratio_[:dimension].sum() * 100
                        ),
                        "mean_absolute_log_correction": float(
                            np.abs(predicted).mean()
                        ),
                    }
                )
        print(
            f"Text residual fold {fold}: PCA fit on {int(training.sum()):,} rows",
            flush=True,
        )
    if not np.isfinite(constant).all() or not all(
        np.isfinite(prediction).all() for prediction in predictions.values()
    ):
        raise AssertionError("Cross-fitted text corrections are incomplete")
    return predictions, constant, pd.DataFrame(rows)


def corrected_trajectory(
    base: np.ndarray,
    correction: np.ndarray,
    strength: float,
) -> np.ndarray:
    clipped = np.clip(
        np.asarray(correction, dtype=float), -CORRECTION_CLIP, CORRECTION_CLIP
    )
    result = views_from_log_predictions(
        np.log1p(np.asarray(base, dtype=float)) + float(strength) * clipped
    )
    for position in range(1, result.shape[1]):
        result[:, position] = np.maximum(
            result[:, position],
            result[:, position - 1] + MINIMUM_INCREMENT_VIEWS,
        )
    return result


def mean_horizon_rmsle(
    targets: dict[int, np.ndarray], predictions: np.ndarray
) -> float:
    return float(
        np.mean(
            [
                regression_metrics(targets[horizon], predictions[:, column])["rmsle"]
                for column, horizon in enumerate(HORIZONS)
            ]
        )
    )


def trajectory_metric_rows(
    method: str,
    targets: dict[int, np.ndarray],
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "horizon_days": horizon,
            "rows": len(predictions),
            **regression_metrics(targets[horizon], predictions[:, column]),
        }
        for column, horizon in enumerate(HORIZONS)
    ]


def select_configuration(
    *,
    targets: dict[int, np.ndarray],
    base_oof: np.ndarray,
    text_oof: dict[tuple[int, float], np.ndarray],
    constant_oof: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows = [
        {
            "correction": "none",
            "pca_dimensions": 0,
            "ridge_alpha": 0.0,
            "strength": 0.0,
            "mean_horizon_rmsle": mean_horizon_rmsle(targets, base_oof),
        }
    ]
    for strength in CORRECTION_STRENGTHS:
        rows.append(
            {
                "correction": "constant",
                "pca_dimensions": 0,
                "ridge_alpha": 0.0,
                "strength": strength,
                "mean_horizon_rmsle": mean_horizon_rmsle(
                    targets,
                    corrected_trajectory(base_oof, constant_oof, strength),
                ),
            }
        )
    for (dimension, alpha), correction in text_oof.items():
        for strength in CORRECTION_STRENGTHS:
            rows.append(
                {
                    "correction": "title_ridge",
                    "pca_dimensions": dimension,
                    "ridge_alpha": alpha,
                    "strength": strength,
                    "mean_horizon_rmsle": mean_horizon_rmsle(
                        targets,
                        corrected_trajectory(base_oof, correction, strength),
                    ),
                }
            )
    comparison = pd.DataFrame(rows).sort_values(
        [
            "mean_horizon_rmsle",
            "correction",
            "pca_dimensions",
            "ridge_alpha",
            "strength",
        ]
    )
    selected_text = comparison.loc[
        comparison["correction"].eq("title_ridge")
    ].iloc[0]
    selected_constant = comparison.loc[
        comparison["correction"].eq("constant")
    ].iloc[0]
    return comparison, selected_text, selected_constant


def fit_final_correction(
    *,
    embeddings: np.ndarray,
    residuals: np.ndarray,
    dimensions: int,
    ridge_alpha: float,
    strength: float,
) -> TitleResidualCorrectionBundle:
    pca = PCA(
        n_components=dimensions,
        svd_solver="randomized",
        iterated_power=3,
        random_state=42,
    )
    projected = pca.fit_transform(embeddings)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(projected)
    regressor = Ridge(alpha=ridge_alpha)
    regressor.fit(standardized, residuals)
    return TitleResidualCorrectionBundle(
        pca=pca,
        scaler=scaler,
        regressor=regressor,
        embedding_dimensions=dimensions,
        correction_strength=strength,
        correction_clip=CORRECTION_CLIP,
        minimum_increment_views=MINIMUM_INCREMENT_VIEWS,
        training_metadata={
            "embedding_model": EMBEDDING_MODEL,
            "target": "log1p(actual_views) - log1p(structured_oof_prediction)",
            "pca_dimensions": dimensions,
            "pca_explained_variance_pct": float(
                pca.explained_variance_ratio_.sum() * 100
            ),
            "ridge_alpha": ridge_alpha,
            "correction_strength": strength,
            "training_rows": len(embeddings),
        },
    )


def run_training(
    *,
    project_root: Path,
    output_dir: Path,
    artifact_version: str,
    baseline_scenario_checkpoint: Path,
    baseline_oof_checkpoint: Path,
    embedding_model: str,
    batch_size: int,
    device: str,
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
    development_targets = {
        horizon: targets[horizon][development] for horizon in HORIZONS
    }
    test_targets = {horizon: targets[horizon][evaluation] for horizon in HORIZONS}

    titles = aligned_titles(project_root, metadata)
    embeddings, embedding_metadata = load_or_create_embeddings(
        titles=titles,
        cache_path=output_dir / "title_embeddings_complete.npz",
        model_name=embedding_model,
        batch_size=batch_size,
        device=device,
    )
    development_embeddings = embeddings[development]
    test_embeddings = embeddings[evaluation]

    base_oof, fold_ids = baseline_oof_trajectory(
        baseline_oof_checkpoint, development_targets
    )
    residuals = residual_targets(development_targets, base_oof)
    print("Cross-fitting PCA plus Ridge title residual models", flush=True)
    text_oof, constant_oof, fold_metrics = cross_fit_text_models(
        embeddings=development_embeddings,
        residuals=residuals,
        fold_ids=fold_ids,
        dimensions=PCA_DIMENSIONS,
        ridge_alphas=RIDGE_ALPHAS,
    )
    configuration_metrics, selected_text, selected_constant = select_configuration(
        targets=development_targets,
        base_oof=base_oof,
        text_oof=text_oof,
        constant_oof=constant_oof,
    )
    selected_key = (
        int(selected_text["pca_dimensions"]),
        float(selected_text["ridge_alpha"]),
    )
    selected_strength = float(selected_text["strength"])
    selected_text_oof = corrected_trajectory(
        base_oof, text_oof[selected_key], selected_strength
    )
    selected_constant_strength = float(selected_constant["strength"])
    selected_constant_oof = corrected_trajectory(
        base_oof, constant_oof, selected_constant_strength
    )

    correction_bundle = fit_final_correction(
        embeddings=development_embeddings,
        residuals=residuals,
        dimensions=selected_key[0],
        ridge_alpha=selected_key[1],
        strength=selected_strength,
    )
    correction_path = models_dir / "title_residual_correction.joblib"
    joblib.dump(correction_bundle, correction_path)
    saved_correction = joblib.load(correction_path)

    scenario_manifest = json.loads(
        (baseline_scenario_checkpoint / "training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    scenario_path = baseline_scenario_checkpoint / scenario_manifest["model_path"]
    if sha256_file(scenario_path) != scenario_manifest["model_sha256"]:
        raise RuntimeError("Baseline scenario checksum mismatch")
    scenario = joblib.load(scenario_path)
    test_X = X.loc[evaluation].reset_index(drop=True)
    base_test = scenario.predict_views(test_X)
    title_test = saved_correction.apply(base_test, test_embeddings)
    constant_value = residuals.mean(axis=0)
    constant_test = corrected_trajectory(
        base_test,
        np.broadcast_to(constant_value, base_test.shape),
        selected_constant_strength,
    )

    development_metrics = pd.DataFrame(
        trajectory_metric_rows("existing_structured_oof", development_targets, base_oof)
        + trajectory_metric_rows(
            "constant_correction_oof", development_targets, selected_constant_oof
        )
        + trajectory_metric_rows(
            "title_ridge_correction_oof", development_targets, selected_text_oof
        )
    )
    test_metrics = pd.DataFrame(
        trajectory_metric_rows("existing_v8", test_targets, base_test)
        + trajectory_metric_rows(
            "constant_correction", test_targets, constant_test
        )
        + trajectory_metric_rows("title_ridge_correction", test_targets, title_test)
    )
    development_base_rmsle = mean_horizon_rmsle(development_targets, base_oof)
    development_constant_rmsle = mean_horizon_rmsle(
        development_targets, selected_constant_oof
    )
    development_title_rmsle = mean_horizon_rmsle(
        development_targets, selected_text_oof
    )
    test_base_rmsle = mean_horizon_rmsle(test_targets, base_test)
    test_constant_rmsle = mean_horizon_rmsle(test_targets, constant_test)
    test_title_rmsle = mean_horizon_rmsle(test_targets, title_test)
    release_gate = bool(
        development_title_rmsle < development_base_rmsle
        and development_title_rmsle < development_constant_rmsle
        and test_title_rmsle < test_base_rmsle
        and test_title_rmsle < test_constant_rmsle
    )

    promoted_scenario_path: Path | None = None
    if release_gate:
        title_trajectory = TitleCorrectedTrajectoryModelBundle(
            base_trajectory=scenario.normal_trajectory,
            correction=saved_correction,
            embedding_model_source=embedding_model,
            training_metadata={
                "artifact_version": artifact_version,
                **saved_correction.training_metadata,
            },
        )
        promoted_scenario = ViralScenarioTrajectoryModelBundle(
            normal_trajectory=title_trajectory,
            breakout_classifier=scenario.breakout_classifier,
            viral_trajectory=scenario.viral_trajectory,
            minimum_viral_uplift_views=scenario.minimum_viral_uplift_views,
            breakout_minimum_views=scenario.breakout_minimum_views,
            breakout_baseline_multiplier=scenario.breakout_baseline_multiplier,
            breakout_minimum_history_count=scenario.breakout_minimum_history_count,
            training_metadata={
                **scenario.training_metadata,
                "normal_title_correction": artifact_version,
            },
        )
        promoted_scenario_path = models_dir / "viral_scenario_trajectory.joblib"
        joblib.dump(promoted_scenario, promoted_scenario_path)
        reloaded_scenario = joblib.load(promoted_scenario_path)
        augmented_test_X = test_X.copy()
        augmented_test_X["title"] = titles.loc[evaluation].reset_index(drop=True)
        reloaded_title_test = (
            reloaded_scenario.normal_trajectory.predict_views_from_embeddings(
                augmented_test_X, test_embeddings
            )
        )
        if not np.allclose(reloaded_title_test, title_test):
            raise AssertionError("Reloaded title trajectory changed predictions")
        smoke = reloaded_scenario.predict_scenario_frame(augmented_test_X.head(2))
        if smoke.shape != (2, 9) or not np.isfinite(
            smoke.to_numpy(dtype=float)
        ).all():
            raise AssertionError("Raw-title scenario smoke prediction failed")

    sample = metadata.loc[
        evaluation, ["source_row_index", "video_id", "channel_id"]
    ].reset_index(drop=True)
    sample["title"] = titles.loc[evaluation].reset_index(drop=True)
    for column, horizon in enumerate(HORIZONS):
        sample[f"actual_day_{horizon}_views"] = test_targets[horizon]
        sample[f"base_day_{horizon}_views"] = base_test[:, column]
        sample[f"title_corrected_day_{horizon}_views"] = title_test[:, column]
    sample.head(100).to_csv(output_dir / "sample_test_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "text_cv_fold_diagnostics.csv", index=False)
    configuration_metrics.to_csv(
        output_dir / "development_configuration_metrics.csv", index=False
    )
    development_metrics.to_csv(
        output_dir / "development_trajectory_metrics.csv", index=False
    )
    test_metrics.to_csv(output_dir / "reserved_test_metrics.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "test": "development and reserved channels disjoint",
                "status": (
                    "PASS"
                    if set(metadata.loc[development, "channel_id"].astype(str)).isdisjoint(
                        set(metadata.loc[evaluation, "channel_id"].astype(str))
                    )
                    else "FAIL"
                ),
            },
            {
                "test": "title embeddings finite and normalized",
                "status": (
                    "PASS"
                    if np.isfinite(embeddings).all()
                    and np.allclose(
                        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4
                    )
                    else "FAIL"
                ),
            },
            {
                "test": "title-corrected trajectories strictly increasing",
                "status": (
                    "PASS" if (np.diff(title_test, axis=1) > 0).all() else "FAIL"
                ),
            },
            {"test": "saved correction reload", "status": "PASS"},
        ]
    )
    validation.to_csv(output_dir / "validation_tests.csv", index=False)
    if validation["status"].ne("PASS").any():
        raise AssertionError("Title residual correction validation failed")

    manifest = {
        "artifact_version": artifact_version,
        "status": (
            "title_residual_release_gate_passed"
            if release_gate
            else "title_residual_benchmarked_not_promoted"
        ),
        "correction_model_path": correction_path.relative_to(output_dir).as_posix(),
        "correction_model_sha256": sha256_file(correction_path),
        "model_path": (
            promoted_scenario_path.relative_to(output_dir).as_posix()
            if promoted_scenario_path is not None
            else None
        ),
        "model_sha256": (
            sha256_file(promoted_scenario_path)
            if promoted_scenario_path is not None
            else None
        ),
        "embedding": embedding_metadata,
        "target": "log1p(actual_views) - log1p(structured_oof_prediction)",
        "candidate_pca_dimensions": list(PCA_DIMENSIONS),
        "candidate_ridge_alphas": list(RIDGE_ALPHAS),
        "candidate_correction_strengths": list(CORRECTION_STRENGTHS),
        "correction_clip": CORRECTION_CLIP,
        "selected_text_configuration": selected_text.to_dict(),
        "selected_constant_configuration": selected_constant.to_dict(),
        "development_rows": int(development.sum()),
        "reserved_test_rows": int(evaluation.sum()),
        "common_split": {
            "type": "channel_grouped_shared_across_all_horizons",
            "selection_seed": split_seed,
            "test_channels": len(test_channels),
            "source_test_row_fractions": {
                "day_7": test_ratios[0],
                "day_7_to_14": test_ratios[1],
                "day_14_to_21": test_ratios[2],
                "day_21_to_30": test_ratios[3],
            },
        },
        "comparison": {
            "development_base_mean_rmsle": development_base_rmsle,
            "development_constant_mean_rmsle": development_constant_rmsle,
            "development_title_mean_rmsle": development_title_rmsle,
            "reserved_base_mean_rmsle": test_base_rmsle,
            "reserved_constant_mean_rmsle": test_constant_rmsle,
            "reserved_title_mean_rmsle": test_title_rmsle,
            "reserved_title_improvement_vs_base_pct": (
                (test_base_rmsle - test_title_rmsle) / test_base_rmsle * 100
            ),
            "release_gate_passed": release_gate,
        },
        "release_gate": (
            "title correction must beat both the unchanged base and an intercept-only "
            "correction on development OOF and reserved channels"
        ),
        "promoted": release_gate,
        "output_contract": scenario_manifest["output_contract"],
        "breakout_definition": scenario_manifest["breakout_definition"],
        "classifier_reserved_test": scenario_manifest["classifier_reserved_test"],
        "limitations": [
            *scenario_manifest["limitations"],
            "Title correction requires the exact frozen multilingual MiniLM encoder.",
        ],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSelected title configuration", flush=True)
    print(selected_text.to_frame().T.to_string(index=False), flush=True)
    print("\nDevelopment trajectory metrics", flush=True)
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
        default=PROJECT_ROOT / "artifacts" / "checkpoint24_title_residual",
    )
    parser.add_argument("--artifact-version", default="checkpoint24_title_residual")
    parser.add_argument(
        "--baseline-scenario-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint22_breakout_ensemble_20260915"
        ),
    )
    parser.add_argument(
        "--baseline-oof-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoint20_growth_ensemble_20260915"
        ),
    )
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        artifact_version=args.artifact_version,
        baseline_scenario_checkpoint=args.baseline_scenario_checkpoint,
        baseline_oof_checkpoint=args.baseline_oof_checkpoint,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
