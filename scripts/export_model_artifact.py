"""Export the selected four-horizon models as a portable deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewcastlk_ml.horizon_preprocessing import (  # noqa: E402
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXCLUDED_MODEL_COLUMNS,
    LLM_SCORE_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    TARGET_ENCODED_COLUMN,
)


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "checkpoint10_wape_blended"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "exports"
DEFAULT_ARTIFACT_VERSION = "viewcastlk_mvp_candidate_v1"
SUPPORTED_HORIZONS = (7, 14, 21, 30)
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUNTIME_MODULES = (
    "__init__.py",
    "horizon_preprocessing.py",
    "modeling.py",
    "preprocessing.py",
)
RUNTIME_REQUIREMENTS = (
    "numpy>=2,<3",
    "pandas>=2.2,<4",
    "scikit-learn>=1.5,<2",
    "xgboost>=3,<4",
    "joblib>=1.4,<2",
)


PREDICT_CLI = '''"""Run a ViewCastLK horizon model against a CSV feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(horizon: int):
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["models"] if item["horizon_days"] == horizon]
    if not matches:
        supported = ", ".join(map(str, manifest["supported_horizons_days"]))
        raise ValueError(f"Unsupported horizon {horizon}; choose one of: {supported}")
    record = matches[0]
    model_path = ROOT / record["model_path"]
    actual_hash = sha256_file(model_path)
    if actual_hash != record["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {model_path.name}")
    return joblib.load(model_path), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, choices=(7, 14, 21, 30))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, manifest = load_model(args.horizon)
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    predictions = np.asarray(model.predict_views(frame), dtype=float)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise RuntimeError("Model returned invalid predictions")
    output = frame.copy()
    output["prediction_horizon_days"] = args.horizon
    output["predicted_views"] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} Day {args.horizon} predictions to {args.output}")


if __name__ == "__main__":
    main()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def installed_runtime_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def input_schema() -> dict[str, Any]:
    expected = (
        [TARGET_ENCODED_COLUMN]
        + list(RAW_NUMERIC_COLUMNS)
        + list(BOOLEAN_COLUMNS)
        + list(CATEGORICAL_COLUMNS)
    )
    return {
        "expected_columns": expected,
        "numeric_columns": list(RAW_NUMERIC_COLUMNS),
        "boolean_columns": list(BOOLEAN_COLUMNS),
        "categorical_columns": [TARGET_ENCODED_COLUMN, *CATEGORICAL_COLUMNS],
        "missing_values": "Allowed in values, but every expected column must exist.",
        "subscriber_tier_note": (
            "Use the exported tier if available. It can also be derived from "
            "ch_subs_at_publish with the boundaries documented in the model code."
        ),
        "future_optional_llm_score_columns": list(LLM_SCORE_COLUMNS),
        "llm_scores_enabled_in_this_artifact": False,
        "explicitly_excluded_columns": list(EXCLUDED_MODEL_COLUMNS),
    }


def sample_input_frame() -> pd.DataFrame:
    values: dict[str, object] = {
        TARGET_ENCODED_COLUMN: "Music",
        "duration_seconds": 300.0,
        "ch_subs_at_publish": 125_000.0,
        "ch_avg_views_per_video_at_publish": 12_000.0,
        "ch_videos_at_publish": 500.0,
        "channel_age_days_at_publish": 1_200.0,
        "default_language": "en",
        "publish_time_bucket": "evening",
        "subscriber_tier": "100k_to_250k",
    }
    for column in BOOLEAN_COLUMNS:
        values[column] = column == "topic_music"
    return pd.DataFrame([values], columns=input_schema()["expected_columns"])


def model_card(manifest: dict[str, Any]) -> str:
    combined = manifest["evaluation"]["combined_development_oof"]
    rows = [
        "# ViewCastLK MVP model candidate",
        "",
        "This artifact predicts independent Day 7, 14, 21, or 30 view totals ",
        "from pre-publication video and channel features.",
        "",
        "## Status",
        "",
        "Candidate only. The reserved test partition has not been evaluated. ",
        "The development result shows weak viral-video performance and is not ",
        "production-ready accuracy.",
        "",
        "## Development evaluation",
        "",
        f"- Combined OOF WAPE: {combined['wape_pct']:.2f}%",
        f"- Combined OOF RMSLE: {combined['rmsle']:.3f}",
        f"- Total view capture: {combined['total_view_capture_pct']:.2f}%",
        f"- Top-decile view capture: {combined['top_decile_view_capture_pct']:.2f}%",
        "",
        "## Usage",
        "",
        "```text",
        "python -m pip install -r requirements.txt",
        "python predict.py --horizon 7 --input sample_input.csv --output predictions.csv",
        "```",
        "",
        "See `manifest.json` for the complete schema, checksums, selected model ",
        "components, and per-horizon metrics.",
    ]
    return "\n".join(rows) + "\n"


def export_artifact(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
) -> tuple[Path, Path, Path]:
    if not VERSION_PATTERN.fullmatch(artifact_version):
        raise ValueError(
            "artifact_version may contain only letters, numbers, dot, dash, and underscore"
        )

    source_dir = source_dir.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_version
    archive_path = output_root / f"{artifact_version}.zip"
    archive_checksum_path = output_root / f"{artifact_version}.zip.sha256"
    conflicts = [path for path in (destination, archive_path, archive_checksum_path) if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing export: "
            + ", ".join(str(path) for path in conflicts)
        )

    source_manifest_path = source_dir / "training_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = source_manifest.get("selected_ensembles", [])
    if sorted(int(item["horizon_days"]) for item in selected) != list(SUPPORTED_HORIZONS):
        raise ValueError("Source manifest must contain exactly Day 7, 14, 21, and 30 models")

    with tempfile.TemporaryDirectory(prefix=f".{artifact_version}-", dir=output_root) as temp:
        staging = Path(temp) / artifact_version
        models_dir = staging / "models"
        runtime_dir = staging / "viewcastlk_ml"
        evaluation_dir = staging / "evaluation"
        models_dir.mkdir(parents=True)
        runtime_dir.mkdir()
        evaluation_dir.mkdir()

        model_records: list[dict[str, Any]] = []
        sample = sample_input_frame()
        smoke_predictions: dict[str, float] = {}
        for source_record in sorted(selected, key=lambda item: int(item["horizon_days"])):
            horizon = int(source_record["horizon_days"])
            source_model = source_dir / Path(source_record["model_path"])
            destination_model = models_dir / f"day_{horizon}_ensemble.joblib"
            if sha256_file(source_model) != source_record["model_sha256"]:
                raise RuntimeError(f"Source checksum mismatch for {source_model}")
            shutil.copy2(source_model, destination_model)
            bundle = joblib.load(destination_model)
            prediction = np.asarray(bundle.predict_views(sample), dtype=float)
            if len(prediction) != 1 or not np.isfinite(prediction).all():
                raise RuntimeError(f"Day {horizon} model failed the export smoke test")
            smoke_predictions[str(horizon)] = float(prediction[0])
            model_records.append(
                {
                    "horizon_days": horizon,
                    "model_path": destination_model.relative_to(staging).as_posix(),
                    "sha256": sha256_file(destination_model),
                    "size_bytes": destination_model.stat().st_size,
                    "components": source_record["components"].split("|"),
                    "weights": [float(value) for value in source_record["weights"].split("|")],
                    "selected_n_estimators": [
                        int(value) for value in source_record["selected_n_estimators"].split("|")
                    ],
                    "development_oof_metrics": {
                        "wape_pct": source_record["cross_fitted_oof_wape_pct"],
                        "rmsle": source_record["cross_fitted_oof_rmsle"],
                        "total_view_capture_pct": source_record[
                            "cross_fitted_oof_total_view_capture_pct"
                        ],
                        "top_decile_view_capture_pct": source_record[
                            "cross_fitted_oof_top_decile_view_capture_pct"
                        ],
                    },
                }
            )

        for module_name in RUNTIME_MODULES:
            shutil.copy2(PROJECT_ROOT / "viewcastlk_ml" / module_name, runtime_dir / module_name)
        for filename in ("selected_ensembles.csv", "validation_tests.csv"):
            shutil.copy2(source_dir / filename, evaluation_dir / filename)

        sample.to_csv(staging / "sample_input.csv", index=False)
        (staging / "predict.py").write_text(PREDICT_CLI, encoding="utf-8")
        (staging / "requirements.txt").write_text(
            "\n".join(RUNTIME_REQUIREMENTS) + "\n", encoding="utf-8"
        )

        manifest = {
            "format_version": 1,
            "artifact_version": artifact_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": source_manifest["status"],
            "supported_horizons_days": list(SUPPORTED_HORIZONS),
            "prediction_interface": "predict.py CSV CLI or bundle.predict_views(DataFrame)",
            "source_checkpoint": source_manifest["artifact_version"],
            "source_checkpoint_manifest_sha256": sha256_file(source_manifest_path),
            "source_git_commit": source_git_commit(PROJECT_ROOT),
            "runtime_versions_used_for_export": installed_runtime_versions(),
            "input_schema": input_schema(),
            "evaluation": {
                "design": source_manifest["validation_design"],
                "primary_metric": source_manifest["primary_metric"],
                "reserved_test_used": source_manifest["reserved_test_used"],
                "combined_development_oof": source_manifest[
                    "selected_oof_combined_metrics"
                ],
            },
            "models": model_records,
            "sample_smoke_predictions": smoke_predictions,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(model_card(manifest), encoding="utf-8")

        checksum_lines = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            checksum_lines.append(f"{sha256_file(path)}  {relative}")
        (staging / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        shutil.move(str(staging), destination)

    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in destination.rglob("*") if item.is_file()):
            archive.write(
                path,
                arcname=(Path(artifact_version) / path.relative_to(destination)).as_posix(),
            )
    archive_checksum_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    return destination, archive_path, archive_checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-version", default=DEFAULT_ARTIFACT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination, archive_path, checksum_path = export_artifact(
        source_dir=args.source_dir,
        output_root=args.output_root,
        artifact_version=args.artifact_version,
    )
    print(f"Export directory: {destination}")
    print(f"ZIP artifact: {archive_path}")
    print(f"ZIP SHA-256: {checksum_path.read_text(encoding='utf-8').split()[0]}")


if __name__ == "__main__":
    main()
