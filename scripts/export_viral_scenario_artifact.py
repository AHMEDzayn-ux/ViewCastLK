"""Export the normal-plus-viral-scenario model as a portable bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_model_artifact import (  # noqa: E402
    RUNTIME_MODULES,
    VERSION_PATTERN,
    input_schema,
    installed_runtime_versions,
    sample_input_frame,
    sha256_file,
    source_git_commit,
)


RUNTIME_REQUIREMENTS = (
    "numpy>=2,<3",
    "pandas>=2.2,<4",
    "scikit-learn>=1.5,<2",
    "xgboost>=3,<4",
    "lightgbm>=4,<5",
    "joblib>=1.4,<2",
)
TITLE_RUNTIME_REQUIREMENTS = (
    "sentence-transformers>=5,<6",
    "torch>=2,<3",
)


PREDICT_CLI = '''"""Generate a normal forecast and conditional viral-upside scenario."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--display-horizon", type=int, choices=(7, 14, 21, 30), default=30
    )
    args = parser.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["model"]
    model_path = ROOT / record["model_path"]
    if sha256_file(model_path) != record["sha256"]:
        raise RuntimeError("Model checksum mismatch")
    model = joblib.load(model_path)
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    scenario = model.predict_scenario_frame(frame)
    normal_columns = [f"normal_day_{day}_views" for day in (7, 14, 21, 30)]
    viral_columns = [f"viral_upside_day_{day}_views" for day in (7, 14, 21, 30)]
    normal = scenario[normal_columns].to_numpy(dtype=float)
    viral = scenario[viral_columns].to_numpy(dtype=float)
    probability = scenario["breakout_probability"].to_numpy(dtype=float)
    if not np.isfinite(normal).all() or not np.isfinite(viral).all():
        raise RuntimeError("Model returned non-finite scenario views")
    if (np.diff(normal, axis=1) <= 0).any() or (np.diff(viral, axis=1) <= 0).any():
        raise RuntimeError("Model returned a flat or decreasing trajectory")
    if (viral <= normal).any():
        raise RuntimeError("Viral upside must be above the normal forecast")
    if ((probability < 0) | (probability > 1)).any():
        raise RuntimeError("Breakout probability is outside [0, 1]")

    horizon = args.display_horizon
    scenario["prediction_summary"] = [
        (
            f"Normally expected around {normal_value:,.0f} views by day {horizon}. "
            f"There is a {chance:.1%} breakout chance; if that happens, "
            f"around {viral_value:,.0f} views."
        )
        for normal_value, chance, viral_value in zip(
            scenario[f"normal_day_{horizon}_views"],
            probability,
            scenario[f"viral_upside_day_{horizon}_views"],
        )
    ]
    output = pd.concat([frame.reset_index(drop=True), scenario.reset_index(drop=True)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} two-scenario forecasts to {args.output}")


if __name__ == "__main__":
    main()
'''


def export_artifact(
    *,
    source_dir: Path,
    output_root: Path,
    artifact_version: str,
) -> tuple[Path, Path, Path]:
    if not VERSION_PATTERN.fullmatch(artifact_version):
        raise ValueError("Invalid artifact version")
    source_dir = source_dir.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / artifact_version
    archive_path = output_root / f"{artifact_version}.zip"
    checksum_path = output_root / f"{artifact_version}.zip.sha256"
    conflicts = [
        path for path in (destination, archive_path, checksum_path) if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing export: "
            + ", ".join(str(path) for path in conflicts)
        )

    source_manifest_path = source_dir / "training_manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_model_record = source_manifest.get("model_path") or source_manifest.get(
        "scenario_model_path"
    )
    source_model_checksum = source_manifest.get(
        "model_sha256"
    ) or source_manifest.get("scenario_model_sha256")
    if not source_model_record or not source_model_checksum:
        raise ValueError("Source checkpoint does not contain a promoted scenario model")
    source_model = source_dir / source_model_record
    if sha256_file(source_model) != source_model_checksum:
        raise RuntimeError("Source model checksum mismatch")
    evaluation_files = [source_manifest_path]
    evaluation_files.extend(sorted(source_dir.glob("*.csv")))

    with tempfile.TemporaryDirectory(
        prefix=f".{artifact_version}-", dir=output_root
    ) as temporary_directory:
        staging = Path(temporary_directory) / artifact_version
        models_dir = staging / "models"
        runtime_dir = staging / "viewcastlk_ml"
        evaluation_dir = staging / "evaluation"
        models_dir.mkdir(parents=True)
        runtime_dir.mkdir()
        evaluation_dir.mkdir()
        destination_model = models_dir / "viral_scenario_trajectory.joblib"
        model = joblib.load(source_model)
        title_trajectory = getattr(model, "normal_trajectory", None)
        uses_title_encoder = hasattr(title_trajectory, "embedding_model_source")
        if uses_title_encoder:
            from huggingface_hub import snapshot_download

            embedding_record = source_manifest.get("embedding", {})
            embedding_model_name = embedding_record.get(
                "model_name", title_trajectory.embedding_model_source
            )
            encoder_source = Path(
                snapshot_download(embedding_model_name, local_files_only=True)
            )
            shutil.copytree(encoder_source, staging / "embedding_model")
            title_trajectory.embedding_model_source = "artifact://embedding_model"
            title_trajectory.local_files_only = True
            joblib.dump(model, destination_model)
        else:
            shutil.copy2(source_model, destination_model)
        for module_name in RUNTIME_MODULES:
            shutil.copy2(
                PROJECT_ROOT / "viewcastlk_ml" / module_name,
                runtime_dir / module_name,
            )
        for source_evaluation in evaluation_files:
            shutil.copy2(source_evaluation, evaluation_dir / source_evaluation.name)

        model = joblib.load(destination_model)
        classifier = model.breakout_classifier
        preprocessor = getattr(classifier, "preprocessor", None)
        if preprocessor is None and getattr(classifier, "components", None):
            preprocessor = classifier.components[0].preprocessor
        if preprocessor is None:
            raise RuntimeError("Could not locate breakout preprocessor")
        sample = sample_input_frame(preprocessor)
        if uses_title_encoder:
            sample["title"] = "Unexpected story explained in Sinhala and English"
        scenario = model.predict_scenario_frame(sample)
        normal = scenario.filter(regex=r"^normal_day_").to_numpy(dtype=float)
        viral = scenario.filter(regex=r"^viral_upside_day_").to_numpy(dtype=float)
        probability = scenario["breakout_probability"].to_numpy(dtype=float)
        if scenario.shape != (1, 9):
            raise RuntimeError("Unexpected scenario output shape")
        if (
            not np.isfinite(normal).all()
            or not np.isfinite(viral).all()
            or (np.diff(normal, axis=1) <= 0).any()
            or (np.diff(viral, axis=1) <= 0).any()
            or (viral <= normal).any()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise RuntimeError("Exported scenario model failed smoke validation")

        sample.to_csv(staging / "sample_input.csv", index=False)
        (staging / "predict.py").write_text(PREDICT_CLI, encoding="utf-8")
        requirements = RUNTIME_REQUIREMENTS + (
            TITLE_RUNTIME_REQUIREMENTS if uses_title_encoder else ()
        )
        (staging / "requirements.txt").write_text(
            "\n".join(requirements) + "\n", encoding="utf-8"
        )
        schema = input_schema(preprocessor)
        runtime_versions = {
            **installed_runtime_versions(),
            "lightgbm": importlib.metadata.version("lightgbm"),
        }
        if uses_title_encoder:
            schema["expected_columns"] = [*schema["expected_columns"], "title"]
            schema["text_columns"] = ["title"]
            runtime_versions.update(
                {
                    "sentence-transformers": importlib.metadata.version(
                        "sentence-transformers"
                    ),
                    "torch": importlib.metadata.version("torch"),
                }
            )
        manifest = {
            "format_version": 1,
            "artifact_version": artifact_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": source_manifest["status"],
            "supported_horizons_days": [7, 14, 21, 30],
            "source_checkpoint": source_manifest["artifact_version"],
            "source_checkpoint_manifest_sha256": sha256_file(
                source_manifest_path
            ),
            "source_git_commit": source_git_commit(PROJECT_ROOT),
            "runtime_versions_used_for_export": runtime_versions,
            "input_schema": schema,
            "output_contract": source_manifest["output_contract"],
            "breakout_definition": source_manifest["breakout_definition"],
            "model": {
                "model_path": destination_model.relative_to(staging).as_posix(),
                "sha256": sha256_file(destination_model),
                "size_bytes": destination_model.stat().st_size,
            },
            "sample_smoke_output": json.loads(
                scenario.to_json(orient="records")
            )[0],
            "evaluation": {
                "classifier_reserved_test": source_manifest[
                    "classifier_reserved_test"
                ],
                "limitations": source_manifest["limitations"],
            },
        }
        if uses_title_encoder:
            manifest["title_embedding"] = {
                **source_manifest["embedding"],
                "packaged_model_path": "embedding_model",
                "correction": source_manifest["selected_text_configuration"],
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "\n".join(
                [
                    "# ViewCastLK two-scenario forecast",
                    "",
                    "Returns one normal trajectory, one calibrated breakout probability,",
                    "and one conditional viral-upside trajectory.",
                    *(
                        [
                            "The normal trajectory includes a frozen multilingual title-embedding correction.",
                        ]
                        if uses_title_encoder
                        else []
                    ),
                    "",
                    "```text",
                    "python predict.py --input sample_input.csv --output predictions.csv",
                    "```",
                    "",
                    "The viral-upside value is conditional on a breakout; it is not a",
                    "second equally likely point prediction.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            checksum_lines.append(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
            )
        (staging / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        shutil.move(str(staging), destination)

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in destination.rglob("*") if item.is_file()):
            archive.write(
                path,
                arcname=(
                    Path(artifact_version) / path.relative_to(destination)
                ).as_posix(),
            )
    checksum_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return destination, archive_path, checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "exports",
    )
    parser.add_argument("--artifact-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination, archive, checksum = export_artifact(
        source_dir=args.source_dir,
        output_root=args.output_root,
        artifact_version=args.artifact_version,
    )
    print(f"Export directory: {destination}")
    print(f"ZIP artifact: {archive}")
    print(f"ZIP SHA-256: {checksum.read_text(encoding='utf-8').split()[0]}")


if __name__ == "__main__":
    main()
