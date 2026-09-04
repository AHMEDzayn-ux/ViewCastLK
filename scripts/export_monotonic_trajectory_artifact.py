"""Export the experimental monotonic trajectory as a portable artifact."""

from __future__ import annotations

import argparse
import json
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

from scripts.export_model_artifact import (  # noqa: E402
    RUNTIME_MODULES,
    VERSION_PATTERN,
    input_schema,
    installed_runtime_versions,
    sample_input_frame,
    sha256_file,
    source_git_commit,
)


DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "checkpoint12_monotonic_trajectory"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "exports"
DEFAULT_ARTIFACT_VERSION = "viewcastlk_monotonic_trajectory_experimental_v1"
RUNTIME_REQUIREMENTS = (
    "numpy>=2,<3",
    "pandas>=2.2,<4",
    "scikit-learn>=1.5,<2",
    "xgboost>=3,<4",
    "lightgbm>=4.7,<5",
    "joblib>=1.4,<2",
)
EVALUATION_FILES = (
    "training_manifest.json",
    "transition_test_metrics.csv",
    "triple_horizon_test_metrics.csv",
    "overlap_summary.csv",
    "trained_components.csv",
    "validation_tests.csv",
    "sample_predictions.csv",
)


PREDICT_CLI = '''"""Predict one monotonic ViewCastLK view trajectory from CSV rows."""

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


def load_model():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["model"]
    model_path = ROOT / record["model_path"]
    if sha256_file(model_path) != record["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {model_path.name}")
    return joblib.load(model_path), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, manifest = load_model()
    frame = pd.read_csv(args.input, low_memory=False)
    expected = manifest["input_schema"]["expected_columns"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing columns: " + ", ".join(missing))

    predictions = np.asarray(model.predict_views(frame), dtype=float)
    horizons = manifest["supported_horizons_days"]
    if predictions.shape != (len(frame), len(horizons)):
        raise RuntimeError("Model returned an unexpected prediction shape")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise RuntimeError("Model returned invalid predictions")
    if (np.diff(predictions, axis=1) < -1e-12).any():
        raise RuntimeError("Model returned a decreasing cumulative trajectory")

    output = frame.copy()
    for position, horizon in enumerate(horizons):
        output[f"predicted_day_{horizon}_views"] = predictions[:, position]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} monotonic trajectories to {args.output}")


if __name__ == "__main__":
    main()
'''


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def model_card(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ViewCastLK monotonic trajectory — experimental artifact",
            "",
            "This artifact predicts cumulative view totals for days 7, 14, 21,",
            "and 30 in one call. Its day-7 base plus nonnegative growth",
            "parameterization guarantees a nondecreasing trajectory.",
            "",
            "## Status",
            "",
            "Experimental only. No video in the frozen training dataset has all",
            "four labels, so end-to-end day-30 accuracy is not measurable yet.",
            "The included experimental channel holdout has already been evaluated.",
            "",
            "## Usage",
            "",
            "```text",
            "python -m pip install -r requirements.txt",
            "python predict.py --input sample_input.csv --output predictions.csv",
            "```",
            "",
            "The output adds predicted_day_7_views, predicted_day_14_views,",
            "predicted_day_21_views, and predicted_day_30_views. See manifest.json",
            "and evaluation/ for the limitations and held-out results.",
            "",
            f"Source checkpoint: {manifest['source_checkpoint']}",
        ]
    ) + "\n"


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
    conflicts = [
        path
        for path in (destination, archive_path, archive_checksum_path)
        if path.exists()
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
    if source_manifest.get("artifact_version") != (
        "checkpoint12_monotonic_trajectory"
    ):
        raise ValueError("Source is not a monotonic trajectory checkpoint")
    if source_manifest.get("complete_four_horizon_rows") != 0:
        raise ValueError("Unexpected complete-label count in source manifest")

    source_model = source_dir / source_manifest["model_path"]
    if sha256_file(source_model) != source_manifest["model_sha256"]:
        raise RuntimeError("Source trajectory model checksum mismatch")
    for filename in EVALUATION_FILES:
        if not (source_dir / filename).is_file():
            raise FileNotFoundError(f"Missing source artifact file: {filename}")

    transition_metrics = pd.read_csv(
        source_dir / "transition_test_metrics.csv"
    )
    triple_metrics = pd.read_csv(
        source_dir / "triple_horizon_test_metrics.csv"
    )

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

        destination_model = models_dir / "monotonic_trajectory.joblib"
        shutil.copy2(source_model, destination_model)
        for module_name in RUNTIME_MODULES:
            shutil.copy2(
                PROJECT_ROOT / "viewcastlk_ml" / module_name,
                runtime_dir / module_name,
            )
        for filename in EVALUATION_FILES:
            shutil.copy2(source_dir / filename, evaluation_dir / filename)

        sample = sample_input_frame()
        bundle = joblib.load(destination_model)
        smoke = np.asarray(bundle.predict_views(sample), dtype=float)
        if smoke.shape != (1, 4):
            raise RuntimeError("Exported model failed trajectory shape smoke test")
        if (
            not np.isfinite(smoke).all()
            or (smoke < 0).any()
            or (np.diff(smoke, axis=1) < -1e-12).any()
        ):
            raise RuntimeError("Exported model failed monotonic smoke test")

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
            "supported_horizons_days": [7, 14, 21, 30],
            "prediction_interface": (
                "predict.py CSV CLI or trajectory_bundle.predict_views(DataFrame)"
            ),
            "trajectory_guarantee": "day_7 <= day_14 <= day_21 <= day_30",
            "source_checkpoint": source_manifest["artifact_version"],
            "source_checkpoint_manifest_sha256": sha256_file(
                source_manifest_path
            ),
            "source_git_commit": source_git_commit(PROJECT_ROOT),
            "runtime_versions_used_for_export": installed_runtime_versions(),
            "input_schema": input_schema(),
            "model": {
                "model_path": destination_model.relative_to(staging).as_posix(),
                "sha256": sha256_file(destination_model),
                "size_bytes": destination_model.stat().st_size,
                "construction": source_manifest["construction"],
                "components": source_manifest["components"],
            },
            "evaluation": {
                "common_split": source_manifest["common_split"],
                "experimental_test_used": True,
                "complete_four_horizon_rows": 0,
                "end_to_end_day_30_testable": False,
                "transition_test_metrics": json_records(transition_metrics),
                "complete_day_7_14_21_test_metrics": json_records(
                    triple_metrics
                ),
                "limitations": source_manifest["limitations"],
            },
            "sample_smoke_prediction": {
                f"day_{horizon}_views": float(smoke[0, position])
                for position, horizon in enumerate((7, 14, 21, 30))
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            model_card(manifest), encoding="utf-8"
        )

        checksum_lines = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            checksum_lines.append(f"{sha256_file(path)}  {relative}")
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
    archive_checksum_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
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
    print(
        "ZIP SHA-256: "
        f"{checksum_path.read_text(encoding='utf-8').split()[0]}"
    )


if __name__ == "__main__":
    main()
