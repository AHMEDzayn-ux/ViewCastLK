from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_monotonic_trajectory_artifact import (  # noqa: E402
    export_artifact,
)


SOURCE_ARTIFACT = (
    PROJECT_ROOT / "artifacts" / "checkpoint12_monotonic_trajectory"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(
    (SOURCE_ARTIFACT / "training_manifest.json").exists(),
    "monotonic trajectory checkpoint is unavailable",
)
class MonotonicTrajectoryArtifactExportTests(unittest.TestCase):
    def test_export_is_complete_and_cli_predictions_are_monotonic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            export_dir, archive_path, checksum_path = export_artifact(
                output_root=output_root,
                artifact_version="test_monotonic_trajectory",
            )

            self.assertTrue(export_dir.is_dir())
            self.assertTrue(archive_path.is_file())
            self.assertEqual(
                checksum_path.read_text(encoding="utf-8").split()[0],
                sha256_file(archive_path),
            )
            manifest = json.loads(
                (export_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["supported_horizons_days"], [7, 14, 21, 30])
            self.assertEqual(
                manifest["trajectory_guarantee"],
                "day_7 <= day_14 <= day_21 <= day_30",
            )
            self.assertEqual(
                manifest["model"]["sha256"],
                sha256_file(export_dir / manifest["model"]["model_path"]),
            )
            self.assertFalse(
                manifest["evaluation"]["end_to_end_day_30_testable"]
            )

            checksums = {}
            for line in (export_dir / "SHA256SUMS.txt").read_text(
                encoding="utf-8"
            ).splitlines():
                digest, relative_path = line.split("  ", maxsplit=1)
                checksums[relative_path] = digest
            for relative_path, expected in checksums.items():
                self.assertEqual(
                    sha256_file(export_dir / relative_path), expected
                )

            output_csv = output_root / "predictions.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(export_dir / "predict.py"),
                    "--input",
                    str(export_dir / "sample_input.csv"),
                    "--output",
                    str(output_csv),
                ],
                cwd=export_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            predictions = pd.read_csv(output_csv)[
                [
                    "predicted_day_7_views",
                    "predicted_day_14_views",
                    "predicted_day_21_views",
                    "predicted_day_30_views",
                ]
            ].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(predictions).all())
            self.assertTrue((predictions >= 0).all())
            self.assertTrue((np.diff(predictions, axis=1) >= 0).all())

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            prefix = "test_monotonic_trajectory/"
            self.assertIn(prefix + "manifest.json", names)
            self.assertIn(prefix + "models/monotonic_trajectory.joblib", names)
            self.assertIn(prefix + "predict.py", names)

    def test_export_refuses_to_overwrite_existing_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            export_artifact(
                output_root=output_root,
                artifact_version="no_overwrite",
            )
            with self.assertRaises(FileExistsError):
                export_artifact(
                    output_root=output_root,
                    artifact_version="no_overwrite",
                )


if __name__ == "__main__":
    unittest.main()
