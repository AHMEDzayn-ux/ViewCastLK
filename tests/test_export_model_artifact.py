import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_model_artifact import export_artifact  # noqa: E402


SOURCE_ARTIFACT = PROJECT_ROOT / "artifacts" / "checkpoint10_wape_blended"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(
    (SOURCE_ARTIFACT / "training_manifest.json").exists(),
    "selected model checkpoint is unavailable",
)
class ModelArtifactExportTests(unittest.TestCase):
    def test_export_is_complete_loadable_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            export_dir, archive_path, checksum_path = export_artifact(
                output_root=output_root,
                artifact_version="test_candidate",
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
            self.assertFalse(manifest["evaluation"]["reserved_test_used"])
            self.assertFalse(
                manifest["input_schema"]["llm_scores_enabled_in_this_artifact"]
            )
            self.assertEqual(len(manifest["models"]), 4)

            for record in manifest["models"]:
                model_path = export_dir / record["model_path"]
                self.assertEqual(sha256_file(model_path), record["sha256"])

            checksums = {}
            for line in (export_dir / "SHA256SUMS.txt").read_text(
                encoding="utf-8"
            ).splitlines():
                digest, relative_path = line.split("  ", maxsplit=1)
                checksums[relative_path] = digest
            for relative_path, expected_digest in checksums.items():
                self.assertEqual(
                    sha256_file(export_dir / relative_path), expected_digest
                )

            output_csv = output_root / "predictions.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(export_dir / "predict.py"),
                    "--horizon",
                    "7",
                    "--input",
                    str(export_dir / "sample_input.csv"),
                    "--output",
                    str(output_csv),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            predictions = pd.read_csv(output_csv)
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions.loc[0, "prediction_horizon_days"], 7)
            self.assertGreaterEqual(predictions.loc[0, "predicted_views"], 0)

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn("test_candidate/manifest.json", names)
            self.assertIn("test_candidate/predict.py", names)
            self.assertIn("test_candidate/models/day_30_ensemble.joblib", names)

    def test_export_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            (output_root / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                export_artifact(
                    output_root=output_root,
                    artifact_version="existing",
                )


if __name__ == "__main__":
    unittest.main()
