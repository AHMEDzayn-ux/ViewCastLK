from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_video_titles_batch as batch
from score_video_titles import MODEL_ID, PROMPT_SHA256, PROMPT_VERSION, SCHEMA_VERSION, SCORE_COLUMNS


def proof_state(requests: list[dict[str, str]]) -> dict[str, object]:
    return {
        "proof_version": batch.BATCH_PROOF_VERSION,
        "status": "SUBMITTED",
        "api_state": "JOB_STATE_PENDING",
        "job_name": "batches/test-job",
        "input_file_name": "files/test-input",
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION,
        "selection_strategy": "test",
        "request_count": len(requests),
        "requests": requests,
        "submitted_at_utc": "2026-08-16T00:00:00+00:00",
        "last_checked_at_utc": "2026-08-16T00:00:00+00:00",
    }


def successful_output(key: str, *, model_version: str = MODEL_ID) -> dict[str, object]:
    score_json = json.dumps(
        {"urgency": 2, "emotional_appeal": 4, "seriousness": 6, "curiosity_gap": 8}
    )
    return {
        "key": key,
        "response": {
            "candidates": [{"content": {"parts": [{"text": score_json}]}}],
            "modelVersion": model_version,
            "usageMetadata": {"promptTokenCount": 350, "candidatesTokenCount": 40},
        },
    }


class BatchScoringTests(unittest.TestCase):
    def test_request_uses_frozen_prompt_and_structured_schema(self) -> None:
        row = pd.Series({"video_id": "v1", "title": 'A title saying "now"', "title_sha256": "abc"})
        request = batch.build_batch_request("title-001", row)
        payload = request["request"]
        self.assertEqual(request["key"], "title-001")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], batch.SYSTEM_PROMPT)
        self.assertIn('Title JSON: "A title saying \\"now\\""', payload["contents"][0]["parts"][0]["text"])
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(
            set(payload["generationConfig"]["responseSchema"]["required"]),
            {"urgency", "emotional_appeal", "seriousness", "curiosity_gap"},
        )

    def test_pending_selection_excludes_completed_ids_and_is_capped(self) -> None:
        manifest = pd.DataFrame(
            {"video_id": ["v1", "v2", "v3"], "title": ["a", "b", "c"], "title_sha256": ["1", "2", "3"]}
        )
        scores = pd.DataFrame(columns=SCORE_COLUMNS)
        scores.loc[0] = ["v1", "1", MODEL_ID, MODEL_ID, PROMPT_VERSION, PROMPT_SHA256, SCHEMA_VERSION, 1, 2, 3, 4, 10, 5, "now"]
        pending = batch.select_pending_rows(manifest, scores, 1)
        self.assertEqual(pending["video_id"].tolist(), ["v2"])
        with self.assertRaises(ValueError):
            batch.select_pending_rows(manifest, scores, batch.MAX_PROOF_REQUESTS + 1)

    def test_jsonl_mapping_and_unicode_round_trip(self) -> None:
        rows = pd.DataFrame(
            {"video_id": ["v1"], "title": ["සිංහල මාතෘකාව"], "title_sha256": ["hash-1"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            mapping = batch.write_batch_jsonl(rows, path)
            decoded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(mapping, [{"key": "title-001", "video_id": "v1", "title_sha256": "hash-1"}])
        self.assertIn("සිංහල මාතෘකාව", decoded["request"]["contents"][0]["parts"][0]["text"])

    def test_batch_output_is_validated_and_merged_idempotently(self) -> None:
        requests = [
            {"key": "title-001", "video_id": "v1", "title_sha256": "hash-1"},
            {"key": "title-002", "video_id": "v2", "title_sha256": "hash-2"},
        ]
        state = proof_state(requests)
        output = "\n".join(
            json.dumps(successful_output(key)) for key in ("title-001", "title-002")
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            scores_path = Path(directory) / "scores.csv"
            first = batch.merge_batch_output(output, state, scores_path)
            second = batch.merge_batch_output(output, state, scores_path)
            scores = pd.read_csv(scores_path)
        self.assertEqual(first["added_count"], 2)
        self.assertEqual(second["added_count"], 0)
        self.assertEqual(second["already_present_count"], 2)
        self.assertEqual(scores["video_id"].tolist(), ["v1", "v2"])
        self.assertEqual(scores["title_curiosity_gap"].tolist(), [8, 8])

    def test_request_error_is_recorded_without_invalid_score(self) -> None:
        requests = [
            {"key": "title-001", "video_id": "v1", "title_sha256": "hash-1"},
            {"key": "title-002", "video_id": "v2", "title_sha256": "hash-2"},
        ]
        state = proof_state(requests)
        output = (
            json.dumps(successful_output("title-001"))
            + "\n"
            + json.dumps({"key": "title-002", "error": {"code": 429, "message": "quota"}})
        ).encode()
        rows, errors = batch.parse_batch_output(output, state, pd.DataFrame(columns=SCORE_COLUMNS))
        self.assertEqual(len(rows), 1)
        self.assertEqual(errors[0]["key"], "title-002")

    def test_missing_key_and_model_version_change_are_rejected(self) -> None:
        requests = [
            {"key": "title-001", "video_id": "v1", "title_sha256": "hash-1"},
            {"key": "title-002", "video_id": "v2", "title_sha256": "hash-2"},
        ]
        state = proof_state(requests)
        one_line = (json.dumps(successful_output("title-001")) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "missing request keys"):
            batch.parse_batch_output(one_line, state, pd.DataFrame(columns=SCORE_COLUMNS))

        existing = pd.DataFrame(columns=SCORE_COLUMNS)
        existing.loc[0] = ["old", "hash", MODEL_ID, MODEL_ID, PROMPT_VERSION, PROMPT_SHA256, SCHEMA_VERSION, 1, 2, 3, 4, 10, 5, "now"]
        changed = "\n".join(
            json.dumps(successful_output(key, model_version="different-version"))
            for key in ("title-001", "title-002")
        ).encode()
        with self.assertRaisesRegex(ValueError, "does not match existing scores"):
            batch.parse_batch_output(changed, state, existing)


if __name__ == "__main__":
    unittest.main()
