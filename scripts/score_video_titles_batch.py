"""Submit and resume one strictly capped Gemini title-scoring Batch API proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from score_video_titles import (
    MODEL_ID,
    PROMPT_SHA256,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    TitleScores,
    append_score,
    build_title_request_text,
    read_manifest,
    read_scores,
)


BATCH_PROOF_VERSION = "viewcastlk-title-batch-proof-v1"
MAX_PROOF_REQUESTS = 100
COMPLETED_API_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
FINAL_LOCAL_STATUSES = {"MERGED", "MERGED_WITH_ERRORS"}

TITLE_SCORE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "urgency": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "emotional_appeal": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "seriousness": {"type": "INTEGER", "minimum": 0, "maximum": 10},
        "curiosity_gap": {"type": "INTEGER", "minimum": 0, "maximum": 10},
    },
    "required": ["urgency", "emotional_appeal", "seriousness", "curiosity_gap"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write resumable state without leaving a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def validate_state(state: dict[str, Any]) -> None:
    """Reject a state file produced by any different scoring contract."""
    expected = {
        "proof_version": BATCH_PROOF_VERSION,
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION,
    }
    for field, expected_value in expected.items():
        if state.get(field) != expected_value:
            raise ValueError(
                f"Batch state {field}={state.get(field)!r}, expected {expected_value!r}"
            )

    requests = state.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Batch state has no request mapping")
    if state.get("request_count") != len(requests):
        raise ValueError("Batch state request_count does not match its request mapping")
    keys = [request.get("key") for request in requests]
    video_ids = [request.get("video_id") for request in requests]
    if len(keys) != len(set(keys)) or None in keys:
        raise ValueError("Batch state contains missing or duplicate request keys")
    if len(video_ids) != len(set(video_ids)) or None in video_ids:
        raise ValueError("Batch state contains missing or duplicate video IDs")


def load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError("Batch state root must be an object")
    validate_state(state)
    return state


def select_pending_rows(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    if limit <= 0 or limit > MAX_PROOF_REQUESTS:
        raise ValueError(f"Batch proof limit must be from 1 through {MAX_PROOF_REQUESTS}")
    completed_ids = set(scores["video_id"].astype(str))
    return manifest.loc[~manifest["video_id"].astype(str).isin(completed_ids)].head(limit).copy()


def build_batch_request(key: str, manifest_row: pd.Series) -> dict[str, Any]:
    """Build one raw GenerateContent JSONL request using the frozen contract."""
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_title_request_text(manifest_row["title"])}],
                }
            ],
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "generationConfig": {
                "candidateCount": 1,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
                "responseSchema": TITLE_SCORE_RESPONSE_SCHEMA,
            },
        },
    }


def write_batch_jsonl(rows: pd.DataFrame, path: Path) -> list[dict[str, str]]:
    """Write API input and return the persistent key-to-video audit mapping."""
    request_mapping: list[dict[str, str]] = []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for position, (_, manifest_row) in enumerate(rows.iterrows(), start=1):
            key = f"title-{position:03d}"
            request = build_batch_request(key, manifest_row)
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            request_mapping.append(
                {
                    "key": key,
                    "video_id": str(manifest_row["video_id"]),
                    "title_sha256": str(manifest_row["title_sha256"]),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    return request_mapping


def api_state_name(batch_job: object) -> str:
    state = getattr(batch_job, "state", None)
    return str(getattr(state, "name", None) or state or "JOB_STATE_UNSPECIFIED")


def file_state_name(file: object) -> str:
    state = getattr(file, "state", None)
    return str(getattr(state, "name", None) or state or "STATE_UNSPECIFIED")


def wait_for_file_active(
    client: genai.Client,
    file_name: str,
    *,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 2.0,
) -> object:
    """Wait until a JSONL upload is ready before creating its Batch job."""
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("File readiness timeout and poll interval must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        uploaded_file = client.files.get(name=file_name)
        state = file_state_name(uploaded_file)
        if state == "ACTIVE":
            return uploaded_file
        if state == "FAILED":
            raise RuntimeError(
                f"Gemini File API processing failed for {file_name}: "
                f"{json_safe(getattr(uploaded_file, 'error', None))}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Gemini File API did not make {file_name} ACTIVE within {timeout_seconds:g}s; "
                f"last state was {state}"
            )
        time.sleep(poll_seconds)


def json_safe(value: object) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return str(value)


def submit_proof(
    client: genai.Client,
    manifest_path: Path,
    scores_path: Path,
    state_path: Path,
    limit: int,
) -> dict[str, Any]:
    if state_path.exists():
        raise FileExistsError(f"Refusing to submit a second proof while {state_path} exists")

    manifest = read_manifest(manifest_path)
    scores = read_scores(scores_path)
    rows = select_pending_rows(manifest, scores, limit)
    if rows.empty:
        raise ValueError("No pending titles remain for the Batch proof")

    submitted_at = utc_now()
    display_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory(prefix="viewcastlk-title-batch-") as temp_directory:
        request_path = Path(temp_directory) / "title-score-proof.jsonl"
        request_mapping = write_batch_jsonl(rows, request_path)
        uploaded_file = client.files.upload(
            file=request_path,
            config=types.UploadFileConfig(
                display_name=f"viewcastlk-title-score-proof-{display_stamp}",
                mime_type="jsonl",
            ),
        )
        input_file_name = str(getattr(uploaded_file, "name", "") or "")
        if not input_file_name:
            raise ValueError("Gemini File API returned no uploaded file name")
        wait_for_file_active(client, input_file_name)
        batch_job = client.batches.create(
            model=MODEL_ID,
            src=input_file_name,
            config={"display_name": f"viewcastlk-title-score-proof-{display_stamp}"},
        )

    job_name = str(getattr(batch_job, "name", "") or "")
    if not job_name:
        raise ValueError("Gemini Batch API returned no job name")

    state = {
        "proof_version": BATCH_PROOF_VERSION,
        "status": "SUBMITTED",
        "api_state": api_state_name(batch_job),
        "job_name": job_name,
        "input_file_name": input_file_name,
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "schema_version": SCHEMA_VERSION,
        "selection_strategy": "first pending rows in frozen manifest order",
        "request_count": len(request_mapping),
        "requests": request_mapping,
        "submitted_at_utc": submitted_at,
        "last_checked_at_utc": submitted_at,
    }
    validate_state(state)
    atomic_write_json(state_path, state)
    return {
        "action": "submitted",
        "job_name": job_name,
        "api_state": state["api_state"],
        "request_count": len(request_mapping),
    }


def extract_response_text(response: dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Batch response has no candidate text parts") from exc
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    if not text:
        raise ValueError("Batch response contains empty score text")
    return text


def parse_batch_output(
    output_bytes: bytes,
    state: dict[str, Any],
    existing_scores: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    """Validate the entire result file before any successful row is appended."""
    validate_state(state)
    expected = {request["key"]: request for request in state["requests"]}
    seen_keys: set[str] = set()
    parsed_rows: list[dict[str, object]] = []
    request_errors: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(output_bytes.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on Batch output line {line_number}") from exc
        key = str(item.get("key", ""))
        if key not in expected:
            raise ValueError(f"Batch output contains unexpected request key {key!r}")
        if key in seen_keys:
            raise ValueError(f"Batch output repeats request key {key!r}")
        seen_keys.add(key)

        if item.get("error"):
            request_errors.append({"key": key, "error": item["error"]})
            continue
        response = item.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"Batch output key {key!r} has neither a response nor an error")

        score_values = TitleScores.model_validate_json(extract_response_text(response))
        api_model_version = str(response.get("modelVersion", "") or "")
        if not api_model_version:
            raise ValueError(f"Batch output key {key!r} has no API model version")
        usage = response.get("usageMetadata") or {}
        mapping = expected[key]
        parsed_rows.append(
            {
                "video_id": mapping["video_id"],
                "title_sha256": mapping["title_sha256"],
                "model_id": MODEL_ID,
                "api_model_version": api_model_version,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "schema_version": SCHEMA_VERSION,
                "title_urgency": score_values.urgency,
                "title_emotional_appeal": score_values.emotional_appeal,
                "title_seriousness": score_values.seriousness,
                "title_curiosity_gap": score_values.curiosity_gap,
                "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "scored_at_utc": utc_now(),
            }
        )

    missing_keys = set(expected) - seen_keys
    if missing_keys:
        raise ValueError(f"Batch output is missing request keys: {sorted(missing_keys)}")

    returned_versions = {str(row["api_model_version"]) for row in parsed_rows}
    if len(returned_versions) > 1:
        raise ValueError(f"Batch output mixes API model versions: {sorted(returned_versions)}")
    existing_versions = set(existing_scores["api_model_version"].astype(str)) - {""}
    if existing_versions and returned_versions and existing_versions != returned_versions:
        raise ValueError(
            f"Batch API model version {sorted(returned_versions)} does not match "
            f"existing scores {sorted(existing_versions)}"
        )
    return parsed_rows, request_errors


def merge_batch_output(
    output_bytes: bytes,
    state: dict[str, Any],
    scores_path: Path,
) -> dict[str, Any]:
    existing = read_scores(scores_path)
    parsed_rows, request_errors = parse_batch_output(output_bytes, state, existing)
    completed_ids = set(existing["video_id"].astype(str))
    existing_hashes = dict(zip(existing["video_id"].astype(str), existing["title_sha256"].astype(str)))
    added = 0
    already_present = 0
    for row in parsed_rows:
        video_id = str(row["video_id"])
        if video_id in completed_ids:
            if existing_hashes.get(video_id) != str(row["title_sha256"]):
                raise ValueError(f"Existing score title hash changed for video {video_id!r}")
            already_present += 1
            continue
        append_score(scores_path, row)
        completed_ids.add(video_id)
        existing_hashes[video_id] = str(row["title_sha256"])
        added += 1

    return {
        "added_count": added,
        "already_present_count": already_present,
        "error_count": len(request_errors),
        "request_errors": request_errors,
    }


def poll_proof(
    client: genai.Client,
    scores_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    state = load_state(state_path)
    if state["status"] in FINAL_LOCAL_STATUSES:
        return {
            "action": "already_complete",
            "status": state["status"],
            "job_name": state["job_name"],
            "added_count": state.get("added_count", 0),
            "error_count": state.get("error_count", 0),
        }

    batch_job = client.batches.get(name=state["job_name"])
    api_state = api_state_name(batch_job)
    state["api_state"] = api_state
    state["last_checked_at_utc"] = utc_now()

    if api_state not in COMPLETED_API_STATES:
        state["status"] = "PROCESSING"
        atomic_write_json(state_path, state)
        return {"action": "polled", "job_name": state["job_name"], "api_state": api_state}

    if api_state != "JOB_STATE_SUCCEEDED":
        state["status"] = api_state
        state["job_error"] = json_safe(getattr(batch_job, "error", None))
        atomic_write_json(state_path, state)
        raise RuntimeError(f"Gemini Batch proof finished with {api_state}: {state['job_error']}")

    destination = getattr(batch_job, "dest", None)
    result_file_name = str(getattr(destination, "file_name", "") or "")
    state["result_file_name"] = result_file_name
    state["status"] = "DOWNLOADING"
    atomic_write_json(state_path, state)
    if not result_file_name:
        raise ValueError("Succeeded Batch job has no result file name")

    output_bytes = client.files.download(file=result_file_name)
    merge_summary = merge_batch_output(output_bytes, state, scores_path)
    state.update(merge_summary)
    state["result_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    state["completed_at_utc"] = utc_now()
    state["status"] = "MERGED_WITH_ERRORS" if merge_summary["error_count"] else "MERGED"
    atomic_write_json(state_path, state)
    return {
        "action": "merged",
        "job_name": state["job_name"],
        "api_state": api_state,
        "status": state["status"],
        **{key: merge_summary[key] for key in ("added_count", "already_present_count", "error_count")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "submit", "poll"), nargs="?", default="run")
    parser.add_argument("--manifest", type=Path, default=Path("Dataset/title_scoring_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("Dataset/title_scores.csv"))
    parser.add_argument("--state", type=Path, default=Path("Dataset/title_batch_proof_state.json"))
    parser.add_argument("--limit", type=int, default=MAX_PROOF_REQUESTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.command == "run" and args.state.exists():
        existing_state = load_state(args.state)
        if existing_state["status"] in FINAL_LOCAL_STATUSES:
            print(
                json.dumps(
                    {
                        "action": "already_complete",
                        "status": existing_state["status"],
                        "job_name": existing_state["job_name"],
                        "added_count": existing_state.get("added_count", 0),
                        "error_count": existing_state.get("error_count", 0),
                    },
                    indent=2,
                )
            )
            return

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key.strip():
        raise ValueError("GEMINI_API_KEY is empty")
    client = genai.Client(api_key=api_key)

    if args.command == "submit" or (args.command == "run" and not args.state.exists()):
        summary = submit_proof(client, args.manifest, args.out, args.state, args.limit)
    else:
        if not args.state.exists():
            raise FileNotFoundError(f"No Batch proof state exists at {args.state}")
        summary = poll_proof(client, args.out, args.state)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
