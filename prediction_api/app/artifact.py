"""Which trained model the API serves.

Every module that needs the artefact reads it from here: the loader, the feature
builder (which imports the artefact's own viewcastlk_ml package), and
personalisation (which reads the artefact's training_video_ids.txt). Swapping
models is a change to ACTIVE_ARTIFACT_VERSION and nothing else.

viewcastlk_reconciled_latest_clean_20261001_v10 replaced
viewcastlk_reconciled_latest_clean_20260918_v9, which replaced
viewcastlk_monotonic_trajectory_experimental_v1. The earlier model was trained
before any video had all four horizon labels, so its day-14, 21 and 30
forecasts were never checked against outcomes, and it saw only the channel's
size. The active release uses the 1 October 2026 table, reads the channel's own
recent record, and passed both reserved-channel and forward date-split
evaluation. Previous folders remain beside it for comparison and rollback.
"""

from __future__ import annotations

from pathlib import Path

ACTIVE_ARTIFACT_VERSION = "viewcastlk_reconciled_latest_clean_20261001_v10"
V9_ARTIFACT_VERSION = "viewcastlk_reconciled_latest_clean_20260918_v9"
V8_ARTIFACT_VERSION = "viewcastlk_viral_scenario_ensemble_20260915_v8"
BREAKOUT_ARTIFACT_VERSION = "viewcastlk_viral_scenario_ensemble_20261001_v10"
V9_BREAKOUT_MODEL_VERSION = f"{V9_ARTIFACT_VERSION}+{V8_ARTIFACT_VERSION}-breakout"
V10_BREAKOUT_MODEL_VERSION = (
    f"{ACTIVE_ARTIFACT_VERSION}+{BREAKOUT_ARTIFACT_VERSION}-breakout"
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "model_artifacts" / ACTIVE_ARTIFACT_VERSION
)

V8_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "model_artifacts" / V8_ARTIFACT_VERSION
)

V9_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "model_artifacts" / V9_ARTIFACT_VERSION
)

BREAKOUT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "model_artifacts"
    / BREAKOUT_ARTIFACT_VERSION
)
