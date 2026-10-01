"""Which trained model the API serves.

Every module that needs the artefact reads it from here: the loader, the feature
builder (which imports the artefact's own viewcastlk_ml package), and
personalisation (which reads the artefact's training_video_ids.txt). Swapping
models is a change to ACTIVE_ARTIFACT_VERSION and nothing else.

viewcastlk_reconciled_latest_clean_20260918_v9 replaced
viewcastlk_monotonic_trajectory_experimental_v1. The earlier model was trained
before any video had all four horizon labels, so its day-14, 21 and 30
forecasts were never checked against outcomes, and it saw only the channel's
size. This one was trained on 36,484 videos with complete labels, reads the
channel's own recent record, and passed its release gate on reserved channels.
The previous folder is kept beside it so a rollback is one line.
"""

from __future__ import annotations

from pathlib import Path

ACTIVE_ARTIFACT_VERSION = "viewcastlk_reconciled_latest_clean_20260918_v9"

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "model_artifacts" / ACTIVE_ARTIFACT_VERSION
)
