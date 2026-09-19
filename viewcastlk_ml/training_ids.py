"""Training-video ID artifacts shared by model training and personalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


TRAINING_VIDEO_IDS_FILENAME = "training_video_ids.txt"


def normalize_training_video_ids(video_ids: Iterable[str]) -> list[str]:
    normalized = {str(video_id).strip() for video_id in video_ids}
    if "" in normalized:
        raise ValueError("Training video IDs must not be empty")
    return sorted(normalized)


def write_training_video_ids(video_ids: Iterable[str], path: Path) -> int:
    """Write an exact, deterministic one-ID-per-line model artifact."""
    normalized = normalize_training_video_ids(video_ids)
    if not normalized:
        raise ValueError("At least one training video ID is required")
    path.write_text("".join(f"{video_id}\n" for video_id in normalized), encoding="utf-8")
    return len(normalized)


def load_training_video_ids(path: Path) -> frozenset[str]:
    if not path.exists():
        raise FileNotFoundError(f"Training video ID artifact not found: {path}")
    identifiers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    normalized = normalize_training_video_ids(identifiers)
    if len(normalized) != len(identifiers):
        raise ValueError("Training video ID artifact contains duplicate IDs")
    return frozenset(normalized)
