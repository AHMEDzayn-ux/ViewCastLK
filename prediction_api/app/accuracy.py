"""Measured accuracy for the served model, when there is any to publish.

The figures live in published_accuracy.json beside the model they describe,
and are only shown while that same model is served. A swap to a model that has
not been measured falls back to the honest "not available yet" answer instead
of showing another model's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.artifact import ARTIFACT_DIR
from app.schemas import AvailableAccuracyResponse

PUBLISHED_ACCURACY_PATH = ARTIFACT_DIR / "published_accuracy.json"


def load_published_accuracy(
    model_name: str, path: Path = PUBLISHED_ACCURACY_PATH
) -> AvailableAccuracyResponse | None:
    """The published results for model_name, or None when there are none.

    A missing, unreadable or malformed file, or one written for a different
    model, all mean the same thing to a visitor: nothing has been measured for
    the model answering their forecasts.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("modelName") != model_name:
        return None
    try:
        return AvailableAccuracyResponse(**data)
    except ValidationError:
        return None
