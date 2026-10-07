"""Package the published EDA into a runtime idea-optimization artifact.

The optimizer is deliberately separate from the view-forecast models.  This
script copies only the evaluated, creator-actionable comparisons needed by the
API and gives them an explicit version and evidence policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "dashboard" / "src" / "data" / "insights.json"
DEFAULT_OUTPUT = (
    REPO
    / "prediction_api"
    / "recommendation_artifacts"
    / "idea_optimization_20261001_v1.json"
)

DURATION_BANDS = {
    "Under 4 min": (0, 240),
    "4-8 min": (240, 480),
    "8-15 min": (480, 900),
    "15-30 min": (900, 1800),
    "30-60 min": (1800, 3600),
    "Over 1 hour": (3600, None),
}
TRAINING_TABLE_SHA256 = (
    "65d7c0f20214ea8324a14f012ef4fd1113a0dd87270368bcd9b1d6f3193e5ab9"
)


def clean_label(value: str) -> str:
    """Normalize the damaged dash present in the historical JSON export."""
    return value.replace("\ufffd", "-").replace("â€“", "-").replace("–", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    hour_blocks = []
    for index, cell in enumerate(source["timing"]["byHourBlock"]):
        hour_blocks.append(
            {
                **cell,
                "label": f"{index * 3:02d}:00-{(index + 1) * 3:02d}:00",
                "startHour": index * 3,
                "endHour": (index + 1) * 3,
            }
        )

    duration_by_category = []
    for category in source["format"]["durationByCategory"]:
        bands = []
        for cell in category["bands"]:
            label = clean_label(cell["label"])
            if label == "Shorts":
                bands.append({**cell, "label": label, "format": "short"})
                continue
            if label not in DURATION_BANDS:
                raise ValueError(f"Unknown duration band: {label}")
            minimum, maximum = DURATION_BANDS[label]
            bands.append(
                {
                    **cell,
                    "label": label,
                    "format": "long",
                    "minSeconds": minimum,
                    "maxSeconds": maximum,
                }
            )
        duration_by_category.append(
            {"category": category["category"], "bands": bands}
        )

    artifact = {
        "artifactVersion": "idea_optimization_20261001_v1",
        "generatedAt": source["generatedAt"],
        "source": {
            "kind": "within-channel day-7 EDA",
            "datasetPeriodStart": source["dataset"]["periodStart"],
            "datasetPeriodEnd": source["dataset"]["periodEnd"],
            "videos": source["dataset"]["videos"],
            "channels": source["dataset"]["channels"],
            "outcome": "Day-7 views relative to the same channel's normal",
            "trainingTableSha256": TRAINING_TABLE_SHA256,
            # Line endings normalised: a Windows checkout stores the file with
            # CRLF and Linux with LF, and the hash must match on both.
            "insightsSha256": hashlib.sha256(
                source_bytes.replace(b"\r\n", b"\n")
            ).hexdigest(),
        },
        "policy": {
            "minVideos": source["dataset"]["minVideos"],
            "minChannels": source["dataset"]["minChannels"],
            "minImprovementPctPoints": 5.0,
            "requireSeparatedIntervals": True,
            "causalityWarning": (
                "Historical associations do not prove that changing this choice "
                "will cause more views."
            ),
        },
        "timing": {
            "days": source["timing"]["byDay"],
            "hourBlocks": hour_blocks,
            "timeZone": "Asia/Colombo",
        },
        "duration": {"byCategory": duration_by_category},
        "format": {"shortsVsLongByCategory": source["format"]["byCategory"]},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
