"""Evidence-gated publishing guidance, isolated from model inference.

This module never changes a forecast or breakout probability.  It compares a
creator's submitted plan with a versioned EDA artifact. Strong changes still
require conservative release rules, while every supported dimension returns a
clearly labelled review: change, aligned choice, or exploratory benchmark.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from app.schemas import ForecastRequest, UnavailableRecommendation


ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "recommendation_artifacts"
    / "idea_optimization_20261001_v1.json"
)


class RecommendationArtifactError(RuntimeError):
    """The released recommendation evidence cannot be loaded or validated."""


@lru_cache(maxsize=1)
def load_recommendation_artifact() -> dict[str, Any]:
    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        required = {"artifactVersion", "policy", "timing", "duration", "format"}
        if not required.issubset(artifact):
            raise ValueError("missing required sections")
        return artifact
    except Exception as exc:
        raise RecommendationArtifactError(
            "The released historical recommendation evidence is unavailable."
        ) from exc


def _supported(cell: dict[str, Any], policy: dict[str, Any]) -> bool:
    return (
        int(cell["videos"]) >= int(policy["minVideos"])
        and int(cell["channels"]) >= int(policy["minChannels"])
    )


def _clearly_better(
    candidate: dict[str, Any], current: dict[str, Any], policy: dict[str, Any]
) -> bool:
    improvement = float(candidate["effectPct"]) - float(current["effectPct"])
    if improvement < float(policy["minImprovementPctPoints"]):
        return False
    if not policy.get("requireSeparatedIntervals", True):
        return True
    return float(candidate["lowPct"]) > float(current["highPct"])


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _effect_evidence(cell: dict[str, Any]) -> str:
    return (
        f"{_pct(float(cell['effectPct']))} relative Day-7 performance "
        f"(95% range {_pct(float(cell['lowPct']))} to "
        f"{_pct(float(cell['highPct']))}); {int(cell['videos']):,} videos "
        f"from {int(cell['channels']):,} channels."
    )


def _unavailable(kind: str, reason: str) -> UnavailableRecommendation:
    return UnavailableRecommendation(type=kind, reason=reason)


def _timing_guidance(
    payload: ForecastRequest, artifact: dict[str, Any]
) -> tuple[dict[str, Any] | None, UnavailableRecommendation | None]:
    if payload.plannedPublishDay is None or payload.plannedPublishHour is None:
        return None, _unavailable(
            "timing",
            "Choose both a planned day and hour to compare your timing with the historical evidence.",
        )

    policy = artifact["policy"]
    days = [cell for cell in artifact["timing"]["days"] if _supported(cell, policy)]
    blocks = [
        cell for cell in artifact["timing"]["hourBlocks"] if _supported(cell, policy)
    ]
    current_day = next(
        (cell for cell in days if cell["label"] == payload.plannedPublishDay), None
    )
    current_block = next(
        (
            cell
            for cell in blocks
            if int(cell["startHour"]) <= payload.plannedPublishHour < int(cell["endHour"])
        ),
        None,
    )
    if current_day is None or current_block is None:
        return None, _unavailable(
            "timing", "The submitted publishing window has insufficient comparison data."
        )

    best_day = max(days, key=lambda cell: float(cell["effectPct"]))
    best_block = max(blocks, key=lambda cell: float(cell["effectPct"]))
    improve_day = _clearly_better(best_day, current_day, policy)
    improve_block = _clearly_better(best_block, current_block, policy)

    if improve_day or improve_block:
        status = "change"
        title = "Test a stronger historical publishing window"
        recommended_day = best_day if improve_day else current_day
        recommended_block = best_block if improve_block else current_block
        changed = []
        if improve_day:
            changed.append(f"day from {current_day['label']} to {best_day['label']}")
        if improve_block:
            changed.append(f"time from {current_block['label']} to {best_block['label']}")
        guidance = (
            f"Consider changing the {' and '.join(changed)}. The evidence cleared the "
            "release threshold, but this remains a planning test rather than a guaranteed lift."
        )
    elif (
        current_day["label"] == best_day["label"]
        and current_block["label"] == best_block["label"]
    ):
        status = "aligned"
        title = "Your publishing window already matches the strongest benchmark"
        recommended_day = current_day
        recommended_block = current_block
        guidance = (
            "The selected day and time are the strongest observed groups in the released "
            "timing EDA. There is no evidence-backed reason to change this part of the plan."
        )
    else:
        status = "benchmark"
        title = "Use the strongest observed window as an optional test"
        recommended_day = best_day
        recommended_block = best_block
        guidance = (
            f"The strongest observed combination was {best_day['label']}, "
            f"{best_block['label']} SLT, but its confidence ranges do not clearly establish "
            f"that it beats your {current_day['label']}, {current_block['label']} plan. "
            "Treat it as an A/B-test idea, not a predicted improvement."
        )

    return {
        "id": f"{artifact['artifactVersion']}-timing",
        "type": "timing",
        "status": status,
        "title": title,
        "guidance": guidance,
        "recommendedPublishingWindow": {
            "day": recommended_day["label"],
            "startHour": int(recommended_block["startHour"]),
            "endHour": int(recommended_block["endHour"]),
            "timeZone": artifact["timing"]["timeZone"],
        },
        "evidence": [
            {
                "label": f"Recommended day - {recommended_day['label']}",
                "detail": _effect_evidence(recommended_day),
            },
            {
                "label": f"Recommended time - {recommended_block['label']} SLT",
                "detail": _effect_evidence(recommended_block),
            },
            {
                "label": "Submitted plan",
                "detail": (
                    f"{current_day['label']}, {current_block['label']} SLT. "
                    f"Day evidence: {_effect_evidence(current_day)} Time evidence: "
                    f"{_effect_evidence(current_block)}"
                ),
            },
        ],
    }, None


def _duration_band(
    duration_seconds: float, bands: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for band in bands:
        if band.get("format") != "long":
            continue
        maximum = band.get("maxSeconds")
        if duration_seconds >= float(band["minSeconds"]) and (
            maximum is None or duration_seconds < float(maximum)
        ):
            return band
    return None


def _duration_guidance(
    payload: ForecastRequest, artifact: dict[str, Any], is_short: bool
) -> tuple[dict[str, Any] | None, UnavailableRecommendation | None]:
    if is_short:
        return None, _unavailable(
            "duration",
            "The EDA treats Shorts as one format and does not support duration advice within Shorts.",
        )

    category = next(
        (
            item
            for item in artifact["duration"]["byCategory"]
            if item["category"] == payload.category
        ),
        None,
    )
    if category is None:
        return None, _unavailable(
            "duration",
            f"There is not enough category-specific duration evidence for {payload.category}.",
        )

    policy = artifact["policy"]
    long_bands = [
        cell
        for cell in category["bands"]
        if cell.get("format") == "long" and _supported(cell, policy)
    ]
    current = _duration_band(payload.durationSeconds, long_bands)
    best = max(long_bands, key=lambda cell: float(cell["effectPct"]))
    if current is None:
        return {
            "id": f"{artifact['artifactVersion']}-duration",
            "type": "duration",
            "status": "benchmark",
            "title": f"Use {best['label']} as the available category benchmark",
            "guidance": (
                f"Your exact duration band does not have enough {payload.category} videos "
                f"for a reliable comparison. Among the supported long-form groups, "
                f"{best['label']} had the strongest observed result. Test that range only "
                "if it suits the content; the data cannot show that it beats your current length."
            ),
            "evidence": [
                {
                    "label": f"Available benchmark - {best['label']}",
                    "detail": _effect_evidence(best),
                },
                {
                    "label": "Current duration",
                    "detail": (
                        f"{payload.durationSeconds / 60:.1f} minutes; its category band did "
                        "not meet the minimum video and channel counts."
                    ),
                },
            ],
        }, None

    if best["label"] == current["label"]:
        status = "aligned"
        title = f"Your duration already matches the strongest {payload.category} benchmark"
        guidance = (
            f"The planned duration sits in the {current['label']} band, which had the "
            "strongest observed Day-7 result among supported long-form groups in this category."
        )
    elif _clearly_better(best, current, policy):
        status = "change"
        title = f"Test the {best['label']} duration range"
        guidance = (
            f"Your planned duration is in the {current['label']} band. In {payload.category}, "
            f"the {best['label']} band had clearly stronger historical Day-7 performance. "
            "Change the length only if the idea still works naturally in that range."
        )
    else:
        status = "benchmark"
        title = f"Compare your plan with the {best['label']} benchmark"
        guidance = (
            f"The {best['label']} band had the strongest observed result in {payload.category}, "
            f"but the evidence does not clearly establish that it beats your {current['label']} "
            "plan. Keep the current length or test the benchmark based on creative fit."
        )

    return {
        "id": f"{artifact['artifactVersion']}-duration",
        "type": "duration",
        "status": status,
        "title": title,
        "guidance": guidance,
        "evidence": [
            {
                "label": f"Suggested band - {best['label']}",
                "detail": _effect_evidence(best),
            },
            {
                "label": f"Current band - {current['label']}",
                "detail": _effect_evidence(current),
            },
        ],
    }, None


def _format_guidance(
    payload: ForecastRequest, artifact: dict[str, Any], is_short: bool
) -> tuple[dict[str, Any] | None, UnavailableRecommendation | None]:
    policy = artifact["policy"]
    contrast = next(
        (
            cell
            for cell in artifact["format"]["shortsVsLongByCategory"]
            if cell["label"] == payload.category and _supported(cell, policy)
        ),
        None,
    )
    if contrast is None:
        return None, _unavailable(
            "format",
            f"There is not enough category-specific format evidence for {payload.category}.",
        )

    effect = float(contrast["effectPct"])
    minimum = float(policy["minImprovementPctPoints"])
    shorts_clearly_better = effect >= minimum and float(contrast["lowPct"]) > 0
    long_clearly_better = effect <= -minimum and float(contrast["highPct"]) < 0
    should_change = (not is_short and shorts_clearly_better) or (
        is_short and long_clearly_better
    )
    current = "Short" if is_short else "standard video"
    if should_change:
        status = "change"
        suggested = "Short" if shorts_clearly_better else "standard video"
        title = f"Consider testing the idea as a {suggested}"
        guidance = (
            f"This plan is currently a {current}. {payload.category} uploads showed a clear "
            f"historical format difference. Switch only if the concept and production goals fit a {suggested}."
        )
    elif (is_short and shorts_clearly_better) or (not is_short and long_clearly_better):
        status = "aligned"
        title = f"Your {current} format already aligns with the category evidence"
        guidance = (
            f"The released {payload.category} comparison supports keeping the current format. "
            "There is no evidence-backed reason to switch formats for this plan."
        )
    else:
        status = "benchmark"
        observed = "Shorts" if effect >= 0 else "standard videos"
        title = "Choose format based on the creative fit"
        guidance = (
            f"{observed} had the higher point estimate in {payload.category}, but the confidence "
            "range includes no difference. The EDA does not support changing your current format."
        )

    comparison = (
        "Shorts compared with standard videos: " + _effect_evidence(contrast)
    )
    return {
        "id": f"{artifact['artifactVersion']}-format",
        "type": "format",
        "status": status,
        "title": title,
        "guidance": guidance,
        "evidence": [
            {"label": "Category format comparison", "detail": comparison},
        ],
    }, None


def optimize_idea(
    payload: ForecastRequest,
) -> tuple[list[dict[str, Any]], list[UnavailableRecommendation], str]:
    """Return independent EDA guidance and its artifact version."""
    artifact = load_recommendation_artifact()
    is_short = payload.isShort
    if is_short is None:
        is_short = payload.durationSeconds <= 60

    recommendations: list[dict[str, Any]] = []
    unavailable: list[UnavailableRecommendation] = []
    for result, missing in (
        _timing_guidance(payload, artifact),
        _duration_guidance(payload, artifact, is_short),
        _format_guidance(payload, artifact, is_short),
    ):
        if result is not None:
            recommendations.append(result)
        if missing is not None:
            unavailable.append(missing)
    return recommendations, unavailable, str(artifact["artifactVersion"])
