from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from app import config
from app.creator_analytics import (
    CreatorSyncException,
    CreatorVideo,
    cumulative_horizons,
    fetch_daily_analytics,
    fetch_upload_video_ids,
    fetch_video_metadata,
)
from app.feature_builder import build_candidate_feature_frame
from app.youtube_oauth import YouTubeChannelIdentity


TRAINING_IDS_PATH = (
    Path(__file__).resolve().parent.parent
    / "model_artifacts"
    / "viewcastlk_monotonic_trajectory_experimental_v1"
    / "training_video_ids.txt"
)
COLOMBO = ZoneInfo("Asia/Colombo")


def load_current_training_video_ids(path: Path = TRAINING_IDS_PATH) -> frozenset[str]:
    if not path.exists():
        raise CreatorSyncException("The model training-ID artifact is unavailable")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise CreatorSyncException("The model training-ID artifact is invalid")
    return frozenset(values)


def compute_adjustments(
    *, records: list[dict[str, Any]], training_video_ids: frozenset[str], model_version: str
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in records
        if row["video_id"] not in training_video_ids
        and row.get("model_version") == model_version
    ]
    output: list[dict[str, Any]] = []
    for horizon in (7, 14, 21, 30):
        for format_name in ("all", "short", "long"):
            candidates = [
                row
                for row in eligible
                if row.get(f"d{horizon}") is not None
                and row.get(f"pred{horizon}") is not None
                and (
                    format_name == "all"
                    or (format_name == "short" and row["is_short"])
                    or (format_name == "long" and not row["is_short"])
                )
            ]
            candidates.sort(key=lambda row: row["published_at"], reverse=True)
            candidates = candidates[:40]
            n_videos = len(candidates)
            if format_name != "all" and n_videos < 5:
                continue
            if n_videos == 0:
                factor = 1.0
            else:
                log_ratios = [
                    math.log1p(float(row[f"d{horizon}"]))
                    - math.log1p(float(row[f"pred{horizon}"]))
                    for row in candidates
                ]
                factor = math.exp(median(log_ratios) * n_videos / (n_videos + 5))
            output.append(
                {
                    "horizon": horizon,
                    "format": format_name,
                    "factor": factor,
                    "n_videos": n_videos,
                }
            )
    return output


def apply_forecast_adjustments(
    *,
    shared_predictions: dict[int, int],
    adjustment_rows: list[dict[str, Any]],
    requested_format: str,
    model_version: str,
) -> tuple[dict[int, int], dict[str, Any]]:
    matching = [
        row
        for row in adjustment_rows
        if row.get("model_version") == model_version
        and row.get("horizon") in (7, 14, 21, 30)
        and row.get("format") in ("all", "short", "long")
        and float(row.get("factor", 0)) > 0
    ]
    personalized: dict[int, int] = {}
    used: list[dict[str, Any]] = []
    for horizon in (7, 14, 21, 30):
        candidates = [row for row in matching if row["horizon"] == horizon]
        selected = next(
            (row for row in candidates if row["format"] == requested_format),
            None,
        ) or next((row for row in candidates if row["format"] == "all"), None)
        factor = float(selected["factor"]) if selected else 1.0
        personalized[horizon] = max(
            0, int(round(float(shared_predictions[horizon]) * factor))
        )
        if selected:
            used.append(
                {
                    "horizonDays": horizon,
                    "format": selected["format"],
                    "factor": factor,
                    "nVideos": int(selected["n_videos"]),
                }
            )

    applied = any(row["nVideos"] > 0 for row in used)
    if not applied:
        personalized = dict(shared_predictions)
    return personalized, {
        "applied": applied,
        "format": requested_format,
        "modelVersion": model_version,
        "sharedEstimates": [
            {"horizonDays": horizon, "cumulativeViews": shared_predictions[horizon]}
            for horizon in (7, 14, 21, 30)
        ],
        "adjustments": used if applied else [],
    }


def _historical_feature_frame(
    *, video: CreatorVideo, position: int, channel: YouTubeChannelIdentity
):
    published_local = video.published_at.astimezone(COLOMBO)
    channel_created = channel.published_at
    channel_age_days = None
    if channel_created is not None:
        channel_age_days = max(0, (video.published_at - channel_created).days)
    request = {
        "category": video.category,
        "durationSeconds": video.duration_seconds,
        "audioLanguage": video.audio_language,
        "plannedPublishDay": published_local.strftime("%A"),
        "plannedPublishHour": published_local.hour,
        "is_short": video.is_short,
    }
    # Historical subscriber/view totals are unavailable from the Data API. Keep
    # them missing instead of leaking current future state into past predictions.
    channel_stats = {
        "ch_subs_at_publish": None,
        "ch_views_at_publish": None,
        "ch_videos_at_publish": position,
        "channel_age_days_at_publish": channel_age_days,
    }
    return build_candidate_feature_frame(request, channel_stats)


async def synchronize_creator_history(
    *,
    user_id: str,
    access_token: str,
    channel: YouTubeChannelIdentity,
    store,
    model_registry,
) -> None:
    if not channel.uploads_playlist_id:
        raise CreatorSyncException("The channel uploads playlist was unavailable")
    video_ids = await fetch_upload_video_ids(
        access_token=access_token,
        uploads_playlist_id=channel.uploads_playlist_id,
        limit=config.CREATOR_HISTORY_VIDEO_LIMIT,
    )
    videos = await fetch_video_metadata(access_token=access_token, video_ids=video_ids)
    videos.sort(key=lambda video: video.published_at)
    analytics = await fetch_daily_analytics(access_token=access_token, videos=videos)
    model_version = model_registry.get_manifest().get("artifact_version")
    if not isinstance(model_version, str) or not model_version:
        raise CreatorSyncException("The active model version is unavailable")

    records: list[dict[str, Any]] = []
    for position, video in enumerate(videos, start=1):
        trajectory = model_registry.predict_trajectory(
            _historical_feature_frame(video=video, position=position, channel=channel)
        )[0]
        horizons = cumulative_horizons(
            published_at=video.published_at,
            daily_views=analytics.get(video.video_id, {}),
        )
        records.append(
            {
                "video_id": video.video_id,
                "title": video.title,
                "category": video.category,
                "duration_seconds": video.duration_seconds,
                "published_at": video.published_at,
                "is_short": video.is_short,
                "d7": horizons[7],
                "d14": horizons[14],
                "d21": horizons[21],
                "d30": horizons[30],
                "pred7": max(0, int(round(float(trajectory[0])))),
                "pred14": max(0, int(round(float(trajectory[1])))),
                "pred21": max(0, int(round(float(trajectory[2])))),
                "pred30": max(0, int(round(float(trajectory[3])))),
                "model_version": model_version,
            }
        )

    await store.upsert_video_history(user_id=user_id, rows=records)
    adjustments = compute_adjustments(
        records=records,
        training_video_ids=load_current_training_video_ids(),
        model_version=model_version,
    )
    await store.replace_adjustments(
        user_id=user_id, model_version=model_version, rows=adjustments
    )
    await store.set_youtube_connection_status(
        user_id=user_id, status="active", refresh_ok=True
    )
