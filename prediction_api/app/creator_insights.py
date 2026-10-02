"""A connected creator's own publishing pattern, from their synced history.

The insights page reports what holds across Sri Lankan channels. This module
asks the same questions of one channel: do its uploads that crowd together do
worse, which time of day works for it, how its Shorts compare with its regular
videos, and how much it keeps growing after day 7.

Every effect is measured against the channel's own normal, the mean log day-7
views of its measured videos, exactly as the population figures are, so the
two can be read side by side. With at most a few hundred videos the ranges are
wide, so each one carries a bootstrap 95% range and a group is reported only
with at least MIN_GROUP videos behind it.

Input rows come from creator.video_history: day-7 and day-30 views are exact
YouTube Analytics figures, not polls, so no label tolerance applies.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np

COLOMBO = timezone(timedelta(hours=5, minutes=30))
MIN_MEASURED = 10      # fewer measured videos than this and no normal is stable
MIN_GROUP = 5          # smallest group an effect is reported for
BOOTSTRAP = 1000
SEED = 20261003

GAP_BUCKETS = (
    ("under_1h", "Within an hour of the next upload", 0.0, 1.0),
    ("1h_to_6h", "1 to 6 hours before the next upload", 1.0, 6.0),
    ("6h_plus", "6 hours or more before the next upload", 6.0, math.inf),
)
TIME_BLOCKS = (
    ("night", "Night (00:00–06:00)", 0, 6),
    ("morning", "Morning (06:00–12:00)", 6, 12),
    ("afternoon", "Afternoon (12:00–18:00)", 12, 18),
    ("evening", "Evening (18:00–24:00)", 18, 24),
)


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _pct(log_effect: float) -> float:
    return round(math.expm1(log_effect) * 100, 1)


def _effect(resid: np.ndarray, members: np.ndarray, rng: np.random.Generator,
            *, against: np.ndarray | None = None) -> dict[str, Any] | None:
    """Mean residual of `members` (or members minus `against`) with a 95% range.

    Resampling draws whole videos with replacement from the measured set, so
    group sizes vary between draws the way they would between channels' worth
    of uploads.
    """
    if members.sum() < MIN_GROUP or (against is not None and against.sum() < MIN_GROUP):
        return None

    def stat(index: np.ndarray) -> float:
        r, m = resid[index], members[index]
        if not m.any():
            return math.nan
        if against is None:
            return float(r[m].mean())
        a = against[index]
        return float(r[m].mean() - r[a].mean()) if a.any() else math.nan

    point = stat(np.arange(len(resid)))
    draws = np.array([stat(rng.integers(0, len(resid), len(resid))) for _ in range(BOOTSTRAP)])
    draws = draws[~np.isnan(draws)]
    low, high = np.percentile(draws, [2.5, 97.5])
    videos = int(members.sum() + (against.sum() if against is not None else 0))
    return dict(videos=videos, effectPct=_pct(point), lowPct=_pct(low), highPct=_pct(high))


def compute_creator_insights(
    rows: Iterable[Mapping[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    videos = sorted(
        ({**row, "published_at": _utc(row["published_at"])} for row in rows),
        key=lambda row: row["published_at"],
    )
    result: dict[str, Any] = dict(
        videosSynced=len(videos), videosMeasured=0, periodStart=None, periodEnd=None,
        mainCategory=None, spacing=None, timing=[], format=None, growth=None,
    )
    if not videos:
        return result
    result["periodStart"] = videos[0]["published_at"].date().isoformat()
    result["periodEnd"] = videos[-1]["published_at"].date().isoformat()
    categories = [v.get("category") for v in videos if v.get("category")]
    if categories:
        result["mainCategory"] = max(set(categories), key=categories.count)

    # Gap to the channel's next upload, over every synced video. The newest has
    # no next upload yet; it counts as spaced once two days have passed.
    gaps: list[float | None] = []
    for current, following in zip(videos, videos[1:] + [None]):
        if following is not None:
            gaps.append((following["published_at"] - current["published_at"]).total_seconds() / 3600)
        elif (now - current["published_at"]).total_seconds() >= 48 * 3600:
            gaps.append(math.inf)
        else:
            gaps.append(None)
    known = [g for g in gaps if g is not None]

    measured = [i for i, v in enumerate(videos) if v.get("d7") is not None and v["d7"] >= 0]
    result["videosMeasured"] = len(measured)
    spacing: dict[str, Any] = dict(
        uploadsCounted=len(known),
        shareWithinHour=round(100 * sum(g < 1 for g in known) / len(known), 1) if known else None,
        medianGapHours=round(median(g for g in known if math.isfinite(g)), 1)
        if any(math.isfinite(g) for g in known) else None,
        buckets=[],
    )
    result["spacing"] = spacing
    if len(measured) < MIN_MEASURED:
        return result

    rng = np.random.default_rng(SEED)
    sub = [videos[i] for i in measured]
    log_views = np.log1p(np.array([v["d7"] for v in sub], dtype=float))
    resid = log_views - log_views.mean()
    sub_gaps = [gaps[i] for i in measured]

    for key, label, lo, hi in GAP_BUCKETS:
        members = np.array([g is not None and lo <= g < hi for g in sub_gaps])
        effect = _effect(resid, members, rng)
        if effect:
            spacing["buckets"].append(dict(key=key, label=label, **effect))

    hours = [v["published_at"].astimezone(COLOMBO).hour for v in sub]
    for key, label, lo, hi in TIME_BLOCKS:
        members = np.array([lo <= h < hi for h in hours])
        effect = _effect(resid, members, rng)
        if effect:
            result["timing"].append(dict(key=key, label=label, **effect))

    shorts = np.array([bool(v.get("is_short")) for v in sub])
    result["format"] = dict(
        shorts=int(shorts.sum()), regular=int((~shorts).sum()),
        shortsVsRegular=_effect(resid, shorts, rng, against=~shorts),
    )

    ratios = [v["d30"] / v["d7"] - 1 for v in sub if v.get("d30") is not None and v["d7"] > 0]
    if len(ratios) >= MIN_GROUP:
        result["growth"] = dict(videos=len(ratios),
                                medianGrowthPct=round(float(median(ratios)) * 100, 1))
    return result
