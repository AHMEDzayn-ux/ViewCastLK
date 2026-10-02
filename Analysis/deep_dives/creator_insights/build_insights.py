"""Compute the figures behind the dashboard's insights page.

Every effect is measured within channel: a video's log day-7 views minus the
mean for its own channel. That asks "did this video beat its channel's
normal?", which is the question a creator can act on, and it stops a
comparison of categories or hours from really being a comparison of which
channels post there (see the EDA findings report, section 4).

Uncertainty comes from a channel-level bootstrap: whole channels are
resampled, because videos from one channel are not independent. A segment is
published only when it has enough videos and enough channels behind it.

Writes dashboard/src/data/insights.json. Run with the project venv:
    python Analysis/deep_dives/creator_insights/build_insights.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "paths.py").is_file())))
from paths import ANALYSIS, REPO, dataset_path  # noqa: E402

OUT = REPO / "dashboard" / "src" / "data" / "insights.json"
THUMBNAILS = ANALYSIS / "deep_dives" / "thumbnails" / "thumbnail_features.parquet"

MIN_CHANNEL_VIDEOS = 5      # channels with fewer usable videos have no stable "normal"
MIN_VIDEOS = 300            # smallest segment cell we publish
MIN_CHANNELS = 20
BOOTSTRAP = 1000
SEED = 20261002

SUB_BANDS = [(0, 1e3, "Under 1K"), (1e3, 1e4, "1K–10K"), (1e4, 1e5, "10K–100K"),
             (1e5, 1e6, "100K–1M"), (1e6, np.inf, "1M+")]
GAP_BANDS = [(0, 1, "Under 1 hour"), (1, 3, "1–3 hours"), (3, 6, "3–6 hours"),
             (6, 12, "6–12 hours"), (12, 24, "12–24 hours"), (24, 48, "1–2 days"),
             (48, np.inf, "2 days or more")]
# Long-form only: Shorts run up to 3 minutes, so a seconds band would mix them in.
DURATION_BANDS = [(0, 240, "Under 4 min"), (240, 480, "4–8 min"), (480, 900, "8–15 min"),
                  (900, 1800, "15–30 min"), (1800, 3600, "30–60 min"),
                  (3600, np.inf, "Over 1 hour")]
HOUR_BLOCKS = [(h, h + 3, f"{h:02d}:00–{h + 3:02d}:00") for h in range(0, 24, 3)]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def band(values, bands):
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for lo, hi, label in bands:
        out[(values >= lo) & (values < hi)] = label
    return out


def pct(log_effect):
    """Log-scale difference as a percentage change in views."""
    return round(float(np.expm1(log_effect)) * 100, 1)


def bootstrap_means(d, group):
    """Mean residual per group with a channel-level bootstrap 95% interval."""
    d = d.dropna(subset=[group])
    channels, ch_idx = np.unique(d.channel_id, return_inverse=True)
    groups, g_idx = np.unique(d[group].astype(str), return_inverse=True)
    sums = np.zeros((len(channels), len(groups)))
    counts = np.zeros_like(sums)
    np.add.at(sums, (ch_idx, g_idx), d.resid.to_numpy())
    np.add.at(counts, (ch_idx, g_idx), 1)
    # A fresh generator per call: each figure is reproducible on its own,
    # whatever else the script computes before it.
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(channels), np.full(len(channels), 1 / len(channels)),
                              size=BOOTSTRAP)
    boot = (weights @ sums) / np.maximum(weights @ counts, 1)
    point = sums.sum(0) / counts.sum(0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    rows = {}
    for j, g in enumerate(groups):
        rows[g] = dict(videos=int(counts[:, j].sum()),
                       channels=int((counts[:, j] > 0).sum()),
                       mean=point[j], lo=lo[j], hi=hi[j], boot=boot[:, j])
    return rows


def as_cells(rows, order):
    cells = []
    for label in order:
        r = rows.get(label)
        if r is None or r["videos"] < MIN_VIDEOS or r["channels"] < MIN_CHANNELS:
            continue
        cells.append(dict(label=label, videos=r["videos"], channels=r["channels"],
                          effectPct=pct(r["mean"]), lowPct=pct(r["lo"]),
                          highPct=pct(r["hi"])))
    return cells


def contrast(d, group, better, worse):
    """better minus worse, within the same bootstrap draws."""
    rows = bootstrap_means(d, group)
    a, b = rows.get(better), rows.get(worse)
    if a is None or b is None:
        return None
    if min(a["videos"], b["videos"]) < MIN_VIDEOS / 2 or \
            min(a["channels"], b["channels"]) < MIN_CHANNELS:
        return None
    diff = a["boot"] - b["boot"]
    return dict(videos=a["videos"] + b["videos"],
                channels=int(d[d[group].isin([better, worse])].channel_id.nunique()),
                effectPct=pct(a["mean"] - b["mean"]),
                lowPct=pct(np.percentile(diff, 2.5)), highPct=pct(np.percentile(diff, 97.5)))


def segmented(d, segment, group, better, worse, order):
    out = []
    for s in order:
        c = contrast(d[d[segment] == s], group, better, worse)
        if c:
            out.append(dict(label=s, **c))
    return out


# ---------------------------------------------------------------- load
src = dataset_path()
raw = pd.read_parquet(src)
raw = raw[raw.eligible & ~raw.is_live_broadcast].copy()
raw["sub_band"] = band(raw.ch_subs_at_publish, SUB_BANDS)
cutoff = raw.published_at.max()

# Gaps use every tracked upload, labelled or not: an unlabelled upload still
# competes for the same subscribers' attention.
raw = raw.sort_values(["channel_id", "published_at"])
nxt = raw.groupby("channel_id").published_at.shift(-1)
prv = raw.groupby("channel_id").published_at.shift(1)
raw["gap_next_h"] = (nxt - raw.published_at).dt.total_seconds() / 3600
raw["gap_prev_h"] = (raw.published_at - prv).dt.total_seconds() / 3600
# Last upload on record: "2 days or more" only if two days have passed since.
quiet = nxt.isna() & ((cutoff - raw.published_at).dt.total_seconds() / 3600 >= 48)
raw.loc[quiet, "gap_next_h"] = np.inf
raw["gap_next"] = band(raw.gap_next_h, GAP_BANDS)
raw["gap_prev"] = band(raw.gap_prev_h, GAP_BANDS)
since_24h = raw.groupby("channel_id", group_keys=False).apply(
    lambda g: g.rolling("24h", on="published_at").video_id.count() - 1, include_groups=False)
raw["uploads_prev_24h"] = since_24h

d = raw[raw.d7_usable & raw.d7_views.notna()].copy()
d["log_views"] = np.log1p(d.d7_views)
size = d.groupby("channel_id").video_id.transform("size")
d = d[size >= MIN_CHANNEL_VIDEOS].copy()
d["resid"] = d.log_views - d.groupby("channel_id").log_views.transform("mean")
d["format"] = np.where(d.is_short, "Shorts", "Long-form")
d["duration_band"] = band(d.duration_seconds, DURATION_BANDS).where(~d.is_short, "Shorts")
d["hour"] = d.publish_hour_slt.astype(str)
d["dow"] = d.publish_dow_slt.map(dict(enumerate(DAYS)))
d["spacing"] = np.select([d.gap_next_h < 3, d.gap_next_h >= 24], ["close", "spaced"], "")

print(f"{src.name}: {len(raw):,} uploads, {len(d):,} usable day-7 videos "
      f"from {d.channel_id.nunique():,} channels, published to {cutoff:%d %b %Y}")

top_categories = (d.groupby("category_name").channel_id.nunique()
                  .loc[lambda s: s >= MIN_CHANNELS].index)
cat_order = d[d.category_name.isin(top_categories)].category_name.value_counts().index.tolist()
band_order = [b[2] for b in SUB_BANDS]

# ---------------------------------------------------------------- 1. spacing
crowded_share = float((raw.gap_next_h < 1).mean())
spacing = dict(
    byGapToNext=as_cells(bootstrap_means(d, "gap_next"), [g[2] for g in GAP_BANDS]),
    byGapSincePrevious=as_cells(bootstrap_means(d, "gap_prev"), [g[2] for g in GAP_BANDS]),
    bySize=segmented(d, "sub_band", "spacing", "spaced", "close", band_order),
    byCategory=segmented(d, "category_name", "spacing", "spaced", "close", cat_order),
    shareWithinHour=round(crowded_share * 100, 1),
    medianGapHours=round(float(raw.gap_next_h.replace(np.inf, np.nan).median()), 1),
    medianUploadsPrev24h=int(raw.uploads_prev_24h.median()),
)

# ---------------------------------------------------------------- 2. timing
timing = dict(
    byHour=as_cells(bootstrap_means(d, "hour"), [str(h) for h in range(24)]),
    byHourBlock=as_cells(bootstrap_means(d.assign(block=band(d.publish_hour_slt, HOUR_BLOCKS)),
                                         "block"), [b[2] for b in HOUR_BLOCKS]),
    byDay=as_cells(bootstrap_means(d, "dow"), DAYS),
)

# ---------------------------------------------------------------- 3. format and duration
fmt = dict(
    overall=contrast(d, "format", "Shorts", "Long-form"),
    byCategory=segmented(d, "category_name", "format", "Shorts", "Long-form", cat_order),
    bySize=segmented(d, "sub_band", "format", "Shorts", "Long-form", band_order),
)
duration_by_category = []
for cat in cat_order:
    cells = as_cells(bootstrap_means(d[d.category_name == cat], "duration_band"),
                     ["Shorts"] + [b[2] for b in DURATION_BANDS])
    if len(cells) >= 3:
        duration_by_category.append(dict(category=cat, bands=cells))
fmt["durationByCategory"] = duration_by_category

# ---------------------------------------------------------------- 4. growth after day 7
g = raw[raw.d7_usable & raw.d30_usable & (raw.d7_views > 0) & raw.d30_views.notna()].copy()
g["after_d7"] = g.d30_views / g.d7_views - 1
g["share_by_d7"] = g.d7_views / g.d30_views.clip(lower=1)


def growth_rows(frame, key, order):
    out = []
    for k in order:
        s = frame[frame[key] == k]
        if len(s) < MIN_VIDEOS or s.channel_id.nunique() < MIN_CHANNELS:
            continue
        out.append(dict(label=k, videos=int(len(s)), channels=int(s.channel_id.nunique()),
                        medianGrowthPct=round(float(s.after_d7.median()) * 100, 1),
                        upperQuartileGrowthPct=round(float(s.after_d7.quantile(.75)) * 100, 1),
                        shareGrowing10Pct=round(float((s.after_d7 >= .10).mean()) * 100, 1)))
    return out


growth = dict(
    videos=int(len(g)),
    medianShareByDay7=round(float(g.share_by_d7.median()) * 100, 1),
    byCategory=sorted(growth_rows(g, "category_name", cat_order),
                      key=lambda r: -r["medianGrowthPct"]),
    bySize=growth_rows(g, "sub_band", band_order),
    byFormat=growth_rows(g.assign(format=np.where(g.is_short, "Shorts", "Long-form")),
                         "format", ["Shorts", "Long-form"]),
)

# ---------------------------------------------------------------- 5. thumbnails (faces)
thumbs = pd.read_parquet(THUMBNAILS, columns=["video_id", "has_face"])
t = d.merge(thumbs, on="video_id")
varies = t.groupby("channel_id").has_face.transform("nunique") > 1
t = t[varies].copy()
t["face"] = np.where(t.has_face, "face", "no_face")
thumbnails = dict(
    videos=int(len(t)),
    channels=int(t.channel_id.nunique()),
    overall=contrast(t, "face", "face", "no_face"),
    byCategory=segmented(t, "category_name", "face", "face", "no_face", cat_order),
)

# ---------------------------------------------------------------- write
payload = dict(
    generatedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    dataset=dict(
        uploads=int(len(raw)), videos=int(len(d)), channels=int(d.channel_id.nunique()),
        periodStart=raw.published_at.min().date().isoformat(),
        periodEnd=cutoff.date().isoformat(),
        minChannelVideos=MIN_CHANNEL_VIDEOS, minVideos=MIN_VIDEOS, minChannels=MIN_CHANNELS,
    ),
    spacing=spacing, timing=timing, format=fmt, growth=growth, thumbnails=thumbnails,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.relative_to(REPO)}")
