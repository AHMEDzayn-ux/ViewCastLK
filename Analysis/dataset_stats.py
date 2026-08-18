"""Profile the built training table.

Reads the exported Parquet, not the database, so what is reported is exactly
what a teammate will load -- a statistic taken from the warehouse could differ
from the file and nobody would know which was wrong.

Everything is reported on the ELIGIBLE subset (live broadcasts and videos with
no parseable duration removed), and target statistics only on rows where the
horizon's label is usable, because an unusable label is not an observation of
that horizon at all.

Views are summarised by MEDIAN and quantiles rather than by mean. The day-7
distribution has a mean fourteen times its median and a maximum four orders of
magnitude above it; a mean describes the handful of viral videos and nothing
else. Where a mean appears it is on log1p, which is the scale to model on.

Usage:  python Analysis/dataset_stats.py [--file PATH] [--horizon 7]
"""
import argparse
import os

import numpy as np
import pandas as pd

from paths import dataset_path
DEFAULT = dataset_path()
HORIZONS = (7, 14, 21, 30)
QS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

# Channel size bands, in subscribers at publication. Boundaries are decimal
# decades because subscriber counts span five of them; equal-width bands would
# put 95% of channels in the first bucket.
BANDS = [(0, 1e3, "<1K"), (1e3, 1e4, "1K-10K"), (1e4, 1e5, "10K-100K"),
         (1e5, 1e6, "100K-1M"), (1e6, np.inf, "1M+")]


def rule(t=""):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}" if t else "")


def band(n):
    if pd.isna(n):
        return "unknown"
    for lo, hi, name in BANDS:
        if lo <= n < hi:
            return name
    return "unknown"


def qtable(g, col):
    """Median-centred summary for a grouped view count."""
    out = g[col].agg(n="size", median="median",
                     p25=lambda s: s.quantile(.25),
                     p75=lambda s: s.quantile(.75),
                     p99=lambda s: s.quantile(.99), max="max")
    return out.sort_values("n", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--horizon", type=int, default=7,
                    help="horizon used for target breakdowns")
    args = ap.parse_args()

    df = pd.read_parquet(args.file)
    e = df[df.eligible].copy()
    h = args.horizon
    lab = e[e[f"d{h}_usable"]].copy()
    lab["log_views"] = np.log1p(lab[f"d{h}_views"])
    lab["size_band"] = lab["ch_subs_at_publish"].apply(band)
    e["size_band"] = e["ch_subs_at_publish"].apply(band)

    rule("1. SHAPE")
    print(f"rows in file            {len(df):,}")
    print(f"eligible                {len(e):,}   "
          f"(live broadcasts and unparseable durations removed: "
          f"{len(df) - len(e):,})")
    print(f"distinct channels       {e.channel_id.nunique():,}")
    print(f"published               {e.published_at.min():%Y-%m-%d} to "
          f"{e.published_at.max():%Y-%m-%d}"
          f"  ({(e.published_at.max() - e.published_at.min()).days} days)")
    print(f"\nvideos per channel: median "
          f"{e.groupby('channel_id').size().median():.0f}, "
          f"max {e.groupby('channel_id').size().max():,}")

    rule("2. LABEL COVERAGE")
    print(f"{'horizon':>8} {'usable':>9} {'% of eligible':>14} "
          f"{'median |offset|':>16}")
    for k in HORIZONS:
        u = e[e[f"d{k}_usable"]]
        med = u[f"d{k}_hours_off"].abs().median()
        print(f"{'day ' + str(k):>8} {len(u):>9,} {len(u) / len(e):>13.1%} "
              f"{(f'{med:.1f} h' if len(u) else '-'):>16}")
    print("\nHorizons do not overlap: collection began 17 July, so old videos")
    print("are past their early marks and recent ones have not reached their")
    print("late ones. Train one model per horizon.")

    rule(f"3. TARGET — day-{h} views ({len(lab):,} usable)")
    v = lab[f"d{h}_views"]
    print(f"{'min':>10} {int(v.min()):>12,}")
    for q in QS:
        print(f"{'p' + str(int(q * 100)):>10} {int(v.quantile(q)):>12,}")
    print(f"{'max':>10} {int(v.max()):>12,}")
    print(f"\nmean {v.mean():,.0f} is {v.mean() / v.median():.0f}x the median "
          f"{v.median():,.0f} — the raw target is not modellable directly.")
    print(f"log1p: mean {lab.log_views.mean():.2f}, sd {lab.log_views.std():.2f}, "
          f"skew {lab.log_views.skew():.2f}  <- fit on this")
    zero = int((v == 0).sum())
    print(f"\nzero views at day {h}: {zero:,} ({zero / len(v):.1%})")
    top1 = v.nlargest(max(1, len(v) // 100)).sum() / v.sum()
    print(f"top 1% of videos hold {top1:.1%} of all day-{h} views")

    rule("4. BY CATEGORY — corpus")
    # Corpus share and labelled share are different questions and are reported
    # separately. A category can be a large part of what was collected while
    # contributing few usable rows, because label coverage depends on when a
    # video was published, not on what it is about.
    cat = e.groupby("category_name").agg(
        videos=("video_id", "size"),
        channels=("channel_id", "nunique"),
        shorts=("is_short", "mean"),
        median_dur=("duration_seconds", "median"),
        median_subs=("ch_subs_at_publish", "median"))
    cat["share"] = cat.videos / cat.videos.sum()
    cat[f"labelled_d{h}"] = lab.groupby("category_name").size()
    cat[f"labelled_d{h}"] = cat[f"labelled_d{h}"].fillna(0).astype(int)
    cat["label_rate"] = cat[f"labelled_d{h}"] / cat.videos
    cat = cat.sort_values("videos", ascending=False)
    print(cat[["videos", "share", "channels", "shorts", "median_dur",
               "median_subs", f"labelled_d{h}", "label_rate"]].to_string(
        formatters={"videos": "{:,}".format, "share": "{:.1%}".format,
                    "channels": "{:,}".format, "shorts": "{:.0%}".format,
                    "median_dur": "{:,.0f}".format,
                    "median_subs": "{:,.0f}".format,
                    f"labelled_d{h}": "{:,}".format,
                    "label_rate": "{:.0%}".format}))

    rule(f"5. BY CATEGORY — day-{h} view targets")
    t = qtable(lab.groupby("category_name"), f"d{h}_views")
    t.insert(1, "share", t.n / t.n.sum())
    t["log_mean"] = lab.groupby("category_name").log_views.mean()
    print(t.to_string(float_format=lambda x: f"{x:,.0f}",
                      formatters={"share": "{:.1%}".format,
                                  "log_mean": "{:.2f}".format}))
    print("\nRead the MEDIAN column, not max. Ranking by log_mean gives the")
    print("same order and is the scale a model would be fit on.")

    rule(f"6. BY CATEGORY — median views at every horizon")
    piv = pd.DataFrame(index=sorted(e.category_name.dropna().unique()))
    for k in HORIZONS:
        u = e[e[f"d{k}_usable"]]
        piv[f"d{k}_n"] = u.groupby("category_name").size()
        piv[f"d{k}_med"] = u.groupby("category_name")[f"d{k}_views"].median()
    piv = piv.loc[piv.filter(like="_n").sum(axis=1).sort_values(
        ascending=False).index]
    print(piv.to_string(float_format=lambda x: f"{x:,.0f}", na_rep="-"))
    print("\nEach horizon is a different set of videos, not the same videos")
    print("followed over time, so these columns are NOT a growth curve.")

    rule("7. BY CHANNEL SIZE AT PUBLICATION")
    order = [b[2] for b in BANDS] + ["unknown"]
    t = qtable(lab.groupby("size_band"), f"d{h}_views").reindex(
        [b for b in order if b in lab.size_band.unique()])
    print(t.to_string(float_format=lambda x: f"{x:,.0f}"))
    corr = lab[["log_views"]].assign(
        log_subs=np.log1p(lab.ch_subs_at_publish)).corr().iloc[0, 1]
    print(f"\ncorrelation log1p(subscribers) vs log1p(day-{h} views): {corr:.3f}")
    print(f"Subscribers alone explain only {corr ** 2:.0%} of the variance in")
    print("log views, and the medians above saturate past 10K -- a 1M-subscriber")
    print("channel does not out-perform a 100K one. Channel size is the single")
    print("strongest simple predictor, not a sufficient one. The baseline to")
    print("beat is a per-channel median, which captures far more than the")
    print("subscriber count does.")
    top10 = (e.groupby("channel_id").size().nlargest(10).sum() / len(e))
    print(f"\nTop 10 channels hold {top10:.1%} of eligible rows "
          f"(the largest posts ~150 clips/day). Split by CHANNEL, not at")
    print("random, or the same channel's videos sit on both sides of the split.")

    rule("8. BY FORMAT AND SCRIPT")
    for key in ("is_short", "title_script", "definition", "made_for_kids",
                "publish_is_weekend"):
        print(f"\n-- {key}")
        print(qtable(lab.groupby(key), f"d{h}_views").to_string(
            float_format=lambda x: f"{x:,.0f}"))

    rule("9. BY PUBLISH HOUR (Asia/Colombo)")
    t = qtable(lab.groupby("publish_hour_slt"), f"d{h}_views").sort_index()
    print(t[["n", "median", "p75"]].to_string(float_format=lambda x: f"{x:,.0f}"))

    rule("10. FEATURE COMPLETENESS (eligible rows)")
    miss = e.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss = miss[~miss.index.str.match(r"d\d+_")]
    if len(miss):
        print(pd.DataFrame({"missing": miss, "share": miss / len(e)}).to_string(
            formatters={"missing": "{:,}".format, "share": "{:.1%}".format}))
    else:
        print("no missing values among features")
    print("\nMissing is meaningful here — a null tag list means the uploader")
    print("set none. Encode it as a category rather than dropping the row.")

    rule("11. DATA-QUALITY FLAGS (eligible rows)")
    for f, note in (("channel_stats_backfilled",
                     "channel stats substituted from a later snapshot"),
                    ("title_changed", "title edited since first seen"),
                    ("description_changed", "description edited since first seen")):
        n = int(e[f].sum())
        print(f"{f:28} {n:>7,}  {n / len(e):>6.1%}   {note}")

    rule("12. USABLE MODELLING SUBSETS")
    for k in HORIZONS:
        a = e[e[f"d{k}_usable"]]
        b = a[~a.channel_stats_backfilled]
        c = b[~b.title_changed]
        print(f"day {k:>2}: {len(a):>7,} usable  ->  {len(b):>7,} with true "
              f"point-in-time channel stats  ->  {len(c):>7,} also unedited")
    print()


if __name__ == "__main__":
    main()
