"""Plain descriptive charts of the collected dataset.

WHY THIS EXISTS SEPARATELY FROM build_eda_notebook.py
That notebook answers "what survives once channel is controlled for", which is
the right question for feature engineering and the wrong one to open a review
with. Every figure in it is a residual, a log scale or a scatter, and none of
them answers "how many views does a Music video get".

These are the descriptive charts: counts and medians, in real view numbers, on
the cuts anyone asks about first -- category, channel size, duration, timing.
They are deliberately unadjusted. Where an unadjusted reading is misleading the
caption says so and points at the controlled figure that corrects it.

Median rather than mean throughout. Day-7 views run from 0 to several million,
so a mean describes almost no video in the corpus.

Usage:
    python Analysis/build_descriptive_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from paths import dataset_path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "eda_figures", "descriptive")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, GREY, RED = "#2F6DB5", "#9AA5B1", "#C0392B"

SUB_EDGES = [0, 1e3, 1e4, 1e5, 1e6, np.inf]
SUB_LABELS = ["< 1K", "1K – 10K", "10K – 100K", "100K – 1M", "1M +"]

DUR_EDGES = [0, 60, 300, 1200, np.inf]
DUR_LABELS = ["< 1 min\n(Shorts)", "1 – 5 min", "5 – 20 min", "20 min +"]

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def thousands(ax, axis="x"):
    f = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def save(name):
    p = os.path.join(OUT, f"{name}.png")
    plt.savefig(p)
    plt.close()
    print(f"  {name}.png")


def annotate(ax, values, counts, fmt="{:,.0f}"):
    """Value at the bar end, sample size beside it -- a median over 12 videos
    and one over 12,000 must not look equally solid."""
    span = max(values) if len(values) else 1
    for i, (v, n) in enumerate(zip(values, counts)):
        ax.text(v + span * .015, i, f"{fmt.format(v)}   n={n:,}",
                va="center", fontsize=8.2, color="#333")


def load():
    d = pd.read_parquet(dataset_path())
    d = d[d.eligible & d.d7_usable & d.d7_views.notna()].copy()
    d["d7_views"] = d.d7_views.astype(float)
    d["subs"] = d.ch_subs_at_publish.astype(float)
    d["sub_band"] = pd.cut(d.subs, SUB_EDGES, labels=SUB_LABELS, right=False)
    d["dur_band"] = pd.cut(d.duration_seconds.astype(float), DUR_EDGES,
                           labels=DUR_LABELS, right=False)
    return d


# ------------------------------------------------------------------ 1. corpus
def fig_corpus(d):
    vids = d.category_name.value_counts()
    chans = d.groupby("category_name").channel_id.nunique().reindex(vids.index)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    y = np.arange(len(vids))[::-1]

    ax[0].barh(y, vids.values, color=BLUE)
    ax[0].set(yticks=y, yticklabels=vids.index, title="Videos collected, by category",
              xlabel="videos")
    for i, v in zip(y, vids.values):
        ax[0].text(v + vids.max() * .015, i, f"{v:,}", va="center", fontsize=8.2)

    ax[1].barh(y, chans.values, color=GREY)
    ax[1].set(yticks=y, yticklabels=[], title="Channels active in each category",
              xlabel="channels")
    for i, v in zip(y, chans.values):
        ax[1].text(v + chans.max() * .015, i, f"{v:,}", va="center", fontsize=8.2)

    for a in ax:
        thousands(a)
    fig.suptitle(f"Corpus composition — {len(d):,} videos, "
                 f"{d.channel_id.nunique():,} channels", y=1.02,
                 fontsize=12, fontweight="bold")
    save("D1_corpus_composition")


# ---------------------------------------------------------------- 2. category
def fig_category(d):
    g = d.groupby("category_name").d7_views.agg(["median", "mean", "size"])
    g = g.sort_values("median")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(g))
    ax.barh(y, g["median"], color=BLUE)
    ax.set(yticks=y, yticklabels=g.index,
           title="Median day-7 views by category",
           xlabel="median views after 7 days")
    annotate(ax, g["median"].values, g["size"].values)
    thousands(ax)
    ax.set_xlim(0, g["median"].max() * 1.32)
    ax.text(.99, .02, "Unadjusted. Reverses once channel is controlled for —\n"
                      "see 04_category_raw_vs_within.png",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=RED, style="italic")
    save("D2_views_by_category")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(y, g["mean"], color=GREY)
    ax.set(yticks=y, yticklabels=g.index,
           title="Mean day-7 views by category (shown for contrast)",
           xlabel="mean views after 7 days")
    annotate(ax, g["mean"].values, g["size"].values)
    thousands(ax)
    ax.set_xlim(0, g["mean"].max() * 1.32)
    ax.text(.99, .02, "The mean sits far above the median in every category:\n"
                      "a few very large videos carry it.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=RED, style="italic")
    save("D3_mean_views_by_category")


# -------------------------------------------------------------- 3. channel size
def fig_size(d):
    g = d.groupby("sub_band", observed=True).d7_views.agg(["median", "size"])
    g = g.reindex([b for b in SUB_LABELS if b in g.index])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(g))
    ax[0].bar(x, g["median"], color=BLUE, width=.65)
    ax[0].set(xticks=x, xticklabels=g.index, title="Median day-7 views by channel size",
              xlabel="subscribers at publication", ylabel="median views")
    thousands(ax[0], "y")
    for i, (v, n) in enumerate(zip(g["median"], g["size"])):
        ax[0].text(i, v * 1.03, f"{v:,.0f}\nn={n:,}", ha="center",
                   va="bottom", fontsize=8.2)
    ax[0].set_ylim(0, g["median"].max() * 1.25)

    cnt = d.groupby("sub_band", observed=True).channel_id.nunique().reindex(g.index)
    ax[1].bar(x, cnt.values, color=GREY, width=.65)
    ax[1].set(xticks=x, xticklabels=g.index, title="Channels in each size band",
              xlabel="subscribers at publication", ylabel="channels")
    for i, v in enumerate(cnt.values):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.2)
    thousands(ax[1], "y")
    save("D4_views_by_channel_size")


# ------------------------------------------------------- 4. category x size
def fig_category_x_size(d):
    piv = d.pivot_table(index="category_name", columns="sub_band",
                        values="d7_views", aggfunc="median", observed=True)
    cnt = d.pivot_table(index="category_name", columns="sub_band",
                        values="d7_views", aggfunc="size", observed=True)
    piv = piv.reindex(columns=[b for b in SUB_LABELS if b in piv.columns])
    cnt = cnt.reindex_like(piv)
    piv = piv.loc[piv.median(axis=1).sort_values(ascending=False).index]
    cnt = cnt.reindex_like(piv)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    shown = piv.mask(cnt < 20)          # a median over <20 videos is not a reading
    im = ax.imshow(np.log10(shown.values.astype(float) + 1),
                   cmap="Blues", aspect="auto")
    ax.set(xticks=range(piv.shape[1]), xticklabels=piv.columns,
           yticks=range(piv.shape[0]), yticklabels=piv.index,
           title="Median day-7 views — category by channel size")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, n = shown.values[i, j], cnt.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=9, color="#bbb")
            else:
                hi = np.log10(v + 1) > np.nanmax(np.log10(
                    shown.values.astype(float) + 1)) * .62
                ax.text(j, i, f"{v:,.0f}\nn={int(n):,}", ha="center",
                        va="center", fontsize=7.4,
                        color="white" if hi else "#222")
    cb = fig.colorbar(im, ax=ax, shrink=.85)
    cb.set_label("log10 median views", fontsize=8.5)
    # imshow inverts the y axis, so a negative data coordinate lands above the
    # title -- anchor the note to the axes instead.
    ax.text(0, -.09, "Cells with fewer than 20 videos are left blank.",
            transform=ax.transAxes, fontsize=8, color="#666", style="italic")
    save("D5_category_by_channel_size")


# ---------------------------------------------------------------- 5. duration
def fig_duration(d):
    g = d.groupby("dur_band", observed=True).d7_views.agg(["median", "size"])
    g = g.reindex([b for b in DUR_LABELS if b in g.index])

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(g))
    ax.bar(x, g["median"], color=BLUE, width=.6)
    ax.set(xticks=x, xticklabels=g.index, title="Median day-7 views by video length",
           ylabel="median views")
    for i, (v, n) in enumerate(zip(g["median"], g["size"])):
        ax.text(i, v * 1.03, f"{v:,.0f}\nn={n:,}", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_ylim(0, g["median"].max() * 1.25)
    thousands(ax, "y")
    save("D6_views_by_duration")


# ------------------------------------------------------------------ 6. timing
def fig_timing(d):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    h = d.groupby("publish_hour_slt").d7_views.agg(["median", "size"])
    ax[0].bar(h.index, h["size"], color=GREY, width=.8)
    ax[0].set(title="When Sri Lankan channels publish", xlabel="hour of day (SLT)",
              ylabel="videos published")
    a0 = ax[0].twinx()
    a0.plot(h.index, h["median"], color=BLUE, marker="o", ms=3.5, lw=1.8)
    a0.set_ylabel("median day-7 views", color=BLUE)
    a0.tick_params(axis="y", colors=BLUE)
    a0.grid(False)
    thousands(a0, "y")

    w = d.groupby("publish_dow_slt").d7_views.agg(["median", "size"])
    w = w.reindex(range(7))
    ax[1].bar(range(7), w["size"], color=GREY, width=.7)
    ax[1].set(xticks=range(7), xticklabels=DOW, title="Publishing by day of week",
              ylabel="videos published")
    a1 = ax[1].twinx()
    a1.plot(range(7), w["median"], color=BLUE, marker="o", ms=4, lw=1.8)
    a1.set_ylabel("median day-7 views", color=BLUE)
    a1.tick_params(axis="y", colors=BLUE)
    a1.grid(False)
    thousands(a1, "y")

    fig.suptitle("Bars = publishing volume,  line = median day-7 views",
                 y=1.03, fontsize=10, color="#444")
    save("D7_publishing_timing")


# ------------------------------------------------------------ 7. distribution
def fig_distribution(d):
    v = d.d7_views[d.d7_views > 0]
    q = v.quantile([.25, .5, .75, .9, .99])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(np.log10(v), bins=60, color=BLUE, alpha=.85)
    ax.set(title="Distribution of day-7 views",
           xlabel="day-7 views", ylabel="videos")
    ticks = [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{t:,.0f}" for t in ticks])
    for lab, qq, c in [("median", q[.5], RED), ("75th", q[.75], "#888")]:
        ax.axvline(np.log10(qq), color=c, ls="--", lw=1.4)
        ax.text(np.log10(qq), ax.get_ylim()[1] * .95, f" {lab} {qq:,.0f}",
                color=c, fontsize=8.5, va="top")
    ax.text(.99, .95, f"25th  {q[.25]:,.0f}\n50th  {q[.5]:,.0f}\n"
                      f"75th  {q[.75]:,.0f}\n90th  {q[.9]:,.0f}\n"
                      f"99th  {q[.99]:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            family="monospace",
            bbox=dict(fc="white", ec="#ddd", boxstyle="round,pad=.5"))
    save("D8_view_distribution")


# ------------------------------------------------------------- 8. engagement
def fig_engagement(d):
    e = d[(d.d7_views > 100) & d.d7_likes.notna() & d.d7_comments.notna()].copy()
    e["likes_per_1k"] = e.d7_likes.astype(float) / e.d7_views * 1000
    e["comm_per_1k"] = e.d7_comments.astype(float) / e.d7_views * 1000
    g = e.groupby("category_name")[["likes_per_1k", "comm_per_1k"]].median()
    n = e.groupby("category_name").size()
    g = g.loc[g.likes_per_1k.sort_values().index]
    n = n.reindex(g.index)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    y = np.arange(len(g))
    ax[0].barh(y, g.likes_per_1k, color=BLUE)
    ax[0].set(yticks=y, yticklabels=g.index, title="Likes per 1,000 views",
              xlabel="median likes per 1,000 views")
    annotate(ax[0], g.likes_per_1k.values, n.values, fmt="{:,.1f}")
    ax[0].set_xlim(0, g.likes_per_1k.max() * 1.4)

    o = g.comm_per_1k.sort_values()
    ax[1].barh(np.arange(len(o)), o.values, color=GREY)
    ax[1].set(yticks=np.arange(len(o)), yticklabels=o.index,
              title="Comments per 1,000 views",
              xlabel="median comments per 1,000 views")
    annotate(ax[1], o.values, n.reindex(o.index).values, fmt="{:,.2f}")
    ax[1].set_xlim(0, o.max() * 1.4)
    fig.suptitle("Audience engagement rate by category "
                 "(videos above 100 day-7 views)", y=1.02, fontsize=11,
                 fontweight="bold")
    save("D9_engagement_by_category")


def summary(d):
    """The headline numbers, for the 'numbers of the study' the review asked for."""
    v = d.d7_views
    lines = [
        "| measure | value |", "|---|---|",
        f"| videos with a usable day-7 label | {len(d):,} |",
        f"| distinct channels | {d.channel_id.nunique():,} |",
        f"| categories represented | {d.category_name.nunique()} |",
        f"| publication window | {d.published_at.min():%d %b %Y} – "
        f"{d.published_at.max():%d %b %Y} |",
        f"| median day-7 views | {v.median():,.0f} |",
        f"| mean day-7 views | {v.mean():,.0f} |",
        f"| 25th – 75th percentile | {v.quantile(.25):,.0f} – "
        f"{v.quantile(.75):,.0f} |",
        f"| 99th percentile | {v.quantile(.99):,.0f} |",
        f"| largest single video | {v.max():,.0f} |",
        f"| videos under 100 views | {(v < 100).mean() * 100:.1f}% |",
        f"| Shorts (under 60s) | {(d.duration_seconds < 60).mean() * 100:.1f}% |",
        f"| median channel size at publication | "
        f"{d.subs.median():,.0f} subscribers |",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "summary_numbers.md"), "w", encoding="utf-8") as f:
        f.write("# Dataset in numbers\n\n" + txt + "\n")
    print("\n" + txt + "\n")


def main():
    d = load()
    print(f"loaded {len(d):,} videos with usable day-7 labels\n")
    fig_corpus(d)
    fig_category(d)
    fig_size(d)
    fig_category_x_size(d)
    fig_duration(d)
    fig_timing(d)
    fig_distribution(d)
    fig_engagement(d)
    summary(d)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
