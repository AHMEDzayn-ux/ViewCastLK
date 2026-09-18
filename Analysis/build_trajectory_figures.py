"""Growth trajectory analysis: the shape of a video's first thirty days.

WHY THIS EXISTS NOW AND NOT BEFORE
It needs videos observed at all four horizons, and until late August there were
almost none -- 1,661 at the mid-evaluation. The model artefact still records
complete_four_horizon_rows: 0 and end_to_end_day_30_testable: false. There are
now over sixteen thousand, so the questions below are answerable for the first
time.

WHAT IT ASKS
  1. How much of day 30 is already decided by day 7? The 98.6% figure in the
     EDA came from a thin sample and is worth rechecking properly.
  2. Do trajectories have distinguishable shapes, or is there one curve with
     noise around it? The dashboard draws a trajectory, so this decides whether
     that drawing carries information.
  3. Does shape depend on category, channel size or format, or only on scale?
  4. How often does an OBSERVED trajectory fall? The model was rebuilt to make
     falling curves impossible; this measures how often reality does it, which
     is the honest check on whether that constraint is right.

Increments are reported alongside, since the model is built as a day-7 base
plus three non-negative increments and these are the distributions it learns.

Usage:
    python Analysis/build_trajectory_figures.py
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
                   "eda_figures", "trajectory")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, GREY, RED, GREEN = "#2F6DB5", "#9AA5B1", "#C0392B", "#2E8B72"
DAYS = [7, 14, 21, 30]
COLS = [f"d{d}_views" for d in DAYS]

SUB_EDGES = [0, 1e3, 1e4, 1e5, 1e6, np.inf]
SUB_LABELS = ["< 1K", "1K – 10K", "10K – 100K", "100K – 1M", "1M +"]


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close()
    print(f"  {name}.png")


def load():
    """Videos observed at every horizon, with a usable flag on each."""
    d = pd.read_parquet(dataset_path())
    usable = np.ones(len(d), bool)
    for day in DAYS:
        usable &= d[f"d{day}_usable"].fillna(False).astype(bool)
        usable &= d[f"d{day}_views"].notna()
    full = d[usable].copy()
    for c in COLS:
        full[c] = full[c].astype(float)
    full["subs"] = full.ch_subs_at_publish.astype(float)
    full["sub_band"] = pd.cut(full.subs, SUB_EDGES, labels=SUB_LABELS, right=False)
    return d, full


# --------------------------------------------------------------- 1. how early
def fig_share_by_day7(full):
    """Share of the day-30 total already present at each earlier horizon."""
    pos = full[full.d30_views > 0]
    shares = pd.DataFrame({f"day {d}": pos[f"d{d}_views"] / pos.d30_views
                           for d in DAYS})

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    med = shares.median()
    ax[0].bar(range(4), med * 100, color=[BLUE, BLUE, BLUE, GREY], width=.6)
    ax[0].set(xticks=range(4), xticklabels=shares.columns, ylim=(0, 105),
              ylabel="% of day-30 views", title="Median share of day-30 views already in")
    for i, v in enumerate(med * 100):
        ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    ax[1].hist(np.clip(shares["day 7"] * 100, 0, 100), bins=60, color=BLUE)
    ax[1].axvline(med["day 7"] * 100, color=RED, ls="--", lw=1.5)
    ax[1].set(xlabel="% of day-30 views present at day 7", ylabel="videos",
              title="Spread across videos, not just the median")
    ax[1].text(.02, .95, f"median {med['day 7']*100:.1f}%\n"
                         f"25th {shares['day 7'].quantile(.25)*100:.1f}%\n"
                         f"75th {shares['day 7'].quantile(.75)*100:.1f}%\n"
                         f"n = {len(pos):,}",
               transform=ax[1].transAxes, va="top", fontsize=8.5,
               family="monospace",
               bbox=dict(fc="white", ec="#ddd", boxstyle="round,pad=.4"))
    save("T1_share_present_by_day7")
    return shares


# ------------------------------------------------------------------ 2. shapes
def fig_shapes(full):
    """Normalised curves. Dividing by day 30 removes scale, so what remains is
    shape alone -- otherwise every plot just re-discovers that big channels are
    big."""
    pos = full[full.d30_views > 0].copy()
    norm = np.vstack([(pos[c] / pos.d30_views).values for c in COLS]).T
    norm = np.clip(norm, 0, 1.2)

    # Grouping on the day-7 share alone: it is the axis the curves actually
    # differ on, and a quartile split is defensible where an unexplained
    # clustering algorithm would need its own justification.
    q = pd.qcut(norm[:, 0], 4, labels=["slowest 25%", "second", "third",
                                       "fastest 25%"])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for lab, colour in zip(["slowest 25%", "second", "third", "fastest 25%"],
                           [RED, "#E08B4B", GREEN, BLUE]):
        # qcut on an ndarray yields a Categorical, so the comparison is already
        # a plain boolean array rather than a Series.
        m = norm[np.asarray(q == lab)].mean(axis=0)
        ax[0].plot(DAYS, m * 100, marker="o", ms=4, color=colour, label=lab)
    ax[0].set(xticks=DAYS, xlabel="days since publication",
              ylabel="% of day-30 views", ylim=(0, 105),
              title="Trajectory shape, by how fast the video started")
    ax[0].legend(fontsize=8, loc="lower right")

    # Same curves as a band: median with the 10th-90th percentile around it.
    lo, md, hi = (np.percentile(norm, p, axis=0) * 100 for p in (10, 50, 90))
    ax[1].fill_between(DAYS, lo, hi, color=BLUE, alpha=.18,
                       label="10th – 90th percentile")
    ax[1].plot(DAYS, md, color=BLUE, marker="o", ms=4, label="median")
    ax[1].set(xticks=DAYS, xlabel="days since publication",
              ylabel="% of day-30 views", ylim=(0, 105),
              title=f"All {len(pos):,} complete trajectories")
    ax[1].legend(fontsize=8, loc="lower right")
    save("T2_trajectory_shapes")


# -------------------------------------------------------- 3. shape by segment
def fig_shape_by_segment(full):
    pos = full[full.d30_views > 0].copy()
    pos["share7"] = pos.d7_views / pos.d30_views

    # The category panel's inline "n=" labels run wide, so the panels need
    # explicit width ratios and spacing rather than the default even split.
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8),
                           gridspec_kw={"width_ratios": [1.55, 1, .75],
                                        "wspace": .34})

    cat = pos.groupby("category_name").share7.agg(["median", "size"])
    cat = cat[cat["size"] >= 30].sort_values("median")
    ax[0].barh(range(len(cat)), cat["median"] * 100, color=BLUE)
    ax[0].set(yticks=range(len(cat)), yticklabels=cat.index,
              xlabel="% of day-30 views by day 7", title="By category")
    for i, (v, n) in enumerate(zip(cat["median"], cat["size"])):
        ax[0].text(v * 100 + .4, i, f"{v*100:.0f}%  n={n:,}", va="center",
                   fontsize=7.6)
    ax[0].set_xlim(0, 108)

    band = pos.groupby("sub_band", observed=True).share7.agg(["median", "size"])
    band = band.reindex([b for b in SUB_LABELS if b in band.index])
    ax[1].bar(range(len(band)), band["median"] * 100, color=BLUE, width=.6)
    ax[1].set(xticks=range(len(band)), xticklabels=band.index, ylim=(0, 108),
              ylabel="% by day 7", title="By channel size")
    ax[1].tick_params(axis="x", labelsize=8.2, rotation=20)
    for i, (v, n) in enumerate(zip(band["median"], band["size"])):
        ax[1].text(i, v * 100 + 1.5, f"{v*100:.0f}%\nn={n:,}", ha="center",
                   fontsize=8)

    fmt = pos.groupby(pos.is_short.fillna(False).astype(bool)).share7.agg(
        ["median", "size"])
    lbl = {True: "Shorts (< 60s)", False: "Long form"}
    ax[2].bar(range(len(fmt)), fmt["median"] * 100, color=BLUE, width=.5)
    ax[2].set(xticks=range(len(fmt)),
              xticklabels=[lbl.get(i, str(i)) for i in fmt.index],
              ylim=(0, 108), ylabel="% by day 7", title="By format")
    for i, (v, n) in enumerate(zip(fmt["median"], fmt["size"])):
        ax[2].text(i, v * 100 + 1.5, f"{v*100:.0f}%\nn={n:,}", ha="center",
                   fontsize=8)
    save("T3_shape_by_segment")


# -------------------------------------------------------------- 4. increments
def fig_increments(full):
    """What the model actually learns: three non-negative steps off a base."""
    inc = pd.DataFrame({
        "day 7 → 14": full.d14_views - full.d7_views,
        "day 14 → 21": full.d21_views - full.d14_views,
        "day 21 → 30": full.d30_views - full.d21_views,
    })
    falling = (inc < 0).sum()
    any_fall = (inc < 0).any(axis=1).mean() * 100

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for (name, s), c in zip(inc.items(), [BLUE, GREEN, "#E08B4B"]):
        v = s[s > 0]
        ax[0].hist(np.log10(v), bins=55, histtype="step", lw=1.8, color=c,
                   label=f"{name}  (median {s.median():,.0f})")
    ticks = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    ax[0].set(xticks=np.log10(ticks), xticklabels=[f"{t:,.0f}" for t in ticks],
              xlabel="views added during the step", ylabel="videos",
              title="Growth added at each step (positive steps)")
    ax[0].legend(fontsize=8)

    ax[1].bar(range(3), (falling / len(inc) * 100).values, color=RED, width=.5)
    ax[1].set(xticks=range(3), xticklabels=inc.columns,
              ylabel="% of videos where the count fell",
              title="Observed decreases — impossible for a cumulative count")
    for i, (n, pct) in enumerate(zip(falling, falling / len(inc) * 100)):
        ax[1].text(i, pct, f"{pct:.2f}%\nn={n:,}", ha="center", va="bottom",
                   fontsize=8.5)
    ax[1].set_ylim(0, max(1.0, (falling / len(inc) * 100).max() * 1.5))
    ax[1].text(.98, .95, f"{any_fall:.2f}% of videos fall at some step.\n"
                         "Any decrease is measurement, not reality:\n"
                         "YouTube revises counts for spam and\n"
                         "the label may anchor either side of it.",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=8,
               style="italic", color="#555")
    save("T4_increments")
    return inc, falling, any_fall


def summary(d, full, shares, inc, falling, any_fall):
    pos = full[full.d30_views > 0]
    lines = [
        "| measure | value |", "|---|---|",
        f"| videos in the table | {len(d):,} |",
        f"| observed at all four horizons | **{len(full):,}** |",
        f"| median share of day-30 views present at day 7 | "
        f"**{shares['day 7'].median()*100:.1f}%** |",
        f"| interquartile range of that share | "
        f"{shares['day 7'].quantile(.25)*100:.1f}% – "
        f"{shares['day 7'].quantile(.75)*100:.1f}% |",
        f"| median share present at day 14 | {shares['day 14'].median()*100:.1f}% |",
        f"| median share present at day 21 | {shares['day 21'].median()*100:.1f}% |",
        f"| median views added, day 7 to 14 | {inc['day 7 → 14'].median():,.0f} |",
        f"| median views added, day 21 to 30 | {inc['day 21 → 30'].median():,.0f} |",
        f"| videos whose observed count fell at some step | {any_fall:.2f}% |",
        f"| complete trajectories with zero day-30 views | "
        f"{len(full) - len(pos):,} |",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "trajectory_numbers.md"), "w",
              encoding="utf-8") as f:
        f.write("# Growth trajectories in numbers\n\n" + txt + "\n")
    print("\n" + txt)


def main():
    d, full = load()
    print(f"{len(d):,} videos in the table; {len(full):,} observed at all "
          f"four horizons\n")
    if len(full) < 500:
        print("too few complete trajectories for this analysis")
        return
    shares = fig_share_by_day7(full)
    fig_shapes(full)
    fig_shape_by_segment(full)
    inc, falling, any_fall = fig_increments(full)
    summary(d, full, shares, inc, falling, any_fall)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
