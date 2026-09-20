"""How good are the labels the model is trained against?

Every accuracy number in this project is measured against these labels, so if
they carry error, part of what looks like model error is really target noise.
Nothing so far has measured that.

Four questions:

  1. ANCHORING. A day-7 label is the observation nearest 168 hours after that
     video's own publication, not a row taken on a fixed schedule. Collection
     runs every six hours, so the nearest observation can be up to three hours
     either side. How far off is it in practice?

  2. WHAT THAT COSTS. An offset only matters if views move during it. Views per
     hour near each horizon are estimated from the next increment, so the
     anchoring error can be expressed in views and as a share of the label
     itself -- which is the number that belongs next to any accuracy claim.

  3. COVERAGE. Of the videos that could carry a label, how many do, and where
     are the rest lost: no observation in the window, outside tolerance, or no
     channel statistics predating publication.

  4. CONTAMINATION. Titles and descriptions edited after publication mean a
     feature was not what it was when the video went out.

Usage:
    python Analysis/deep_dives/label_quality/build_label_quality_figures.py
"""
import os
import sys

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import dataset_path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, GREY, RED, GREEN = "#2F6DB5", "#9AA5B1", "#C0392B", "#2E8B72"
DAYS = [7, 14, 21, 30]
TOLERANCE = 12.0        # hours, matches build_training_table.py


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close()
    print(f"  {name}.png")


def load():
    d = pd.read_parquet(dataset_path())
    for day in DAYS:
        for c in (f"d{day}_views", f"d{day}_hours_off"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ------------------------------------------------------------- 1. anchoring
def fig_anchoring(d):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for day, c in zip(DAYS, [BLUE, GREEN, "#E08B4B", RED]):
        off = d[f"d{day}_hours_off"].dropna()
        ax[0].hist(off, bins=80, histtype="step", lw=1.7, color=c,
                   label=f"day {day}  (median |off| {off.abs().median():.1f} h)")
    ax[0].axvline(0, color="#444", lw=.8)
    ax[0].set(xlabel="hours from the nominal horizon (− = early, + = late)",
              ylabel="videos", title="Where the label actually landed")
    ax[0].legend(fontsize=8)

    rows = []
    for day in DAYS:
        off = d[f"d{day}_hours_off"].dropna().abs()
        rows.append([(off <= 1).mean(), (off <= 3).mean(), (off <= 6).mean(),
                     (off <= TOLERANCE).mean()])
    rows = np.array(rows) * 100
    x = np.arange(len(DAYS))
    for i, (lab, c) in enumerate(zip(["within 1 h", "within 3 h", "within 6 h",
                                      f"within {TOLERANCE:.0f} h"],
                                     [BLUE, GREEN, "#E08B4B", GREY])):
        ax[1].bar(x + (i - 1.5) * .2, rows[:, i], width=.19, color=c, label=lab)
    ax[1].set(xticks=x, xticklabels=[f"day {v}" for v in DAYS], ylim=(0, 108),
              ylabel="% of labels", title="How tightly the labels are anchored")
    ax[1].legend(fontsize=8, ncol=2, loc="lower right")
    save("L1_label_anchoring")
    return rows


# ------------------------------------------------- 2. what the offset costs
def fig_offset_cost(d):
    """Views per hour near each horizon, taken from the following increment,
    times the offset. This converts an anchoring error in hours into an error
    in views -- the form in which it can be compared to model error."""
    pairs = [(7, 14), (14, 21), (21, 30)]
    out = []
    for a, b in pairs:
        m = d[[f"d{a}_views", f"d{b}_views", f"d{a}_hours_off"]].dropna()
        m = m[(m[f"d{a}_views"] > 0) & (m[f"d{b}_views"] >= m[f"d{a}_views"])]
        per_hour = (m[f"d{b}_views"] - m[f"d{a}_views"]) / ((b - a) * 24)
        err = per_hour * m[f"d{a}_hours_off"].abs()
        pct = (err / m[f"d{a}_views"] * 100).replace([np.inf, -np.inf], np.nan)
        out.append({"horizon": f"day {a}", "n": len(m),
                    "views_per_hour": per_hour.median(),
                    "err_views": err.median(),
                    "err_pct_median": pct.median(),
                    "err_pct_p90": pct.quantile(.90)})
    res = pd.DataFrame(out)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(res))
    ax[0].bar(x, res.err_views, color=BLUE, width=.55)
    ax[0].set(xticks=x, xticklabels=res.horizon, ylabel="views",
              title="Median label error implied by the anchoring offset")
    for i, (v, n) in enumerate(zip(res.err_views, res.n)):
        ax[0].text(i, v, f"{v:,.1f}\nn={n:,}", ha="center", va="bottom",
                   fontsize=8.5)
    ax[0].set_ylim(0, max(res.err_views.max() * 1.35, 1))

    ax[1].bar(x - .2, res.err_pct_median, width=.38, color=BLUE, label="median")
    ax[1].bar(x + .2, res.err_pct_p90, width=.38, color=GREY, label="90th pct")
    ax[1].set(xticks=x, xticklabels=res.horizon,
              ylabel="% of the label value",
              title="Same error as a share of the label")
    ax[1].legend(fontsize=8)
    for i, v in enumerate(res.err_pct_median):
        ax[1].text(i - .2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(res.err_pct_p90):
        ax[1].text(i + .2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Anchoring error is small because views barely move by the time "
                 "the label is taken", y=1.03, fontsize=10, color="#444")
    save("L2_offset_cost")
    return res


# -------------------------------------------------------------- 3. coverage
def fig_coverage(d):
    elig = d[d.eligible]
    rows = []
    for day in DAYS:
        has = elig[f"d{day}_views"].notna()
        usable = elig[f"d{day}_usable"].fillna(False).astype(bool)
        pit = usable & ~elig.channel_stats_backfilled.fillna(False).astype(bool)
        rows.append({"horizon": f"day {day}", "eligible": len(elig),
                     "observed": int(has.sum()), "usable": int(usable.sum()),
                     "modellable": int(pit.sum())})
    cov = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(cov))
    for i, (col, c, lab) in enumerate([
            ("eligible", "#E3E8ED", "eligible videos"),
            ("observed", GREY, "an observation exists"),
            ("usable", BLUE, f"within ±{TOLERANCE:.0f} h tolerance"),
            ("modellable", GREEN, "+ point-in-time channel stats")]):
        ax.bar(x + (i - 1.5) * .21, cov[col], width=.2, color=c, label=lab)
        for j, v in enumerate(cov[col]):
            ax.text(x[j] + (i - 1.5) * .21, v, f"{v/1000:.0f}k", ha="center",
                    va="bottom", fontsize=7.4, rotation=90)
    ax.set(xticks=x, xticklabels=cov.horizon, ylabel="videos",
           title="Where labels are lost between eligible and modellable")
    ax.set_ylim(0, cov.eligible.max() * 1.18)
    ax.legend(fontsize=8)
    save("L3_label_coverage")
    return cov


# --------------------------------------------------------- 4. contamination
def fig_edits(d):
    t = d.title_changed.fillna(False).astype(bool)
    de = d.description_changed.fillna(False).astype(bool)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].bar(range(3), [t.mean() * 100, de.mean() * 100, (t | de).mean() * 100],
              color=[BLUE, GREEN, GREY], width=.55)
    ax[0].set(xticks=range(3),
              xticklabels=["title edited", "description edited", "either"],
              ylabel="% of videos", title="Metadata changed after publication")
    for i, (v, n) in enumerate(zip(
            [t.mean() * 100, de.mean() * 100, (t | de).mean() * 100],
            [t.sum(), de.sum(), (t | de).sum()])):
        ax[0].text(i, v, f"{v:.2f}%\nn={n:,}", ha="center", va="bottom",
                   fontsize=8.5)
    ax[0].set_ylim(0, max(1.0, (t | de).mean() * 100 * 1.45))

    g = d.assign(edited=(t | de)).groupby("category_name").edited.agg(
        ["mean", "size"])
    g = g[g["size"] >= 200].sort_values("mean")
    ax[1].barh(range(len(g)), g["mean"] * 100, color=BLUE)
    ax[1].set(yticks=range(len(g)), yticklabels=g.index,
              xlabel="% of videos edited after publication", title="By category")
    for i, (v, n) in enumerate(zip(g["mean"], g["size"])):
        ax[1].text(v * 100 + .03, i, f"{v*100:.2f}%  n={n:,}", va="center",
                   fontsize=7.6)
    ax[1].set_xlim(0, g["mean"].max() * 100 * 1.45)
    save("L4_metadata_edits")
    return t, de


def summary(d, anchor, cost, cov, t, de):
    lines = ["| measure | value |", "|---|---|",
             f"| videos in the table | {len(d):,} |",
             f"| eligible | {int(d.eligible.sum()):,} |"]
    for i, day in enumerate(DAYS):
        off = d[f"d{day}_hours_off"].dropna().abs()
        lines.append(f"| day {day}: median anchoring offset | "
                     f"**{off.median():.1f} h** |")
    lines.append(f"| labels within 3 hours of the horizon | "
                 f"{anchor[0][1]:.1f}% (day 7) |")
    for r in cost.itertuples():
        lines.append(f"| {r.horizon}: implied label error from that offset | "
                     f"**{r.err_pct_median:.2f}%** of the label "
                     f"({r.err_views:,.0f} views) |")
    for r in cov.itertuples():
        lines.append(f"| {r.horizon}: usable → modellable | "
                     f"{r.usable:,} → {r.modellable:,} |")
    lines += [
        f"| titles edited after publication | {t.sum():,} ({t.mean()*100:.2f}%) |",
        f"| descriptions edited after publication | "
        f"{de.sum():,} ({de.mean()*100:.2f}%) |",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "label_quality_numbers.md"), "w",
              encoding="utf-8") as f:
        f.write("# Label quality in numbers\n\n" + txt + "\n")
    print("\n" + txt)


def main():
    d = load()
    print(f"{len(d):,} videos, {int(d.eligible.sum()):,} eligible\n")
    anchor = fig_anchoring(d)
    cost = fig_offset_cost(d)
    cov = fig_coverage(d)
    t, de = fig_edits(d)
    summary(d, anchor, cost, cov, t, de)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
