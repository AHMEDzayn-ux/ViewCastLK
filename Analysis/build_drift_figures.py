"""Do the findings hold across the collection window, or only on average?

The model trains on roughly six weeks of publications. Every result so far
pools that window, which assumes the period is homogeneous -- and it is not
obviously so, because the channel roster grew from 603 to 2,781 during it. If
the corpus changed underneath the analysis, a pooled finding could be an
artefact of composition rather than a fact about Sri Lankan YouTube.

Three checks:

  1. Is the target drifting week to week, and is any movement explained by
     which channels joined rather than by behaviour changing?
  2. Does the variance budget hold in the first half and the second?
  3. Does the category inversion -- the headline result -- replicate in both
     halves independently?

Day 7 only. Later horizons are truncated for recent publications by
construction: a video published ten days ago cannot have a day-30 label, so a
week-by-week comparison on day 30 would measure the calendar, not the data.
Even at day 7 the final week is thin for the same reason and is dropped rather
than plotted as if it were comparable.

Usage:
    python Analysis/build_drift_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import dataset_path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "eda_figures", "drift")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, GREY, RED, GREEN = "#2F6DB5", "#9AA5B1", "#C0392B", "#2E8B72"
MIN_WEEK = 400          # below this a week is too thin to compare


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close()
    print(f"  {name}.png")


def load():
    d = pd.read_parquet(dataset_path())
    d = d[d.eligible & d.d7_usable.fillna(False).astype(bool)
          & d.d7_views.notna()].copy()
    d["d7_views"] = d.d7_views.astype(float)
    d["y"] = np.log1p(d.d7_views)
    d["published_at"] = pd.to_datetime(d.published_at, utc=True)
    # to_period drops the timezone and warns about it; the week bucket does not
    # depend on the offset, so drop it deliberately rather than be told.
    d["week"] = (d.published_at.dt.tz_convert(None)
                 .dt.to_period("W").dt.start_time.dt.date)
    # Channel-mean residual: the same control the main notebook uses, so the
    # figures here are comparable to the pooled ones.
    d["channel_resid"] = d.y - d.groupby("channel_id").y.transform("mean")
    keep = d.week.value_counts()
    keep = set(keep[keep >= MIN_WEEK].index)
    dropped = sorted(set(d.week) - keep)
    d = d[d.week.isin(keep)].copy()
    return d, dropped


# ------------------------------------------------------------ 1. target drift
def fig_target_drift(d):
    g = d.groupby("week").agg(median=("d7_views", "median"),
                              n=("d7_views", "size"),
                              chans=("channel_id", "nunique"),
                              subs=("ch_subs_at_publish",
                                    lambda s: s.astype(float).median()))
    # Both panels carry a twin axis, so the inner labels need room that the
    # default spacing does not give them.
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6),
                           gridspec_kw={"wspace": .42})
    x = np.arange(len(g))

    ax[0].bar(x, g.n, color="#E3E8ED")
    ax[0].set(xticks=x, xticklabels=[f"{w:%d %b}" for w in g.index],
              ylabel="videos published", title="Volume and median views by week")
    a0 = ax[0].twinx()
    a0.plot(x, g["median"], color=BLUE, marker="o", ms=4.5, lw=1.8)
    a0.set_ylabel("median day-7 views", color=BLUE)
    a0.tick_params(axis="y", colors=BLUE)
    a0.grid(False)
    a0.set_ylim(0, g["median"].max() * 1.25)

    # If the median moves with the channels being sampled rather than on its
    # own, composition is the explanation, not behaviour.
    ax[1].plot(x, g.subs / 1000, color=GREEN, marker="s", ms=4.5, lw=1.8,
               label="median channel size (thousands of subs)")
    ax[1].set(xticks=x, xticklabels=[f"{w:%d %b}" for w in g.index],
              ylabel="thousand subscribers", title="What changed underneath it")
    a1 = ax[1].twinx()
    a1.plot(x, g.chans, color=GREY, marker="^", ms=4.5, lw=1.8,
            label="channels publishing")
    a1.set_ylabel("distinct channels", color="#666")
    a1.grid(False)
    h0, l0 = ax[1].get_legend_handles_labels()
    h1, l1 = a1.get_legend_handles_labels()
    ax[1].legend(h0 + h1, l0 + l1, fontsize=8, loc="lower left")
    save("W1_target_drift")
    return g


# ------------------------------------------------------- 2. composition drift
def fig_composition(d):
    top = d.category_name.value_counts().head(6).index
    share = (d.assign(cat=d.category_name.where(d.category_name.isin(top), "other"))
              .pivot_table(index="week", columns="cat", values="d7_views",
                           aggfunc="size", fill_value=0))
    share = share.div(share.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bottom = np.zeros(len(share))
    colours = [BLUE, GREEN, "#E08B4B", RED, "#7B5EA7", "#4FA3C7", GREY]
    for c, col in zip(share.columns, colours):
        ax.bar(range(len(share)), share[c], bottom=bottom, label=c,
               color=col, width=.72)
        bottom += share[c].values
    ax.set(xticks=range(len(share)),
           xticklabels=[f"{w:%d %b}" for w in share.index],
           ylabel="% of videos published", ylim=(0, 100),
           title="Category mix by publication week")
    ax.legend(fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(.5, -.13))
    save("W2_composition_drift")
    return share


# ---------------------------------------------------- 3. variance, half by half
def variance_budget(frame, cols):
    """Share of variance in y explained by each single categorical, the same
    one-way decomposition the main notebook reports."""
    out = {}
    tot = frame.y.var()
    for c in cols:
        g = frame.groupby(c).y
        grand = frame.y.mean()
        between = (g.count() * (g.mean() - grand) ** 2).sum() / (len(frame) - 1)
        out[c] = between / tot if tot else np.nan
    return pd.Series(out)


def fig_variance_halves(d):
    mid = d.published_at.quantile(.5)
    early, late = d[d.published_at <= mid], d[d.published_at > mid]
    cols = ["channel_id", "category_name", "publish_hour_slt", "is_short"]
    e, l = variance_budget(early, cols), variance_budget(late, cols)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(cols))
    ax.bar(x - .2, e * 100, width=.38, color=BLUE,
           label=f"first half  (n={len(early):,}, to {mid:%d %b})")
    ax.bar(x + .2, l * 100, width=.38, color=GREEN,
           label=f"second half (n={len(late):,})")
    ax.set(xticks=x, xticklabels=["channel identity", "category",
                                  "publish hour", "Shorts"],
           ylabel="% of variance in log day-7 views explained",
           title="Does the variance budget hold across the window?")
    for i, (a, b) in enumerate(zip(e, l)):
        ax.text(i - .2, a * 100, f"{a*100:.1f}", ha="center", va="bottom",
                fontsize=8.5)
        ax.text(i + .2, b * 100, f"{b*100:.1f}", ha="center", va="bottom",
                fontsize=8.5)
    ax.legend(fontsize=8)
    save("W3_variance_halves")
    return e, l, mid


# ------------------------------------------------- 4. does the inversion hold
def fig_inversion_halves(d, mid):
    def within(frame):
        multi = frame.groupby("channel_id").category_name.nunique()
        dm = frame[frame.channel_id.isin(multi[multi > 1].index)]
        g = dm.groupby("category_name").channel_resid.agg(["size", "mean", "std"])
        g["ci95"] = 1.96 * g["std"] / np.sqrt(g["size"])
        return g

    e, l = within(d[d.published_at <= mid]), within(d[d.published_at > mid])
    common = [c for c in e.index if c in l.index
              and e.loc[c, "size"] >= 25 and l.loc[c, "size"] >= 25]
    e, l = e.loc[common], l.loc[common]
    order = ((e["mean"] + l["mean"]) / 2).sort_values().index
    e, l = e.loc[order], l.loc[order]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(order))
    ax.barh(y - .2, e["mean"], xerr=e.ci95, height=.38, color=BLUE,
            label="first half", error_kw=dict(lw=.9))
    ax.barh(y + .2, l["mean"], xerr=l.ci95, height=.38, color=GREEN,
            label="second half", error_kw=dict(lw=.9))
    ax.axvline(0, color="#444", lw=.9)
    ax.set(yticks=y, yticklabels=order, xlabel="within-channel residual log views",
           title="The category inversion, measured separately in each half")
    ax.legend(fontsize=8, loc="lower right")
    save("W4_inversion_halves")
    return e, l


def summary(d, dropped, g, e_var, l_var, mid, e_inv, l_inv):
    lines = ["| measure | value |", "|---|---|",
             f"| labelled videos in the drift window | {len(d):,} |",
             f"| weeks compared | {d.week.nunique()} |",
             f"| weeks dropped as too thin | "
             f"{', '.join(f'{w:%d %b}' for w in dropped) or 'none'} |",
             f"| median day-7 views, first week to last | "
             f"{g['median'].iloc[0]:,.0f} → {g['median'].iloc[-1]:,.0f} |",
             f"| median channel size, first week to last | "
             f"{g.subs.iloc[0]:,.0f} → {g.subs.iloc[-1]:,.0f} subscribers |",
             f"| channel identity, variance explained | "
             f"{e_var['channel_id']*100:.1f}% → {l_var['channel_id']*100:.1f}% |",
             f"| category, variance explained | "
             f"{e_var['category_name']*100:.1f}% → "
             f"{l_var['category_name']*100:.1f}% |"]
    if "News & Politics" in e_inv.index and "News & Politics" in l_inv.index:
        lines.append(
            f"| News & Politics within-channel effect | "
            f"**{e_inv.loc['News & Politics','mean']:+.3f}** first half, "
            f"**{l_inv.loc['News & Politics','mean']:+.3f}** second |")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "drift_numbers.md"), "w", encoding="utf-8") as f:
        f.write("# Temporal drift in numbers\n\n" + txt + "\n")
    print("\n" + txt)


def main():
    d, dropped = load()
    print(f"{len(d):,} labelled videos across {d.week.nunique()} weeks"
          f"{f'; dropped thin weeks {dropped}' if dropped else ''}\n")
    g = fig_target_drift(d)
    fig_composition(d)
    e_var, l_var, mid = fig_variance_halves(d)
    e_inv, l_inv = fig_inversion_halves(d, mid)
    summary(d, dropped, g, e_var, l_var, mid, e_inv, l_inv)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
