"""Channel-level patterns: what a creator's own uploads say that pooled data cannot.

WHY THIS EXISTS
The product is moving to signed-in creators who share their channel's data. That
only earns its keep if a channel's own history carries information the pooled
corpus does not. Every earlier analysis in this folder averaged across channels,
so none of it can answer that. This does, using public data only; private
analytics would add to whatever is measured here, never subtract from it.

Five questions:

  G1  HISTORY. Does a channel's own recent record predict its next upload better
      than the category averages a first-time visitor would get? Strictly point in
      time: a previous upload counts only once its day-7 label exists, seven days
      after it was published. The category baselines are computed on the whole
      table, which flatters them, so any win for channel history is conservative.

  G2  HETEROGENEITY. Do the same levers -- Shorts versus long form, publishing
      time, video length -- push every channel the same way? If channels disagree,
      pooled advice is wrong for many of them and per-channel guidance is justified.

  G3  SPACING. Does an upload do worse when the same channel posted shortly
      before? Measured within channel, so a high-volume channel's lower average is
      removed before the comparison.

  G4  CONSISTENCY. Are channels that post at regular intervals different from
      bursty ones? Association only: a regular schedule may follow success rather
      than cause it.

  G5  PERSONAL MODELS. For channels with enough uploads, does a small model fitted
      on that channel's own past beat one pooled model on its future uploads? Both
      are scored against simply predicting the channel's own average, on the same
      later videos.

Usage:
    python Analysis/deep_dives/channel_guidelines/build_channel_guideline_figures.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from paths import dataset_path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, GREY, RED, GREEN, ORANGE = "#2F6DB5", "#9AA5B1", "#C0392B", "#2E8B72", "#E08B4B"
SUB_EDGES = [0, 1e3, 1e4, 1e5, 1e6, np.inf]
SUB_LABELS = ["< 1K", "1K – 10K", "10K – 100K", "100K – 1M", "1M +"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
# Collection began on 17 July; a channel's first week has no previous upload in
# the table, so spacing and consistency skip it rather than read it as a gap.
WARMUP = pd.Timestamp("2026-07-24", tz="UTC")
LN2 = np.log(2)
MULT_TICKS = [.25, .5, 1, 2, 4]


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close()
    print(f"  {name}.png")


def mult_axis(ax, lim=np.log(6)):
    """Effects are differences in log views; label them as multipliers."""
    ax.set_xticks(np.log(MULT_TICKS))
    ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"])
    ax.set_xlim(-lim, lim)


def load():
    d = pd.read_parquet(dataset_path())
    d = d[d.eligible.fillna(False).astype(bool)].copy()
    d["published_at"] = pd.to_datetime(d.published_at, utc=True)
    views = pd.to_numeric(d.d7_views, errors="coerce")
    usable = d.d7_usable.fillna(False).astype(bool) & views.notna()
    d["y"] = np.where(usable, np.log1p(views), np.nan)
    d["subs"] = pd.to_numeric(d.ch_subs_at_publish, errors="coerce")
    d["sub_band"] = pd.cut(d.subs, SUB_EDGES, labels=SUB_LABELS,
                           right=False).astype(str)
    d["short"] = d.is_short.fillna(False).astype(bool)
    d["hour"] = pd.to_numeric(d.publish_hour_slt, errors="coerce")
    d["dur"] = pd.to_numeric(d.duration_seconds, errors="coerce")
    d["weekend"] = d.publish_is_weekend.fillna(False).astype(bool)
    d = d.sort_values(["channel_id", "published_at"]).reset_index(drop=True)

    d["gap_h"] = d.groupby("channel_id").published_at.diff().dt.total_seconds() / 3600
    secs = (d.published_at - EPOCH).dt.total_seconds().values
    prior = np.zeros(len(d), dtype=int)
    for idx in d.groupby("channel_id").indices.values():
        ts = secs[idx]
        prior[idx] = np.arange(len(idx)) - np.searchsorted(ts, ts - 86400, side="left")
    d["prior24"] = prior

    lab = d[d.y.notna()].copy()
    lab["resid"] = lab.y - lab.groupby("channel_id").y.transform("mean")
    return d, lab


# ----------------------------------------------------------------- G1 history
def channel_history(lab, window=10, min_prior=3, mature_days=7):
    secs = (lab.published_at - EPOCH).dt.total_seconds().values
    y = lab.y.values
    med = np.full(len(lab), np.nan)
    for idx in lab.groupby("channel_id").indices.values():
        ts, ys = secs[idx], y[idx]
        avail = np.searchsorted(ts, ts - mature_days * 86400, side="right")
        for k, j in enumerate(avail):
            if j >= min_prior:
                med[idx[k]] = np.median(ys[max(0, j - window):j])
    return lab.assign(
        hist_med=med,
        cat_med=lab.groupby("category_name").y.transform("median"),
        catsize_med=lab.groupby(["category_name", "sub_band"]).y.transform("median"))


def fig_history(lab):
    h = channel_history(lab)
    h = h[h.hist_med.notna()]
    preds = [("category average", "cat_med", GREY),
             ("category × channel size", "catsize_med", ORANGE),
             ("channel's last 10 uploads", "hist_med", BLUE)]
    res = {}
    for name, col, _ in preds:
        err = (h.y - h[col]).abs()
        res[name] = {"mdae": err.median(), "within2": (err < LN2).mean() * 100}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"wspace": .3})
    x = np.arange(len(preds))
    vals = [res[p[0]]["within2"] for p in preds]
    ax[0].bar(x, vals, color=[p[2] for p in preds], width=.6)
    ax[0].set(xticks=x, xticklabels=[p[0] for p in preds], ylim=(0, 100),
              ylabel="% of uploads predicted within 2×",
              title=f"What a channel's own record is worth (n={len(h):,})")
    ax[0].tick_params(axis="x", labelsize=8.4)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    for (name, col, c) in preds:
        by = h.groupby("sub_band").apply(
            lambda g: ((g.y - g[col]).abs() < LN2).mean() * 100)
        by = by.reindex([b for b in SUB_LABELS if b in by.index])
        ax[1].plot(range(len(by)), by.values, marker="o", ms=4, color=c, label=name)
    ax[1].set(xticks=range(len(SUB_LABELS)), xticklabels=SUB_LABELS, ylim=(0, 100),
              ylabel="% within 2×", title="By channel size")
    ax[1].tick_params(axis="x", labelsize=8.2, rotation=15)
    ax[1].legend(fontsize=8)
    save("G1_channel_history_vs_averages")
    return res, len(h)


# ----------------------------------------------------------- G2 heterogeneity
def per_channel_effects(lab, min_videos=30, min_group=8):
    n = lab.groupby("channel_id").size()
    big = lab[lab.channel_id.isin(n[n >= min_videos].index)].copy()
    big["band"] = pd.cut(big.hour, [0, 6, 12, 18, 24], right=False,
                         labels=["00–06", "06–12", "12–18", "18–24"]).astype(str)
    shorts, bands, durs = [], [], []
    for cid, g in big.groupby("channel_id"):
        a, b = g.resid[g.short], g.resid[~g.short]
        if len(a) >= min_group and len(b) >= min_group:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            shorts.append({"channel_id": cid, "effect": a.mean() - b.mean(), "p": p})
        bm = g[g.band != "nan"].groupby("band").resid.agg(["mean", "size"])
        bm = bm[bm["size"] >= min_group]
        if len(bm) >= 2:
            _, p = stats.f_oneway(*[g.resid[g.band == k].values for k in bm.index])
            bands.append({"channel_id": cid, "best": bm["mean"].idxmax(),
                          "spread": bm["mean"].max() - bm["mean"].min(), "p": p})
        lf = g[(~g.short) & (g.dur > 0)]
        if len(lf) >= 20:
            rho, p = stats.spearmanr(np.log(lf.dur), lf.resid)
            if not np.isnan(rho):
                durs.append({"channel_id": cid, "rho": rho, "p": p})
    return (pd.DataFrame(shorts), pd.DataFrame(bands), pd.DataFrame(durs),
            big.channel_id.nunique())


def fig_heterogeneity(lab):
    sh, bd, du, n_big = per_channel_effects(lab)
    both = lab.groupby("channel_id").short.transform("nunique") == 2
    pooled_short = (lab[both & lab.short].resid.mean()
                    - lab[both & ~lab.short].resid.mean())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7), gridspec_kw={"wspace": .3})

    lim = np.log(6)
    ax[0].hist(np.clip(sh.effect, -lim, lim), bins=35, color=BLUE)
    # Values beyond the axis are clipped into the end bins; say how many, or the
    # tall edge bar reads as a mode rather than a pile-up of extremes.
    pinned = int((sh.effect.abs() >= lim).sum())
    ax[0].axvline(0, color="#444", lw=.8)
    ax[0].axvline(pooled_short, color=RED, ls="--", lw=1.5,
                  label=f"pooled effect {np.exp(pooled_short):.2f}×")
    mult_axis(ax[0])
    ax[0].set(xlabel="Shorts relative to the same channel's long form",
              ylabel="channels", title=f"Shorts effect, per channel (n={len(sh)})")
    ax[0].legend(fontsize=8, loc="upper left")
    sig_up = ((sh.p < .05) & (sh.effect > 0)).mean() * 100
    sig_dn = ((sh.p < .05) & (sh.effect < 0)).mean() * 100
    ax[0].text(.9, .95, f"clearly better: {sig_up:.0f}%\nclearly worse: {sig_dn:.0f}%\n"
                        f"beyond 6× either way: {pinned}",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=8.5,
               bbox=dict(fc="white", ec="#ddd", boxstyle="round,pad=.4"))

    best = bd.best.value_counts().reindex(["00–06", "06–12", "12–18", "18–24"]).fillna(0)
    ax[1].bar(range(4), best.values / len(bd) * 100, color=GREEN, width=.6)
    ax[1].set(xticks=range(4), xticklabels=[f"{b}\nSLT" for b in best.index],
              ylabel="% of channels", title=f"Each channel's best time band (n={len(bd)})")
    for i, v in enumerate(best.values / len(bd) * 100):
        ax[1].text(i, v + .8, f"{v:.0f}%", ha="center", fontsize=9)
    # The two tallest bands are on the right, so the note goes top left, with
    # headroom so the bar labels clear it.
    ax[1].set_ylim(0, best.max() / len(bd) * 100 * 1.45)
    ax[1].text(.02, .97, f"median best–worst gap: {np.exp(bd.spread.median()):.2f}×\n"
                         f"differs clearly (p<.05): {(bd.p < .05).mean()*100:.0f}%",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=8.5,
               bbox=dict(fc="white", ec="#ddd", boxstyle="round,pad=.4"))

    ax[2].hist(du.rho, bins=30, color=ORANGE, range=(-1, 1))
    ax[2].axvline(0, color="#444", lw=.8)
    ax[2].set(xlabel="Spearman correlation of length with performance",
              ylabel="channels", title=f"Longer long-form videos, per channel (n={len(du)})")
    ax[2].text(.98, .95, f"longer clearly better: {((du.p<.05)&(du.rho>0)).mean()*100:.0f}%\n"
                         f"longer clearly worse: {((du.p<.05)&(du.rho<0)).mean()*100:.0f}%",
               transform=ax[2].transAxes, ha="right", va="top", fontsize=8.5,
               bbox=dict(fc="white", ec="#ddd", boxstyle="round,pad=.4"))
    fig.suptitle(f"Channels with at least 30 labelled uploads ({n_big}); effects are "
                 "within channel", y=1.03, fontsize=10, color="#444")
    save("G2_lever_effects_per_channel")
    return sh, bd, du, pooled_short, n_big


# ---------------------------------------------------------------- G3 spacing
def fig_spacing(lab):
    s = lab[lab.gap_h.notna() & (lab.published_at >= WARMUP)].copy()
    s["gap_bin"] = pd.cut(s.gap_h, [0, 1, 6, 24, 72, np.inf], right=False,
                          labels=["< 1 h", "1–6 h", "6–24 h", "1–3 days", "> 3 days"])
    s["prior_bin"] = pd.cut(s.prior24, [-.5, .5, 1.5, 4.5, 9.5, np.inf],
                            labels=["0", "1", "2–4", "5–9", "10+"])
    s["group"] = np.where(s.category_name == "News & Politics",
                          "News & Politics", "other categories")
    tables = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"wspace": .28})
    for k, (col, title, xlabel) in enumerate([
            ("gap_bin", "Time since the channel's previous upload", "gap"),
            ("prior_bin", "Uploads by the same channel in the previous 24 h",
             "earlier uploads in the last 24 h")]):
        for name, sub, c in [("all", s, "#444"),
                             ("News & Politics", s[s.group == "News & Politics"], RED),
                             ("other categories", s[s.group != "News & Politics"], BLUE)]:
            g = sub.groupby(col, observed=True).resid.agg(["mean", "sem", "size"])
            g = g[g["size"] >= 50]
            xs = [list(s[col].cat.categories).index(i) for i in g.index]
            ax[k].errorbar(xs, np.exp(g["mean"]), yerr=[
                np.exp(g["mean"]) - np.exp(g["mean"] - 1.96 * g["sem"]),
                np.exp(g["mean"] + 1.96 * g["sem"]) - np.exp(g["mean"])],
                marker="o", ms=4, capsize=3, lw=1.6, color=c, label=name)
            tables[(col, name)] = g
        cats = list(s[col].cat.categories)
        ax[k].axhline(1, color="#888", lw=.8)
        ax[k].set(xticks=range(len(cats)), xticklabels=cats, xlabel=xlabel,
                  ylabel="views relative to the channel's own average", title=title)
        ax[k].legend(fontsize=8)
    save("G3_upload_spacing")
    return tables, len(s)


# ------------------------------------------------------------ G4 consistency
def fig_consistency(d, lab):
    act = d[d.published_at >= WARMUP]
    span = act.groupby("channel_id").published_at.agg(["min", "max", "size"])
    weeks = ((span["max"] - span["min"]).dt.total_seconds() / 604800).clip(lower=1)
    gap_cv = act.groupby("channel_id").gap_h.agg(
        lambda x: x.std() / x.mean() if x.notna().sum() >= 5 and x.mean() > 0 else np.nan)
    c = pd.DataFrame({
        "per_week": span["size"] / weeks, "gap_cv": gap_cv,
        "med_y": lab.groupby("channel_id").y.median(),
        "n_lab": lab.groupby("channel_id").size(),
        "band": lab.groupby("channel_id").sub_band.agg(lambda x: x.mode().iat[0]),
    }).dropna()
    c = c[c.n_lab >= 10]

    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": .25})
    colours = dict(zip(SUB_LABELS, [GREY, ORANGE, GREEN, BLUE, RED]))
    for band in SUB_LABELS:
        sub = c[c.band == band]
        if len(sub) < 20:
            continue
        r_cv = stats.spearmanr(sub.gap_cv, sub.med_y)
        r_pw = stats.spearmanr(sub.per_week, sub.med_y)
        rows.append({"band": band, "channels": len(sub), "rho_gap_cv": r_cv[0],
                     "p_gap_cv": r_cv[1], "rho_per_week": r_pw[0], "p_per_week": r_pw[1]})
        ax[0].scatter(sub.gap_cv, np.expm1(sub.med_y), s=9, alpha=.5,
                      color=colours[band], label=f"{band}  ρ={r_cv[0]:+.2f}")
        ax[1].scatter(sub.per_week, np.expm1(sub.med_y), s=9, alpha=.5,
                      color=colours[band], label=f"{band}  ρ={r_pw[0]:+.2f}")
    for a, xl, t in [(ax[0], "irregularity of upload gaps (coefficient of variation)",
                      "Regular versus bursty posting"),
                     (ax[1], "uploads per week", "Posting volume")]:
        a.set(xscale="log", yscale="log", xlabel=xl,
              ylabel="channel's median day-7 views", title=t)
        a.legend(fontsize=7.6, title="size band, within-band ρ", title_fontsize=7.6)
    save("G4_posting_consistency")
    return pd.DataFrame(rows), len(c)


# --------------------------------------------------------- G5 personal models
def features(x):
    gap = x.gap_h.fillna(x.gap_h.median()).clip(upper=24 * 30)
    return pd.DataFrame({
        "short": x.short.astype(float), "weekend": x.weekend.astype(float),
        "h_sin": np.sin(2 * np.pi * x.hour.fillna(12) / 24),
        "h_cos": np.cos(2 * np.pi * x.hour.fillna(12) / 24),
        "log_dur": np.log1p(x.dur.fillna(x.dur.median())),
        "title_length": pd.to_numeric(x.title_length, errors="coerce").fillna(0),
        "tag_count": pd.to_numeric(x.tag_count, errors="coerce").fillna(0),
        "log_prior24": np.log1p(x.prior24), "log_gap": np.log1p(gap),
    }, index=x.index)


def fig_personal(lab, min_train=40, min_test=15):
    cut = lab.published_at.quantile(.7)
    train, test = lab[lab.published_at < cut], lab[lab.published_at >= cut]
    cmean = train.groupby("channel_id").y.mean()
    train = train[train.channel_id.isin(cmean.index)].copy()
    test = test[test.channel_id.isin(cmean.index)].copy()
    train["r"] = train.y - train.channel_id.map(cmean)
    test["r"] = test.y - test.channel_id.map(cmean)

    pooled = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                           min_samples_leaf=40, random_state=0)
    pooled.fit(features(train), train.r)
    test["pool"] = pooled.predict(features(test))

    ntr, nte = train.groupby("channel_id").size(), test.groupby("channel_id").size()
    chans = [c for c in ntr.index if ntr[c] >= min_train and nte.get(c, 0) >= min_test]
    rows = []
    for cid in chans:
        tr, te = train[train.channel_id == cid], test[test.channel_id == cid]
        sc = StandardScaler().fit(features(tr))
        own = Ridge(alpha=3.0).fit(sc.transform(features(tr)), tr.r)
        pred = own.predict(sc.transform(features(te)))
        base = te.r.abs().mean()
        if base <= 0:
            continue
        rows.append({"channel_id": cid, "n_test": len(te),
                     "pool_gain": (base - (te.r - te.pool).abs().mean()) / base * 100,
                     "own_gain": (base - (te.r - pred).abs().mean()) / base * 100})
    res = pd.DataFrame(rows)

    gkf_r2 = []
    X, yv, groups = features(lab), lab.resid, lab.channel_id
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, yv, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                          min_samples_leaf=40, random_state=0)
        m.fit(X.iloc[tr_i], yv.iloc[tr_i])
        gkf_r2.append(r2_score(yv.iloc[te_i], m.predict(X.iloc[te_i])))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"wspace": .28,
                                                               "width_ratios": [1.6, 1]})
    bins = np.linspace(-40, 40, 33)
    ax[0].hist(np.clip(res.pool_gain, -40, 40), bins=bins, histtype="step", lw=1.8,
               color=GREY, label=f"one pooled model  (median {res.pool_gain.median():+.1f}%)")
    ax[0].hist(np.clip(res.own_gain, -40, 40), bins=bins, histtype="step", lw=1.8,
               color=BLUE, label=f"model of the channel's own past  "
                                 f"(median {res.own_gain.median():+.1f}%)")
    ax[0].axvline(0, color="#444", lw=.8)
    beyond = int(((res.own_gain.abs() >= 40) | (res.pool_gain.abs() >= 40)).sum())
    ax[0].set(xlabel="error reduction versus predicting the channel's own average (%)",
              ylabel="channels",
              title=f"Later uploads of {len(res)} channels, trained before {cut:%d %b}")
    # The clipping note rides in the legend title; as free text it collided
    # with the legend.
    ax[0].legend(fontsize=8, loc="upper left", title_fontsize=7.6,
                 title=f"end bars hold everything beyond ±40% ({beyond} channels)")

    ax[1].bar([0], [np.mean(gkf_r2)], yerr=[np.std(gkf_r2)], color=GREEN, width=.5,
              capsize=5)
    ax[1].axhline(0, color="#444", lw=.8)
    # The figure goes in the tick label: placed above the error bar it collided
    # with the title.
    ax[1].set(xticks=[0], xlim=(-.8, .8),
              xticklabels=[f"public levers, unseen channels\n"
                           f"R² {np.mean(gkf_r2):.3f} ± {np.std(gkf_r2):.3f}"],
              ylabel="R² of within-channel variation",
              title="How much public metadata explains")
    save("G5_personal_vs_pooled_models")
    return res, gkf_r2, cut


def summary(n_hist, hist, sh, bd, du, pooled_short, n_big, spacing, n_sp,
            cons, n_cons, pers, gkf_r2, cut):
    g = lambda key: spacing.get(key)
    lines = ["| measure | value |", "|---|---|"]
    for name, r in hist.items():
        lines.append(f"| G1 · {name}: uploads predicted within 2× | "
                     f"{r['within2']:.1f}% (n={n_hist:,}) |")
    lines += [
        f"| G2 · channels with ≥30 labelled uploads | {n_big} |",
        f"| G2 · pooled Shorts effect (within channel) | {np.exp(pooled_short):.2f}× |",
        f"| G2 · channels where Shorts clearly beat their long form | "
        f"{((sh.p<.05)&(sh.effect>0)).mean()*100:.0f}% of {len(sh)} |",
        f"| G2 · channels where Shorts clearly lose | "
        f"{((sh.p<.05)&(sh.effect<0)).mean()*100:.0f}% of {len(sh)} |",
        f"| G2 · median gap between a channel's best and worst time band | "
        f"{np.exp(bd.spread.median()):.2f}× (n={len(bd)}) |",
        f"| G2 · channels whose time bands differ clearly | {(bd.p<.05).mean()*100:.0f}% |",
        f"| G2 · most common best band | {bd.best.value_counts().idxmax()} SLT "
        f"({bd.best.value_counts(normalize=True).max()*100:.0f}% of channels) |",
        f"| G2 · longer long-form clearly better / worse | "
        f"{((du.p<.05)&(du.rho>0)).mean()*100:.0f}% / {((du.p<.05)&(du.rho<0)).mean()*100:.0f}% "
        f"of {len(du)} |",
    ]
    for label, key in [("gap < 1 h", ("gap_bin", "all")), ("gap > 3 days", ("gap_bin", "all"))]:
        t = g(key)
        idx = "< 1 h" if "1 h" in label else "> 3 days"
        if t is not None and idx in t.index:
            lines.append(f"| G3 · {label}, all categories | {np.exp(t.loc[idx,'mean']):.2f}× "
                         f"the channel's average (n={int(t.loc[idx,'size']):,}) |")
    for grp in ["News & Politics", "other categories"]:
        t = g(("prior_bin", grp))
        if t is not None and "0" in t.index and "10+" in t.index:
            lines.append(f"| G3 · {grp}: no upload in prior 24 h vs 10+ | "
                         f"{np.exp(t.loc['0','mean']):.2f}× vs {np.exp(t.loc['10+','mean']):.2f}× |")
    for r in cons.itertuples():
        lines.append(f"| G4 · {r.band}: irregular gaps vs median views (ρ) | "
                     f"{r.rho_gap_cv:+.2f} (p={r.p_gap_cv:.3f}, {r.channels} channels) |")
    lines += [
        f"| G5 · channels evaluated (trained before {cut:%d %b}) | {len(pers)} |",
        f"| G5 · median error reduction, one pooled model | {pers.pool_gain.median():+.1f}% |",
        f"| G5 · median error reduction, channel's own model | {pers.own_gain.median():+.1f}% |",
        f"| G5 · channels where own model beats pooled | "
        f"{(pers.own_gain > pers.pool_gain).mean()*100:.0f}% |",
        f"| G5 · within-channel R² from public levers, unseen channels | "
        f"{np.mean(gkf_r2):.3f} ± {np.std(gkf_r2):.3f} |",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "channel_guideline_numbers.md"), "w", encoding="utf-8") as f:
        f.write("# Channel-level patterns in numbers\n\n" + txt + "\n")
    print("\n" + txt)


def main():
    d, lab = load()
    print(f"{len(d):,} eligible videos, {len(lab):,} with usable day-7 labels, "
          f"{lab.channel_id.nunique():,} channels\n")
    hist, n_hist = fig_history(lab)
    sh, bd, du, pooled_short, n_big = fig_heterogeneity(lab)
    spacing, n_sp = fig_spacing(lab)
    cons, n_cons = fig_consistency(d, lab)
    pers, gkf_r2, cut = fig_personal(lab)
    summary(n_hist, hist, sh, bd, du, pooled_short, n_big, spacing, n_sp,
            cons, n_cons, pers, gkf_r2, cut)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
