"""Simple feature-by-feature EDA: does day-7 viewership move with each input?

One chart per feature showing the median day-7 views at each of its values,
with the number of videos behind every bar. One ranking of how much of the
variation in views each feature accounts for on its own. One table with a
plain keep or drop reason. No models.

How to read the numbers
  explains %   share of the variation in log day-7 views that the feature's
               groups account for by themselves (between-group variance over
               total variance). 0% means views look the same in every group.
  best/worst   median views in the best group divided by the worst group,
               counting only groups with at least 200 videos.

These are raw comparisons. Big channels dominate some of them (category most
of all), which is noted where it matters.

Usage:
    python Analysis/build_simple_feature_eda.py
"""
import os
import sys
import warnings
from urllib.parse import unquote, urlparse

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from paths import dataset_path

OUT = os.path.join(HERE, "eda_figures", "simple_features")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
    "axes.spines.right": False,
})
COLOURS = {"keep": "#2F6DB5", "useful, needs a new form field": "#E08B4B",
           "useful, but unknown before publishing": "#9AA5B1", "drop": "#C0392B"}
MIN_N = 200


def short(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.1f}K" if v >= 1e3 else f"{v:.0f}"


def cut(series, edges, labels):
    # Kept categorical so bins stay in their natural order on the charts.
    return pd.cut(series, edges, labels=labels, right=False)


def first_topic(value):
    if pd.isna(value):
        return pd.NA
    item = str(value).split("|")[-1]
    return unquote(urlparse(item).path.rsplit("/", 1)[-1]).replace("_", " ")


def load():
    d = pd.read_parquet(dataset_path())
    for c in ("eligible", "is_live_broadcast", "d7_usable"):
        d[c] = d[c].fillna(False).astype(bool)
    d = d[d.eligible & ~d.is_live_broadcast & d.d7_usable & d.d7_views.notna()].copy()
    d["views"] = pd.to_numeric(d.d7_views, errors="coerce")
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    inf = np.inf

    f = pd.DataFrame(index=d.index)
    subs, vids, age = num("ch_subs_at_publish"), num("ch_videos_at_publish"), num("channel_age_days_at_publish")
    f["subscribers"] = cut(subs, [0, 1e3, 1e4, 1e5, 1e6, inf], ["<1K", "1K–10K", "10K–100K", "100K–1M", "1M+"])
    f["channel video count"] = cut(vids, [0, 100, 500, 2000, 10000, inf], ["<100", "100–500", "500–2K", "2K–10K", "10K+"])
    f["channel age"] = cut(age / 365, [0, 1, 3, 5, 10, inf], ["<1 yr", "1–3 yr", "3–5 yr", "5–10 yr", "10+ yr"])
    f["channel average views per video"] = cut(num("ch_views_at_publish") / vids.where(vids > 0),
                                               [0, 100, 1e3, 1e4, 1e5, inf],
                                               ["<100", "100–1K", "1K–10K", "10K–100K", "100K+"])
    f["channel uploads per day"] = cut(vids / age.where(age > 0), [0, .1, .5, 2, 10, inf],
                                       ["<0.1", "0.1–0.5", "0.5–2", "2–10", "10+"])

    f["duration"] = cut(num("duration_seconds") / 60, [0, 1, 5, 20, 60, inf],
                        ["<1 min", "1–5 min", "5–20 min", "20–60 min", "60+ min"])
    # is_short is the real format from the player shape (vertical or square
    # and at most 3 minutes), not the old duration <= 60 s rule.
    f["Short or long-form"] = pd.Categorical(
        d.is_short.fillna(False).astype(bool).map({True: "Short", False: "long-form"}),
        categories=["Short", "long-form"], ordered=True)
    f["category"] = d.category_name.astype("string")
    lang = d.default_audio_language.astype("string").str.lower().str.split("-").str[0]
    f["audio language"] = lang.where(lang.isin(["si", "en", "ta"]), "other/unknown").map(
        {"si": "Sinhala", "en": "English", "ta": "Tamil", "other/unknown": "other/unknown"})
    f["definition"] = d.definition.astype("string")
    f["captions"] = d.caption.astype("string")
    f["made for kids"] = d.made_for_kids.astype("string").str.lower()

    slt = pd.to_datetime(d.published_at, utc=True).dt.tz_convert("Asia/Colombo")
    f["publish hour (SLT)"] = slt.dt.hour.astype("string").str.zfill(2)
    f["publish weekday"] = slt.dt.day_name().str[:3]
    f["weekend"] = d.publish_is_weekend.fillna(False).astype(bool).map({True: "yes", False: "no"})

    f["title length (characters)"] = cut(num("title_length"), [0, 31, 51, 71, 91, inf],
                                         ["≤30", "31–50", "51–70", "71–90", "91+"])
    f["title word count"] = cut(num("title_word_count"), [0, 5, 9, 13, 17, inf],
                                ["≤4", "5–8", "9–12", "13–16", "17+"])
    for c, lab in (("title_has_number", "number in title"), ("title_has_question", "question mark in title"),
                   ("title_has_exclaim", "exclamation mark in title")):
        f[lab] = d[c].fillna(False).astype(bool).map({True: "yes", False: "no"})
    f["capital letters in title"] = cut(num("title_upper_ratio"), [-.001, .0001, .1, .3, 1.01],
                                        ["none", "up to 10%", "10–30%", "over 30%"])
    f["title script"] = d.title_script.astype("string").str.replace("_script", "")

    f["tag count"] = cut(num("tag_count"), [0, 1, 6, 16, 31, inf], ["0", "1–5", "6–15", "16–30", "31+"])
    f["description length"] = cut(num("description_length"), [0, 1, 201, 1001, 3001, inf],
                                  ["empty", "1–200", "201–1K", "1K–3K", "3K+"])
    top = d.topic_categories.map(first_topic)
    common = top.value_counts().head(9).index
    f["YouTube topic"] = top.where(top.isin(common), "other").astype("string")
    return d, f


# Where each feature comes from when a creator asks for a forecast.
SOURCE = {
    "subscribers": "looked up from the channel", "channel video count": "looked up from the channel",
    "channel age": "looked up from the channel",
    "channel average views per video": "looked up from the channel",
    "channel uploads per day": "looked up from the channel",
    "duration": "on the form", "Short or long-form": "should be asked on the form", "category": "on the form",
    "audio language": "on the form", "definition": "not on the form", "captions": "not on the form",
    "made for kids": "not on the form", "publish hour (SLT)": "on the form (optional)",
    "publish weekday": "on the form (optional)", "weekend": "on the form (optional)",
    "title length (characters)": "worked out from the title", "title word count": "worked out from the title",
    "number in title": "worked out from the title", "question mark in title": "worked out from the title",
    "exclamation mark in title": "worked out from the title",
    "capital letters in title": "worked out from the title", "title script": "worked out from the title",
    "tag count": "not on the form", "description length": "not on the form",
    "YouTube topic": "not available before publishing",
}
DUPLICATES = {"title word count": "title length (characters)", "weekend": "publish weekday"}
# Features whose raw comparison is misleading, with the reason (see SF7).
OVERRIDES = {"Short or long-form": ("keep", "hidden overall because channel size is mixed in; within "
                                             "each size band Shorts get more views, 6 to 13 times more "
                                             "under 10K subscribers (SF7)")}
PAGES = {
    "SF2_channel_features": ["subscribers", "channel video count", "channel age",
                             "channel average views per video", "channel uploads per day"],
    "SF3_video_features": ["duration", "Short or long-form", "category", "audio language",
                           "definition", "captions", "made for kids"],
    "SF4_publish_time": ["publish hour (SLT)", "publish weekday", "weekend"],
    "SF5_title_features": ["title length (characters)", "title word count", "number in title",
                           "question mark in title", "exclamation mark in title",
                           "capital letters in title", "title script"],
    "SF6_tags_description_topics": ["tag count", "description length", "YouTube topic"],
}
ORDERED = {"publish weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
           "publish hour (SLT)": [f"{h:02d}" for h in range(24)]}


def summarise(d, f):
    ylog = np.log1p(d.views)
    total = ((ylog - ylog.mean()) ** 2).sum()
    rows, tables = [], {}
    for col in f.columns:
        g = f[col]
        if isinstance(g.dtype, pd.CategoricalDtype):
            g = g.cat.add_categories(["missing"]).fillna("missing")
        else:
            g = g.astype("string").fillna("missing")
        grouped = d.views.groupby(g, observed=True)
        t = pd.DataFrame({"median": grouped.median(), "n": grouped.size()})
        tables[col] = t
        yg = ylog.groupby(g, observed=True)
        between = (yg.count() * (yg.mean() - ylog.mean()) ** 2).sum()
        big = t[t.n >= MIN_N]
        rows.append({"feature": col, "explains_pct": between / total * 100,
                     "best_worst": big["median"].max() / max(big["median"].min(), 1) if len(big) > 1 else np.nan,
                     "top_share": t.n.max() / t.n.sum() * 100,
                     "missing_pct": f[col].isna().mean() * 100, "source": SOURCE[col]})
    s = pd.DataFrame(rows).set_index("feature")

    # One rule, stated the same way everywhere: a feature earns a place if it
    # explains at least 1% of the variation, or its best group gets at least
    # double the views of its worst.
    constant = s.top_share >= 95
    weak = (s.explains_pct < 1) & (s.best_worst.isna() | (s.best_worst < 2))

    def verdict(name, r):
        if name in OVERRIDES:
            return OVERRIDES[name]
        if constant[name]:
            return "drop", f"almost every video has the same value ({r.top_share:.0f}%)"
        if weak[name]:
            return "drop", "views barely change across its values"
        other = DUPLICATES.get(name)
        if other and not (constant[other] or weak[other]):
            return "drop", f"repeats {other}, which carries the same information"
        if r.source == "not available before publishing":
            return ("useful, but unknown before publishing",
                    "YouTube only assigns it after a video is uploaded")
        if r.source == "not on the form":
            return "useful, needs a new form field", "views change with it, but the form doesn't ask for it"
        return "keep", "views change clearly across its values"
    s[["verdict", "reason"]] = [verdict(n, r) for n, r in s.iterrows()]
    return s.sort_values("explains_pct", ascending=False), tables


def panel(ax, name, t, s):
    order = ORDERED.get(name)
    if order:
        t = t.reindex([o for o in order if o in t.index])
    elif name in ("category", "YouTube topic", "audio language", "title script"):
        t = t.sort_values("median", ascending=False)
    x = np.arange(len(t))
    colour = COLOURS[s.loc[name, "verdict"]]
    ax.bar(x, t["median"], color=colour, alpha=.9)
    top = t["median"].max()
    # Log scale only where medians span more than 20x. Otherwise bars start at
    # zero, so their heights can be compared by eye without exaggeration.
    use_log = top / max(t["median"].min(), 1) > 20
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(top=top * 4)
    else:
        ax.set_ylim(0, top * 1.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: short(v)))
    labels = [str(i) for i in t.index]
    rotate = len(t) > 6 or max(len(l) for l in labels) > 9
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center",
                       fontsize=7.5 if len(t) > 12 else 8)
    if len(t) <= 12:
        for i, (m, n) in enumerate(zip(t["median"], t.n)):
            y = m * 1.08 if use_log else m + top * .02
            ax.text(i, y, f"{short(m)}\nn={n:,}", ha="center", va="bottom", fontsize=6.6)
    r = s.loc[name]
    bw = "" if np.isnan(r.best_worst) else f" · best/worst {r.best_worst:.1f}×"
    ax.set_title(f"{name}\nexplains {r.explains_pct:.1f}%{bw} · {r.verdict}", fontsize=9)
    ax.set_ylabel("median day-7 views" + (" (log scale)" if use_log else ""))


def page(file, names, tables, s):
    cols = 4 if len(names) > 4 else len(names)
    rows = int(np.ceil(len(names) / cols))
    wide = [n for n in names if n in ("publish hour (SLT)",)]
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        panel(ax, name, tables[name], s)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT, f"{file}.png"))
    plt.close()
    print(f"  {file}.png")


def ranking(s):
    fig, ax = plt.subplots(figsize=(10, 8))
    r = s.iloc[::-1]
    ax.barh(range(len(r)), r.explains_pct, color=[COLOURS[v] for v in r.verdict])
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r.index)
    for i, (v, bw) in enumerate(zip(r.explains_pct, r.best_worst)):
        ax.text(v + .3, i, f"{v:.1f}%" + ("" if np.isnan(bw) else f"   best/worst {bw:.1f}×"),
                va="center", fontsize=7.8)
    ax.set_xlim(0, r.explains_pct.max() * 1.35)
    ax.set(xlabel="% of the variation in day-7 views the feature accounts for on its own",
           title="Which features go with more or fewer views")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS.values()],
              labels=list(COLOURS), loc="lower right", fontsize=8.5)
    plt.savefig(os.path.join(OUT, "SF1_feature_ranking.png"))
    plt.close()
    print("  SF1_feature_ranking.png")


def write_table(s, n):
    lines = [f"# Simple feature table ({n:,} videos with a day-7 view count)", "",
             "Explains % = share of the variation in day-7 views the feature accounts for on its own. "
             "Best/worst = median views in the best group divided by the worst (groups of 200+ videos).", "",
             "Rule: keep a feature if it explains at least 1%, or its best group gets at least double "
             "the views of its worst. Drop it if one value covers 95% or more of videos.", "",
             "| feature | explains | best/worst | most common value | where it comes from | verdict | reason |",
             "|---|---|---|---|---|---|---|"]
    for name, r in s.iterrows():
        bw = "" if np.isnan(r.best_worst) else f"{r.best_worst:.1f}×"
        lines.append(f"| {name} | {r.explains_pct:.1f}% | {bw} | {r.top_share:.0f}% | {r.source} | "
                     f"{r.verdict} | {r.reason} |")
    lines += ["", "Note: these are raw comparisons. Large channels dominate some of them, "
              "category most of all: News and Politics looks weak here only because a few channels "
              "post very many clips each. Short or long-form is the clearest case of an effect hidden "
              "this way, so its verdict is set from the channel-size split in SF7."]
    with open(os.path.join(OUT, "simple_feature_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def shorts_by_channel_size(d, f):
    """SF7: Shorts against long-form within each channel size, and within each channel."""
    fmt, tier = f["Short or long-form"], f["subscribers"]
    names = list(fmt.cat.categories)
    colours = ["#2F6DB5", "#E08B4B"]
    tiers = [t for t in tier.cat.categories if (tier == t).sum() >= MIN_N]
    med = d.views.groupby([tier, fmt], observed=True).median().unstack()
    cnt = d.views.groupby([tier, fmt], observed=True).size().unstack()

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    x, w = np.arange(len(tiers)), .36
    for i, (name, c) in enumerate(zip(names, colours)):
        m = med.reindex(tiers)[name]
        a.bar(x + (i - .5) * w, m, w, color=c, label=name)
        for xi, (v, n) in enumerate(zip(m, cnt.reindex(tiers)[name])):
            a.text(xi + (i - .5) * w, v * 1.08, f"{short(v)}\nn={n:,}", ha="center", va="bottom", fontsize=6.4)
    a.set_yscale("log")
    a.yaxis.set_minor_formatter(mticker.NullFormatter())
    a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: short(v)))
    a.set_ylim(top=med.max().max() * 5)
    a.set_xticks(x)
    a.set_xticklabels(tiers)
    a.set(xlabel="channel subscribers", ylabel="median day-7 views (log scale)",
          title="Shorts get more day-7 views than long-form at every channel size")
    a.legend(fontsize=8, loc="upper left")

    # Same channel, both formats: does its Shorts' median beat its own long-form median?
    short_, long_ = names[0], names[1]
    g = pd.DataFrame({"channel": d.channel_id, "fmt": fmt, "tier": tier, "views": d.views})
    g = g[g.fmt.isin([short_, long_])]
    per = g.groupby(["channel", "fmt"], observed=True).views.agg(["median", "size"]).unstack()
    per = per[(per["size"] >= 5).all(axis=1)]
    ratio = per["median"][short_] / per["median"][long_]
    ch_tier = g.groupby("channel").tier.agg(lambda t: t.mode().iloc[0] if len(t.mode()) else np.nan)
    r = pd.DataFrame({"ratio": ratio, "tier": ch_tier.reindex(ratio.index)})
    share = r.groupby("tier", observed=True).ratio.agg(lambda v: (v > 1).mean() * 100)
    n_ch = r.groupby("tier", observed=True).size()
    keep = [t for t in tiers if n_ch.get(t, 0) >= 10]
    b.bar(range(len(keep)), share.reindex(keep), color="#2F6DB5")
    for i, t in enumerate(keep):
        b.text(i, share[t] - 2, f"{share[t]:.0f}%\n{n_ch[t]} channels\n{r[r.tier == t].ratio.median():.1f}× typical",
               ha="center", va="top", fontsize=7.5, color="white")
    b.axhline(50, color="k", lw=.8, ls="--")
    b.set_ylim(0, 115)
    b.set_xticks(range(len(keep)))
    b.set_xticklabels(keep)
    b.set(xlabel="channel subscribers", ylabel="% of channels whose Shorts beat their own long-form",
          title=f"Same channel, both formats ({len(r)} channels, 5+ of each)")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT, "SF7_shorts_by_channel_size.png"))
    plt.close()
    print("  SF7_shorts_by_channel_size.png")
    print(med.reindex(tiers).round(0).to_string())
    print(f"\nsame-channel: Shorts higher for {(r.ratio > 1).mean() * 100:.0f}% of {len(r)} channels, "
          f"typical ratio {r.ratio.median():.2f}")
    print(pd.DataFrame({"channels": n_ch, "shorts_win_pct": share.round(0)}).to_string())


def main():
    d, f = load()
    print(f"{len(d):,} videos with a usable day-7 view count\n")
    s, tables = summarise(d, f)
    ranking(s)
    for file, names in PAGES.items():
        page(file, names, tables, s)
    shorts_by_channel_size(d, f)
    write_table(s, len(d))


if __name__ == "__main__":
    main()
