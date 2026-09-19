"""Helpers for viewership_eda.ipynb: every chart in that notebook is drawn from here.

Everything is a count, a median or a percentage. Two comparisons are used
throughout, and both are explained at the top of the notebook:

  x typical        median views of a group divided by the median of all
                   videos at the same horizon. 2.0 means twice the typical
                   video, 0.5 means half.
  x own channel    each video's views divided by its own channel's average
                   (on the log scale, channels with at least 5 labelled videos
                   that use more than one value of the input), averaged over
                   the group. It removes channel size from the comparison.

Figures are shown inline and also saved to figures/ next to this file.
"""
import os
import warnings
from urllib.parse import unquote, urlparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.colors import LogNorm

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").is_file())))
from paths import dataset_path

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

H = (7, 14, 21, 30)
H_COLOURS = {7: "#1B4F8A", 14: "#3E7CC0", 21: "#E08B4B", 30: "#B03A2E"}
RAW, OWN = "#2F6DB5", "#E08B4B"
MIN_CH = 5          # labelled videos a channel needs for the own-channel comparison
MIN_N = 200         # group size counted for best/worst
MIN_SHOW = 50       # groups smaller than this are left off the views chart
MIN_CELL = 100      # grid cells smaller than this are left blank
MIN_CELL_CH = 5     # ... or drawn from fewer channels than this
MIN_BAND_CH = 20    # channels each group needs in the per-size-band keep rule
OWN_MIN = 1.2       # keep rule for video inputs (best/worst gap, own-channel view)
RAW_MIN = 1.0       # keep rule for channel inputs (% explained)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
    "axes.spines.right": False,
})
_fig_no = [0]


def save(fig, name):
    _fig_no[0] += 1
    fig.savefig(os.path.join(OUT, f"{_fig_no[0]:02d}_{name}.png"))
    plt.show()
    plt.close(fig)


def say(text):
    display(Markdown(text))


def short(v):
    if pd.isna(v):
        return ""
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.1f}K" if v >= 1e3 else f"{v:.0f}"


def times(v):
    if pd.isna(v):
        return ""
    return f"{v:.2f}×" if v < 2 else f"{v:.1f}×" if v < 10 else f"{v:.0f}×"


def cut(series, edges, labels):
    return pd.cut(series, edges, labels=labels, right=False)


def yesno(s):
    return pd.Categorical(s.fillna(False).astype(bool).map({True: "yes", False: "no"}),
                          categories=["no", "yes"], ordered=True)


def first_topic(value):
    if pd.isna(value) or not str(value):
        return np.nan
    item = str(value).split("|")[-1]
    return unquote(urlparse(item).path.rsplit("/", 1)[-1]).replace("_", " ")


# ------------------------------------------------------------------ loading
def load():
    raw = pd.read_parquet(dataset_path())
    for c in ("eligible", "is_live_broadcast", "channel_stats_backfilled", "title_changed",
              "description_changed", "is_short") + tuple(f"d{h}_usable" for h in H):
        raw[c] = raw[c].fillna(False).astype(bool)
    d = raw[raw.eligible & ~raw.is_live_broadcast].copy()
    d["published"] = pd.to_datetime(d.published_at, utc=True).dt.tz_convert("Asia/Colombo")
    for h in H:
        v = pd.to_numeric(d[f"d{h}_views"], errors="coerce")
        d[f"v{h}"] = v.where(d[f"d{h}_usable"])
        lv = np.log1p(d[f"v{h}"])
        g = lv.groupby(d.channel_id)
        d[f"rel{h}"] = (lv - g.transform("mean")).where(g.transform("count") >= MIN_CH)
    d["all4"] = d[[f"d{h}_usable" for h in H]].all(axis=1)
    return raw, d


# ----------------------------------------------------------------- features
# name -> (section, where it comes from at forecast time, what it is, kind)
# kind "channel": same for every video of a channel, so the own-channel view
# does not apply and the keep rule uses the raw view instead.
META = {}


def _m(name, section, source, what, kind="video"):
    META[name] = dict(section=section, source=source, what=what, kind=kind)


def build_features(d):
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    inf = np.inf
    f = pd.DataFrame(index=d.index)
    pit = ~d.channel_stats_backfilled   # channel inputs only where recorded before publishing
    subs, vids, age = num("ch_subs_at_publish"), num("ch_videos_at_publish"), num("channel_age_days_at_publish")

    f["subscribers"] = cut(subs.where(pit), [0, 1e3, 1e4, 1e5, 1e6, inf],
                           ["<1K", "1K–10K", "10K–100K", "100K–1M", "1M+"])
    _m("subscribers", "channel", "looked up from the channel", "subscriber count on the day the video was published", "channel")
    f["channel video count"] = cut(vids.where(pit), [0, 100, 500, 2000, 10000, inf],
                                   ["<100", "100–500", "500–2K", "2K–10K", "10K+"])
    _m("channel video count", "channel", "looked up from the channel", "videos the channel had already published", "channel")
    f["channel age"] = cut((age / 365).where(pit), [0, 1, 3, 5, 10, inf],
                           ["<1 yr", "1–3 yr", "3–5 yr", "5–10 yr", "10+ yr"])
    _m("channel age", "channel", "looked up from the channel", "years since the channel was created", "channel")
    f["channel average views per video"] = cut((num("ch_views_at_publish") / vids.where(vids > 0)).where(pit),
                                               [0, 100, 1e3, 1e4, 1e5, inf],
                                               ["<100", "100–1K", "1K–10K", "10K–100K", "100K+"])
    _m("channel average views per video", "channel", "looked up from the channel",
       "channel's total views divided by its video count", "channel")
    f["channel uploads per day"] = cut((vids / age.where(age > 0)).where(pit), [0, .1, .5, 2, 10, inf],
                                       ["<0.1", "0.1–0.5", "0.5–2", "2–10", "10+"])
    _m("channel uploads per day", "channel", "worked out from the channel",
       "video count divided by channel age: how often it posts", "channel")
    f["channel country"] = d.channel_country.fillna("not set").astype("string")
    _m("channel country", "channel", "looked up from the channel", "country the channel lists", "channel")

    f["duration"] = cut(num("duration_seconds") / 60, [0, 1, 3, 5, 20, 60, inf],
                        ["<1 min", "1–3 min", "3–5 min", "5–20 min", "20–60 min", "60+ min"])
    _m("duration", "format", "on the form", "video length")
    f["Short or long-form"] = pd.Categorical(d.is_short.map({True: "Short", False: "long-form"}),
                                             categories=["Short", "long-form"], ordered=True)
    _m("Short or long-form", "format", "not sent yet (form needs the field)",
       "real YouTube Short: vertical or square and at most 3 minutes")
    f["category"] = d.category_name.replace("", np.nan).astype("string")
    _m("category", "format", "on the form", "YouTube category the creator picks", "channel")
    f["HD or SD"] = d.definition.str.upper().astype("string")
    _m("HD or SD", "format", "not on the form", "video resolution")
    f["captions"] = d.caption.astype("string").str.lower()
    _m("captions", "format", "not on the form", "whether the uploader added captions", "channel")
    f["made for kids"] = d.made_for_kids.astype("string").str.lower()
    _m("made for kids", "format", "not on the form", "YouTube made-for-kids setting", "channel")

    audio = d.default_audio_language.astype("string").replace("", pd.NA).str.lower().str.split("-").str[0]
    names = {"si": "Sinhala", "en": "English", "ta": "Tamil"}
    f["audio language"] = audio.map(names).fillna(audio.where(audio.isna(), "other")).fillna("not set")
    _m("audio language", "language", "on the form", "spoken language the uploader set")
    meta_lang = d.default_language.astype("string").replace("", pd.NA).str.lower().str.split("-").str[0]
    f["default language"] = meta_lang.map(names).fillna(meta_lang.where(meta_lang.isna(), "other")).fillna("not set")
    _m("default language", "language", "on the form (mapped from audio language)",
       "language of the title and description the uploader set")
    f["title script"] = d.title_script.astype("string").str.replace("_script", "").str.capitalize()
    _m("title script", "language", "worked out from the title (not sent yet)", "writing system the title is in")

    f["publish hour"] = pd.Categorical(d.published.dt.hour.map(lambda x: f"{x:02d}"),
                                       categories=[f"{x:02d}" for x in range(24)], ordered=True)
    _m("publish hour", "timing", "on the form (optional)", "hour of publishing, Sri Lanka time")
    band = pd.cut(d.published.dt.hour, [0, 6, 15, 21, 24], right=False,
                  labels=["night (0–5)", "morning–afternoon (6–14)", "evening (15–20)", "late night (21–23)"])
    f["publish time band"] = band
    _m("publish time band", "timing", "on the form (optional)",
       "the four bands the API groups the hour into")
    f["publish weekday"] = pd.Categorical(d.published.dt.day_name().str[:3],
                                          categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ordered=True)
    _m("publish weekday", "timing", "on the form (optional)", "day of publishing, Sri Lanka time")
    f["weekend"] = yesno(d.publish_is_weekend)
    _m("weekend", "timing", "on the form (optional)", "published on Saturday or Sunday")

    f["title length"] = cut(num("title_length"), [0, 31, 51, 71, 91, inf], ["≤30", "31–50", "51–70", "71–90", "91+"])
    _m("title length", "title", "worked out from the title (not sent yet)", "characters in the title")
    f["title word count"] = cut(num("title_word_count"), [0, 5, 9, 13, 17, inf], ["≤4", "5–8", "9–12", "13–16", "17+"])
    _m("title word count", "title", "worked out from the title (not sent yet)", "words in the title")
    for c, lab in (("title_has_number", "number in title"), ("title_has_question", "question mark in title"),
                   ("title_has_exclaim", "exclamation mark in title")):
        f[lab] = yesno(d[c])
        _m(lab, "title", "worked out from the title (not sent yet)", lab)
    f["capital letters in title"] = cut(num("title_upper_ratio"), [-.001, .0001, .1, .3, 1.01],
                                        ["none", "up to 10%", "10–30%", "over 30%"])
    _m("capital letters in title", "title", "worked out from the title (not sent yet)",
       "share of the title's letters that are capitals")

    f["tag count"] = cut(num("tag_count"), [0, 1, 6, 16, 31, inf], ["0", "1–5", "6–15", "16–30", "31+"])
    _m("tag count", "text", "not on the form", "number of tags")
    f["description length"] = cut(num("description_length"), [0, 1, 201, 1001, 3001, inf],
                                  ["empty", "1–200", "201–1K", "1K–3K", "3K+"])
    _m("description length", "text", "not on the form", "characters in the description")

    top = d.topic_categories.map(first_topic)
    common = top.value_counts().head(9).index
    f["YouTube topic"] = top.where(top.isin(common) | top.isna(), "other").astype("string")
    _m("YouTube topic", "channel", "looked up from the channel (not sent yet)",
       "topic YouTube assigns to the channel (from the channel, not the video)", "channel")
    for c in f.columns:
        if not isinstance(f[c].dtype, pd.CategoricalDtype):
            f[c] = f[c].astype("object").where(f[c].notna(), np.nan)
    return f


# ------------------------------------------------------------ core numbers
def explains(y, g):
    ok = y.notna() & g.notna()
    y, g = y[ok], g[ok]
    if len(y) < 2:
        return np.nan
    tot = ((y - y.mean()) ** 2).sum()
    m = y.groupby(g, observed=True).agg(["mean", "count"])
    return (m["count"] * (m["mean"] - y.mean()) ** 2).sum() / tot * 100


def own_rows(d, g, h=7):
    """Videos usable in the own-channel view of input g: the channel has at
    least MIN_CH labelled videos and uses two or more values of g. A channel
    that only ever uses one value sits at exactly 1x by definition and says
    nothing about the input, so it is left out."""
    rel = d[f"rel{h}"]
    ok = rel.notna() & g.notna()
    n_values = g[ok].groupby(d.channel_id[ok], observed=True).nunique()
    return ok & d.channel_id.map(n_values).fillna(0).ge(2)


def group_table(d, g, h=7, rows=None):
    if rows is not None:
        d, g = d[rows], g[rows]
    v, rel = d[f"v{h}"], d[f"rel{h}"]
    base = v.median()
    ok = v.notna() & g.notna()
    grp = v[ok].groupby(g[ok], observed=True)
    t = pd.DataFrame({"videos": grp.size(), "median": grp.median(),
                      "q25": grp.quantile(.25), "q75": grp.quantile(.75)})
    t["share"] = t.videos / t.videos.sum() * 100
    t["raw_x"] = t["median"] / base
    t["q25_x"], t["q75_x"] = t.q25 / base, t.q75 / base
    okr = own_rows(d, g, h)
    own = rel[okr].groupby(g[okr], observed=True)
    t["own_x"] = np.exp(own.mean())
    t["own_n"] = own.size()
    t["own_ch"] = d.channel_id[okr].groupby(g[okr], observed=True).nunique()
    t[["own_n", "own_ch"]] = t[["own_n", "own_ch"]].fillna(0).astype(int)
    t.loc[(t.own_n < MIN_SHOW) | (t.own_ch < MIN_CELL_CH), "own_x"] = np.nan
    return t


def gap_by_size(d, f, name, h=7):
    """Own-channel gap inside each channel-size band. An input can matter a lot
    for small channels and not for big ones (or the reverse), which a gap over
    all videos averages away because big channels post most of the videos."""
    out = {}
    for band in f["subscribers"].cat.categories:
        rows = f["subscribers"] == band
        t = group_table(d, f[name], h, rows)
        t = t[(t.own_n >= MIN_N) & (t.own_ch >= MIN_BAND_CH) & t.own_x.notna()]
        if len(t) > 1:
            out[band] = t.own_x.max() / t.own_x.min()
    return pd.Series(out, dtype=float)


def stats(d, f, name, h=7, rows=None):
    g = f[name] if rows is None else f[name][rows]
    dd = d if rows is None else d[rows]
    t = group_table(dd, g, h)
    bands = gap_by_size(dd, f if rows is None else f[rows], name, h)         if META[name]["kind"] == "video" else pd.Series(dtype=float)
    big = t[t.videos >= MIN_N]
    bigo = t[(t.own_n >= MIN_N) & t.own_x.notna()]
    okr = own_rows(dd, g, h)
    return {
        "explains_raw": explains(np.log1p(dd[f"v{h}"]), g),
        "explains_own": explains(dd[f"rel{h}"][okr], g[okr]),
        "own_videos": int(okr.sum()),
        "own_channels": int(dd.channel_id[okr].nunique()),
        "bw_raw": big["median"].max() / max(big["median"].min(), 1) if len(big) > 1 else np.nan,
        "bw_own": bigo.own_x.max() / bigo.own_x.min() if len(bigo) > 1 else np.nan,
        "bw_own_band": bands.max() if len(bands) else np.nan,
        "band": bands.idxmax() if len(bands) else None,
        "top_share": t.share.max() if len(t) else np.nan,
        "top_value": t.share.idxmax() if len(t) else None,
        "missing_pct": g.isna().mean() * 100,
    }


# ------------------------------------------------------------------ charts
def _log_axis(ax, lo, hi):
    ax.set_yscale("log")
    ticks = [t for t in (.01, .03, .1, .3, 1, 3, 10, 30, 100) if lo / 1.5 <= t <= hi * 1.5]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(lo / 1.5, hi * 1.5)


def _order(name, t):
    if name in ("category", "YouTube topic", "audio language", "default language", "title script",
                "channel country", "HD or SD", "captions", "made for kids"):
        return t.sort_values("median", ascending=False)
    return t


def show(d, f, name):
    """The standard view of one input: how common each value is, and day-7 views."""
    info = META[name]
    t = _order(name, group_table(d, f[name]))
    s = stats(d, f, name)
    shown = t[t.videos >= MIN_SHOW]
    channel = info["kind"] == "channel"

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [2, 3]})
    x = np.arange(len(t))
    a.bar(x, t.videos, color="#9AA5B1")
    for i, (n, p) in enumerate(zip(t.videos, t.share)):
        if len(t) <= 12:
            a.text(i, n, f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5)
    a.set_xticks(x)
    rot = len(t) > 6 or max(len(str(i)) for i in t.index) > 9
    a.set_xticklabels(t.index, rotation=45 if rot else 0, ha="right" if rot else "center",
                      fontsize=7.5 if len(t) > 12 else 8.5)
    a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: short(v)))
    a.set(title="How common each value is", ylabel="videos with a day-7 label")

    xs = np.arange(len(shown))
    off = 0 if channel else .14
    b.axhline(1, color="#444", lw=.9, ls="--")
    b.vlines(xs - off, shown.q25_x, shown.q75_x, color=RAW, alpha=.3, lw=7, label="middle half of videos")
    b.plot(xs - off, shown.raw_x, "o", color=RAW, ms=7, label="typical video (median) vs all videos")
    vals = list(shown.q25_x) + list(shown.q75_x)
    if not channel:
        b.plot(xs + off, shown.own_x, "D", color=OWN, ms=6, label="vs its own channel's average")
        vals += list(shown.own_x.dropna())
    vals = [v for v in vals if v > 0 and np.isfinite(v)]
    _log_axis(b, min(vals + [.5]), max(vals + [2]))
    b.set_xticks(xs)
    b.set_xticklabels(shown.index, rotation=45 if rot else 0, ha="right" if rot else "center",
                      fontsize=7.5 if len(shown) > 12 else 8.5)
    b.set(title="Day-7 views (1× = the typical video)", ylabel="× typical (log scale)")
    b.legend(fontsize=7.5, loc="upper center", ncol=3, frameon=False,
             bbox_to_anchor=(.5, -.3 if rot else -.12))
    fig.suptitle(f"{name}: {info['what']}", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, name.replace(" ", "_"))

    tab = pd.DataFrame({
        "videos": t.videos.map("{:,}".format),
        "share": t.share.map("{:.1f}%".format),
        "median day-7 views": t["median"].map(short),
        "middle half": [f"{short(q1)} – {short(q3)}" for q1, q3 in zip(t.q25, t.q75)],
        "× typical": t.raw_x.map(times),
    })
    if not channel:
        tab["× own channel"] = t.own_x.map(times)
        tab["videos in own-channel view"] = t.own_n.map("{:,}".format)
    display(tab)

    top = t.share.idxmax()
    lines = [f"**Source at forecast time:** {info['source']}."]
    if s["missing_pct"] >= 1:
        lines.append(f"Missing for {s['missing_pct']:.0f}% of videos"
                     + (" (channel statistics recorded after publishing are left out)." if channel else "."))
    lines.append(f"Most common value: **{top}** ({t.share.max():.0f}% of videos).")
    big = t[t.videos >= MIN_N]
    if len(big) > 1:
        lines.append(f"Best: **{big['median'].idxmax()}** ({short(big['median'].max())} views, "
                     f"{times(big.raw_x.max())} typical). Worst: **{big['median'].idxmin()}** "
                     f"({short(big['median'].min())}, {times(big.raw_x.min())}).")
    if not channel:
        bo = t[(t.own_n >= MIN_N) & t.own_x.notna()]
        if len(bo) > 1:
            lines.append(f"Against its own channel ({s['own_videos']:,} videos from the {s['own_channels']:,} "
                         f"channels that use more than one value): best **{bo.own_x.idxmax()}** "
                         f"({times(bo.own_x.max())}), worst **{bo.own_x.idxmin()}** ({times(bo.own_x.min())}): "
                         f"a **{s['bw_own']:.2f}×** gap.")
        if pd.notna(s["bw_own_band"]):
            lines.append(f"Largest gap inside one channel-size band: **{s['bw_own_band']:.2f}×** "
                         f"(channels with {s['band']} subscribers).")
    lines.append(f"Explains **{s['explains_raw']:.1f}%** of the variation in day-7 views.")
    say(" ".join(lines))
    return s


def constants_table(d, f, names):
    rows = []
    for n in names:
        t = group_table(d, f[n])
        rows.append({"input": n, "most common value": t.share.idxmax(),
                     "share of videos": f"{t.share.max():.1f}%",
                     "source at forecast time": META[n]["source"]})
    display(pd.DataFrame(rows).set_index("input"))


def horizon_lines(d, f, name, groups=None):
    """× typical at each horizon, on the same videos (all four labels)."""
    rows = d.all4
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    tabs = {h: group_table(d, f[name], h, rows) for h in H}
    idx = tabs[7][tabs[7].videos >= MIN_N].index
    if groups is not None:
        idx = [g for g in groups if g in idx]
    cmap = plt.get_cmap("tab10")
    vals = []
    for i, gname in enumerate(idx):
        ys = [tabs[h].loc[gname, "raw_x"] for h in H]
        vals += ys
        ax.plot(H, ys, "o-", color=cmap(i % 10), label=f"{gname} (n={tabs[7].loc[gname, 'videos']:,})")
    ax.axhline(1, color="#444", lw=.9, ls="--")
    _log_axis(ax, min(vals), max(vals))
    ax.set_xticks(H)
    ax.set_xticklabels([f"day {h}" for h in H])
    ax.set(ylabel="× typical video (log scale)",
           title=f"{name} at each horizon (same {int(rows.sum()):,} videos throughout)")
    ax.legend(fontsize=7.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    save(fig, f"horizons_{name.replace(' ', '_')}")
    out = pd.DataFrame({f"day {h}": tabs[h].loc[idx, "raw_x"].map(times) for h in H})
    display(out)


# --------------------------------------------------------------- grids
def grid(values, counts, title, fmt, cmap, norm, xlabel, ylabel, cbar_label, figsize=None, cbar_ticks=None):
    values = values.where(counts >= MIN_CELL)
    figsize = figsize or (1.25 * values.shape[1] + 3, .42 * values.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values.values.astype(float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(values.shape[1]))
    tilt = values.shape[1] > 4 or max(len(str(c)) for c in values.columns) > 10
    ax.set_xticklabels(values.columns, rotation=30 if tilt else 0, ha="right" if tilt else "center")
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(values.index)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            if pd.notna(v):
                r, g_, b_, _ = im.cmap(im.norm(v))
                dark = .299 * r + .587 * g_ + .114 * b_ < .5
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                        color="white" if dark else "black")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label(cbar_label)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v)))
    cb.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    fig.tight_layout()
    return fig


def main_categories(d, n=1000):
    vc = d.loc[d.v7.notna(), "category_name"].value_counts()
    return list(vc[vc >= n].index)


def category_of(d, cats):
    c = d.category_name.where(d.category_name.isin(cats), "Other")
    return pd.Categorical(c, categories=cats + ["Other"], ordered=True)


def own_grid(d, f, name, cats):
    """Rows categories, columns values of an input, cell = × own channel's usual."""
    cat = pd.Series(category_of(d, cats), index=d.index)
    ok = own_rows(d, f[name])
    grp = d.rel7[ok].groupby([cat[ok], f[name][ok]], observed=True)
    vals = np.exp(grp.mean()).unstack()
    cnt = grp.size().unstack().fillna(0)
    chans = d.channel_id[ok].groupby([cat[ok], f[name][ok]], observed=True).nunique().unstack().fillna(0)
    cnt = cnt.where(chans >= MIN_CELL_CH, 0)
    cols = [c for c in (f[name].cat.categories if isinstance(f[name].dtype, pd.CategoricalDtype)
                        else cnt.sum().sort_values(ascending=False).index) if c in vals.columns]
    vals, cnt = vals[cols], cnt[cols]
    fig = grid(vals, cnt, f"{name} by category: day-7 views vs own channel's average\n"
                          f"(– = under {MIN_CELL} videos or {MIN_CELL_CH} channels)",
               times, "RdBu", LogNorm(vmin=.5, vmax=2), name, "category",
               "× own channel's average (blue = better)", cbar_ticks=[.5, .7, 1, 1.4, 2])
    save(fig, f"category_grid_{name.replace(' ', '_')}")
    return vals, cnt


def best_worst(vals, cnt, cat, min_gap=1.2):
    row = vals.loc[cat].where(cnt.loc[cat] >= MIN_CELL).dropna()
    if len(row) < 2:
        return None
    gap = row.max() / row.min()
    if gap < min_gap:
        return "no clear difference"
    return f"{row.idxmax()} best ({times(row.max())}), {row.idxmin()} worst ({times(row.min())})"


DUPLICATES = {"title word count": "title length", "weekend": "publish weekday",
              "publish time band": "publish hour", "default language": "audio language"}


def verdict(name, s, kept=None):
    """One keep/drop rule for every input, stated in the notebook's opening section."""
    info = META[name]
    if s["top_share"] >= 95:
        return "drop", f"almost every video has the same value ({s['top_share']:.0f}%)"
    if info["kind"] == "channel":
        ok = s["explains_raw"] >= RAW_MIN
        why = f"explains {s['explains_raw']:.1f}% of day-7 views"
    else:
        ok = s["bw_own"] >= OWN_MIN or s["bw_own_band"] >= OWN_MIN
        why = f"{s['bw_own']:.2f}× gap against own channel"
        if s["bw_own"] < OWN_MIN and s["bw_own_band"] >= OWN_MIN:
            why += f", {s['bw_own_band']:.2f}× for {s['band']} channels"
    if not ok:
        return "drop", why + " (too small)"
    if "not sent" in info["source"] or "not on the form" in info["source"]:
        return "keep, needs serving work", why + f"; {info['source']}"
    return "keep", why


# -------------------------------------------------- inside one category
PAGE_INPUTS = ["Short or long-form", "duration", "publish time band", "audio language", "title length",
               "number in title", "capital letters in title", "tag count", "description length"]


def _dots(ax, shown, own=True):
    xs = np.arange(len(shown))
    off = .14 if own else 0
    ax.axhline(1, color="#444", lw=.9, ls="--")
    ax.vlines(xs - off, shown.q25_x, shown.q75_x, color=RAW, alpha=.3, lw=6)
    ax.plot(xs - off, shown.raw_x, "o", color=RAW, ms=6)
    vals = list(shown.q25_x) + list(shown.q75_x)
    if own:
        ax.plot(xs + off, shown.own_x, "D", color=OWN, ms=5)
        vals += list(shown.own_x.dropna())
    vals = [v for v in vals if v > 0 and np.isfinite(v)]
    _log_axis(ax, min(vals + [.5]), max(vals + [2]))
    rot = len(shown) > 4 or max(len(str(i)) for i in shown.index) > 9
    ax.set_xticks(xs)
    ax.set_xticklabels(shown.index, rotation=35 if rot else 0, ha="right" if rot else "center", fontsize=7.5)


def category_page(d, f, cat_name, names=PAGE_INPUTS):
    """How each creator-controlled input moves day-7 views inside one category.
    1x = the typical video in this category, not all videos."""
    rows = d.category_name.eq(cat_name)
    n_vid, n_ch = int((rows & d.v7.notna()).sum()), d.channel_id[rows & d.v7.notna()].nunique()
    base = d.v7[rows].median()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    summary = []
    for ax, name in zip(axes.flat, names):
        t = group_table(d, f[name], rows=rows)
        shown = t[t.videos >= MIN_SHOW]
        if len(shown) < 2:
            ax.axis("off")
            ax.set_title(f"{name}\n(too few videos)", fontsize=9)
            continue
        _dots(ax, shown)
        ax.set_title(name, fontsize=9.5)
        big = t[t.videos >= MIN_CELL]
        bo = t[(t.own_n >= MIN_CELL) & t.own_x.notna()]
        summary.append({
            "input": name,
            "best (vs typical in category)": f"{big.raw_x.idxmax()} ({times(big.raw_x.max())})" if len(big) > 1 else "",
            "worst": f"{big.raw_x.idxmin()} ({times(big.raw_x.min())})" if len(big) > 1 else "",
            "best vs own channel": f"{bo.own_x.idxmax()} ({times(bo.own_x.max())})" if len(bo) > 1 else "too few channels",
            "worst vs own channel": f"{bo.own_x.idxmin()} ({times(bo.own_x.min())})" if len(bo) > 1 else "",
            "own-channel gap": bo.own_x.max() / bo.own_x.min() if len(bo) > 1 else np.nan,
        })
    handles = [plt.Line2D([], [], marker="o", color=RAW, ls="", label="typical video vs typical in this category"),
               plt.Rectangle((0, 0), 1, 1, color=RAW, alpha=.3, label="middle half of videos"),
               plt.Line2D([], [], marker="D", color=OWN, ls="", label="vs its own channel's average")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle(f"{cat_name}: {n_vid:,} videos from {n_ch:,} channels. "
                 f"1× = the typical {cat_name} video ({short(base)} day-7 views)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, .03, 1, .97))
    save(fig, f"inside_{cat_name.replace(' & ', '_').replace(' ', '_')}")
    s = pd.DataFrame(summary).set_index("input")
    s["own-channel gap"] = s["own-channel gap"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}×")
    display(s)


def news_check(d, f, names):
    """Every input's day-7 result with and without News & Politics."""
    news = d.category_name.eq("News & Politics")
    rows = []
    for n in names:
        a, b = stats(d, f, n), stats(d[~news], f[~news], n)
        ta = group_table(d, f[n]); ta = ta[ta.videos >= MIN_N]
        tb = group_table(d[~news], f[n][~news]); tb = tb[tb.videos >= MIN_N]
        video = META[n]["kind"] == "video"
        best_a, best_b = ta["median"].idxmax(), tb["median"].idxmax()
        gap_a, gap_b = (a["bw_own"], b["bw_own"]) if video else (np.nan, np.nan)
        changed = []
        if best_a != best_b:
            changed.append("best value changes")
        if a["explains_raw"] > .5 and abs(b["explains_raw"] - a["explains_raw"]) / a["explains_raw"] > .3:
            changed.append("strength changes")
        if video and abs(gap_b - gap_a) >= .1:
            changed.append("own-channel gap changes")
        rows.append({"input": n, "explains, all": f"{a['explains_raw']:.1f}%",
                     "explains, without News": f"{b['explains_raw']:.1f}%",
                     "best value, all": best_a, "best value, without News": best_b,
                     "own-channel gap, all": "" if not video else f"{gap_a:.2f}×",
                     "own-channel gap, without News": "" if not video else f"{gap_b:.2f}×",
                     "what News changes": ", ".join(changed) or "nothing much"})
    display(pd.DataFrame(rows).set_index("input"))
