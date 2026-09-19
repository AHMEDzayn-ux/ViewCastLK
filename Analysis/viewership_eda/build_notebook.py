"""Generate and run viewership_eda.ipynb: a simple, explained EDA of every input.

Every chart is a count, a median or a percentage (helpers in helpers.py).
The interpretation text in notes.py is written against the build named in
BUILD; the numbers inside the charts and tables are recomputed on every run.

Usage:
    python Analysis/viewership_eda/build_notebook.py            # build and run
    python Analysis/viewership_eda/build_notebook.py --no-run   # build only
"""
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = "18 September 2026"
nb = nbf.v4.new_notebook()
C = []


def md(text):
    C.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    C.append(nbf.v4.new_code_cell(text.strip()))


def note(key):
    if NOTES.get(key):
        md(NOTES[key])


NOTES = {}
if os.path.exists(os.path.join(HERE, "notes.py")):
    sys.path.insert(0, HERE)
    from notes import NOTES  # noqa: E402

# ==================================================================== intro
md(f"""
# ViewCastLK: simple EDA of every input

What does each input tell us about how many views a Sri Lankan YouTube video
gets? One input at a time, with plain charts and a short explanation under each.

Dataset build: **{BUILD}**. The interpretation text was written against that
build; every number in a chart or table is recomputed when the notebook runs.

## How to read the charts

Each input gets the same two charts.

* **Left: how common each value is.** Number of videos with that value.
* **Right: day-7 views for each value, as "× typical".**
  * **Blue dot:** the typical (median) video with that value, compared with the
    typical video overall. **1×** is average, **2×** is twice the views, **0.5×**
    is half. The axis is a log scale, so 2× and 0.5× are the same distance from
    1×.
  * **Blue band:** where the middle half of those videos fall. A tall band means
    videos with that value vary a lot.
  * **Orange diamond: compared with its own channel.** Each video's views
    divided by its channel's average views (on the log scale), averaged over
    the group. This removes channel size. Only channels that post
    videos with more than one value of the input are counted (a channel that
    only posts Shorts cannot tell us whether Shorts beat long-form), and a
    diamond needs at least 5 such channels. If the blue dot is high but the
    orange diamond is near 1×, the value only looks good because big channels
    use it.

A table under each chart gives the actual numbers, followed by a summary.

## Why the orange diamond matters

Channel size explains far more of the views than anything else. So a raw
comparison ("Comedy gets more views than News") is often really a comparison of
which channels post it. The orange diamond answers the question a creator
actually has: **for my channel, does this help?**

Some inputs are the same for almost every video of a channel (subscribers,
category, topic), so there is no within-channel comparison to make. Those are
judged on the raw view only.

## The keep or drop rule

* **Drop** if 95% or more of videos have the same value.
* **Inputs fixed per channel** (channel statistics, category, topic): **keep**
  if the input explains at least **1%** of the variation in day-7 views.
* **Inputs that vary within a channel** (everything else): **keep** if, compared
  with their own channel, the best value's videos get at least **1.2×** the
  worst value's (groups of at least 200 videos). Because some inputs matter for
  small channels and not big ones, an input also passes if it reaches 1.2×
  inside one channel-size band, as long as every group there comes from at
  least 20 channels.
* "Explains %" is the share of the variation in log day-7 views that the
  input's groups account for on their own. 0% means views look the same in
  every group.

## Which videos

Live streams and videos without a duration are left out. Each horizon (7, 14,
21, 30 days) uses only videos with a label at that horizon. The per-input charts
use **day 7**, which has the most videos. Where the notebook compares horizons,
it uses the same videos (those with all four labels) so that time is the only
difference.
""")

code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from IPython.display import display
import helpers as L
from helpers import H, short, times, say, save

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
raw, d = L.load()
f = L.build_features(d)
S = {}   # summary numbers for each input, filled as the notebook runs
""")

# ================================================================ 1 overview
md("""
---
# 1. Data overview
""")
code("""
pub = d.published
overview = pd.DataFrame({"value": [
    f"{len(raw):,}", f"{int(raw.is_live_broadcast.sum()):,}",
    f"{int((~raw.eligible & ~raw.is_live_broadcast).sum()):,}", f"{len(d):,}",
    f"{d.channel_id.nunique():,}", f"{pub.min():%d %b %Y} to {pub.max():%d %b %Y}",
] + [f"{int(d[f'v{h}'].notna().sum()):,}" for h in H] + [f"{int(d.all4.sum()):,}"]},
    index=["rows in the file", "live streams (left out)", "no duration, unavailable (left out)",
           "videos analysed", "channels", "published"]
          + [f"videos with a day-{h} label" for h in H] + ["videos with all four labels"])
display(overview)
""")
note("overview")

md("""
## Data dictionary

Every column in the file, grouped by what it is. **Inputs** are what the model
may use. **Labels** are what it predicts. **Bookkeeping** columns describe how
the row was built and are never inputs.
""")
code("""
DICT = [
    ("video_id, channel_id", "id", "identifiers"),
    ("published_at", "input (timing)", "publishing time, UTC; hour, weekday and weekend come from it"),
    ("title, tags, thumbnail_url", "raw text", "title and tag features are worked out from these"),
    ("category_id, category_name", "input (format)", "YouTube category"),
    ("duration_seconds", "input (format)", "video length"),
    ("is_short", "input (format)", "real Short: vertical or square and at most 3 minutes"),
    ("is_short_source, is_vertical", "bookkeeping", "whether the player shape or the old 60 s rule decided is_short"),
    ("definition, caption, made_for_kids", "input (format)", "HD or SD, captions, made-for-kids setting"),
    ("default_audio_language, default_language", "input (language)", "spoken language, and title/description language"),
    ("publish_hour_slt, publish_dow_slt, publish_is_weekend", "input (timing)", "Sri Lanka time"),
    ("publish_hour_sin/cos, publish_dow_sin/cos", "input (timing)", "the same hour and weekday as points on a circle, so 23:00 sits next to 00:00. Not charted separately"),
    ("title_length ... title_script", "input (title)", "length, word count, number, ?, !, capitals, writing system"),
    ("description_length, tag_count", "input (text)", "description size and number of tags"),
    ("ch_subs/views/videos_at_publish, channel_age_days_at_publish", "input (channel)", "the channel on the day the video was published"),
    ("channel_country, topic_categories", "input (channel)", "channel country and the channel's YouTube topics"),
    ("dN_views", "label", "views N days after publishing (N = 7, 14, 21, 30)"),
    ("dN_likes, dN_comments", "outcome, never an input", "happen after publishing, so using them would leak the answer"),
    ("dN_hours_off, dN_usable", "bookkeeping", "how far the observation was from exactly N days, and whether it is close enough"),
    ("eligible, is_live_broadcast", "bookkeeping", "rows left out of analysis"),
    ("channel_stats_backfilled, ch_stats_as_of", "bookkeeping", "channel statistics recorded only after publishing"),
    ("title_changed, description_changed", "bookkeeping", "edited since first seen"),
]
display(pd.DataFrame(DICT, columns=["column(s)", "type", "meaning"]).set_index("column(s)"))
""")

md("## Types and missing values of the input columns")
code("""
cols = ["category_name", "duration_seconds", "is_short", "definition", "caption", "made_for_kids",
        "default_audio_language", "default_language", "publish_hour_slt", "publish_dow_slt",
        "publish_is_weekend", "title_length", "title_word_count", "title_has_number",
        "title_has_question", "title_has_exclaim", "title_upper_ratio", "title_script",
        "description_length", "tag_count", "ch_subs_at_publish", "ch_views_at_publish",
        "ch_videos_at_publish", "channel_age_days_at_publish", "channel_country", "topic_categories"]
blank = lambda s: s.isna() | s.astype("string").str.strip().eq("")
display(pd.DataFrame({
    "type": [str(d[c].dtype) for c in cols],
    "missing or blank": [f"{blank(d[c]).mean()*100:.1f}%" for c in cols],
    "distinct values": [f"{d[c].nunique():,}" for c in cols],
    "example": [str(d[c].dropna().iloc[0])[:40] for c in cols],
}, index=cols))
""")
note("types")

md("## Videos published per day")
code("""
day = d.published.dt.floor("D").dt.tz_localize(None)
per_day = pd.DataFrame({"all": day.value_counts(), "with a day-7 label": day[d.v7.notna()].value_counts()}).sort_index().fillna(0)
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.bar(per_day.index, per_day["all"], color="#C9D3DD", label="all videos", width=.9)
ax.bar(per_day.index, per_day["with a day-7 label"], color="#2F6DB5", label="with a day-7 label", width=.9)
ax.set(ylabel="videos", title="Videos published per day (Sri Lanka time)")
ax.legend(fontsize=8)
save(fig, "videos_per_day")
""")
note("per_day")

# ================================================================ 2 quality
md("""
---
# 2. Data quality
""")
code("""
week = d.published.dt.tz_localize(None).dt.to_period("W").dt.start_time
bf = d.groupby(week).channel_stats_backfilled.mean() * 100
fig, ax = plt.subplots(figsize=(11, 3))
ax.bar(bf.index, bf.values, width=5, color="#E08B4B")
ax.set(ylabel="% of videos", title="Channel statistics recorded only after publishing, by publish week")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
save(fig, "backfilled_by_week")

rows = []
for h in H:
    off = pd.to_numeric(d[f"d{h}_hours_off"], errors="coerce").abs()[d[f"v{h}"].notna()]
    rows.append({"horizon": f"day {h}", "labels": f"{int(d[f'v{h}'].notna().sum()):,}",
                 "median distance from exact day": f"{off.median():.1f} h",
                 "within 3 h": f"{(off <= 3).mean()*100:.0f}%", "within 12 h": f"{(off <= 12).mean()*100:.0f}%"})
display(pd.DataFrame(rows).set_index("horizon"))
display(pd.DataFrame({"share of videos": [
    f"{d.channel_stats_backfilled.mean()*100:.1f}%", f"{d.title_changed.mean()*100:.1f}%",
    f"{d.description_changed.mean()*100:.1f}%", f"{(d.is_short_source == 'shape').mean()*100:.1f}%"]},
    index=["channel stats recorded after publishing", "title edited after first seen",
           "description edited after first seen", "Short flag decided by player shape"]))
""")
note("quality")

# ================================================================== 3 target
md("""
---
# 3. The target: views at 7, 14, 21 and 30 days
""")
code("""
rows = []
for h in H:
    v = d[f"v{h}"].dropna()
    rows.append({"horizon": f"day {h}", "videos": f"{len(v):,}", "median": short(v.median()),
                 "25th pct": short(v.quantile(.25)), "75th pct": short(v.quantile(.75)),
                 "90th pct": short(v.quantile(.9)), "99th pct": short(v.quantile(.99)),
                 "largest": short(v.max()), "share with 0 views": f"{(v == 0).mean()*100:.1f}%"})
display(pd.DataFrame(rows).set_index("horizon"))

v7 = d.v7.dropna()
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.5))
a.hist(v7.clip(upper=v7.quantile(.99)), bins=80, color="#9AA5B1")
a.set(title="Day-7 views as they are (top 1% cut off)", xlabel="views", ylabel="videos")
a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: short(x)))
b.hist(np.log10(v7 + 1), bins=80, color="#2F6DB5")
b.set(title="The same on a log scale", xlabel="views (log scale)", ylabel="videos")
b.xaxis.set_major_locator(mticker.FixedLocator(range(0, 8)))
b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: short(10 ** x)))
save(fig, "target_distribution")
""")
note("target")

md("## How views grow from day 7 to day 30 (same videos)")
code("""
c = d[d.all4]
med = pd.Series({h: c[f"v{h}"].median() for h in H})
ratio = (c.v30 / c.v7.where(c.v7 > 0)).dropna()
display(pd.DataFrame({"median views": med.map(short).values, "× day 7": (med / med[7]).map(times).values},
                     index=[f"day {h}" for h in H]))
say(f"On the {len(c):,} videos with all four labels, the typical video's day-30 views are "
    f"**{times(ratio.median())}** its day-7 views. For a quarter of videos they are at most "
    f"{times(ratio.quantile(.25))}; for another quarter at least {times(ratio.quantile(.75))}.")
""")
note("growth_intro")

# ================================================================ 4..9 inputs
SECTIONS = [
    ("4. Channel inputs", "channel", ["subscribers", "channel average views per video", "channel video count",
                                     "channel uploads per day", "channel age", "YouTube topic"],
     ["channel country"]),
    ("5. Video format", "format", ["duration", "Short or long-form", "category"],
     ["HD or SD", "captions", "made for kids"]),
    ("6. Language", "language", ["audio language", "default language", "title script"], []),
    ("7. Publish timing", "timing", ["publish hour", "publish time band", "publish weekday", "weekend"], []),
    ("8. Title", "title", ["title length", "title word count", "number in title", "question mark in title",
                          "exclamation mark in title", "capital letters in title"], []),
    ("9. Tags and description", "text", ["tag count", "description length"], []),
]
HORIZON_LINES = {"subscribers": None, "duration": None, "Short or long-form": None,
                 "category": "top"}

for title, key, names, consts in SECTIONS:
    md(f"---\n# {title}")
    note(f"section_{key}")
    for name in names:
        md(f"## {name}")
        code(f'S["{name}"] = L.show(d, f, "{name}")')
        note(name)
        if name in HORIZON_LINES:
            md(f"### {name} at 7, 14, 21 and 30 days")
            groups = "L.main_categories(d, 2000)" if HORIZON_LINES[name] == "top" else "None"
            code(f'L.horizon_lines(d, f, "{name}", {groups})')
            note(f"{name} horizons")
        if name == "Short or long-form":
            md("### Shorts against long-form by channel size")
            code("""
t = d[d.v7.notna()].assign(size=f["subscribers"], fmt=f["Short or long-form"]).dropna(subset=["size"])
med = t.groupby(["size", "fmt"], observed=True).v7.median().unstack()
cnt = t.groupby(["size", "fmt"], observed=True).v7.size().unstack()
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(med))
for i, (fmt, col) in enumerate(zip(med.columns, [L.RAW, L.OWN])):
    ax.bar(x + (i - .5) * .38, med[fmt], .38, color=col, label=fmt)
    for xi, (m, n) in enumerate(zip(med[fmt], cnt[fmt])):
        ax.text(xi + (i - .5) * .38, m * 1.1, f"{short(m)}\\nn={n:,}", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: short(v)))
ax.set_ylim(top=med.max().max() * 6)
ax.set_xticks(x); ax.set_xticklabels(med.index)
ax.set(xlabel="channel subscribers", ylabel="median day-7 views (log scale)",
       title="Shorts against long-form within each channel size")
ax.legend(fontsize=8)
save(fig, "shorts_by_channel_size")
same = {}
for band in med.index:
    tb = L.group_table(d, f["Short or long-form"], rows=f["subscribers"] == band)
    same[band] = tb.loc["Short", "own_x"] / tb.loc["long-form", "own_x"]
display(pd.DataFrame({"all videos in the band (median Short ÷ median long-form)": (med["Short"] / med["long-form"]).map(times),
                      "same channel (Short ÷ long-form, own-channel view)": pd.Series(same).map(times)}).T)
""")
            note("shorts_by_size")
    if consts:
        md("## Inputs where almost every video has the same value")
        code(f"L.constants_table(d, f, {consts!r})\n" + "\n".join(f'S["{n}"] = L.stats(d, f, "{n}")' for n in consts))
        note(f"constants_{key}")

# ============================================================== 10 category
md("""
---
# 10. Category by category

Category changes what works. This section looks at each category separately,
using day-7 views. Categories with fewer than 1,000 labelled videos are grouped
as "Other".
""")
code("""
CATS = L.main_categories(d, 1000)
cat = pd.Series(L.category_of(d, CATS), index=d.index)
lab = d.v7.notna()
subs = pd.to_numeric(d.ch_subs_at_publish, errors="coerce").where(~d.channel_stats_backfilled)
upd = (pd.to_numeric(d.ch_videos_at_publish, errors="coerce")
       / pd.to_numeric(d.channel_age_days_at_publish, errors="coerce").where(lambda s: s > 0))
prof = pd.DataFrame({
    "videos": d[lab].groupby(cat[lab], observed=True).size(),
    "channels": d[lab].groupby(cat[lab], observed=True).channel_id.nunique(),
    "median day-7 views": d[lab].groupby(cat[lab], observed=True).v7.median(),
    "% Shorts": d[lab].groupby(cat[lab], observed=True).is_short.mean() * 100,
    "median length (min)": pd.to_numeric(d.duration_seconds, errors="coerce")[lab].groupby(cat[lab], observed=True).median() / 60,
    "median channel subscribers": subs[lab].groupby(cat[lab], observed=True).median(),
    "median channel uploads per day": upd[lab].groupby(cat[lab], observed=True).median(),
}).sort_values("median day-7 views", ascending=False)
prof["videos per channel"] = prof.videos / prof.channels
display(prof.style.format({"videos": "{:,}", "channels": "{:,}", "median day-7 views": short,
                           "% Shorts": "{:.0f}%", "median length (min)": "{:.1f}",
                           "median channel subscribers": short, "median channel uploads per day": "{:.1f}",
                           "videos per channel": "{:.0f}"}))
""")
note("cat_profile")

md("""
## Is News & Politics skewing the results?

News & Politics is 31% of the labelled videos but comes from only about 60
channels, many posting dozens of clips a day. If it behaves differently from
everything else, it could be pulling the whole-dataset results in sections 4
to 9. This table repeats every input's day-7 result without it.
""")
code("""
L.news_check(d, f, [n for n in f.columns if n not in ("channel country", "captions", "made for kids", "HD or SD", "category")])
""")
note("news_check")

md("""
## Inside each category

One page per category. Each small chart is one input a creator controls, and
**1× is now the typical video in that category**, not across all videos. So the
chart answers: within Comedy (say), do Shorts, longer videos or evening posts do
better? The orange diamond again compares each video with its own channel. The
table under each page lists the best and worst value of every input.
""")
# Categories with at least 1,000 day-7 labelled videos in the build named above.
# A category missing from a newer build is skipped rather than failing.
PAGE_CATS = ["News & Politics", "Education", "Entertainment", "Travel & Events", "Music",
             "People & Blogs", "Gaming", "Howto & Style", "Autos & Vehicles", "Sports", "Comedy",
             "Pets & Animals", "Film & Animation", "Science & Technology"]
for c_ in PAGE_CATS:
    md(f"### {c_}")
    code(f'if "{c_}" in CATS:\n    L.category_page(d, f, "{c_}")')
    note(f"inside {c_}")

md("""
## Raw ranking against the own-channel ranking

Most channels post in one category only, so the own-channel comparison uses
just the channels that post in two or more. For them it asks: is a video in
this category better or worse than the channel's usual?
""")
code("""
ncat = d[lab].groupby("channel_id").category_name.nunique()
t = L.group_table(d, cat)
say(f"{int((ncat >= 2).sum()):,} of {len(ncat):,} channels post in two or more categories.")
t = t[t.videos >= 200].sort_values("raw_x", ascending=False)
t["raw rank"] = range(1, len(t) + 1)
t["own-channel rank"] = t.own_x.where(t.own_n >= 100).rank(ascending=False).astype("Int64")
display(pd.DataFrame({"videos": t.videos.map("{:,}".format), "× typical": t.raw_x.map(times),
                      "raw rank": t["raw rank"], "× own channel": t.own_x.map(times),
                      "videos in the own-channel view": t.own_n.map("{:,}".format),
                      "channels": t.own_ch,
                      "own-channel rank": t["own-channel rank"]}))
""")
note("cat_rank")

md("## Category by channel size (median day-7 views)")
code("""
t = d[lab].assign(cat=cat[lab], size=f["subscribers"][lab]).dropna(subset=["size"])
vals = t.groupby(["cat", "size"], observed=True).v7.median().unstack()
cnt = t.groupby(["cat", "size"], observed=True).v7.size().unstack().fillna(0)
fig = L.grid(vals, cnt, "Median day-7 views by category and channel size (– = under 100 videos)",
             short, "Blues", LogNorm(vmin=10, vmax=max(vals.max().max(), 100)),
             "channel subscribers", "category", "median day-7 views", cbar_ticks=[10, 100, 1000, 10000])
save(fig, "category_by_size")
""")
note("cat_size")

md("""
## What a creator can change, category by category

Each grid compares videos with their **own channel's usual** views, so channel
size is removed. **Blue** is better than usual, **red** worse, white about the
same. Cells with fewer than 100 videos are left blank.
""")
GRID_INPUTS = ["Short or long-form", "duration", "publish time band", "weekend", "audio language",
               "number in title", "capital letters in title", "title length", "tag count"]
for name in GRID_INPUTS:
    md(f"### {name}")
    code(f'G_{name.replace(" ", "_").replace("-", "_")} = L.own_grid(d, f, "{name}", CATS)')
    note(f"grid {name}")

md("## One line per category")
code("""
lines = []
for c_ in CATS + ["Other"]:
    parts = []
    for label, (vals, cnt) in [("format", G_Short_or_long_form), ("length", G_duration),
                               ("time", G_publish_time_band), ("title capitals", G_capital_letters_in_title),
                               ("tags", G_tag_count)]:
        r = L.best_worst(vals, cnt, c_)
        if r:
            parts.append(f"{label}: {r}")
    lines.append(f"* **{c_}**: " + "; ".join(parts))
say("\\n".join(lines))
""")
note("cat_lines")

# ================================================================ 11 growth
md("""
---
# 11. Growth from day 7 to day 30

How many extra views a video gains between day 7 and day 30, on the videos
with all four labels. 20% means day-30 views are 1.2 times day-7 views.
""")
code("""
c = d[d.all4 & (d.v7 > 0)].copy()
c["ratio"] = c.v30 / c.v7
cc = f.loc[c.index]
def ratio_table(g):
    grp = c.ratio.groupby(g, observed=True)
    t = pd.DataFrame({"videos": grp.size(), "extra": (grp.median() - 1) * 100})
    return t[t.videos >= 200]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 3, 2]})
for ax, (label, g) in zip(axes, [("format", cc["Short or long-form"]),
                                ("category", pd.Series(L.category_of(c, CATS), index=c.index)),
                                ("channel size", cc["subscribers"])]):
    t = ratio_table(g)
    if label == "category":
        t = t.sort_values("extra", ascending=False)
    ax.bar(range(len(t)), t["extra"], color="#2F6DB5")
    ax.axhline((c.ratio.median() - 1) * 100, color="#444", ls="--", lw=.9)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t.index, rotation=45 if len(t) > 3 else 0, ha="right" if len(t) > 3 else "center", fontsize=8)
    for i, v in enumerate(t["extra"]):
        ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set(title=f"by {label}", ylabel="extra views, day 7 to day 30")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
fig.suptitle(f"Views gained after the first week (typical video; dashed = all videos, "
             f"{(c.ratio.median() - 1) * 100:.0f}%)", fontweight="bold")
fig.tight_layout()
save(fig, "growth_7_to_30")
""")
note("growth")

# ============================================================== 12 thin data
md("""
---
# 12. Where the data is thin

Number of videos with a day-7 label for each category and channel size. A model
learns little where a cell is small, so forecasts there deserve less trust.
""")
code("""
t = d[lab].assign(cat=cat[lab], size=f["subscribers"][lab]).dropna(subset=["size"])
cnt = t.groupby(["cat", "size"], observed=True).size().unstack().fillna(0)
fig = L.grid(cnt.where(cnt > 0), pd.DataFrame(1e9, index=cnt.index, columns=cnt.columns),
             "Videos with a day-7 label (channel statistics recorded before publishing)",
             lambda v: f"{int(v):,}", "Greens", LogNorm(vmin=10, vmax=cnt.max().max()),
             "channel subscribers", "category", "videos", cbar_ticks=[10, 100, 1000, 10000])
save(fig, "thin_data_grid")
thin = int((cnt < 100).sum().sum())
say(f"**{thin}** of {cnt.size} cells have fewer than 100 videos.")
""")
note("thin")

# ============================================================ 13 over time
md("""
---
# 13. Does the data change over time?

Typical day-7 views by the week a video was published. The model is trained on
earlier weeks and tested on later ones, so a steady line is what we want.
""")
code("""
t = d[lab]
wk = t.published.dt.tz_localize(None).dt.to_period("W").dt.start_time
w = pd.DataFrame({"videos": t.groupby(wk).size(), "median day-7 views": t.groupby(wk).v7.median(),
                  "% Shorts": t.groupby(wk).is_short.mean() * 100,
                  "% News & Politics": t.groupby(wk).category_name.apply(lambda s: (s == "News & Politics").mean() * 100)})
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.5))
a.bar(w.index, w.videos, width=5, color="#C9D3DD", label="videos")
a2 = a.twinx()
a2.plot(w.index, w["median day-7 views"], "o-", color="#2F6DB5", label="median day-7 views")
a2.set_ylim(0, w["median day-7 views"].max() * 1.3)
a.set(title="Videos and typical day-7 views by publish week", ylabel="videos")
a2.set_ylabel("median day-7 views")
b.plot(w.index, w["% Shorts"], "o-", color=L.OWN, label="% Shorts")
b.plot(w.index, w["% News & Politics"], "o-", color="#6C7A89", label="% News & Politics")
b.set(title="Mix of videos by publish week", ylabel="% of videos")
b.legend(fontsize=8)
for ax in (a, b):
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
fig.tight_layout()
save(fig, "change_over_time")
w.index = [f"week of {i:%d %b}" for i in w.index]
display(w.style.format({"videos": "{:,}", "median day-7 views": short, "% Shorts": "{:.0f}%",
                        "% News & Politics": "{:.0f}%"}))
""")
note("drift")

# ================================================================= 14 viral
md("""
---
# 14. What the most-viewed videos look like

The top 1% of videos by day-7 views, compared with all videos.
""")
code("""
t = d[lab].copy()
cut_ = t.v7.quantile(.99)
top = t.v7 >= cut_
def mix(g, n=None):
    a = g[lab].value_counts(normalize=True) * 100
    b_ = g[lab][top.reindex(g[lab].index, fill_value=False)].value_counts(normalize=True) * 100
    out = pd.DataFrame({"all videos": a, "top 1%": b_}).fillna(0).sort_values("top 1%", ascending=False)
    return out.head(n) if n else out
say(f"Top 1% = at least **{short(cut_)}** day-7 views ({int(top.sum()):,} videos from "
    f"{t[top].channel_id.nunique():,} channels). The top 10 channels account for "
    f"**{t[top].channel_id.value_counts().head(10).sum() / top.sum() * 100:.0f}%** of them.")
for label, g, n in [("channel size", f["subscribers"].astype("object").fillna("recorded after publishing"), None),
                    ("format", f["Short or long-form"].astype("object"), None),
                    ("category", d.category_name, 8), ("duration", f["duration"].astype("object"), None)]:
    say(f"**By {label}** (% of videos)")
    display(mix(g, n).style.format("{:.0f}%"))
""")
note("viral")

# ============================================================== 15 repeats
md("""
---
# 15. Inputs that repeat each other

When two inputs carry the same information, the model only needs one.
**Rank agreement** runs from 0 (unrelated) to 1 (always in the same order).
""")
code("""
num = lambda c: pd.to_numeric(d[c], errors="coerce")
sp = lambda a, b: a.corr(b, method="spearman")
rows = [
    ("title length vs title word count", f"rank agreement {sp(num('title_length'), num('title_word_count')):.2f}", "keep one"),
    ("weekend vs publish weekday", "weekend is Sat or Sun: fully decided by weekday", "keep one"),
    ("publish time band vs publish hour", "the band is a grouping of the hour", "keep one"),
    ("hour sin/cos vs publish hour", "the same hour written as a point on a circle", "encoding, not a new input"),
    ("Short vs duration", f"{(d.is_short & (num('duration_seconds') > 180)).sum():,} Shorts over 3 min; "
     f"{(~d.is_short & (num('duration_seconds') <= 60)).sum():,} videos of 60 s or less are not Shorts",
     "both needed: shape adds what length cannot"),
    ("audio language vs default language", f"agree on {(f['audio language'] == f['default language']).mean()*100:.0f}% of videos", "the API sends one, mapped from the other"),
    ("subscribers vs channel average views", f"rank agreement {sp(num('ch_subs_at_publish'), num('ch_views_at_publish') / num('ch_videos_at_publish')):.2f}", "related, both useful"),
    ("channel video count vs uploads per day", f"rank agreement {sp(num('ch_videos_at_publish'), num('ch_videos_at_publish') / num('channel_age_days_at_publish')):.2f}", "related"),
    ("tag count vs description length", f"rank agreement {sp(num('tag_count'), num('description_length')):.2f}", "mostly separate"),
]
display(pd.DataFrame(rows, columns=["pair", "how related", "what to do"]).set_index("pair"))
""")
note("repeats")

# =========================================================== 16 availability
md("""
---
# 16. What the forecast actually receives

An input only helps if the app can supply it when a creator asks for a forecast.
This is what `prediction_api/app/feature_builder.py` sends today.
""")
code("""
AV = [
    ("category, duration", "on the form", "sent"),
    ("audio language", "on the form", "sent, mapped to en / si / ta"),
    ("publish day and hour", "on the form, optional", "sent as weekend yes/no and a four-band time"),
    ("subscribers, video count, channel age, average views", "looked up from the channel", "sent"),
    ("Short or long-form", "not asked", "always missing: the form needs the field"),
    ("YouTube topic", "available from the same channel lookup", "always missing: needs fetching"),
    ("title", "typed on the form", "used for tone advice only, not the forecast"),
    ("tags, description", "not asked", "not used"),
    ("HD or SD, captions, made for kids", "not asked", "not used (nearly constant anyway)"),
]
display(pd.DataFrame(AV, columns=["input", "where it comes from", "status today"]).set_index("input"))
""")
note("availability")

# ================================================================ 17 summary
md("""
---
# 17. Summary: every input

Day-7 numbers use all videos with a day-7 label. The four "explains" columns
use the same videos at every horizon (those with all four labels), so they can
be compared across time.
""")
code("""
rows = []
for name in f.columns:
    s = S.get(name) or L.stats(d, f, name)
    v, why = L.verdict(name, s)
    row = {"input": name, "section": L.META[name]["section"], "source at forecast time": L.META[name]["source"],
           "explains (day 7)": s["explains_raw"],
           "gap vs own channel": s["bw_own"] if L.META[name]["kind"] == "video" else np.nan}
    for h in H:
        row[f"explains d{h} (same videos)"] = L.explains(np.log1p(d.loc[d.all4, f"v{h}"]), f.loc[d.all4, name])
    row["verdict"], row["why"] = v, why
    if v == "drop":
        row["strength"] = ""
    elif L.META[name]["kind"] == "channel":
        row["strength"] = "strong" if s["explains_raw"] >= 5 else "moderate"
    elif s["bw_own"] >= 1.5:
        row["strength"] = "strong"
    elif s["bw_own"] >= L.OWN_MIN:
        row["strength"] = "moderate"
    else:
        row["strength"] = f"only for {s['band']} channels"
    rows.append(row)
summary = pd.DataFrame(rows).set_index("input").sort_values("explains (day 7)", ascending=False)
fmt = {c: "{:.1f}%" for c in summary.columns if c.startswith("explains")}
fmt["gap vs own channel"] = lambda v: "" if pd.isna(v) else f"{v:.2f}×"
display(summary.style.format(fmt))

colours = {"keep": "#2F6DB5", "keep, needs serving work": "#E08B4B", "drop": "#C0392B"}
r = summary.iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(r)), r["explains (day 7)"], color=[colours[v] for v in r.verdict])
ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index)
for i, (x, g) in enumerate(zip(r["explains (day 7)"], r["gap vs own channel"])):
    ax.text(x + .3, i, f"{x:.1f}%" + ("" if pd.isna(g) else f"   own-channel gap {g:.2f}×"), va="center", fontsize=7.5)
ax.set_xlim(0, r["explains (day 7)"].max() * 1.35)
ax.set(xlabel="% of the variation in day-7 views the input accounts for on its own",
       title="Every input, with its verdict")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()], labels=list(colours),
          loc="lower right", fontsize=8.5)
save(fig, "summary_all_inputs")
""")
note("summary")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = os.path.join(HERE, "viewership_eda.ipynb")
with open(path, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"wrote {path} ({len(C)} cells)")

if "--no-run" not in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbf.read(path, as_version=4)
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": HERE}})
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("executed")
