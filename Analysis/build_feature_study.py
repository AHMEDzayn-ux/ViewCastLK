"""Which inputs the forecasting model should keep, drop, fix or add.

WHY THIS EXISTS
The deployed model (viewcastlk_monotonic_trajectory_experimental_v1) takes 30
inputs. Its feature choices were made on an earlier, smaller table, and three
things about them had never been measured:

  * Nineteen of those inputs (the topic_* flags) and is_short are ALWAYS
    missing when the live API builds a forecast. prediction_api/app/
    feature_builder.py says so: "Serving logic for topic category extraction
    is unresolved". The model was trained with them present for 99% of rows.
    Accuracy measured on training-style inputs is therefore not the accuracy a
    creator gets.
  * The SRS requires splits by publication date (FR-75) and an ablation study
    by feature group (FR-80). The product models use channel-grouped splits and
    no ablation has been run.
  * Several features the EDA found informative within channel (title features,
    posting cadence, the channel's own recent record) were excluded without a
    test.

WHAT IT DOES, per horizon (7, 14, 21, 30)
  S1  Serving reality. Train the deployed feature set exactly as the team's
      pipeline does, then score the test period three ways: with training-style
      inputs, with the inputs the live API actually sends when the creator gives
      a publish day and hour, and when they leave both blank.
  S2  Serve-safe base. Retrain on inputs as they exist at serving time. This is
      the honest reference every other comparison uses.
  S3  Ablation. Remove one group at a time from the serve-safe base and retrain.
      A group matters only if removing it hurts by more than seed-to-seed noise.
  S4  Add-backs. Add one candidate group at a time and retrain, noting what the
      API would need to supply it.
  S5  Data questions. Rows with post-publication channel stats; unseen channels
      in the test period; Gemini title scores on the subset that has them;
      redundancy and near-constant columns.

REPRODUCING THE PIPELINE
Feature construction mirrors notebooks/01_horizon_isolation.ipynb (cell 2) and
the model uses the team's own HorizonDatasetPreprocessor, build_xgb_regressor
and 3-sigma log-target inlier rule, imported from viewcastlk_ml. That package
lives on main; on a branch without it, point VIEWCASTLK_ML_PATH at a folder that
contains viewcastlk_ml/.

The split is by publication date: the last 20% of each horizon's rows by
publication time are the test set, and the last 15% of the remaining training
period is the early-stopping validation set. Nothing from the future of a test
video is used to fit anything it is scored by.

Usage:
    python Analysis/build_feature_study.py
"""
import os
import sys
import time
import warnings
from urllib.parse import unquote, urlparse

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for candidate in (os.environ.get("VIEWCASTLK_ML_PATH"), os.path.dirname(HERE)):
    if candidate and os.path.isdir(os.path.join(candidate, "viewcastlk_ml")):
        sys.path.insert(0, candidate)
        break

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

try:
    from viewcastlk_ml.horizon_preprocessing import (
        HorizonDatasetPreprocessor, TOPIC_COLUMNS, subscriber_tier_from_count)
    from viewcastlk_ml.modeling import build_xgb_regressor, log_target_inlier_mask
except ImportError as exc:
    raise SystemExit("viewcastlk_ml not importable. Merge main, or set "
                     "VIEWCASTLK_ML_PATH to a folder containing viewcastlk_ml/. "
                     f"({exc})")

from paths import dataset_path

OUT = os.path.join(HERE, "eda_figures", "feature_study")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, GREY, RED, GREEN, ORANGE, PURPLE = ("#2F6DB5", "#9AA5B1", "#C0392B",
                                          "#2E8B72", "#E08B4B", "#7B5EA7")
HORIZONS = (7, 14, 21, 30)
TEST_Q, VAL_Q = .80, .85
SEEDS = (42, 7, 123)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close()
    log(f"  {name}.png")


# ---------------------------------------------------------------- features
# Mirrors notebooks/01_horizon_isolation.ipynb cell 2.
TOPIC_CANONICAL_GROUPS = {
    "Music": {"Christian music", "Classical music", "Electronic music", "Hip hop music",
              "Music", "Music of Asia", "Music of Latin America", "Pop music",
              "Rock music", "Soul music"},
    "Gaming": {"Action game", "Action-adventure game", "Casual game", "Puzzle video game",
               "Racing video game", "Role-playing video game", "Simulation video game",
               "Sports game", "Strategy video game", "Video game culture"},
    "Sports": {"Association football", "Boxing", "Cricket", "Motorsport", "Sport", "Volleyball"},
    "Entertainment": {"Entertainment", "Film", "Performing arts", "Television program"},
    "Health": {"Health", "Physical fitness"},
    "Lifestyle": {"Lifestyle (sociology)"},
    "Politics": {"Politics", "Military"},
    "Knowledge": {"Knowledge", "Business"},
}
TOPIC_LOOKUP = {o: c for c, os_ in TOPIC_CANONICAL_GROUPS.items() for o in os_}
TOPIC_PARENT_CHILDREN = {
    "Society": {"Politics", "Religion"},
    "Lifestyle": {"Tourism", "Vehicle", "Hobby", "Food", "Fashion", "Pet", "Technology", "Health"},
    "Entertainment": {"Humour"},
}
MODEL_TOPIC_LABELS = ("Entertainment", "Fashion", "Food", "Gaming", "Health", "Hobby",
                      "Humour", "Knowledge", "Lifestyle", "Music", "Pet", "Politics",
                      "Religion", "Society", "Sports", "Technology", "Tourism", "Vehicle")
CATEGORICAL = ("category_name", "default_language", "publish_time_bucket", "subscriber_tier")


def topic_labels(value):
    if pd.isna(value):
        return None
    labels = []
    for item in str(value).split("|"):
        item = item.strip()
        if not item:
            continue
        label = unquote(urlparse(item).path.rsplit("/", 1)[-1]).replace("_", " ").strip()
        label = TOPIC_LOOKUP.get(label, label)
        if label and label not in labels:
            labels.append(label)
    s = set(labels)
    for parent, children in TOPIC_PARENT_CHILDREN.items():
        if parent in s and s & children:
            s.discard(parent)
    return s or None


def load():
    d = pd.read_parquet(dataset_path())
    for c in ("eligible", "is_live_broadcast", "title_changed", "channel_stats_backfilled",
              "is_short", "publish_is_weekend", "title_has_number", "title_has_question",
              "title_has_exclaim"):
        d[c] = d[c].fillna(False).astype(bool)
    for h in HORIZONS:
        d[f"d{h}_usable"] = d[f"d{h}_usable"].fillna(False).astype(bool)
        d[f"d{h}_views"] = pd.to_numeric(d[f"d{h}_views"], errors="coerce")
    d["published_at"] = pd.to_datetime(d.published_at, utc=True)
    d = d[d.eligible & ~d.is_live_broadcast].copy()
    return d


def deployed_frame(d):
    """The 30-input contract as the training notebook builds it."""
    f = pd.DataFrame(index=d.index)
    f["category_name"] = d.category_name
    for c in ("duration_seconds", "ch_subs_at_publish", "ch_videos_at_publish",
              "channel_age_days_at_publish"):
        f[c] = pd.to_numeric(d[c], errors="coerce")
    views = pd.to_numeric(d.ch_views_at_publish, errors="coerce")
    vids = f.ch_videos_at_publish.where(f.ch_videos_at_publish > 0)
    f["ch_avg_views_per_video_at_publish"] = (views / vids).replace(
        [np.inf, -np.inf], np.nan).mask(d.channel_stats_backfilled)
    f["is_short"] = d.is_short
    f["publish_is_weekend"] = d.publish_is_weekend
    sets = d.topic_categories.map(topic_labels)
    for t in MODEL_TOPIC_LABELS:
        f["topic_" + t.casefold()] = sets.map(lambda s, t=t: bool(s) and t in s)
    f["topic_missing"] = sets.isna()
    f["default_language"] = d.default_language
    hour = pd.to_numeric(d.publish_hour_slt, errors="coerce")
    f["publish_time_bucket"] = pd.cut(
        hour, bins=[-0.001, 5.999, 14.999, 20.999, 23.999],
        labels=["early_morning", "morning_afternoon", "evening", "late_night"],
        include_lowest=True).astype("string").fillna("unknown")
    f["subscriber_tier"] = subscriber_tier_from_count(f.ch_subs_at_publish)
    return f


def served(frame, d, day_hour_given=True, short_rule=None):
    """What prediction_api/app/feature_builder.py actually sends.

    Topics are unresolved at serving, so every topic flag is False and
    topic_missing is True. is_short is never supplied, so it is missing unless
    a duration rule is chosen. default_language comes from the creator's audio
    language, mapped to en/si/ta or missing."""
    s = frame.copy()
    for c in TOPIC_COLUMNS:
        s[c] = False
    s["topic_missing"] = True
    dur = pd.to_numeric(d.duration_seconds, errors="coerce")
    s["is_short"] = (dur <= short_rule) if short_rule else np.nan
    lang = d.default_audio_language.astype("string").str.lower().str.split("-").str[0]
    s["default_language"] = lang.where(lang.isin(["en", "si", "ta"]))
    if not day_hour_given:
        s["publish_is_weekend"] = np.nan
        s["publish_time_bucket"] = np.nan
    return s


def extras(d):
    """Candidate inputs the deployed model excludes, all computable before
    publication. Returned as numeric columns keyed like the table."""
    e = pd.DataFrame(index=d.index)
    e["title_length"] = pd.to_numeric(d.title_length, errors="coerce")
    e["title_word_count"] = pd.to_numeric(d.title_word_count, errors="coerce")
    e["title_upper_ratio"] = pd.to_numeric(d.title_upper_ratio, errors="coerce")
    for c in ("title_has_number", "title_has_question", "title_has_exclaim"):
        e[c] = d[c].astype(float)
    for sc in ("sinhala_script", "latin_script", "tamil_script"):
        e[f"script_{sc.split('_')[0]}"] = (d.title_script == sc).astype(float)
    e["tag_count"] = pd.to_numeric(d.tag_count, errors="coerce")
    e["log_description_length"] = np.log1p(pd.to_numeric(d.description_length, errors="coerce"))
    for c in ("publish_hour_sin", "publish_hour_cos", "publish_dow_sin", "publish_dow_cos"):
        e[c] = pd.to_numeric(d[c], errors="coerce")

    o = d.sort_values(["channel_id", "published_at"])
    secs = (o.published_at - EPOCH).dt.total_seconds().values
    gap = np.full(len(o), np.nan)
    prior = np.zeros(len(o))
    hist = np.full(len(o), np.nan)
    y7 = np.log1p(o.d7_views.where(o.d7_usable).values)
    for idx in o.groupby("channel_id").indices.values():
        ts = secs[idx]
        if len(idx) > 1:
            gap[idx[1:]] = np.diff(ts) / 3600
        prior[idx] = np.arange(len(idx)) - np.searchsorted(ts, ts - 86400, side="left")
        yy = y7[idx]
        avail = np.searchsorted(ts, ts - 7 * 86400, side="right")
        for k, j in enumerate(avail):
            past = yy[:j][~np.isnan(yy[:j])][-10:]
            if len(past) >= 3:
                hist[idx[k]] = np.median(past)
    e.loc[o.index, "log_gap_prev_h"] = np.log1p(np.clip(gap, 0, 720))
    e.loc[o.index, "log_uploads_prev_24h"] = np.log1p(prior)
    e.loc[o.index, "ch_recent_median_log_d7"] = hist
    return e


GROUPS = {   # removed from the serve-safe base in S3
    "channel statistics": ["ch_subs_at_publish", "ch_avg_views_per_video_at_publish",
                           "ch_videos_at_publish", "channel_age_days_at_publish",
                           "subscriber_tier"],
    "category": ["category_name"],
    "duration and format": ["duration_seconds", "is_short"],
    "publish day and time": ["publish_is_weekend", "publish_time_bucket"],
    "language": ["default_language"],
}
ADDBACKS = {   # added to the serve-safe base in S4, with what serving would need
    "title surface features": (["title_length", "title_word_count", "title_upper_ratio",
                                "title_has_number", "title_has_question", "title_has_exclaim",
                                "script_sinhala", "script_latin", "script_tamil"],
                               "derived from the title the creator already types"),
    "exact hour and weekday": (["publish_hour_sin", "publish_hour_cos",
                                "publish_dow_sin", "publish_dow_cos"],
                               "the optional day/hour already on the form"),
    "tags and description": (["tag_count", "log_description_length"],
                             "new form fields"),
    "posting cadence": (["log_gap_prev_h", "log_uploads_prev_24h"],
                        "channel's recent upload times (free RSS feed) plus planned time"),
    "channel's recent record": (["ch_recent_median_log_d7"],
                                "views of the channel's recent uploads (about 2 quota units)"),
}


def neutralise(frame, cols):
    f = frame.copy()
    for c in cols:
        f[c] = "__NEUTRAL__" if c in CATEGORICAL else np.nan
    return f


# ------------------------------------------------------------------ fitting
def metrics(y_views, pred_log):
    y = np.asarray(y_views, float)
    ylog = np.log1p(y)
    pv = np.clip(np.expm1(pred_log), 0, None)
    pos = y > 0
    return {"log_mae": float(np.mean(np.abs(ylog - pred_log))),
            "r2_log": float(r2_score(ylog, pred_log)),
            "medape": float(np.median(np.abs(pv[pos] - y[pos]) / y[pos]) * 100),
            "within_2x": float(np.mean(np.abs(ylog - pred_log) < np.log(2)) * 100),
            "wape": float(np.abs(pv - y).sum() / y.sum() * 100),
            "n": int(len(y))}


def fit_predict(train_frame, y_train, t_train, vcut, scoring, extra=None, seed=42):
    """Fit with the team's preprocessor and XGBoost settings; return predictions
    for every (frame, index) in `scoring`."""
    ylog = np.log1p(y_train.values.astype(float))
    keep, _, _ = log_target_inlier_mask(ylog)
    train_frame, ylog, t_train = train_frame[keep], ylog[keep], t_train[keep]
    fit_rows = (t_train < vcut).values
    pre = HorizonDatasetPreprocessor().fit(train_frame[fit_rows], ylog[fit_rows])

    def matrix(frame):
        m = pre.transform(frame)
        if extra is not None:
            m = pd.concat([m, extra.loc[frame.index].astype(float)], axis=1)
        return m

    model = build_xgb_regressor(random_state=seed)
    model.fit(matrix(train_frame[fit_rows]), ylog[fit_rows],
              eval_set=[(matrix(train_frame[~fit_rows]), ylog[~fit_rows])], verbose=False)
    return {k: model.predict(matrix(fr)) for k, fr in scoring.items()}, pre, model


def horizon_split(d, h):
    rows = d[d[f"d{h}_usable"] & d[f"d{h}_views"].notna() & ~d.title_changed]
    cut = rows.published_at.quantile(TEST_Q)
    tr, te = rows[rows.published_at < cut], rows[rows.published_at >= cut]
    return tr, te, cut, tr.published_at.quantile(VAL_Q)


# -------------------------------------------------------------------- study
def run(d, F, E):
    R = {"serving": [], "noise": [], "ablation": [], "addback": [], "backfill": [],
         "unseen": [], "splits": []}
    S_given = served(F, d, day_hour_given=True)
    S_blank = served(F, d, day_hour_given=False)
    for h in HORIZONS:
        t0 = time.time()
        tr, te, cut, vcut = horizon_split(d, h)
        y, yt = tr[f"d{h}_views"], te[f"d{h}_views"]
        unseen = ~te.channel_id.isin(tr.channel_id)
        R["splits"].append({"h": h, "train": len(tr), "test": len(te), "cut": cut,
                            "unseen_share": unseen.mean() * 100})
        log(f"\nday {h}: train {len(tr):,}  test {len(te):,} (published from {cut:%d %b}), "
            f"{unseen.mean()*100:.0f}% of test rows from channels unseen in training")

        # S1 deployed model, three ways of scoring
        p, _, _ = fit_predict(F.loc[tr.index], y, tr.published_at, vcut, {
            "training-style inputs": F.loc[te.index],
            "as served, day and hour given": S_given.loc[te.index],
            "as served, day and hour blank": S_blank.loc[te.index]})
        for k, v in p.items():
            R["serving"].append({"h": h, "model": "deployed inputs", "scored": k, **metrics(yt, v)})
        p, _, _ = fit_predict(neutralise(F, list(TOPIC_COLUMNS)).loc[tr.index], y,
                              tr.published_at, vcut,
                              {"training-style inputs": neutralise(F, list(TOPIC_COLUMNS)).loc[te.index]})
        R["serving"].append({"h": h, "model": "deployed without topics",
                             "scored": "training-style inputs",
                             **metrics(yt, p["training-style inputs"])})

        # S2 serve-safe base (trained on the inputs serving actually has)
        for seed in SEEDS:
            p, _, _ = fit_predict(S_given.loc[tr.index], y, tr.published_at, vcut, {
                "given": S_given.loc[te.index], "blank": S_blank.loc[te.index]}, seed=seed)
            m = metrics(yt, p["given"])
            R["noise"].append({"h": h, "seed": seed, **m})
            if seed == SEEDS[0]:
                base_pred = p["given"]
                R["serving"].append({"h": h, "model": "serve-safe base",
                                     "scored": "as served, day and hour given", **m})
                R["serving"].append({"h": h, "model": "serve-safe base",
                                     "scored": "as served, day and hour blank",
                                     **metrics(yt, p["blank"])})
                R["unseen"].append({"h": h, "subset": "channels seen in training",
                                    **metrics(yt[~unseen], base_pred[~unseen.values])})
                R["unseen"].append({"h": h, "subset": "channels unseen in training",
                                    **metrics(yt[unseen], base_pred[unseen.values])})
        base = R["noise"][-len(SEEDS)]

        for rule in (60, 180):
            Sr = served(F, d, day_hour_given=True, short_rule=rule)
            p, _, _ = fit_predict(Sr.loc[tr.index], y, tr.published_at, vcut,
                                  {"x": Sr.loc[te.index]})
            R["addback"].append({"h": h, "group": f"is_short = duration ≤ {rule}s",
                                 "needs": "nothing new: duration is on the form",
                                 **metrics(yt, p["x"])})

        # S3 ablation
        for g, cols in GROUPS.items():
            N = neutralise(S_given, cols)
            p, _, _ = fit_predict(N.loc[tr.index], y, tr.published_at, vcut, {"x": N.loc[te.index]})
            R["ablation"].append({"h": h, "group": g, **metrics(yt, p["x"])})

        # S4 add-backs
        for g, (cols, needs) in ADDBACKS.items():
            p, _, _ = fit_predict(S_given.loc[tr.index], y, tr.published_at, vcut,
                                  {"x": S_given.loc[te.index]}, extra=E[cols])
            R["addback"].append({"h": h, "group": g, "needs": needs, **metrics(yt, p["x"])})
        allc = [c for cols, _ in ADDBACKS.values() for c in cols]
        p, _, _ = fit_predict(S_given.loc[tr.index], y, tr.published_at, vcut,
                              {"x": S_given.loc[te.index]}, extra=E[allc])
        R["addback"].append({"h": h, "group": "all candidates together", "needs": "all of the above",
                             **metrics(yt, p["x"])})

        # S5 post-publication channel stats: train with and without, score clean rows
        clean_te = ~te.channel_stats_backfilled
        clean_tr = ~tr.channel_stats_backfilled
        p_all, _, _ = fit_predict(S_given.loc[tr.index], y, tr.published_at, vcut,
                                  {"x": S_given.loc[te.index[clean_te.values]]})
        p_cl, _, _ = fit_predict(S_given.loc[tr.index[clean_tr.values]], y[clean_tr.values],
                                 tr.published_at[clean_tr.values], vcut,
                                 {"x": S_given.loc[te.index[clean_te.values]]})
        yc = yt[clean_te.values]
        R["backfill"] += [
            {"h": h, "variant": "trained on all rows", **metrics(yc, p_all["x"])},
            {"h": h, "variant": "trained without backfilled rows", **metrics(yc, p_cl["x"])}]
        log(f"  day {h} done in {time.time()-t0:.0f}s; serve-safe log-MAE {base['log_mae']:.3f}")
    return {k: pd.DataFrame(v) for k, v in R.items()}


def llm_subset(d, S_given, scores_path):
    """Gemini title scores exist for a small, older subset, so a date split would
    leave almost nothing to test on. Channel-grouped folds within the subset
    give an indicative answer, clearly labelled as such."""
    if not scores_path or not os.path.exists(scores_path):
        log("\nGemini scores file not found; skipping")
        return None
    cols = ["title_urgency", "title_emotional_appeal", "title_seriousness", "title_curiosity_gap"]
    ts = pd.read_csv(scores_path, usecols=["video_id"] + cols).drop_duplicates("video_id")
    sub = d[d.d7_usable & d.d7_views.notna() & ~d.title_changed]
    sub = sub.reset_index().merge(ts, on="video_id").set_index("index")
    if len(sub) < 1000:
        log(f"\nonly {len(sub)} scored rows; skipping")
        return None
    ex = sub[cols].astype(float)
    rows = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(sub, groups=sub.channel_id)):
        tr, te = sub.iloc[a], sub.iloc[b]
        vcut = tr.published_at.quantile(VAL_Q)
        for name, extra in (("without scores", None), ("with Gemini scores", ex)):
            p, _, _ = fit_predict(S_given.loc[tr.index], tr.d7_views, tr.published_at, vcut,
                                  {"x": S_given.loc[te.index]}, extra=extra)
            rows.append({"fold": fold, "variant": name, **metrics(te.d7_views, p["x"])})
    out = pd.DataFrame(rows)
    log(f"\nGemini subset: {len(sub):,} day-7 rows from {sub.channel_id.nunique():,} channels")
    return out, len(sub)


def redundancy(F, E, d):
    num = pd.concat([F[["duration_seconds", "ch_subs_at_publish", "ch_avg_views_per_video_at_publish",
                        "ch_videos_at_publish", "channel_age_days_at_publish"]],
                     E.drop(columns=["script_sinhala", "script_latin", "script_tamil"])], axis=1)
    rho = num.corr(method="spearman")
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(rho.columns)
             for b in rho.columns[i + 1:] if abs(rho.loc[a, b]) >= .7]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(rho)), yticks=range(len(rho)),
           title="Spearman correlation between candidate numeric inputs")
    ax.set_xticklabels(rho.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(rho.columns, fontsize=8)
    ax.grid(False)
    for i in range(len(rho)):
        for j in range(len(rho)):
            if i != j and abs(rho.values[i, j]) >= .5:
                ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(rho.values[i, j]) > .75 else "#222")
    fig.colorbar(im, ax=ax, shrink=.8)
    save("FS5_redundancy")
    cand = ["caption", "made_for_kids", "definition", "channel_country", "is_live_broadcast",
            "publish_is_weekend", "is_short", "title_has_question", "title_has_exclaim"]
    const = [(c, d[c].astype("string").value_counts(normalize=True, dropna=False).iloc[0] * 100)
             for c in cand if c in d]
    return pairs, const


# -------------------------------------------------------------------- plots
# FS1 and FS2 take plain tables so they can be redrawn from the written
# numbers (--replot) without repeating a 35-minute run.
def plot_fs1(sv):
    order = [("deployed inputs", "training-style inputs", GREY),
             ("deployed inputs", "as served, day and hour given", RED),
             ("deployed inputs", "as served, day and hour blank", "#E6A0A0"),
             ("serve-safe base", "as served, day and hour given", BLUE),
             ("serve-safe base", "as served, day and hour blank", "#9CC0E8")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"wspace": .25})
    x = np.arange(len(HORIZONS))
    w = .16
    for k, (model, scored, c) in enumerate(order):
        s = sv[(sv.model == model) & (sv.scored == scored)].set_index("h").reindex(HORIZONS)
        ax[0].bar(x + (k - 2) * w, s.log_mae, w, color=c, label=f"{model}: {scored}")
        ax[1].bar(x + (k - 2) * w, s.within_2x, w, color=c)
    for a, yl, t in ((ax[0], "mean absolute error, log views (lower is better)",
                      "Error on the test period"),
                     (ax[1], "% of test videos predicted within 2×",
                      "Share predicted within 2× (higher is better)")):
        a.set(xticks=x, xticklabels=[f"day {h}" for h in HORIZONS], ylabel=yl, title=t)
    # Inside the axes the legend covered the tallest bars; below, it covers nothing.
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .02), ncol=3, fontsize=8, frameon=False)
    save("FS1_serving_reality")


def plot_fs2(abl, add, band):
    """abl and add: columns group, h, delta (change in log-MAE vs the base)."""
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.4))
    fig.subplots_adjust(left=.14, right=.98, wspace=.62)
    for a, df, title, loc in (
            (ax[0], abl, "Removing a group (positive = the group was helping)", "center right"),
            (ax[1], add, "Adding a group (negative = the group helps)", "lower left")):
        groups = list(dict.fromkeys(df.group))
        hh = .8 / len(HORIZONS)
        for j, h in enumerate(HORIZONS):
            s = df[df.h == h].set_index("group").reindex(groups)
            a.barh(np.arange(len(groups)) + (j - 1.5) * hh, s.delta, hh,
                   color=[BLUE, GREEN, ORANGE, RED][j], label=f"day {h}")
        a.axvspan(-band, band, color="#ddd", alpha=.6, zorder=0,
                  label=f"±2 sd seed noise ({band:.3f})")
        a.axvline(0, color="#444", lw=.8)
        a.set(yticks=range(len(groups)), yticklabels=groups, title=title,
              xlabel="change in log-scale MAE versus the serve-safe base")
        a.legend(fontsize=7.6, loc=loc)
    save("FS2_ablation_and_addbacks")


def from_md(path):
    """Recover the FS1 and FS2 tables from feature_study_numbers.md."""
    lines = open(path, encoding="utf-8").read().splitlines()

    def table(prefix):
        start = next(i for i, ln in enumerate(lines) if ln.startswith(prefix))
        rows = []
        for ln in lines[start + 1:]:
            if ln.startswith("|"):
                rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
            elif rows:
                break
        return rows[2:]

    num = lambda s: float(s.replace("*", "").replace("%", "").strip())
    label = lambda s: s.replace("is_short derived from duration <= ", "is_short = duration ≤ ")
    sv = pd.DataFrame([{"h": int(r[0].split()[1]), "model": r[1], "scored": r[2],
                        "log_mae": num(r[3]), "within_2x": num(r[4])} for r in table("## S1")])
    abl = pd.DataFrame([(label(r[0]), h, num(r[1 + k])) for r in table("## S3")
                        for k, h in enumerate(HORIZONS)], columns=["group", "h", "delta"])
    add = pd.DataFrame([(label(r[0]), h, num(r[2 + k])) for r in table("## S4")
                        for k, h in enumerate(HORIZONS)], columns=["group", "h", "delta"])
    band = 2 * max(float(r[2]) for r in table("## Seed noise"))
    return sv, abl, add, band


def plots(R, noise):
    sd = noise.groupby("h").log_mae.std()
    base = noise[noise.seed == SEEDS[0]].set_index("h").log_mae
    x = np.arange(len(HORIZONS))
    plot_fs1(R["serving"])
    delta = lambda t: t.assign(delta=t.log_mae - t.h.map(base))[["group", "h", "delta"]]
    plot_fs2(delta(R["ablation"]), delta(R["addback"]), 2 * sd.max())

    bf = R["backfill"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"wspace": .25})
    for k, (v, c) in enumerate((("trained on all rows", GREY),
                                ("trained without backfilled rows", BLUE))):
        s = bf[bf.variant == v].set_index("h").reindex(HORIZONS)
        ax[0].bar(x + (k - .5) * .35, s.log_mae, .35, color=c, label=v)
    ax[0].set(xticks=x, xticklabels=[f"day {h}" for h in HORIZONS],
              ylabel="log-scale MAE on clean test rows",
              title="Rows with post-publication channel stats")
    ax[0].legend(fontsize=8)
    un = R["unseen"]
    for k, (v, c) in enumerate((("channels seen in training", BLUE),
                                ("channels unseen in training", ORANGE))):
        s = un[un.subset == v].set_index("h").reindex(HORIZONS)
        ax[1].bar(x + (k - .5) * .35, s.within_2x, .35, color=c,
                  label=f"{v}")
        for i, (val, n) in enumerate(zip(s.within_2x, s.n)):
            ax[1].text(i + (k - .5) * .35, val + .8, f"n={n:,}", ha="center", fontsize=7)
    ax[1].set(xticks=x, xticklabels=[f"day {h}" for h in HORIZONS],
              ylabel="% within 2×", title="Serve-safe base: known versus new channels")
    ax[1].legend(fontsize=8)
    save("FS3_backfill_and_new_channels")


def write_summary(R, noise, llm, pairs, const):
    sd = noise.groupby("h").log_mae.std()
    base = noise[noise.seed == SEEDS[0]].set_index("h")
    L = ["# Feature study in numbers", "",
         "Log-MAE is mean absolute error on log1p(views); 0.69 is a factor of 2. "
         "Date-based split: last 20% of each horizon's rows by publication time.", ""]
    L += ["## Splits", "", "| horizon | train | test | test published from | test rows from unseen channels |",
          "|---|---|---|---|---|"]
    for r in R["splits"].itertuples():
        L.append(f"| day {r.h} | {r.train:,} | {r.test:,} | {r.cut:%d %b} | {r.unseen_share:.0f}% |")
    L += ["", "## S1 Serving reality", "", "| horizon | model | scored with | log-MAE | within 2× | MedAPE | R² (log) |",
          "|---|---|---|---|---|---|---|"]
    for r in R["serving"].itertuples():
        L.append(f"| day {r.h} | {r.model} | {r.scored} | {r.log_mae:.3f} | {r.within_2x:.1f}% | "
                 f"{r.medape:.0f}% | {r.r2_log:.3f} |")
    L += ["", "## Seed noise (serve-safe base)", "", "| horizon | log-MAE mean | sd | 2 sd |", "|---|---|---|---|"]
    for h in HORIZONS:
        s = noise[noise.h == h].log_mae
        L.append(f"| day {h} | {s.mean():.3f} | {s.std():.4f} | {2*s.std():.4f} |")
    L += ["", "## S3 Ablation from the serve-safe base (Δ log-MAE; positive means the group helps)", "",
          "| group | " + " | ".join(f"day {h}" for h in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    for g in dict.fromkeys(R["ablation"].group):
        s = R["ablation"][R["ablation"].group == g].set_index("h")
        cells = []
        for h in HORIZONS:
            dv = s.loc[h, "log_mae"] - base.loc[h, "log_mae"]
            flag = " *" if abs(dv) > 2 * sd[h] else ""
            cells.append(f"{dv:+.3f}{flag}")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L += ["", "## S4 Add-backs to the serve-safe base (Δ log-MAE; negative means it helps)", "",
          "| group | needs at serving | " + " | ".join(f"day {h}" for h in HORIZONS) + " |",
          "|---|---|" + "---|" * len(HORIZONS)]
    for g in dict.fromkeys(R["addback"].group):
        s = R["addback"][R["addback"].group == g].set_index("h")
        cells = []
        for h in HORIZONS:
            dv = s.loc[h, "log_mae"] - base.loc[h, "log_mae"]
            flag = " *" if abs(dv) > 2 * sd[h] else ""
            cells.append(f"{dv:+.3f}{flag}")
        L.append(f"| {g} | {s.needs.iloc[0]} | " + " | ".join(cells) + " |")
    L += ["", "`*` exceeds two standard deviations of seed-to-seed noise at that horizon.", "",
          "## S5 Post-publication channel stats (scored on clean test rows)", "",
          "| horizon | variant | log-MAE | within 2× | n |", "|---|---|---|---|---|"]
    for r in R["backfill"].itertuples():
        L.append(f"| day {r.h} | {r.variant} | {r.log_mae:.3f} | {r.within_2x:.1f}% | {r.n:,} |")
    L += ["", "## Known versus new channels (serve-safe base)", "",
          "| horizon | subset | log-MAE | within 2× | n |", "|---|---|---|---|---|"]
    for r in R["unseen"].itertuples():
        L.append(f"| day {r.h} | {r.subset} | {r.log_mae:.3f} | {r.within_2x:.1f}% | {r.n:,} |")
    if llm is not None:
        df, n = llm
        agg = df.groupby("variant").log_mae.agg(["mean", "std"])
        piv = df.pivot(index="fold", columns="variant", values="log_mae")
        t, p = stats.ttest_rel(piv["with Gemini scores"], piv["without scores"])
        L += ["", f"## Gemini title scores (indicative: {n:,} scored day-7 rows, channel-grouped 5-fold)", "",
              "| variant | log-MAE mean | sd |", "|---|---|---|"]
        for v, r in agg.iterrows():
            L.append(f"| {v} | {r['mean']:.3f} | {r['std']:.3f} |")
        L.append(f"| paired difference | {(piv['with Gemini scores']-piv['without scores']).mean():+.3f} | "
                 f"t={t:+.2f}, p={p:.3f} |")
    L += ["", "## Redundant pairs (|Spearman| ≥ 0.7)", "", "| a | b | ρ |", "|---|---|---|"]
    for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
        L.append(f"| {a} | {b} | {r:+.2f} |")
    L += ["", "## Dominant value share (near-constant columns)", "", "| column | top value share |", "|---|---|"]
    for c, s in sorted(const, key=lambda x: -x[1]):
        L.append(f"| {c} | {s:.1f}% |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "feature_study_numbers.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="redraw FS1 and FS2 from feature_study_numbers.md without refitting")
    if ap.parse_args().replot:
        sv, abl, add, band = from_md(os.path.join(OUT, "feature_study_numbers.md"))
        plot_fs1(sv)
        plot_fs2(abl, add, band)
        return
    t0 = time.time()
    d = load()
    log(f"{len(d):,} eligible, non-live videos")
    F = deployed_frame(d)
    E = extras(d)
    log(f"features built in {time.time()-t0:.0f}s")
    R = run(d, F, E)
    noise = R.pop("noise")
    # Raw result tables, so any figure can be redrawn later without refitting.
    for k, v in {**R, "noise": noise}.items():
        v.to_csv(os.path.join(OUT, f"results_{k}.csv"), index=False)
    scores = os.environ.get("VIEWCASTLK_TITLE_SCORES",
                            os.path.join(os.path.dirname(HERE), "Dataset", "title_scores.csv"))
    llm = llm_subset(d, served(F, d, day_hour_given=True), scores)
    pairs, const = redundancy(F, E, d)
    plots(R, noise)
    write_summary(R, noise, llm, pairs, const)
    log(f"\nfinished in {(time.time()-t0)/60:.1f} min; written to {OUT}")


if __name__ == "__main__":
    main()
