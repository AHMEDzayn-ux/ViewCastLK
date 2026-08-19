"""Cumulative-by-construction forecasting: a level plus three growth multipliers.

THE PROBLEM THIS SOLVES
Four independently trained per-horizon models have nothing tying them
together, so their errors are uncorrelated and the predicted trajectory can
fall -- which is impossible, since views accumulate. Measured on the shipped
artefact, 74.6% of forecasts decreased somewhere. Clamping the output
afterwards hides that rather than fixing it.

THE DESIGN
Predict the day-7 level, then three multipliers, and chain them:

    d7  = expm1(level_model(x))
    d14 = d7  * exp(m1(x))
    d21 = d14 * exp(m2(x))
    d30 = d21 * exp(m3(x))

Each multiplier model predicts log(ratio) and is trained only on non-negative
growth, so exp() of its output is >= 1 whenever the prediction is >= 0. The
curve is non-decreasing by arithmetic; there is nothing to enforce.

WHY THE COMPOUNDING IS AFFORDABLE
Errors add in quadrature along the chain, so day 30 inherits day 7's error.
That is acceptable because the multipliers are tiny and tightly distributed --
measured on 1,651 complete trajectories, the median video gains 0.8% between
day 7 and day 14 and 1.4% between day 7 and day 30, with sd(log) of 0.127,
0.060 and 0.065 per step against roughly 1.5 log units of day-7 error. The
chain adds about 0.008 log units in total.

They are not, however, constant. The same measurement by segment ranges from
1.005 for News & Politics to 1.651 for Autos & Vehicles, and from 1.007 for
1M+ channels to 1.135 for the 1K-10K band: large channels collect nearly
everything in the first week, small ones keep accumulating through search and
recommendation. So the multipliers are modelled, not hard-coded.

LABEL NOISE
About 0.1-0.3% of observed ratios are below 1 -- a view count that went down,
which happens when YouTube purges inflated views. Those are clipped to 1.0
rather than dropped: the video is real, the direction is not.

Evaluated against independent per-horizon models on identical channel-grouped
splits, five seeds.

Usage:  python Analysis/train_increment_model.py
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder

from paths import dataset_path

HORIZONS = (7, 14, 21, 30)
STEPS = ((7, 14), (14, 21), (21, 30))
SEEDS = range(5)

NUM = ["duration_seconds", "title_length", "title_word_count",
       "title_upper_ratio", "tag_count", "description_length",
       "publish_hour_slt", "ch_subs_at_publish", "ch_views_at_publish",
       "ch_videos_at_publish", "channel_age_days_at_publish"]
CAT = ["category_name", "title_script"]
BIN = ["is_short", "title_has_number", "publish_is_weekend"]

PARAMS = dict(n_estimators=700, learning_rate=.04, num_leaves=63,
              min_child_samples=40, subsample=.9, subsample_freq=1,
              colsample_bytree=.9, verbose=-1, random_state=0)
# The multipliers are small, low-variance targets on far fewer rows than the
# level. The same capacity used for the level would memorise noise here.
PARAMS_MULT = dict(PARAMS, n_estimators=400, num_leaves=31,
                   min_child_samples=80)


def load():
    df = pd.read_parquet(dataset_path())
    d = df[df.eligible].copy()
    for h in HORIZONS:
        d[f"y{h}"] = np.where(d[f"d{h}_usable"], d[f"d{h}_views"], np.nan)
    return d


def encode(d):
    X = d[NUM + CAT + BIN].copy()
    for c in CAT:
        X[c] = OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1).fit_transform(
                                  X[[c]].astype(str)).ravel()
    for c in BIN:
        X[c] = X[c].astype(float)
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def fit_increment(d, X, tr):
    """Level model on day 7, plus one multiplier model per step."""
    sub = d.iloc[tr]
    med = X.iloc[tr].median(numeric_only=True)
    Xtr = X.iloc[tr].fillna(med).fillna(-1)

    ok = sub[f"y{HORIZONS[0]}"].notna().values
    level = lgb.LGBMRegressor(**PARAMS)
    level.fit(Xtr[ok], np.log1p(sub.loc[sub.index[ok], "y7"].values))

    mults = {}
    for a, b in STEPS:
        pair = (sub[f"y{a}"].notna() & sub[f"y{b}"].notna()).values
        if pair.sum() < 200:
            mults[(a, b)] = None
            continue
        ya = sub.loc[sub.index[pair], f"y{a}"].values
        yb = sub.loc[sub.index[pair], f"y{b}"].values
        ratio = np.clip(np.where(ya > 0, yb / np.maximum(ya, 1), 1.0), 1.0, None)
        m = lgb.LGBMRegressor(**PARAMS_MULT)
        m.fit(Xtr[pair], np.log(ratio))
        mults[(a, b)] = m
    return level, mults, med


def predict_curve(level, mults, X, te, med):
    Xte = X.iloc[te].fillna(med).fillna(-1)
    cur = np.expm1(level.predict(Xte))
    cur = np.clip(cur, 0, None)
    out = {7: cur}
    for a, b in STEPS:
        m = mults[(a, b)]
        step = np.exp(np.clip(m.predict(Xte), 0, None)) if m is not None else 1.0
        cur = cur * step
        out[b] = cur
    return out


def fit_independent(d, X, tr):
    sub = d.iloc[tr]
    med = X.iloc[tr].median(numeric_only=True)
    Xtr = X.iloc[tr].fillna(med).fillna(-1)
    models = {}
    for h in HORIZONS:
        ok = sub[f"y{h}"].notna().values
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(Xtr[ok], np.log1p(sub.loc[sub.index[ok], f"y{h}"].values))
        models[h] = m
    return models, med


def metrics(actual, pred):
    actual = np.asarray(actual, float)
    pred = np.clip(np.asarray(pred, float), 0, None)
    nz = actual > 0
    ape = np.abs(actual[nz] - pred[nz]) / actual[nz]
    return {"n": int(len(actual)),
            "log_R2": r2_score(np.log1p(actual), np.log1p(pred)),
            "spearman": pd.Series(pred).corr(pd.Series(actual), method="spearman"),
            "WAPE_%": 100 * np.abs(actual - pred).sum() / actual.sum(),
            "MedAPE_%": 100 * np.median(ape),
            "capture_%": 100 * pred.sum() / actual.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    d = load()
    X = encode(d)
    print(f"eligible videos: {len(d):,}   channels: {d.channel_id.nunique():,}")
    for h in HORIZONS:
        print(f"  day {h:>2} labels: {d[f'y{h}'].notna().sum():>7,}")
    for a, b in STEPS:
        print(f"  d{a}&d{b} pairs: {(d[f'y{a}'].notna() & d[f'y{b}'].notna()).sum():>7,}")

    rows, mono = [], []
    for seed in list(SEEDS)[:args.seeds]:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                        random_state=seed)
                      .split(X, groups=d.channel_id))
        lvl, mults, med = fit_increment(d, X, tr)
        inc = predict_curve(lvl, mults, X, te, med)
        ind_models, med_i = fit_independent(d, X, tr)
        Xte = X.iloc[te].fillna(med_i).fillna(-1)
        ind = {h: np.clip(np.expm1(ind_models[h].predict(Xte)), 0, None)
               for h in HORIZONS}

        sub = d.iloc[te]
        for h in HORIZONS:
            ok = sub[f"y{h}"].notna().values
            if ok.sum() < 50:
                continue
            act = sub.loc[sub.index[ok], f"y{h}"].values
            rows.append({"seed": seed, "horizon": h, "model": "increment",
                         **metrics(act, inc[h][ok])})
            rows.append({"seed": seed, "horizon": h, "model": "independent",
                         **metrics(act, ind[h][ok])})

        for name, cv in (("increment", inc), ("independent", ind)):
            P = np.vstack([cv[h] for h in HORIZONS])
            dec = (np.diff(P, axis=0) < 0).any(axis=0)
            mono.append({"seed": seed, "model": name, "decreasing_%": 100 * dec.mean()})

    r = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("HELD-OUT CHANNELS — mean over splits")
    print("=" * 92)
    agg = (r.groupby(["horizon", "model"])
             [["log_R2", "spearman", "WAPE_%", "MedAPE_%", "capture_%"]].mean())
    print(agg.round(3).to_string())

    print("\n--- increment minus independent (positive = increment better) ---")
    for h in HORIZONS:
        a = agg.loc[(h, "increment")]
        b = agg.loc[(h, "independent")]
        print(f"  day {h:>2}: log_R2 {a.log_R2 - b.log_R2:+.4f}   "
              f"spearman {a.spearman - b.spearman:+.4f}   "
              f"WAPE {a['WAPE_%'] - b['WAPE_%']:+.2f}pp   "
              f"MedAPE {a['MedAPE_%'] - b['MedAPE_%']:+.2f}pp")

    m = pd.DataFrame(mono).groupby("model")["decreasing_%"].mean()
    print("\n--- physically impossible (decreasing) trajectories ---")
    for k, v in m.items():
        print(f"  {k:12} {v:6.2f}%")


if __name__ == "__main__":
    main()
