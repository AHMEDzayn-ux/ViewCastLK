"""Train and evaluate forecasting baselines for ViewCastLK.

Exploratory: this exists to establish what is achievable and to give the
modelling task a floor to beat, not to be the deliverable model.

TWO REGIMES, ALWAYS BOTH
  cold start - grouped split, the channel is never seen in training. The
               honest test of whether video attributes carry anything, and the
               right number for a creator the system has no history for.
  warm start - random split, the channel has history. The realistic deployment
               case for a monitored roster.
Quoting one alone misleads: warm start credits the features with the channel's
contribution, cold start hides that deployment usually knows the channel.

WHERE LEAKAGE WOULD CREEP IN
Every channel-level aggregate -- target encoding, median duration, median title
length -- is fitted on the TRAINING FOLD ONLY and applied to the test fold.
Computing them over the whole frame first would let a channel's test videos
inform its own encoding. That inflates warm-start scores substantially while
looking entirely reasonable in the code, which is what makes it dangerous.

Repeated over five splits, because a single grouped split on this data swings
by +/-0.1 R^2 depending only on which channels land in the test set.

Usage:
    python Analysis/train_baseline.py
    python Analysis/train_baseline.py --horizon 7 --with-thumbnails
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score

from paths import dataset_path

HORIZONS = (7, 14, 21, 30)
SEEDS = range(5)

NUM = ["duration_seconds", "title_length", "title_word_count",
       "title_upper_ratio", "tag_count", "description_length",
       "publish_hour_slt", "ch_subs_at_publish", "ch_views_at_publish",
       "ch_videos_at_publish", "channel_age_days_at_publish",
       "gap_prev_h", "same_ch_24h_before"]
CAT = ["category_name", "title_script"]
BIN = ["title_has_number"]


def load(horizon, with_thumbs=False):
    df = pd.read_parquet(dataset_path())

    # Cadence needs EVERY eligible upload, not just labelled ones, or gaps of
    # days get invented where the channel actually posted hourly.
    allv = (df[df.eligible][["video_id", "channel_id", "published_at"]]
            .sort_values(["channel_id", "published_at"]))
    allv["gap_prev_h"] = (allv.groupby("channel_id").published_at
                          .diff().dt.total_seconds() / 3600)
    ts = allv.set_index("published_at")
    allv["same_ch_24h_before"] = (
        ts.groupby("channel_id", group_keys=False)
          .apply(lambda g: g.rolling("24h", closed="both").video_id.count() - 1)
          .values)

    d = df[df.eligible & df[f"d{horizon}_usable"]].copy()
    d = d.merge(allv[["video_id", "gap_prev_h", "same_ch_24h_before"]],
                on="video_id", how="left")
    d["y"] = np.log1p(d[f"d{horizon}_views"])

    cols = list(NUM)
    if with_thumbs:
        tp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "thumbnail_features.parquet")
        if os.path.exists(tp):
            tf = pd.read_parquet(tp)
            img = [c for c in tf.columns if c not in ("video_id", "bytes")]
            d = d.merge(tf[["video_id"] + img], on="video_id", how="left")
            for c in img:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            cols += img
            print(f"  thumbnail features joined "
                  f"({d[img[0]].notna().mean():.1%} coverage)")
        else:
            print("  no thumbnail_features.parquet; skipping")
    return d, cols


def encode(d, cols):
    X = d[cols + CAT + BIN].copy()
    for c in CAT:
        X[c] = OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1).fit_transform(
                                  X[[c]].astype(str)).ravel()
    for c in BIN:
        X[c] = (X[c].astype(float) if X[c].dtype == bool
                else pd.to_numeric(X[c], errors="coerce"))
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def add_fold_features(X, d, tr):
    """Channel aggregates fitted on the TRAINING fold only. See module docstring."""
    X = X.copy()
    train = d.iloc[tr]
    gm = float(train.y.median())

    enc = train.groupby("channel_id").y.median()
    counts = train.groupby("channel_id").size()
    dur = train.groupby("channel_id").duration_seconds.median()
    tlen = train.groupby("channel_id").title_length.median()

    X["ch_target_enc"] = d.channel_id.map(enc).fillna(gm).values
    X["ch_history_n"] = d.channel_id.map(counts).fillna(0).values
    X["dur_vs_channel"] = (d.duration_seconds
                           / d.channel_id.map(dur).replace(0, np.nan)).values
    X["titlelen_vs_channel"] = (d.title_length
                                / d.channel_id.map(tlen).replace(0, np.nan)).values

    med = X.iloc[tr].median(numeric_only=True)
    return X.fillna(med).fillna(-1), gm


def metrics(y_true, pred):
    err = np.abs(y_true - pred)
    fac = np.exp(err)
    return {"R2": r2_score(y_true, pred), "MAE": err.mean(),
            "med_factor": float(np.median(fac)),
            "within_2x": float((fac <= 2).mean()),
            "within_3x": float((fac <= 3).mean()),
            "spearman": pd.Series(pred).corr(pd.Series(y_true),
                                             method="spearman")}


# Features that only exist because the channel has history in the training
# fold. For a channel the model has never seen they collapse to a constant --
# the global median for the encoding, zero for the count -- while varying
# freely during training. The model leans on them hard and then finds them
# useless at prediction time, which is worse than never having offered them.
HISTORY_FEATURES = ["ch_target_enc", "ch_history_n", "dur_vs_channel",
                    "titlelen_vs_channel"]


def evaluate(d, X0, regime, use_history=True):
    rows, model, Xlast = [], None, None
    for seed in SEEDS:
        if regime == "cold":
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=seed)
                          .split(X0, groups=d.channel_id))
        else:
            tr, te = train_test_split(np.arange(len(d)), test_size=.25,
                                      random_state=seed)
        X, gm = add_fold_features(X0, d, tr)
        y = d.y.values

        Xm = X if use_history else X.drop(columns=HISTORY_FEATURES)
        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=.04,
                              num_leaves=63, min_child_samples=40,
                              subsample=.9, subsample_freq=1,
                              colsample_bytree=.9, verbose=-1, random_state=0)
        m.fit(Xm.iloc[tr], y[tr])
        pred = m.predict(Xm.iloc[te])

        for name, p in [("global median", np.full(len(te), gm)),
                        ("per-channel median", X.ch_target_enc.values[te]),
                        ("LightGBM", pred)]:
            rows.append({"seed": seed, "model": name, **metrics(y[te], p)})
        model, Xlast = m, Xm
    return pd.DataFrame(rows).groupby("model").agg(["mean", "std"]), model, Xlast


def report(agg, label):
    print(f"\n  --- {label} ---")
    print(f"  {'model':22}{'R2':>17}{'MAE':>9}{'med err':>10}"
          f"{'<=2x':>8}{'<=3x':>8}{'rho':>8}")
    for name in ["global median", "per-channel median", "LightGBM"]:
        if name not in agg.index:
            continue
        a = agg.loc[name]
        print(f"  {name:22}"
              f"{a[('R2', 'mean')]:>+10.3f} ±{a[('R2', 'std')]:<5.3f}"
              f"{a[('MAE', 'mean')]:>9.3f}"
              f"{a[('med_factor', 'mean')]:>9.1f}x"
              f"{a[('within_2x', 'mean')]:>8.1%}"
              f"{a[('within_3x', 'mean')]:>8.1%}"
              f"{a[('spearman', 'mean')]:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=0, help="0 = all horizons")
    ap.add_argument("--with-thumbnails", action="store_true")
    args = ap.parse_args()

    for h in ([args.horizon] if args.horizon else list(HORIZONS)):
        print(f"\n{'=' * 78}\nHORIZON: day {h}\n{'=' * 78}")
        d, cols = load(h, args.with_thumbnails)
        print(f"  {len(d):,} labelled videos, "
              f"{d.channel_id.nunique():,} channels")
        X0 = encode(d, cols)

        runs = [("cold", False, "COLD START (unseen channels, no history features)"),
                ("cold", True, "COLD START but WITH history features -- the trap"),
                ("warm", True, "WARM START (channels seen before)")]
        for regime, use_hist, label in runs:
            agg, model, X = evaluate(d, X0, regime, use_history=use_hist)
            report(agg, label)
            if regime == "warm" and h == 7:
                imp = pd.Series(model.feature_importances_,
                                index=X.columns).sort_values(ascending=False)
                print("\n  top features by gain:")
                for k, v in imp.head(12).items():
                    print(f"    {k:28} {v:>7}")


if __name__ == "__main__":
    main()
