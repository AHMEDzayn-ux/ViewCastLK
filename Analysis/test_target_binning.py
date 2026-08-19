"""Does binning the target beat regressing on it?

THE QUESTION
Day-7 views span zero to 17 million and MAPE is dominated by videos with
fewer than a hundred views. If exact counts are unpredictable anyway, maybe
the model should predict a BAND -- "this will land in the 1k-10k range" --
rather than a number.

Three approaches are compared on identical splits:

  A. REGRESS THEN BIN   fit on log views as now, bin the prediction afterwards.
  B. CLASSIFY BINS      fit a multiclass model directly on the bin label.
  C. CHANNEL QUARTILE   predict which quartile of ITS OWN CHANNEL's recent
                        output the video lands in.

A and B answer "is training on bins better than binning afterwards", which is
the literal question. They are not the same experiment: A keeps the ordering
information inside a bin and can be re-binned at any boundary without
retraining, while B throws that away at fit time in exchange for optimising
the decision boundary directly.

C is a different product. Absolute views are mostly a fact about the channel,
so a creator asking "will this video do well?" usually means "well FOR ME".
That question is immune to the heavy tail -- every channel contributes the
same four classes regardless of size -- and it is the one the EDA suggests is
answerable.

Accuracy alone would flatter any of these, because the bins are unbalanced.
Macro-F1 and within-one-bin accuracy are reported alongside, plus the majority
-class baseline every classifier must beat.

Usage:  python Analysis/test_target_binning.py [--horizon 7]
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
from sklearn.metrics import accuracy_score, f1_score

import train_baseline as tb

SEEDS = range(5)
# Decade bins: views span five orders of magnitude, so equal-width bands would
# put almost everything in the first one.
EDGES = [-1, 100, 1_000, 10_000, 100_000, np.inf]
NAMES = ["<100", "100-1k", "1k-10k", "10k-100k", "100k+"]


def bin_views(v):
    return pd.cut(v, EDGES, labels=NAMES)


def score(y_true, y_pred, n_classes):
    """Accuracy, macro-F1, and how often the prediction is at most one band out."""
    exact = accuracy_score(y_true, y_pred)
    within1 = np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)
    return {"accuracy": exact, "macro_f1": f1_score(y_true, y_pred,
                                                    average="macro"),
            "within_1_bin": within1}


def run(horizon, regime):
    d, cols = tb.load(horizon, with_thumbs=False)
    X0 = tb.encode(d, cols)
    y_log = d.y.values
    views = d[f"d{horizon}_views"].values.astype(float)

    y_bin = bin_views(views).codes.astype(int)

    # Channel quartile: computed within channel, so it means the same thing
    # for a 500-subscriber channel and a 3-million one. Channels with too few
    # labelled videos cannot support quartiles and are dropped from C only.
    tmp = pd.DataFrame({"ch": d.channel_id.values, "y": y_log})
    counts = tmp.groupby("ch").y.transform("size")
    q = (tmp.groupby("ch").y
         .transform(lambda s: pd.qcut(s.rank(method="first"), 4,
                                      labels=False, duplicates="drop")
                    if len(s) >= 8 else np.nan))
    ok_q = q.notna().values & (counts.values >= 8)
    y_q = pd.Series(q).fillna(-1).astype(int).values

    rows = []
    for seed in SEEDS:
        if regime == "cold":
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=seed)
                          .split(X0, groups=d.channel_id))
        else:
            tr, te = train_test_split(np.arange(len(d)), test_size=.25,
                                      random_state=seed)
        X, _ = tb.add_fold_features(X0, d, tr)
        Xm = X if regime == "warm" else X.drop(columns=tb.HISTORY_FEATURES)

        # --- A: regress, then bin the prediction -------------------------
        reg = lgb.LGBMRegressor(n_estimators=800, learning_rate=.04,
                                num_leaves=63, min_child_samples=40,
                                verbose=-1, random_state=0)
        reg.fit(Xm.iloc[tr], y_log[tr])
        pred_views = np.expm1(reg.predict(Xm.iloc[te]))
        a_pred = bin_views(pred_views).codes.astype(int)
        rows.append({"seed": seed, "approach": "A regress-then-bin",
                     **score(y_bin[te], a_pred, len(NAMES))})

        # --- B: classify the bins directly -------------------------------
        clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=.04,
                                 num_leaves=63, min_child_samples=40,
                                 verbose=-1, random_state=0)
        clf.fit(Xm.iloc[tr], y_bin[tr])
        b_pred = clf.predict(Xm.iloc[te])
        rows.append({"seed": seed, "approach": "B classify-bins",
                     **score(y_bin[te], b_pred, len(NAMES))})

        # --- majority class, the floor -----------------------------------
        maj = np.bincount(y_bin[tr], minlength=len(NAMES)).argmax()
        rows.append({"seed": seed, "approach": "  majority-class baseline",
                     **score(y_bin[te], np.full(len(te), maj), len(NAMES))})

        # --- C: channel quartile -----------------------------------------
        tr_q = np.array([i for i in tr if ok_q[i]])
        te_q = np.array([i for i in te if ok_q[i]])
        if len(tr_q) > 500 and len(te_q) > 200:
            cq = lgb.LGBMClassifier(n_estimators=800, learning_rate=.04,
                                    num_leaves=63, min_child_samples=40,
                                    verbose=-1, random_state=0)
            cq.fit(Xm.iloc[tr_q], y_q[tr_q])
            rows.append({"seed": seed, "approach": "C channel-quartile",
                         **score(y_q[te_q], cq.predict(Xm.iloc[te_q]), 4)})
            rows.append({"seed": seed, "approach": "  quartile chance (25%)",
                         "accuracy": .25, "macro_f1": .25, "within_1_bin": .75})
    return pd.DataFrame(rows), len(NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    for regime, label in [("cold", "COLD START (unseen channels)"),
                          ("warm", "WARM START (channels seen before)")]:
        df, _ = run(args.horizon, regime)
        agg = df.groupby("approach").agg(["mean", "std"])
        print(f"\n=== day {args.horizon} — {label} ===")
        print(f"{'approach':30}{'accuracy':>18}{'macro F1':>12}{'within 1 bin':>15}")
        for name in ["  majority-class baseline", "A regress-then-bin",
                     "B classify-bins", "  quartile chance (25%)",
                     "C channel-quartile"]:
            if name not in agg.index:
                continue
            a = agg.loc[name]
            print(f"{name:30}"
                  f"{a[('accuracy', 'mean')]:>11.1%} ±{a[('accuracy', 'std')]:<5.3f}"
                  f"{a[('macro_f1', 'mean')]:>12.3f}"
                  f"{a[('within_1_bin', 'mean')]:>15.1%}")
    print("\nA and B share every split and feature; the difference between them")
    print("is training on bins versus binning afterwards. C is a different")
    print("question and its numbers are not comparable to A and B.")


if __name__ == "__main__":
    main()
