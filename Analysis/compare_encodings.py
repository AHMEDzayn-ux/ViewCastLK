"""Full correlation table, and does categorical encoding choice matter?

WHY THIS EXISTS
train_baseline.py encodes category_name and title_script with OrdinalEncoder,
which asserts an ordering that does not exist -- News & Politics -> 5 and
Music -> 7 does not mean Music is greater. For a linear model that would be
fatal. For trees it is softer but not free: a split on an ordinal column can
only separate CONTIGUOUS runs of those arbitrary integers, so isolating
{Comedy, Entertainment, Film & Animation} takes three splits instead of one
whenever the alphabetical accident of the encoding scatters them.

LightGBM can partition categories properly when told they are categorical, and
14 categories is small enough to one-hot. This measures whether either actually
helps, paired across identical splits so the comparison is not just split noise.

The correlation table is reported first, with two caveats built in:
  * Correlations for the categorical columns are meaningless -- they measure
    the encoding order, not the category. They are printed and flagged rather
    than omitted, so nobody re-derives them and believes them.
  * A single rank correlation understates any non-monotonic feature.
    publish_hour_slt is cyclical and duration_seconds has a sweet spot, so both
    look weaker here than their permutation importance says they are.

Usage:  python Analysis/compare_encodings.py [--horizon 7]
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy import stats

import train_baseline as tb

SEEDS = range(5)


def correlation_table(d, X, y):
    rows = []
    for c in X.columns:
        v = X[c].values.astype(float)
        sp = stats.spearmanr(v, y)
        rows.append({
            "feature": c,
            "spearman_views": sp.statistic,
            "pearson_views": stats.pearsonr(v, y).statistic,
            "spearman_resid": stats.spearmanr(v, d.resid.values).statistic,
            "p_value": sp.pvalue,
            "meaningful": "no - ordinal code" if c in tb.CAT else "yes",
        })
    t = pd.DataFrame(rows).set_index("feature")
    t = t.reindex(t.spearman_views.abs().sort_values(ascending=False).index)
    t["p_value"] = t.p_value.map(lambda p: "<1e-300" if p == 0 else f"{p:.1e}")
    return t


def build(Xf, d, mode):
    """Return (frame, extra kwargs for fit) for one encoding strategy."""
    if mode == "ordinal":
        return Xf, {}
    if mode == "native":
        X = Xf.copy()
        for c in tb.CAT:
            X[c] = X[c].astype(int).astype("category")
        return X, {"categorical_feature": tb.CAT}
    base = Xf.drop(columns=tb.CAT)
    oh = pd.get_dummies(d[tb.CAT].astype(str), dummy_na=True)
    oh.index = base.index
    return pd.concat([base, oh], axis=1), {}


def score(d, X0, y, regime, mode):
    out = []
    for seed in SEEDS:
        if regime == "cold":
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=seed)
                          .split(X0, groups=d.channel_id))
        else:
            tr, te = train_test_split(np.arange(len(d)), test_size=.25,
                                      random_state=seed)
        Xf, _ = tb.add_fold_features(X0, d, tr)
        if regime == "cold":
            Xf = Xf.drop(columns=tb.HISTORY_FEATURES)
        X, kw = build(Xf, d, mode)

        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=.04,
                              num_leaves=63, min_child_samples=40,
                              subsample=.9, subsample_freq=1,
                              colsample_bytree=.9, verbose=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr], **kw)
        out.append(r2_score(y[te], m.predict(X.iloc[te])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    d, cols = tb.load(args.horizon, with_thumbs=False)
    X0 = tb.encode(d, cols)
    y = d.y.values
    d["resid"] = y - d.groupby("channel_id").y.transform("mean")
    X_all, _ = tb.add_fold_features(X0, d, np.arange(len(d)))

    print("=" * 94)
    print(f"CORRELATION TABLE — day {args.horizon}, n = {len(d):,}")
    print("=" * 94)
    print("spearman_views : rank correlation with log1p(views)")
    print("spearman_resid : rank correlation with the within-channel residual")
    print("p_value        : for spearman_views\n")
    print(correlation_table(d, X_all, y).round(3).to_string())
    print("\nA single rank correlation understates non-monotonic features.")
    print("publish_hour_slt is cyclical and duration_seconds has a sweet spot,")
    print("so both rank far higher on permutation importance than here.")

    print("\n" + "=" * 94)
    print("ENCODING COMPARISON — R2 on log1p(views), 5 identical splits")
    print("=" * 94)
    for regime in ["cold", "warm"]:
        res = {m: score(d, X0, y, regime, m)
               for m in ["ordinal", "native", "onehot"]}
        print(f"\n{regime.upper()} START")
        for m in ["ordinal", "native", "onehot"]:
            print(f"  {m:9} R2 = {res[m].mean():+.4f} ± {res[m].std():.4f}")
        for m in ["native", "onehot"]:
            t, p = stats.ttest_rel(res[m], res["ordinal"])
            print(f"  {m:9} vs ordinal: "
                  f"{res[m].mean() - res['ordinal'].mean():+.4f}  "
                  f"t={t:+.2f}  p={p:.3f}  "
                  f"better on {int((res[m] > res['ordinal']).sum())}/5 splits")


if __name__ == "__main__":
    main()
