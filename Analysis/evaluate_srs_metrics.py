"""Evaluate the forecasting model against the metrics the SRS actually specifies.

WHY THIS EXISTS SEPARATELY FROM train_baseline.py
train_baseline.py reports R^2 and MAE on log-transformed views, which is the
right scale to *fit* on. The SRS names something else as the headline:

  "MAPE  Mean Absolute Percentage Error; the primary accuracy metric for this
   system, expressing error as a proportion of the true value."   -- SRS v1.1

and the feasibility study fixes the comparison:

  "...with documented performance metrics (MAPE, R^2, MAE, RMSE) relative to a
   naive category-average baseline."

So the numbers that go in the report are proportional error on RAW view counts,
against a category-average baseline, per horizon and combined (FR-13).

WHAT MAPE DOES TO A DISTRIBUTION LIKE THIS
MAPE divides by the true value, so a video that got 12 views and was predicted
1,200 contributes 9,900% on its own. Day-7 views have a median of 1,120 and a
10th percentile of 36, and 106 videos have exactly zero -- for which MAPE is
undefined and which must be excluded, not silently treated as a small number.
The result is a metric dominated by the smallest videos in the corpus rather
than by the model's typical behaviour.

That is a property of MAPE, not a defect in the model, and it is the reason
MedAPE (the median rather than the mean) and sMAPE are reported alongside. All
three are shown so the report can quote the specified metric honestly while
also showing a figure that describes the typical case.

Usage:
    python Analysis/evaluate_srs_metrics.py
    python Analysis/evaluate_srs_metrics.py --horizon 7
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
from sklearn.metrics import r2_score
from scipy import stats

import train_baseline as tb
from paths import dataset_path

SEEDS = range(5)


def proportional_metrics(actual, pred):
    """MAPE and friends, on RAW views. Zeros excluded and counted."""
    actual = np.asarray(actual, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)

    nz = actual > 0
    ape = np.abs(actual[nz] - pred[nz]) / actual[nz]
    denom = (np.abs(actual) + np.abs(pred)) / 2
    smape = np.abs(actual - pred) / np.where(denom == 0, np.nan, denom)

    return {
        "MAPE_%": 100 * ape.mean(),
        "MedAPE_%": 100 * np.median(ape),
        "sMAPE_%": 100 * np.nanmean(smape),
        "MAE_views": np.abs(actual - pred).mean(),
        "RMSE_views": np.sqrt(((actual - pred) ** 2).mean()),
        "MedAE_views": np.median(np.abs(actual - pred)),
        "excluded_zero": int((~nz).sum()),
    }


def run(horizon, regime):
    d, cols = tb.load(horizon, with_thumbs=False)
    X0 = tb.encode(d, cols)
    y_log = d.y.values
    y_raw = d[f"d{horizon}_views"].values.astype(float)

    rows = []
    for seed in SEEDS:
        if regime == "cold":
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=seed)
                          .split(X0, groups=d.channel_id))
        else:
            tr, te = train_test_split(np.arange(len(d)), test_size=.25,
                                      random_state=seed)
        X, gm = tb.add_fold_features(X0, d, tr)
        Xm = X if regime == "warm" else X.drop(columns=tb.HISTORY_FEATURES)

        # The naive baseline the feasibility study names: the average of the
        # category, learned on the training fold only. Computed as a mean of
        # log views then back-transformed, so one viral video in a category
        # does not set the baseline for every video in it.
        train = d.iloc[tr]
        cat_mean = train.groupby("category_name").y.mean()
        base_cat = np.expm1(d.category_name.map(cat_mean)
                            .fillna(train.y.mean()).values[te])

        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=.04,
                              num_leaves=63, min_child_samples=40,
                              subsample=.9, subsample_freq=1,
                              colsample_bytree=.9, verbose=-1, random_state=0)
        m.fit(Xm.iloc[tr], y_log[tr])
        pred = np.expm1(m.predict(Xm.iloc[te]))

        for name, p in [("naive category-average", base_cat),
                        ("LightGBM", pred)]:
            r = proportional_metrics(y_raw[te], p)
            r["R2_log"] = r2_score(y_log[te],
                                   np.log1p(np.clip(p, 0, None)))
            rows.append({"seed": seed, "model": name, **r})
    return pd.DataFrame(rows)


def show(df, label):
    print(f"\n  --- {label} ---")
    agg = df.groupby("model").mean(numeric_only=True)
    cols = ["MAPE_%", "MedAPE_%", "sMAPE_%", "MedAE_views", "MAE_views",
            "RMSE_views", "R2_log"]
    print(f"  {'model':24}" + "".join(f"{c:>14}" for c in cols))
    for name in ["naive category-average", "LightGBM"]:
        if name not in agg.index:
            continue
        a = agg.loc[name]
        print(f"  {name:24}" + "".join(
            f"{a[c]:>14,.1f}" if "R2" not in c else f"{a[c]:>14.3f}"
            for c in cols))

    # Paired across the same splits, so split-to-split noise cancels.
    piv = df.pivot(index="seed", columns="model")
    out = []
    for metric in ["MAPE_%", "MedAPE_%", "R2_log"]:
        a = piv[(metric, "LightGBM")]
        b = piv[(metric, "naive category-average")]
        t, p = stats.ttest_rel(a, b)
        better = (a < b).sum() if "%" in metric else (a > b).sum()
        out.append(f"    {metric:12} model vs baseline: "
                   f"t={t:+6.2f}  p={p:.4f}  better on {better}/{len(a)} splits")
    print("  paired significance (5 splits):")
    print("\n".join(out))
    z = int(df.excluded_zero.mean())
    print(f"    ({z} videos with zero views excluded from MAPE -- undefined)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=0)
    args = ap.parse_args()

    print("Metrics on RAW view counts, as the SRS specifies. MAPE is the named")
    print("primary metric; MedAPE and sMAPE accompany it because MAPE on this")
    print("distribution is dominated by the smallest videos (see module docs).")

    combined = []
    for h in ([args.horizon] if args.horizon else list(tb.HORIZONS)):
        print(f"\n{'=' * 100}\nHORIZON: day {h}\n{'=' * 100}")
        for regime, label in [("cold", "COLD START (unseen channels)"),
                              ("warm", "WARM START (channels seen before)")]:
            df = run(h, regime)
            show(df, label)
            df["horizon"], df["regime"] = h, regime
            combined.append(df)

    if len(combined) > 1:
        allr = pd.concat(combined)
        print(f"\n{'=' * 100}\nCOMBINED ACROSS HORIZONS (FR-13)\n{'=' * 100}")
        for regime in ["cold", "warm"]:
            sub = allr[allr.regime == regime]
            g = sub.groupby("model")[["MAPE_%", "MedAPE_%", "sMAPE_%",
                                      "R2_log"]].mean()
            print(f"\n  {regime} start:")
            print(g.round(2).to_string())


if __name__ == "__main__":
    main()
