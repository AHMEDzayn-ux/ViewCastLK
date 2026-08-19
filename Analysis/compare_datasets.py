"""Compare two builds of the training table.

A rebuild is not just "more rows". Label coverage moves unevenly across
horizons, the usable subsets move differently again once data-quality filters
are applied, and the target distribution can shift if the newly labelled videos
are not like the ones already there. Anyone who has already started modelling
on the older file needs to know which of those happened.

Usage:
    python Analysis/compare_datasets.py OLD.parquet NEW.parquet
"""
import argparse

import numpy as np
import pandas as pd

HORIZONS = (7, 14, 21, 30)


def pct(new, old):
    if old == 0:
        return "  n/a" if new == 0 else "   new"
    return f"{(new - old) / old:+.0%}"


def line(label, old, new, width=34):
    print(f"{label:<{width}} {old:>10,} {new:>10,} {new - old:>+10,} "
          f"{pct(new, old):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    args = ap.parse_args()

    o, n = pd.read_parquet(args.old), pd.read_parquet(args.new)
    oe, ne = o[o.eligible], n[n.eligible]

    print("=" * 76)
    print(f"{'':34} {'OLD':>10} {'NEW':>10} {'change':>10} {'':>8}")
    print("=" * 76)
    print("\nSIZE")
    line("rows", len(o), len(n))
    line("eligible", len(oe), len(ne))
    line("distinct channels", oe.channel_id.nunique(), ne.channel_id.nunique())
    print(f"{'published through':<34} "
          f"{oe.published_at.max():%Y-%m-%d:>10} "
          f"{ne.published_at.max():%Y-%m-%d:>10}".replace(":>10", ""))

    print("\nLABEL COVERAGE (usable)")
    for h in HORIZONS:
        line(f"  day {h}", int(oe[f"d{h}_usable"].sum()),
             int(ne[f"d{h}_usable"].sum()))

    print("\nMODELLING SUBSET (usable + true point-in-time channel stats)")
    for h in HORIZONS:
        a = int((oe[f"d{h}_usable"] & ~oe.channel_stats_backfilled).sum())
        b = int((ne[f"d{h}_usable"] & ~ne.channel_stats_backfilled).sum())
        line(f"  day {h}", a, b)

    print("\nDATA-QUALITY FLAGS")
    line("  channel stats backfilled", int(oe.channel_stats_backfilled.sum()),
         int(ne.channel_stats_backfilled.sum()))
    line("  title changed", int(oe.title_changed.sum()),
         int(ne.title_changed.sum()))
    line("  description changed", int(oe.description_changed.sum()),
         int(ne.description_changed.sum()))

    print("\nDAY-7 TARGET DISTRIBUTION")
    ov = oe.loc[oe.d7_usable, "d7_views"]
    nv = ne.loc[ne.d7_usable, "d7_views"]
    for q, lbl in ((.25, "  p25"), (.50, "  median"), (.75, "  p75"),
                   (.99, "  p99")):
        line(lbl, int(ov.quantile(q)), int(nv.quantile(q)))
    line("  max", int(ov.max()), int(nv.max()))
    print(f"{'  log1p mean':<34} {np.log1p(ov).mean():>10.2f} "
          f"{np.log1p(nv).mean():>10.2f} "
          f"{np.log1p(nv).mean() - np.log1p(ov).mean():>+10.2f}")

    print("\nCOLUMNS")
    added = [c for c in n.columns if c not in o.columns]
    removed = [c for c in o.columns if c not in n.columns]
    print(f"  added:   {', '.join(added) if added else 'none'}")
    print(f"  removed: {', '.join(removed) if removed else 'none'}")

    print("\nROW OVERLAP")
    ids_o, ids_n = set(o.video_id), set(n.video_id)
    print(f"  in both:          {len(ids_o & ids_n):,}")
    print(f"  only in old:      {len(ids_o - ids_n):,}")
    print(f"  new in this build:{len(ids_n - ids_o):,}")

    # Newly labelled videos are the ones that change a model, so check whether
    # they resemble what was already there. A large shift means the older
    # sample was not representative and results on it will not carry over.
    was_labelled = set(oe.loc[oe.d7_usable, "video_id"])
    fresh = ne[ne.d7_usable & ~ne.video_id.isin(was_labelled)]
    if len(fresh):
        print(f"\nNEWLY LABELLED AT DAY 7: {len(fresh):,}")
        print(f"  median views      {fresh.d7_views.median():>12,.0f}   "
              f"(existing: {ov.median():,.0f})")
        print(f"  log1p mean        {np.log1p(fresh.d7_views).mean():>12.2f}   "
              f"(existing: {np.log1p(ov).mean():.2f})")
        top = fresh.category_name.value_counts(normalize=True).head(3)
        print("  top categories    " + ", ".join(
            f"{k} {v:.0%}" for k, v in top.items()))
    print()


if __name__ == "__main__":
    main()
