"""Two checks on the prior batch's dataset:

1. How much its channel set overlaps ViewCastLK's tracked roster.
2. Whether its `country = LK` label is a verified channel property or just a
   by-product of LK-regioned search. Samples channels and asks the YouTube API
   what country each one actually declares.
"""
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Project Code"))
from youtube_client import youtube          # noqa: E402
from storage import connect                 # noqa: E402

D = os.path.join(os.path.dirname(__file__), "..", "Reference Datasets")
SAMPLE = 300
random.seed(7)

f1 = pd.read_csv(os.path.join(D, "FinalProcessedDataset_05 (1).csv"), low_memory=False)
f2 = pd.read_csv(os.path.join(D, "youtube_video_data_lk.csv"), low_memory=False)

ch1, ch2 = set(f1.channel_id.dropna()), set(f2.channel_id.dropna())
prior = ch1 | ch2

conn = connect()
with conn.cursor() as cur:
    cur.execute("SELECT channel_id FROM channels")
    ours = {r[0] for r in cur.fetchall()}
conn.close()

print("=" * 70)
print("CHANNEL OVERLAP")
print("=" * 70)
print(f"prior file 1 channels : {len(ch1):,}")
print(f"prior file 2 channels : {len(ch2):,}")
print(f"prior combined        : {len(prior):,}")
print(f"ViewCastLK roster     : {len(ours):,}")
print(f"shared with our roster: {len(prior & ours):,} "
      f"({100*len(prior & ours)/max(len(ours),1):.1f}% of our roster)")
print(f"  via file 1: {len(ch1 & ours):,}   via file 2: {len(ch2 & ours):,}")

# videos per channel — tells us how each file was collected
print(f"\nvideos per channel  file1: {len(f1)/len(ch1):.2f}   file2: {len(f2)/len(ch2):.2f}")

print()
print("=" * 70)
print(f"IS 'country = LK' REAL?  (sampling {SAMPLE} channels from file 1)")
print("=" * 70)
sample = random.sample(sorted(ch1), min(SAMPLE, len(ch1)))

counts, checked, missing = {}, 0, 0
for i in range(0, len(sample), 50):
    batch = sample[i:i + 50]
    resp = youtube.channels().list(part="snippet", id=",".join(batch)).execute()
    items = resp.get("items", [])
    found = {it["id"] for it in items}
    missing += len(batch) - len(found)
    for it in items:
        c = it["snippet"].get("country", "(not set)")
        counts[c] = counts.get(c, 0) + 1
        checked += 1

print(f"channels resolved: {checked}   not found/deleted: {missing}")
print("\ndeclared country of sampled channels:")
for c, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"    {c:12} {n:>5}  ({100*n/max(checked,1):.1f}%)")

lk = counts.get("LK", 0)
notset = counts.get("(not set)", 0)
other = checked - lk - notset
print(f"\n  verified LK      : {lk:>5}  ({100*lk/max(checked,1):.1f}%)")
print(f"  country not set  : {notset:>5}  ({100*notset/max(checked,1):.1f}%)")
print(f"  explicitly NOT LK: {other:>5}  ({100*other/max(checked,1):.1f}%)")
print(f"\napprox quota used: {(len(sample)+49)//50} units")
