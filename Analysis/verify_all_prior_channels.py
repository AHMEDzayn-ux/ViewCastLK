"""Verify the declared country of every channel in the prior batch's dataset,
then report how many videos survive an LK-only filter at each forecast horizon.

Writes results incrementally so a quota cutoff never loses work; re-running
resumes from what has already been checked.
"""
import os
import sys
import csv

import pandas as pd
from googleapiclient.errors import HttpError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Project Code"))
from youtube_client import youtube          # noqa: E402

D = os.path.join(os.path.dirname(__file__), "..", "Reference Datasets")
OUT = os.path.join(os.path.dirname(__file__), "prior_channel_countries.csv")

f1 = pd.read_csv(os.path.join(D, "FinalProcessedDataset_05 (1).csv"), low_memory=False)
f2 = pd.read_csv(os.path.join(D, "youtube_video_data_lk.csv"), low_memory=False)
all_ch = sorted(set(f1.channel_id.dropna()) | set(f2.channel_id.dropna()))

done = {}
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as fh:
        done = {r["channel_id"]: r["country"] for r in csv.DictReader(fh)}
    print(f"resuming — {len(done):,} already checked")

todo = [c for c in all_ch if c not in done]
print(f"{len(all_ch):,} channels total, {len(todo):,} to check "
      f"(~{(len(todo)+49)//50} quota units)")

new = open(OUT, "a", newline="", encoding="utf-8")
w = csv.writer(new)
if not done:
    w.writerow(["channel_id", "country", "title", "subscriber_count"])

stopped = False
for i in range(0, len(todo), 50):
    batch = todo[i:i + 50]
    try:
        resp = youtube.channels().list(part="snippet,statistics",
                                       id=",".join(batch)).execute()
    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            print("\nQUOTA EXCEEDED — stopping cleanly; re-run to resume.")
            stopped = True
            break
        raise
    found = set()
    for it in resp.get("items", []):
        cid = it["id"]
        found.add(cid)
        country = it["snippet"].get("country", "")
        done[cid] = country
        w.writerow([cid, country, it["snippet"].get("title", ""),
                    it.get("statistics", {}).get("subscriberCount", "")])
    for missing in set(batch) - found:            # deleted / terminated
        done[missing] = "__GONE__"
        w.writerow([missing, "__GONE__", "", ""])
    new.flush()
    if (i // 50) % 40 == 0:
        print(f"  {i + len(batch):,}/{len(todo):,} checked...")
new.close()

# ---------------------------------------------------------------- reporting
print("\n" + "=" * 70)
print("COUNTRY BREAKDOWN")
print("=" * 70)
ser = pd.Series(done)
vc = ser.value_counts()
lk = int(vc.get("LK", 0))
gone = int(vc.get("__GONE__", 0))
unset = int(vc.get("", 0))
total = len(ser)
print(f"channels checked : {total:,}")
print(f"  verified LK    : {lk:,} ({100*lk/total:.1f}%)")
print(f"  country not set: {unset:,} ({100*unset/total:.1f}%)")
print(f"  deleted/gone   : {gone:,} ({100*gone/total:.1f}%)")
print(f"  other countries: {total-lk-unset-gone:,} "
      f"({100*(total-lk-unset-gone)/total:.1f}%)")
print("\ntop declared countries:")
for c, n in vc.head(12).items():
    label = {"": "(not set)", "__GONE__": "(deleted)"}.get(c, c)
    print(f"    {label:12} {n:>7,} ({100*n/total:.1f}%)")

if stopped:
    print("\nIncomplete — re-run after quota reset for the full picture.")
    sys.exit(0)

lk_ch = {c for c, v in done.items() if v == "LK"}
print("\n" + "=" * 70)
print("VIDEOS SURVIVING AN LK-VERIFIED FILTER")
print("=" * 70)
for name, df in [("file 1", f1), ("file 2", f2)]:
    keep = df[df.channel_id.isin(lk_ch)]
    print(f"\n{name}: {len(keep):,} of {len(df):,} videos "
          f"({100*len(keep)/len(df):.1f}%) from {keep.channel_id.nunique():,} LK channels")
    age = pd.to_numeric(keep.days_since_publish, errors="coerce")
    for lo, hi, lbl in [(6, 8, "day 7"), (13, 15, "day 14"),
                        (20, 22, "day 21"), (28, 32, "day 30")]:
        print(f"    {lbl:7}: {age.between(lo, hi).sum():,}")

both = pd.concat([f1[["video_id", "channel_id", "days_since_publish"]],
                  f2[["video_id", "channel_id", "days_since_publish"]]]).drop_duplicates("video_id")
keep = both[both.channel_id.isin(lk_ch)]
age = pd.to_numeric(keep.days_since_publish, errors="coerce")
print(f"\nCOMBINED (deduped): {len(keep):,} LK-verified videos")
for lo, hi, lbl in [(6, 8, "day 7"), (13, 15, "day 14"),
                    (20, 22, "day 21"), (28, 32, "day 30")]:
    print(f"    {lbl:7}: {age.between(lo, hi).sum():,}")
print(f"\nchannel country map saved -> {OUT}")
