"""What roster size fits inside the daily API allowance, and under what settings.

Three costs, per day:

    discovery  = runs_per_day x channels        one playlistItems.list each
    refresh    = 2 x ceil(channels / 50)        channels.list, fifty per call
    snapshots  = 4 x ceil(active_videos / 50)   videos.list, fifty per call

Snapshots stay at four runs a day in every scenario. That cadence is what puts
each horizon within about three hours of its mark, and it is not the expensive
part per channel -- but it scales with the *video* population, so it grows with
the roster and with the tracking window, and at a large roster it becomes the
dominant term.

active_videos is videos_per_day x tracking_window, because every video inside
the window is re-snapshotted on every run. The window is therefore a direct
multiplier on the biggest cost, which is why it appears here as a lever
alongside the discovery cadence.
"""
import math

DAILY_QUOTA = 10_000
BUDGET = 8_500          # headroom rather than running at the ceiling

CURRENT_CHANNELS = 603
CURRENT_VIDEOS_DAY = 800      # measured over the last ten complete days
NEW_CHANNEL_RATE = 0.275      # mean uploads/day of the 2,178 candidates,
                              # measured from their prior-batch history
CANDIDATES = 2_178


def cost(channels, runs, videos_day, window):
    discovery = runs * channels
    refresh = 2 * math.ceil(channels / 50)
    snapshots = 4 * math.ceil(videos_day * window / 50)
    return discovery, refresh, snapshots, discovery + refresh + snapshots


def affordable(runs, window, budget=BUDGET):
    """Largest number of new channels that fits, by direct search."""
    for n in range(CANDIDATES, -1, -1):
        vids = CURRENT_VIDEOS_DAY + n * NEW_CHANNEL_RATE
        if cost(CURRENT_CHANNELS + n, runs, vids, window)[3] <= budget:
            return n
    return 0


print(f"budget {BUDGET:,} of {DAILY_QUOTA:,} units/day")
print(f"new-channel upload rate: {NEW_CHANNEL_RATE}/day (measured, mean of the "
      f"{CANDIDATES:,} candidates)\n")

print("today, steady state at the current roster:")
for w in (40, 60):
    d, r, s, t = cost(CURRENT_CHANNELS, 4, CURRENT_VIDEOS_DAY, w)
    print(f"  {w}-day window, 4x discovery: "
          f"discovery {d:,}  refresh {r}  snapshots {s:,}  total {t:,}")

print(f"\nhow many of the {CANDIDATES:,} candidates fit:")
print(f"{'discovery':>12}{'40-day window':>18}{'60-day window':>18}")
for runs in (4, 3, 2, 1):
    a40, a60 = affordable(runs, 40), affordable(runs, 60)
    f40 = "ALL" if a40 >= CANDIDATES else f"{a40:,}"
    f60 = "ALL" if a60 >= CANDIDATES else f"{a60:,}"
    print(f"{str(runs) + 'x/day':>12}{f40:>18}{f60:>18}")

print("\nfull roster (603 + 2,178 = 2,781) under each setting:")
vids = CURRENT_VIDEOS_DAY + CANDIDATES * NEW_CHANNEL_RATE
print(f"  {vids:,.0f} videos/day once all are added")
print(f"{'discovery':>12}{'window':>9}{'disc':>9}{'refresh':>9}{'snaps':>9}"
      f"{'total':>9}{'':>4}")
for runs in (4, 2, 1):
    for w in (60, 40):
        d, r, s, t = cost(CURRENT_CHANNELS + CANDIDATES, runs, vids, w)
        flag = "ok" if t <= BUDGET else ("tight" if t <= DAILY_QUOTA else "OVER")
        print(f"{str(runs) + 'x/day':>12}{w:>9}{d:>9,}{r:>9}{s:>9,}{t:>9,}  {flag}")
