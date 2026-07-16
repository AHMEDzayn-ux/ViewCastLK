"""
One-time / occasional: resolve seed_channels.csv (handle or channel_id + category)
into full channel data via the YouTube Data API, classify size_tier, and write
resolved_channels.csv locally. Safe to re-run as more rows are added to the input CSV.

DB upsert (channels table) is not wired up yet - Supabase connection pending.
"""
import csv
import os
import sys

from dotenv import load_dotenv
from googleapiclient.discovery import build

# Fixed global subscriber thresholds (not per-category relative - see project decision).
MEGA_MIN = 100_000
MID_MIN = 10_000
SMALL_MIN = 2_000

INPUT_CSV = "seed_channels.csv"
OUTPUT_CSV = "resolved_channels.csv"


def classify_tier(subscriber_count):
    if subscriber_count is None:
        return None
    if subscriber_count >= MEGA_MIN:
        return "mega"
    if subscriber_count >= MID_MIN:
        return "mid"
    if subscriber_count >= SMALL_MIN:
        return "small"
    return None  # below 2,000 - not eligible for any tier


def read_seed_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            handle_or_id = row["channel_handle_or_id"].strip()
            category = row["category"].strip()
            if handle_or_id:
                rows.append((handle_or_id, category))
    return rows


def resolve_channels(youtube, seed_rows):
    """Returns list of (category, channel_item_dict) for successfully resolved channels,
    and a list of (handle_or_id, category, reason) for ones that failed to resolve."""
    resolved = []
    failed = []

    ids = [(h, c) for h, c in seed_rows if h.startswith("UC")]
    handles = [(h, c) for h, c in seed_rows if not h.startswith("UC")]

    # Batch channel IDs, up to 50 per call.
    id_to_category = {h: c for h, c in ids}
    id_list = list(id_to_category.keys())
    calls_made = 0
    for i in range(0, len(id_list), 50):
        batch = id_list[i : i + 50]
        resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
        ).execute()
        calls_made += 1
        found_ids = {item["id"] for item in resp.get("items", [])}
        for item in resp.get("items", []):
            resolved.append((id_to_category[item["id"]], item))
        for missing_id in set(batch) - found_ids:
            failed.append((missing_id, id_to_category[missing_id], "channel_id_not_found"))

    # Handles have to be looked up one at a time (forHandle takes a single value).
    for handle, category in handles:
        resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            forHandle=handle,
        ).execute()
        calls_made += 1
        items = resp.get("items", [])
        if not items:
            failed.append((handle, category, "handle_not_found"))
            continue
        resolved.append((category, items[0]))

    return resolved, failed, calls_made


def main():
    load_dotenv()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("YOUTUBE_API_KEY not set in .env")

    seed_rows = read_seed_rows(INPUT_CSV)
    if not seed_rows:
        sys.exit(f"No rows found in {INPUT_CSV}")

    youtube = build("youtube", "v3", developerKey=api_key)
    resolved, failed, calls_made = resolve_channels(youtube, seed_rows)

    out_rows = []
    tier_counts = {}
    for category, item in resolved:
        snippet = item["snippet"]
        stats = item["statistics"]
        subscriber_count = (
            int(stats["subscriberCount"]) if "subscriberCount" in stats else None
        )
        country = snippet.get("country")
        tier = classify_tier(subscriber_count)

        flags = []
        if country != "LK":
            flags.append("not_LK_or_missing_country")
        if tier is None:
            flags.append("below_2000_subscribers")

        out_rows.append(
            {
                "channel_id": item["id"],
                "title": snippet["title"],
                "category": category,
                "size_tier": tier or "",
                "country": country or "",
                "subscriber_count": subscriber_count or "",
                "channel_created_at": snippet.get("publishedAt", ""),
                "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
                "flags": ";".join(flags),
            }
        )
        tier_counts.setdefault(category, {}).setdefault(tier or "unclassified", 0)
        tier_counts[category][tier or "unclassified"] += 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel_id",
                "title",
                "category",
                "size_tier",
                "country",
                "subscriber_count",
                "channel_created_at",
                "uploads_playlist_id",
                "flags",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Resolved {len(resolved)} / {len(seed_rows)} channels -> {OUTPUT_CSV}")
    print(f"API calls made: {calls_made} (channels.list, 1 unit each)")
    if failed:
        print(f"\nFailed to resolve ({len(failed)}):")
        for handle_or_id, category, reason in failed:
            print(f"  {handle_or_id} ({category}): {reason}")

    print("\nTier counts by category:")
    for category, counts in tier_counts.items():
        print(f"  {category}: {counts}")


if __name__ == "__main__":
    main()
