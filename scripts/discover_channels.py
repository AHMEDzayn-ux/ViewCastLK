"""
Occasional/manual: use search.list (100 units/call) to find candidate Sri Lankan
channels per category, then resolve them via channels.list (1 unit/50 channels)
to verify country + subscriber count, and append new candidates to seed_channels.csv.

This is the fallback discovery path (spec section 6.5 / proposal section 3) - expensive
relative to the other calls, so run it deliberately per category, not in one big sweep.
"""
import csv
import os
import sys

from dotenv import load_dotenv
from googleapiclient.discovery import build

sys.stdout.reconfigure(encoding="utf-8")

SEED_CSV = "seed_channels.csv"

# category -> search query used against search.list
CATEGORY_QUERIES = {
    "Film & Animation": "Sri Lanka short film",
    "Autos & Vehicles": "Sri Lanka vehicle review",
    "Music": "Sri Lanka music",
    "Pets & Animals": "Sri Lanka pets",
    "Sports": "Sri Lanka sports",
    "Travel & Events": "Sri Lanka travel vlog",
    "Gaming": "Sri Lanka gaming",
    "People & Blogs": "Sri Lanka vlog",
    "Comedy": "Sri Lanka comedy",
    "Entertainment": "Sri Lanka entertainment",
    "News & Politics": "Sri Lanka news",
    "Howto & Style": "Sri Lanka style tips",
    "Education": "Sri Lanka education",
    "Science & Technology": "Sri Lanka technology",
    "Nonprofits & Activism": "Sri Lanka nonprofit",
}


def load_existing_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["channel_handle_or_id"].strip() for row in csv.DictReader(f)}


def search_channels(youtube, query, max_results=25):
    resp = youtube.search().list(
        part="snippet",
        q=query,
        type="channel",
        regionCode="LK",
        maxResults=max_results,
    ).execute()
    return [item["snippet"]["channelId"] for item in resp.get("items", [])]


def resolve_and_filter(youtube, channel_ids):
    """Batch-resolve channel IDs, return list of (channel_id, title, country, subscriber_count)."""
    results = []
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        resp = youtube.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            stats = item["statistics"]
            sub_count = int(stats["subscriberCount"]) if "subscriberCount" in stats else None
            results.append(
                (item["id"], item["snippet"]["title"], item["snippet"].get("country"), sub_count)
            )
    return results


REVIEW_CSV = "discovered_candidates.csv"


def classify_tier_hint(subscriber_count):
    if subscriber_count is None:
        return ""
    if subscriber_count >= 100_000:
        return "mega"
    if subscriber_count >= 10_000:
        return "mid"
    if subscriber_count >= 2_000:
        return "small"
    return "below_threshold"


def main():
    load_dotenv()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("YOUTUBE_API_KEY not set in .env")

    categories = sys.argv[1:]
    if not categories:
        sys.exit(
            "Usage: python scripts/discover_channels.py \"Category Name\" [\"Another Category\" ...]\n"
            "Available: " + ", ".join(CATEGORY_QUERIES.keys())
        )

    youtube = build("youtube", "v3", developerKey=api_key)
    existing_ids = load_existing_ids(SEED_CSV)

    # country field is unreliable (many real LK channels leave it blank, some
    # unrelated channels show a foreign country) - so this writes everything
    # to a review CSV instead of auto-appending to the real seed list.
    file_exists = os.path.exists(REVIEW_CSV)
    review_file = open(REVIEW_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(review_file)
    if not file_exists:
        writer.writerow(["channel_id", "title", "category", "country", "subscriber_count", "tier_hint"])

    search_units = 0
    row_count = 0
    for category in categories:
        query = CATEGORY_QUERIES.get(category)
        if not query:
            print(f"Skipping unknown category: {category}")
            continue

        channel_ids = search_channels(youtube, query)
        search_units += 100
        resolved = resolve_and_filter(youtube, channel_ids)

        print(f"\n=== {category} (query: '{query}') ===")
        for channel_id, title, country, sub_count in resolved:
            if channel_id in existing_ids:
                continue
            tier_hint = classify_tier_hint(sub_count)
            print(f"  {channel_id}  {title}  country={country}  subs={sub_count}  tier={tier_hint}")
            writer.writerow([channel_id, title, category, country or "", sub_count or "", tier_hint])
            row_count += 1

    review_file.close()
    print(f"\nWrote {row_count} candidates to {REVIEW_CSV} for manual review.")
    print("Eyeball each row (real SL channel? correct category? reasonable tier?) before")
    print(f"copying the ones you approve into {SEED_CSV}.")
    print(f"search.list units used this run: {search_units}")


if __name__ == "__main__":
    main()
