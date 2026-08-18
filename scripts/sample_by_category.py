"""
One-shot sample: for each of the 15 standard YouTube categories, pull today's
Sri Lankan videos via search.list (order=date, maxResults=25), resolve each
video's channel via channels.list, keep only channels with country == LK,
and classify by subscriber tier. Writes results/today_sample.csv.
"""
import csv
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build

sys.stdout.reconfigure(encoding="utf-8")


# (name, query) - search.list on this API key returns 0 results with no `q`
# term at all, regardless of order/regionCode/videoCategoryId (verified: even
# type=video+order=date+regionCode=LK alone returns totalResults=0). A q is
# mandatory to get anything back, so each category gets one deliberately
# generic keyword to minimize relevance-driven bias beyond what's unavoidable.
CATEGORIES = {
    "1": ("Film & Animation", "film"),
    "2": ("Autos & Vehicles", "car"),
    "10": ("Music", "song"),
    "15": ("Pets & Animals", "pet"),
    "17": ("Sports", "cricket"),
    "19": ("Travel & Events", "travel"),
    "20": ("Gaming", "gaming"),
    "22": ("People & Blogs", "vlog"),
    "23": ("Comedy", "comedy"),
    "24": ("Entertainment", "entertainment"),
    "25": ("News & Politics", "news"),
    "26": ("Howto & Style", "style"),
    "27": ("Education", "education"),
    "28": ("Science & Technology", "technology"),
    "29": ("Nonprofits & Activism", "charity"),
}

OUT_CSV = "results/today_sample.csv"


def classify_tier(sub_count):
    if sub_count is None:
        return "unknown"
    if sub_count >= 100_000:
        return "mega"
    if sub_count >= 10_000:
        return "mid"
    return "small"


def search_videos(youtube, category_id, query, published_after, max_results=25):
    resp = youtube.search().list(
        part="snippet",
        type="video",
        q=query,
        regionCode="LK",
        videoCategoryId=category_id,
        publishedAfter=published_after,
        order="date",
        maxResults=max_results,
    ).execute()
    return [(item["id"]["videoId"], item["snippet"]["channelId"]) for item in resp.get("items", [])]


def resolve_channels(youtube, channel_ids):
    info = {}
    ids = list(dict.fromkeys(channel_ids))
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        resp = youtube.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            stats = item["statistics"]
            sub_count = int(stats["subscriberCount"]) if "subscriberCount" in stats else None
            info[item["id"]] = {
                "country": item["snippet"].get("country"),
                "subscriber_count": sub_count,
            }
    return info


def main():
    load_dotenv()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("YOUTUBE_API_KEY not set in .env")

    published_after = sys.argv[1] if len(sys.argv) > 1 else None
    if not published_after:
        start_of_today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        published_after = start_of_today.strftime("%Y-%m-%dT%H:%M:%SZ")

    youtube = build("youtube", "v3", developerKey=api_key)

    os.makedirs("results", exist_ok=True)
    rows = []
    summary = {}
    search_units = 0
    channel_units = 0

    for cat_id, (cat_name, query) in CATEGORIES.items():
        print(f"\n=== {cat_name} (id={cat_id}, q='{query}') ===")
        try:
            vids = search_videos(youtube, cat_id, query, published_after)
        except Exception as e:
            print(f"  search.list failed: {e}")
            summary[cat_name] = {"mega": 0, "mid": 0, "small": 0, "unknown": 0, "non_lk_excluded": 0, "error": str(e)}
            continue
        search_units += 100
        print(f"  {len(vids)} videos returned")

        channel_ids = [c for _, c in vids]
        chan_info = resolve_channels(youtube, channel_ids)
        channel_units += (len(set(channel_ids)) // 50 + 1)

        cat_counts = {"mega": 0, "mid": 0, "small": 0, "unknown": 0, "non_lk_excluded": 0}
        for video_id, channel_id in vids:
            info = chan_info.get(channel_id, {})
            country = info.get("country")
            if country != "LK":
                cat_counts["non_lk_excluded"] += 1
                continue
            sub_count = info.get("subscriber_count")
            tier = classify_tier(sub_count)
            cat_counts[tier] += 1
            rows.append([video_id, cat_name, channel_id, sub_count if sub_count is not None else "", tier])

        print(f"  kept LK channels: mega={cat_counts['mega']} mid={cat_counts['mid']} "
              f"small={cat_counts['small']} unknown={cat_counts['unknown']} "
              f"(excluded non-LK: {cat_counts['non_lk_excluded']})")
        summary[cat_name] = cat_counts

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "category", "channel_id", "subscriber_count", "tier"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    print(f"API units used: search={search_units}, channels~={channel_units}")
    print("\nPer-category summary:")
    for cat_name, counts in summary.items():
        print(f"  {cat_name}: {counts}")


if __name__ == "__main__":
    main()
