"""Broad, incremental channel discovery to grow the roster toward ~1000 channels.

Runs many distinct category+keyword search.list queries (breadth over depth —
per earlier findings, paging deeper into one query mostly resurfaces the same
large channels rather than finding new ones).

Safety: every newly-verified channel is appended to channel_handles.txt
immediately, not batched at the end. If the YouTube API reports quotaExceeded
mid-run, the script stops cleanly instead of crashing — everything found up to
that point is already safely on disk, and re-running later (e.g. tomorrow, once
quota resets) resumes with no lost work and automatically skips channels
already known.
"""
import csv
import os

from googleapiclient.errors import HttpError

from youtube_client import youtube

HANDLES_PATH = "channel_handles.txt"
LOG_PATH = "output/discovery_log.csv"

CATEGORY_QUERIES = {
    "Film & Animation": ["Sri Lanka short film", "Sri Lanka animation"],
    "Autos & Vehicles": ["Sri Lanka car review", "Sri Lanka bike vlog"],
    "Music": ["Sri Lanka music cover", "Sri Lanka original song"],
    "Pets & Animals": ["Sri Lanka pets", "Sri Lanka dog training"],
    "Sports": ["Sri Lanka cricket highlights", "Sri Lanka football"],
    "Travel & Events": ["Sri Lanka travel vlog", "Sri Lanka tour guide"],
    "Gaming": ["Sri Lanka gaming", "Sri Lanka gameplay sinhala"],
    "People & Blogs": ["Sri Lanka vlog", "Sri Lanka daily vlog"],
    "Comedy": ["Sri Lanka comedy sinhala", "Sri Lanka funny"],
    "Entertainment": ["Sri Lanka entertainment", "Sri Lanka teledrama"],
    "News & Politics": ["Sri Lanka news", "Sri Lanka political analysis"],
    "Howto & Style": ["Sri Lanka makeup tutorial", "Sri Lanka fashion sinhala"],
    "Education": ["Sri Lanka education sinhala", "Sri Lanka tuition class"],
    "Science & Technology": ["Sri Lanka tech review", "Sri Lanka technology sinhala"],
    "Nonprofits & Activism": ["Sri Lanka charity", "Sri Lanka nonprofit"],
}


class QuotaExceeded(Exception):
    pass


def load_existing_channel_ids() -> set[str]:
    ids = set()
    if os.path.isfile("output/channels.csv"):
        with open("output/channels.csv", encoding="utf-8-sig") as f:
            ids.update(row["channel_id"] for row in csv.DictReader(f))
    return ids


def load_existing_handles() -> set[str]:
    if not os.path.isfile(HANDLES_PATH):
        return set()
    with open(HANDLES_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_roster_entry(value: str) -> None:
    with open(HANDLES_PATH, "a", encoding="utf-8") as f:
        f.write(value + "\n")


def run_query(query: str, category: str, known_ids: set, known_roster: set, log_rows: list) -> int:
    try:
        resp = youtube.search().list(
            part="snippet", q=query, type="channel", regionCode="LK", maxResults=50,
        ).execute()
    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            raise QuotaExceeded()
        raise

    candidate_ids = [item["snippet"]["channelId"] for item in resp.get("items", [])]
    new_ids = [cid for cid in candidate_ids if cid not in known_ids]
    if not new_ids:
        print("  no new candidates")
        return 0

    added = 0
    for i in range(0, len(new_ids), 50):
        batch = new_ids[i:i + 50]
        try:
            info_resp = youtube.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                raise QuotaExceeded()
            raise

        for item in info_resp.get("items", []):
            cid = item["id"]
            title = item["snippet"]["title"]
            country = item["snippet"].get("country", "")
            subs = item["statistics"].get("subscriberCount", "")
            custom_url = item["snippet"].get("customUrl", "")
            is_lk = country == "LK"
            known_ids.add(cid)

            roster_entry = custom_url.lstrip("@") if custom_url else cid
            promoted = is_lk and roster_entry not in known_roster
            if promoted:
                append_roster_entry(roster_entry)
                known_roster.add(roster_entry)
                added += 1
                print(f"  + {title} ({roster_entry}, {subs} subs)")

            log_rows.append({
                "channel_id": cid, "title": title, "category": category, "query": query,
                "country": country, "subscriber_count": subs, "roster_entry": roster_entry,
                "promoted": promoted,
            })
    return added


def main():
    from storage import append_rows

    known_ids = load_existing_channel_ids()
    known_roster = load_existing_handles()
    print(f"Starting with {len(known_ids)} known channel IDs, {len(known_roster)} known roster entries.\n")

    total_added = 0
    stopped_early = False
    try:
        for category, queries in CATEGORY_QUERIES.items():
            for query in queries:
                print(f"Searching: '{query}' ({category})...")
                log_rows = []
                try:
                    total_added += run_query(query, category, known_ids, known_roster, log_rows)
                finally:
                    if log_rows:
                        append_rows(log_rows, LOG_PATH)
    except QuotaExceeded:
        stopped_early = True
        print("\nQUOTA EXCEEDED — stopped cleanly. Everything found so far is already saved to disk.")

    print(f"\n{'Stopped early due to quota.' if stopped_early else 'All queries completed.'}")
    print(f"Added {total_added} new channels this run.")
    print(f"Roster now has {len(known_roster)} total entries.")


if __name__ == "__main__":
    main()
