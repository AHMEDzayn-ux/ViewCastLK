"""One-time (or rarely re-run) setup script. YouTube's category taxonomy
barely changes, so this doesn't belong in the recurring daily poll — run it
once, and again only if you suspect the taxonomy changed.
"""
from youtube_client import get_video_categories
from storage import write_rows


def main():
    category_names = get_video_categories()
    rows = [{"category_id": cid, "category_name": name} for cid, name in category_names.items()]
    write_rows(rows, "video_categories")
    print(f"Wrote {len(rows)} categories to the video_categories table")


if __name__ == "__main__":
    main()
