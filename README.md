# ViewCastLK — YouTube Data Collection

Collects video and channel data for Sri Lankan YouTube channels via the YouTube
Data API v3, and builds a day-by-day view/like/comment time series per video
(the data the forecasting model trains on).

## Setup

1. Create a Google Cloud project, enable **YouTube Data API v3**, create an API
   key restricted to that API (Console → APIs & Services → Credentials).
2. Copy `.env.example` to `.env` and paste the key in as `YOUTUBE_API_KEY=...`
   (no quotes needed).
3. `pip install -r requirements.txt`

## File structure

```
youtube_client.py         # All YouTube API calls + flatten functions (raw API
                           # response -> plain flat dict). No file I/O.
storage.py                 # ALL persistence goes through here. The one file to
                           # rewrite for the Supabase migration (see below).
channel_handles.txt        # The tracked roster — one @handle or channel ID per
                           # line. Edit this to add/remove channels; no code
                           # changes needed.
channel_roster.py          # Loads channel_handles.txt.
fetch_categories.py        # One-time script: YouTube category id -> name.
discover_more_channels.py  # Occasional script: broad search.list sweep across
                           # category+keyword queries to grow the roster.
run_daily_poll.py          # THE recurring job. Wrap this in a scheduled
                           # GitHub Actions workflow.
```

## Data model: identity vs. snapshot tables

Every entity (channel, video) is split into two tables:

- **Identity** (`channels.csv`, `videos.csv`) — fields that don't change day
  to day: title, description, category, duration. Written once per
  channel/video when first seen.
- **Snapshot** (`channel_snapshots.csv`, `video_snapshots.csv`) — fields that
  change every poll: subscriber count, view/like/comment count. One new row
  per entity **every run**, tagged with a `captured_at` timestamp. This is
  the actual time-series data the model needs — `days_since_publish` for any
  row is just `captured_at - published_at`, always derived from the video's
  own publish date, never from when tracking started.

Identity tables intentionally accumulate some duplicate rows over time (see
"Known quirks" below) — that's fine for CSV, since it's meant to be
**upserted** (dedupe on `video_id`/`channel_id`) once in a real database.
Snapshot tables should be **inserted** — every row is a legitimate distinct
observation.

## Supabase / GitHub Actions migration (for whoever picks this up)

Only **`storage.py`** needs to change. Its three functions are already the
exact shape a Supabase migration needs:

| Current (CSV) | Supabase equivalent |
|---|---|
| `append_rows(rows, path)` on an identity table | `supabase.table(name).upsert(rows).execute()` |
| `append_rows(rows, path)` on a snapshot table | `supabase.table(name).insert(rows).execute()` |
| `write_rows(rows, path)` (categories, one-time) | `supabase.table(name).upsert(rows).execute()` |
| `load_active_video_ids(path, since_date)` | `supabase.table("videos").select("video_id").gte("published_at", since_date).execute()` |

Nothing in `youtube_client.py`, `run_daily_poll.py`, `channel_roster.py`, or
`fetch_categories.py` needs to know whether it's writing to CSV or Supabase —
they only ever call `storage.py`'s functions with flat dicts.

To wrap `run_daily_poll.py` in GitHub Actions: a scheduled workflow
(`schedule: cron:`) that checks out the repo, installs `requirements.txt`,
and runs `python run_daily_poll.py`, with `YOUTUBE_API_KEY` (and eventually
Supabase credentials) as repo secrets, not committed to `.env`.

**Known operational gotcha**: GitHub disables scheduled workflows after 60
days of no repository activity — make sure someone pushes periodically
during the collection window, or add a step that self-pings.

## Known quirks (learned the hard way — don't rediscover these)

- **`search.list` never returns more than ~500 results for a single query**,
  regardless of `pageInfo.totalResults` claiming far more, and costs 100
  units/page. Favor many distinct queries (breadth) over deep pagination
  into one (depth) — see `discover_more_channels.py`.
- **`channels.list().snippet.country`** is the same self-declared field shown
  on a channel's About page — reliable when set, but far from universal.
  Established individual creators mostly have it; news-org sub-channels and
  auto-generated "-Topic" channels often don't. Treat as high-precision,
  incomplete-recall.
- **`snippet.defaultAudioLanguage`** is not a trustworthy language signal —
  observed a Sinhala-titled, Sinhala-description video tagged `en`. Creators
  don't fill this in accurately. For real language detection, check Unicode
  script ranges on the title/description instead (Sinhala U+0D80–U+0DFF,
  Tamil U+0B80–U+0BFF).
- **Live videos** have no `contentDetails.duration` while live (shows
  `P0D`), and `liveStreamingDetails.concurrentViewers` only exists *while*
  the stream is actually live at query time — it disappears once the
  broadcast ends. Check `snippet.liveBroadcastContent` (`live`/`upcoming`/
  `none`) before treating `duration` as meaningful.
- **`playlistItems.list` on a channel's uploads playlist is newest-first**
  (confirmed empirically, not something Google documents explicitly) — this
  is what makes `get_channel_videos_since()` cheap: it stops paging the
  moment it hits a video older than the cutoff, instead of walking a
  channel's entire history on every single poll.
- **Windows console + Sinhala/Tamil text**: `sys.stdout.reconfigure(encoding="utf-8")`
  is set in `youtube_client.py` — without it, printing non-Latin titles
  crashes with `UnicodeEncodeError` on Windows' default `cp1252` console.
  Ad-hoc one-off scripts that don't import `youtube_client` need
  `PYTHONIOENCODING=utf-8` set manually.
- **CSV files opened in Excel lock the file** — a pipeline run will fail with
  `PermissionError` if you're viewing an output CSV in Excel at the same
  time. Close it first.
