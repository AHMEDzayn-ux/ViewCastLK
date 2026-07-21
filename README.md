# ViewCastLK — YouTube Data Collection

Collects video and channel data for Sri Lankan YouTube channels via the YouTube
Data API v3, and builds a day-by-day view/like/comment time series per video
(the data the forecasting model trains on).

## Setup

1. Create a Google Cloud project, enable **YouTube Data API v3**, create an API
   key restricted to that API (Console → APIs & Services → Credentials).
2. Create a Supabase project (Mumbai/`ap-south-1` region — closest to Sri
   Lanka for the team's own direct access and the eventual dashboard; the
   scheduled poll itself runs from GitHub Actions' US-based runners either
   way, so region doesn't change that part's latency), and grab the
   **Transaction-mode pooler** connection string (Project Settings → Database
   → Connection string, port `6543` — not the direct connection on port
   `5432`; GitHub Actions jobs are short-lived and the pooler avoids
   connection-exhaustion issues).
3. Copy `.env.example` to `.env` and fill in `YOUTUBE_API_KEY` and
   `SUPABASE_DB_URL` (no quotes needed).
4. `pip install -r requirements.txt`
5. Schema is managed via the **Supabase CLI** (`npx supabase`, no global
   install needed — requires Node.js and Docker Desktop locally):
   - Migrations live in `supabase/migrations/*.sql`, applied in order.
   - To push pending migrations: `npx supabase db push --db-url "<your SUPABASE_DB_URL, without the ?pgbouncer=true suffix>"`
     (or `npx supabase link --project-ref <ref>` once, then plain
     `npx supabase db push`, if you'd rather authenticate interactively).
   - To make a schema change: `npx supabase migration new <description>`,
     edit the generated file, then push again. Never edit an already-pushed
     migration file — add a new one instead.
6. If migrating existing CSV-collected data: `python scripts/migrate_csv_to_supabase.py`
   (safe to re-run, upserts on each table's primary key).
7. Add `YOUTUBE_API_KEY` and `SUPABASE_DB_URL` as **GitHub Actions repo
   secrets** (Settings → Secrets and variables → Actions) so the scheduled
   workflow can use them — never commit real values to `.env`.

## File structure

```
youtube_client.py          # All YouTube API calls + flatten functions (raw API
                            # response -> plain flat dict). No DB access here.
storage.py                  # ALL persistence goes through here — Supabase
                            # (Postgres) backend via psycopg2.
supabase/migrations/*.sql   # Schema migrations, managed via the Supabase CLI
                            # (npx supabase db push / migration new).
channel_handles.txt         # The tracked roster — one @handle or channel ID per
                            # line, plain text (not in the database). Edit this
                            # to add/remove channels; no code changes needed.
channel_roster.py           # Loads channel_handles.txt.
fetch_categories.py         # One-time script: YouTube category id -> name.
discover_more_channels.py   # Occasional script: broad search.list sweep across
                            # category+keyword queries to grow the roster.
run_daily_poll.py           # THE recurring job, run on a schedule by
                            # .github/workflows/collect.yml.
migrate_csv_to_supabase.py  # One-time: load pre-Supabase CSV data into the DB.
.github/workflows/collect.yml  # Scheduled GitHub Actions workflow (6-hourly).
```

## Data model: identity vs. snapshot tables

Every entity (channel, video) is split into two tables:

- **Identity** (`channels`, `videos`) — fields that don't change day to day:
  title, description, category, duration. Written once per channel/video
  when first seen (upserted on `channel_id`/`video_id`, so a rerun never
  duplicates identity data).
- **Snapshot** (`channel_snapshots`, `video_snapshots`) — fields that change
  every poll: subscriber count, view/like/comment count. One new row per
  entity **every run**, tagged with a `captured_at` timestamp (composite
  primary key `(id, captured_at)`). This is the actual time-series data the
  model needs — `days_since_publish` for any row is just
  `captured_at - published_at`, always derived from the video's own publish
  date, never from when tracking started.

## GitHub Actions

`.github/workflows/collect.yml` runs `run_daily_poll.py` every 6 hours
(`cron: "0 */6 * * *"`), plus a manual `workflow_dispatch` trigger for
testing or catching up after an outage. Reads `YOUTUBE_API_KEY` and
`SUPABASE_DB_URL` from repo secrets — set those under Settings → Secrets and
variables → Actions before the schedule can run successfully.

**Known operational gotcha**: GitHub disables scheduled workflows after 60
days of no repository activity — make sure someone pushes periodically
during the collection window, or add a step that self-pings.

**Quota note**: at ~2,800-3,800 units per full run (channel refresh +
discovery + snapshot across the full roster), 4 runs/day sits right at or
over the 10,000/day YouTube API quota ceiling, and gets tighter as the
roster or per-channel video volume grows. Watch actual usage in Google
Cloud Console once this is running continuously.

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
