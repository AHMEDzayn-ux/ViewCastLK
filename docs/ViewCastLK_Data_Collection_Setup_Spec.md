# ViewCastLK — YouTube Data Collection Pipeline: Setup Specification

This document is a complete build spec for the data collection side of ViewCastLK (YouTube Data API integration, quota-aware collection design, scheduled polling service, and Postgres/Supabase data warehouse). Hand this directly to Claude Code to scaffold and implement.

---

## 1. Overview

Build an automated pipeline that:
1. Tracks ~225 Sri Lankan YouTube channels (15 standard categories × 15 channels each, split 5 mega / 5 mid / 5 small subscriber tier per category)
2. **Selects a fixed set of videos to monitor once, at the start of the project** — each channel's most recent existing video(s), capped at 7–8 per channel. There is no ongoing daily discovery of new uploads; the monitored video list does not change once selected.
3. Polls the same fixed video IDs **twice per day**, for the duration of the collection window, updating each video's views/likes/comments to build a day-by-day time-series
4. Stores everything in an existing Supabase (Postgres) project
5. Runs unattended via GitHub Actions on a twice-daily schedule, with quota-aware guardrails
6. Keeps polling each video until it reaches ~day 30–35 since its own publish date, then marks it complete — since the list is fixed, this is naturally self-limiting rather than requiring a hard calendar cutoff

Language: **Python 3.11+** (chosen for the official `google-api-python-client`, native GitHub Actions support, and consistency with the team's later pandas/XGBoost work).

### Critical design rule — read before implementing §6.3
**A video's "day N" is always counted from its own `published_at` date, never from when tracking started.** This is non-negotiable: the whole point of the model is forecasting from pre-publish metadata to day 7/14/21/30 *after that video goes live*. If age were instead counted from tracking-start, two videos both labeled "day 5" could mean completely different things — one freshly published, one already weeks old — and the model would be learning noise, not a usable pattern. Full reasoning is in §6.3 and §6.1.

---

## 2. Prerequisites & Credentials

### 2.1 Google Cloud + YouTube Data API v3 (not yet set up — full steps below)
1. Go to https://console.cloud.google.com and create a new project (e.g. `viewcastlk`).
2. In "APIs & Services" → "Library", search for and enable **YouTube Data API v3**.
3. In "APIs & Services" → "Credentials", click "Create Credentials" → "API key".
4. Restrict the key: under "API restrictions", select "Restrict key" and choose only **YouTube Data API v3**. (Skip IP restriction — GitHub Actions runners don't have static IPs.)
5. Copy the key value. It will be stored as a GitHub Actions secret (see §8).
6. Default daily quota is 10,000 units. This project's ongoing usage is very light (see §7 quota estimate) — a one-time ~7,500-unit spend for initial channel discovery via `search.list` is the only heavy day; a quota increase is not expected to be necessary.

### 2.2 Supabase (already created)
1. In the Supabase dashboard: **Project Settings → Database → Connection string**.
2. Select the **Transaction** mode pooler connection string (port `6543`), not the direct connection (port `5432`) — this matters because GitHub Actions jobs are short-lived, and the transaction pooler avoids connection-exhaustion issues.
3. Store this full connection string as a GitHub Actions secret (see §8).
4. Run `sql/schema.sql` (§4) once against this database — via the Supabase SQL Editor or `psql` — before the first pipeline run.

---

## 3. Repository Structure

```
/scripts
  seed_channels.py        # one-time / occasional: ingest curated channel list
  select_videos.py        # one-time: pick the fixed video list to monitor
  poll_stats.py           # every scheduled run: update stats for the fixed video list
  quota_guard.py          # shared: quota ledger + backoff wrapper for all API calls
  db.py                   # shared: DB connection + upsert helpers
  socialblade_import.py   # manual: import a Social Blade CSV export for early backfill
/sql
  schema.sql
/.github/workflows
  collect.yml
seed_channels.csv          # input: channel list (see §5)
requirements.txt
.env.example
README.md
```

---

## 4. Database Schema

```sql
CREATE TABLE channels (
    channel_id          TEXT PRIMARY KEY,
    title               TEXT,
    category            TEXT,               -- one of the 15 standard YouTube categories
    size_tier           TEXT,               -- 'mega' | 'mid' | 'small', auto-computed at ingest (see §5)
    country             TEXT,
    subscriber_count     BIGINT,
    channel_created_at   TIMESTAMP,
    uploads_playlist_id  TEXT,
    last_checked_at      TIMESTAMP
);

CREATE TABLE videos (
    video_id             TEXT PRIMARY KEY,
    channel_id            TEXT REFERENCES channels(channel_id),
    title                 TEXT,
    category_id           INT,
    duration_seconds       INT,
    made_for_kids          BOOLEAN,
    published_at           TIMESTAMP,       -- source of truth for all "day N" calculations
    selected_at             TIMESTAMP,       -- when this video was added to the fixed monitoring list
    age_at_selection_days   INT,             -- (selected_at - published_at), computed once at selection
    forecast_usable         BOOLEAN DEFAULT TRUE,  -- FALSE if age_at_selection_days > 30 (see §6.2)
    tracking_status         TEXT DEFAULT 'active'  -- active | completed
);

CREATE TABLE video_daily_stats (
    video_id            TEXT REFERENCES videos(video_id),
    days_since_publish   INT,           -- (stat_date - published_at), NOT a tracking-loop counter
    stat_date            DATE,
    views                BIGINT,
    likes                BIGINT,
    comments             BIGINT,
    source                TEXT DEFAULT 'youtube_api',  -- 'youtube_api' | 'social_blade'
    collected_at          TIMESTAMP DEFAULT now(),
    PRIMARY KEY (video_id, days_since_publish)
);

CREATE TABLE api_quota_usage (
    usage_date   DATE,
    endpoint     TEXT,
    calls_made   INT DEFAULT 0,
    units_used   INT DEFAULT 0,
    PRIMARY KEY (usage_date, endpoint)
);

CREATE TABLE collection_log (
    run_id         SERIAL PRIMARY KEY,
    run_at         TIMESTAMP DEFAULT now(),
    job_name       TEXT,
    status         TEXT,          -- success | partial | failed
    videos_touched INT,
    units_used     INT,
    error_message  TEXT
);

CREATE INDEX idx_videos_channel_id ON videos(channel_id);
CREATE INDEX idx_videos_published_at ON videos(published_at);
```

All timestamps stored in UTC. Convert to Sri Lanka time (UTC+5:30) only at the analysis/feature-engineering layer, per the project's stated cleaning step — do not convert at ingest.

---

## 5. Channel Seed List

**Status: not yet finalized by the team.** The target is 225 channels — 15 standard YouTube categories × 15 channels per category, aiming for a 5 mega / 5 mid / 5 small subscriber-tier split within each category. The team is sourcing candidate channels directly from Social Blade's country + category filters (general web search undercounts niche-category channels).

`seed_channels.csv` input format (Claude Code should build `seed_channels.py` around this, and treat the CSV as something the user will expand over time, not a fixed list):

```csv
channel_handle_or_id,category
@examplehandle,Pets & Animals
UCxxxxxxxxxxxxxxxxxxxxxx,Music
```

Note: **do not require the user to manually pre-assign the mega/mid/small tier.** `seed_channels.py` should:
1. Batch-call `channels.list` (up to 50 IDs/call) to resolve each handle/ID to its channel data, subscriber count, and uploads playlist ID.
2. Auto-classify `size_tier` **per category**, not with one fixed global subscriber threshold — e.g. split each category's channels into top/middle/bottom third by subscriber count. "Mega" in Entertainment (1M+) and "mega" in Pets & Animals (hundreds of thousands) are different scales; a single global threshold would leave sparse-looking categories with zero mega-tier channels even when a locally-dominant channel exists.
3. Upsert into `channels`.

This script should be safely re-runnable as the user adds more rows to `seed_channels.csv` over time (upsert on `channel_id`, don't fail on already-existing channels).

---

## 6. Video Selection — one-time, fixed list

### 6.1 `select_videos.py` (run once, not scheduled)
- For each channel in `channels`, call `playlistItems.list` on its `uploads_playlist_id` (already sorted newest-first) to get its most recent uploads.
- Take up to 7–8 videos per channel (fewer if the channel doesn't have that many, or hasn't published that many recently — small/infrequent channels may contribute only 1–2 videos, or occasionally none).
- For each selected video, batch-call `videos.list` (up to 50 IDs/call) to pull `title`, `category_id`, `duration_seconds`, `made_for_kids`, and — critically — `published_at`.
- Compute `age_at_selection_days = (now - published_at).days` and store it.
- Set `forecast_usable = FALSE` if `age_at_selection_days > 30` (see §6.2 for why), otherwise `TRUE`.
- Set `selected_at = now()`, `tracking_status = 'active'`.
- Upsert into `videos`. **This script is not part of the recurring GitHub Actions schedule** — run it manually once at project start (or re-run deliberately if the team wants to expand the video list later, understanding that re-running adds more videos rather than replacing the existing fixed set).

### 6.2 Why `forecast_usable` matters
Since videos are selected as "whatever a channel's most recent upload currently is" rather than "brand new videos published today," some selected videos will already be old at selection time:
- If a video is, say, 5 days old at selection and gets polled for the ~30-day window, you'll get valid `days_since_publish` values from 5 through ~35 — day-7/14/21/30 labels are all still obtainable.
- If a video is **already 35+ days old at selection**, all four forecast horizons (7/14/21/30) are already in the past before tracking even begins — that video is dead weight for the forecasting task specifically, though its stats are still useful for the channel's historical-performance feature.
- `forecast_usable = FALSE` flags this at ingest so the modeling team doesn't have to rediscover it later by re-deriving ages from `published_at` themselves.

### 6.3 `poll_stats.py` (runs on every scheduled cycle, twice daily)
- For every video where `tracking_status = 'active'`, batch-call `videos.list` (up to 50 IDs/call) to get current `views`, `likes`, `comments`.
- Compute `days_since_publish = (today - published_at).days` **fresh, from the stored `published_at`** — never from a run counter or from `selected_at`. This is what keeps every video's trajectory correctly aligned to its own actual life, regardless of when it happened to be selected.
- Upsert into `video_daily_stats` with `INSERT ... ON CONFLICT (video_id, days_since_publish) DO UPDATE` — never a plain insert, since twice-daily runs on the same calendar day must not create duplicate/conflicting rows, and any rerun must not fail.
- Once a video's `days_since_publish` exceeds ~35 (a small buffer past day 30), set `tracking_status = 'completed'` and stop polling it. Because the list is fixed, once every video has reached this point the job naturally has nothing left to do — no hard calendar cutoff is needed; the pipeline can simply keep running cheaply until then.

### 6.4 `socialblade_import.py` (manual, not scheduled)
- Social Blade blocks automated fetching (bot detection), so this cannot be part of the automated pipeline.
- Provide a script that reads a manually-exported CSV (a team member pulls the data by hand from Social Blade) and upserts rows into `video_daily_stats` with `source = 'social_blade'`, for early-history backfill on videos that were already a few days old at selection.

### 6.5 `quota_guard.py` (shared by all scripts)
- Wraps every YouTube Data API call.
- Before each call: read today's row from `api_quota_usage` (create if missing), check remaining budget.
- Cheap batched calls (`channels.list`, `playlistItems.list`, `videos.list`) always proceed — ongoing usage is now very light (see §7).
- `search.list` calls (100 units) only proceed if remaining daily quota exceeds a safety threshold (e.g. 2,000 units) — used only for one-time channel discovery, never as a recurring call.
- Wrap all calls in retry-with-backoff (e.g. 3–5 attempts, exponential delay) on HTTP 403 `quotaExceeded` and transient 5xx errors.

### 6.6 Logging (all scripts)
- Every run writes one row to `collection_log` with status, videos touched, quota units used, and any error message — wrapped in a top-level try/except so a crash still produces a log row instead of failing silently.
- Optional (not required for v1): a Discord/Slack webhook `curl` step at the end of the GitHub Actions workflow. Leave a placeholder env var (`ALERT_WEBHOOK_URL`) for this; wire it up only if the team wants it.

---

## 7. GitHub Actions Workflow (`.github/workflows/collect.yml`)

**Twice-daily scheduled workflow** running only `poll_stats.py` — there is no recurring discovery step, since the video list is fixed after `select_videos.py` runs once at project start.

- Cron: e.g. `0 6,18 * * *` (adjust the two times to whatever spacing suits the team; consistent ~12-hour spacing is enough since there's no early-hour-velocity capture requirement anymore)
- Steps: checkout → setup-python (3.11) → `pip install -r requirements.txt` → run `poll_stats.py`
- No repo commits needed — all state lives in Supabase

**Quota estimate with the fixed-list design** (much lighter than a continuous-discovery design):
- ~1,575–1,800 videos total (225 channels × 7–8) ÷ 50 per batched `videos.list` call ≈ 32–36 calls per run
- × 2 runs/day ≈ **64–72 units/day ongoing** — well under 1% of the 10,000-unit daily budget
- One-time costs (not daily): `search.list` channel discovery (~7,500 units, already largely done) and `select_videos.py`'s one-time `playlistItems.list` + `videos.list` calls (~225 + ~35 units)

**Known operational risk to flag in the README:** GitHub disables scheduled workflows automatically after 60 days with no repository activity. Make sure someone commits/pushes to the repo periodically during the collection window, or note this explicitly as a risk to monitor.

---

## 8. GitHub Actions Secrets

| Secret name | Value |
|---|---|
| `YOUTUBE_API_KEY` | API key from §2.1 |
| `SUPABASE_DB_URL` | Transaction-mode pooler connection string from §2.2 (port 6543) |
| `ALERT_WEBHOOK_URL` | (optional) Discord/Slack webhook, only if alerting is wired up |

Never commit these to the repo — GitHub Actions secrets only.

---

## 9. Reliability Requirements (apply throughout)

- **Idempotent writes everywhere**: every insert into `video_daily_stats` must be an upsert (`ON CONFLICT ... DO UPDATE`), since twice-daily runs on the same day, or any rerun, must not duplicate or crash.
- **Quota guard on every call**: no script should call the YouTube API directly without going through `quota_guard.py`.
- **Retry with backoff** on quota errors and transient 5xx responses; fail gracefully and log rather than crashing the whole run if one video fails.
- **`days_since_publish` is always derived from `published_at`, never from a tracking-start counter or run sequence.** This is the single most important invariant in the whole pipeline — see §1 and §6.3.
- **The fixed video list does not grow via the scheduled workflow.** If the team wants to add more videos later, that's a deliberate manual re-run of `select_videos.py`, not something the twice-daily job does automatically.

---

## 10. Open Items for Claude Code to Flag Back to the User

- The final 225-channel `seed_channels.csv` is not yet complete — build `seed_channels.py` to be safely re-run as more rows are added over time, not around a fixed list.
- Confirm whether Discord/Slack alerting should be wired up now or deferred.
- The Google Cloud API key must be created by the user's own Google account — Claude Code can only guide the steps in §2.1, not perform them.
- `select_videos.py` should be run manually once the channel list is finalized — it is a deliberate one-time action, not part of the recurring schedule. Confirm with the team before running it, since it locks in the fixed video set for the rest of the project.
