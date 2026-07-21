-- ViewCastLK data warehouse — initial schema.
-- Mirrors the identity/snapshot split in youtube_client.py's flatten_* functions:
-- identity tables hold fields that don't change (upsert on their id), snapshot
-- tables hold fields that change every poll (plain insert, one row per poll).

CREATE TABLE IF NOT EXISTS channels (
    channel_id            TEXT PRIMARY KEY,
    title                 TEXT NOT NULL,
    description           TEXT,
    country               TEXT,
    channel_published_at  TIMESTAMPTZ,
    uploads_playlist_id   TEXT,
    topic_categories      TEXT
);

CREATE TABLE IF NOT EXISTS channel_snapshots (
    channel_id              TEXT NOT NULL REFERENCES channels(channel_id),
    captured_at             TIMESTAMPTZ NOT NULL,
    subscriber_count        BIGINT,
    hidden_subscriber_count BOOLEAN,
    view_count              BIGINT,
    video_count             BIGINT,
    PRIMARY KEY (channel_id, captured_at)
);

CREATE TABLE IF NOT EXISTS video_categories (
    category_id    TEXT PRIMARY KEY,
    category_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id                TEXT PRIMARY KEY,
    channel_id              TEXT NOT NULL REFERENCES channels(channel_id),
    title                   TEXT NOT NULL,
    description             TEXT,
    tags                    TEXT,
    published_at            TIMESTAMPTZ NOT NULL,
    category_id             TEXT REFERENCES video_categories(category_id),
    category_name           TEXT,
    duration                TEXT,
    definition              TEXT,
    caption                 TEXT,
    default_audio_language  TEXT,
    default_language        TEXT,
    thumbnail_url           TEXT,
    made_for_kids           BOOLEAN
);

CREATE TABLE IF NOT EXISTS video_snapshots (
    video_id                 TEXT NOT NULL REFERENCES videos(video_id),
    captured_at              TIMESTAMPTZ NOT NULL,
    view_count               BIGINT,
    like_count               BIGINT,
    comment_count            BIGINT,
    live_broadcast_content   TEXT,
    live_actual_start_time   TIMESTAMPTZ,
    live_actual_end_time     TIMESTAMPTZ,
    live_concurrent_viewers  BIGINT,
    PRIMARY KEY (video_id, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
CREATE INDEX IF NOT EXISTS idx_video_snapshots_video_id ON video_snapshots(video_id);
CREATE INDEX IF NOT EXISTS idx_channel_snapshots_channel_id ON channel_snapshots(channel_id);
