-- Convert video_snapshots to daily range partitions on captured_at.
--
-- WHY
-- Snapshots are the only table that grows without bound: every tracked video
-- gets four rows a day for as long as it stays inside the tracking window, and
-- the Supabase free tier stops at 500 MB. Reclaiming space from an ordinary
-- table means DELETE followed by VACUUM FULL, which rewrites the whole table
-- and needs free space roughly equal to its size -- precisely what you do not
-- have when the disk is nearly full. Dropping a partition is instant, returns
-- space immediately, takes no lock on live data and needs no spare capacity.
--
-- This is deliberately done while the table is small (56 MB, 385,930 rows).
-- The conversion copies every row once, so it needs room for a second copy;
-- doing it in three weeks at 300 MB would need 300 MB free and would not run.
--
-- DAILY, NOT WEEKLY
-- Partition granularity sets the smallest unit that can be archived and
-- dropped. With weekly partitions the database holds between 7 and 14 days of
-- snapshots depending on where the week boundary falls, and the peak exceeds
-- the free tier. Daily partitions with a nightly archive hold 7-8 days
-- steadily, and tolerate four or five consecutive failed archive runs before
-- space becomes a concern.
--
-- WHAT ELSE THIS CHANGES
--   * idx_video_snapshots_video_id is not recreated. The primary key is
--     (video_id, captured_at) and a B-tree's leftmost prefix already answers
--     every lookup on video_id alone, so that index was 4 MB of duplication.
--   * live_broadcast_content stored the literal string 'none' on 99.64% of
--     rows. It is normalised to NULL on copy, which the collector now does at
--     write time as well.
--
-- SAFETY
-- The whole migration runs in one transaction: if any step fails nothing is
-- applied and the collector keeps writing to the original table. A Parquet
-- copy of all 385,930 rows was taken before running this.

-- Bounds are written as explicit UTC instants, and the session is pinned to
-- UTC so current_date agrees with them. Without this the day a partition
-- covers depends on the timezone the migration happens to run under.
SET LOCAL timezone TO 'UTC';

-- ---------------------------------------------------------------- new parent
ALTER TABLE video_snapshots RENAME TO video_snapshots_legacy;
ALTER INDEX video_snapshots_pkey RENAME TO video_snapshots_legacy_pkey;
DROP INDEX IF EXISTS idx_video_snapshots_video_id;

CREATE TABLE video_snapshots (
    video_id                text        NOT NULL REFERENCES videos(video_id),
    captured_at             timestamptz NOT NULL,
    view_count              bigint,
    like_count              bigint,
    comment_count           bigint,
    live_broadcast_content  text,
    live_actual_start_time  timestamptz,
    live_actual_end_time    timestamptz,
    live_concurrent_viewers bigint,
    PRIMARY KEY (video_id, captured_at)
) PARTITION BY RANGE (captured_at);

COMMENT ON TABLE video_snapshots IS
    'Engagement observations, one row per video per collection run, partitioned '
    'daily on captured_at. Partitions older than the retention window are '
    'exported to Parquet and dropped by the nightly archive workflow; the '
    'exported files are the authoritative long-term dataset.';

-- ------------------------------------------------------- partition management
CREATE OR REPLACE FUNCTION ensure_snapshot_partitions(days_ahead int DEFAULT 14)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    d     date;
    part  text;
    made  int := 0;
BEGIN
    FOR d IN
        SELECT generate_series(current_date, current_date + days_ahead, interval '1 day')::date
    LOOP
        part := 'video_snapshots_' || to_char(d, 'YYYYMMDD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = part AND n.nspname = 'public'
        ) THEN
            EXECUTE format(
                'CREATE TABLE public.%I PARTITION OF public.video_snapshots '
                'FOR VALUES FROM (%L) TO (%L)',
                part,
                to_char(d,     'YYYY-MM-DD') || ' 00:00:00+00',
                to_char(d + 1, 'YYYY-MM-DD') || ' 00:00:00+00');
            made := made + 1;
        END IF;
    END LOOP;
    RETURN made;
END $$;

COMMENT ON FUNCTION ensure_snapshot_partitions(int) IS
    'Creates any missing daily partitions from today up to days_ahead. Called '
    'by the collector at the start of every run and by the nightly archive '
    'workflow, so a partition always exists before a write needs it.';

-- Historical partitions for the rows already collected, created before the
-- copy so that none of them lands in the default partition.
DO $$
DECLARE
    d    date;
    lo   date;
    part text;
BEGIN
    SELECT min(captured_at)::date INTO lo FROM video_snapshots_legacy;
    IF lo IS NULL THEN
        RETURN;
    END IF;
    FOR d IN SELECT generate_series(lo, current_date, interval '1 day')::date
    LOOP
        part := 'video_snapshots_' || to_char(d, 'YYYYMMDD');
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = part AND n.nspname = 'public'
        ) THEN
            EXECUTE format(
                'CREATE TABLE public.%I PARTITION OF public.video_snapshots '
                'FOR VALUES FROM (%L) TO (%L)',
                part,
                to_char(d,     'YYYY-MM-DD') || ' 00:00:00+00',
                to_char(d + 1, 'YYYY-MM-DD') || ' 00:00:00+00');
        END IF;
    END LOOP;
END $$;

SELECT ensure_snapshot_partitions(14);

-- Catches any row whose partition does not exist rather than failing the
-- insert and losing a collection run. It should always be empty; the archive
-- workflow reports it if it is not, because a non-empty default means
-- partition creation fell behind and blocks creating that day's partition.
CREATE TABLE video_snapshots_default PARTITION OF video_snapshots DEFAULT;

-- ------------------------------------------------------------------ copy over
INSERT INTO video_snapshots (
    video_id, captured_at, view_count, like_count, comment_count,
    live_broadcast_content, live_actual_start_time, live_actual_end_time,
    live_concurrent_viewers)
SELECT video_id, captured_at, view_count, like_count, comment_count,
       nullif(live_broadcast_content, 'none'),
       live_actual_start_time, live_actual_end_time, live_concurrent_viewers
FROM video_snapshots_legacy;

DROP TABLE video_snapshots_legacy;
