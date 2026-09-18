-- Player shape of each video, so Shorts can be told apart from long-form.
--
-- WHY
-- The Data API has no Shorts flag. Until now the training table called a
-- video a Short if it ran 60 seconds or less, but YouTube has accepted Shorts
-- up to 3 minutes since October 2024, and a short horizontal clip is not a
-- Short at all. That left 61 s to 3 min as an unlabelled mix, and it is the
-- worst-performing duration band in the data. For channels under 10K
-- subscribers, Shorts get 6 to 9 times the day-7 views of long-form, so the
-- format has to be known exactly.
--
-- videos.list with the player part and a maxHeight returns embedWidth and
-- embedHeight in the video's own aspect ratio (4608x8192 for 9:16 vertical,
-- 14564x8192 for 16:9). A Short is a video that is vertical or square and at
-- most 3 minutes long. Checked against youtube.com/shorts/<id>, which serves
-- Shorts and redirects everything else, on a sample that included a
-- 50-second horizontal video (not a Short) and a 52-minute vertical one (not
-- a Short).
--
-- COST
-- None for new videos: the player part rides on the videos.list call the poll
-- already makes, and videos.list costs 1 unit per 50 ids whatever parts are
-- requested. Videos seen before this table existed need a one-off backfill
-- (scripts/backfill_video_shapes.py), 1 unit per 50 videos.
--
-- WHY A SEPARATE TABLE
-- Adding these as columns on videos would mean an UPDATE of every existing
-- row for the backfill, and Postgres rewrites the whole row on UPDATE. On a
-- database at its size limit that is the dead-row bloat fixed on 30 August.
-- This table only ever receives inserts.

CREATE TABLE IF NOT EXISTS video_shapes (
    video_id     text        PRIMARY KEY,
    -- Null when the video was no longer available (deleted or private) when
    -- checked, so the backfill does not pay to ask about it again.
    embed_width  integer,
    embed_height integer,
    captured_at  timestamptz NOT NULL
);

COMMENT ON TABLE video_shapes IS
    'Player dimensions per video from videos.list (part=player, maxHeight). '
    'Short = embed_height >= embed_width and duration <= 180 s.';
