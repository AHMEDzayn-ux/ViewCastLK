-- Track post-publication edits to a video's title, description and tags.
--
-- WHY
-- videos is an identity table: title, description, tags and thumbnail_url are
-- written once when a video is first discovered and never updated. So what we
-- store is not "the metadata at publication" but "the metadata when we first
-- saw it" -- and 73% of the corpus was first seen more than 26 hours after
-- publication, median 15 days, because of the two backfills.
--
-- That matters beyond accuracy. Creators retitle videos that underperform, so
-- a title captured a fortnight late may already be a reaction to the video's
-- performance. In a model whose whole claim is pre-publication forecasting,
-- that is the target leaking into a feature.
--
-- It also breaks label attribution: if the title changed on day 3, the day-7
-- figure is cumulative views accrued under two different titles, and training
-- "title A -> day-7 views" credits title A with an outcome title B helped
-- cause. Per-horizon usability flags can express that, but only once there is
-- a measured change rate to justify them.
--
-- COST
-- None. videos.list is already called with the snippet part on every snapshot
-- pass, so title, description and tags arrive in responses we already pay for
-- and are currently discarded. Rows are written only when something changes.
--
-- WHAT IS NOT COVERED
-- Thumbnails. YouTube serves the current image from a stable URL
-- (i.ytimg.com/vi/<id>/hqdefault.jpg) with no version component, so a changed
-- thumbnail is byte-identical in the API response. Detecting it would mean
-- fetching and hashing 49,502 images per pass. It cannot be tracked from the
-- API and that is a documented limitation, not an oversight.

CREATE TABLE IF NOT EXISTS video_metadata_changes (
    video_id        text        NOT NULL REFERENCES videos(video_id),
    observed_at     timestamptz NOT NULL,
    title           text,
    description_len integer,
    description_sha text,
    tags_sha        text,
    -- True for rows written by the initial sweep, where the change happened at
    -- some unknown point between first observation and that sweep. Without
    -- this every pre-existing edit would appear to have happened at the same
    -- instant, which would quietly corrupt any timing analysis.
    baseline        boolean     NOT NULL DEFAULT false,
    PRIMARY KEY (video_id, observed_at)
);

COMMENT ON TABLE video_metadata_changes IS
    'One row each time a video''s title, description or tags are observed to '
    'differ from the previous observation. videos still holds the first-seen '
    'values; this is the history on top of them.';
COMMENT ON COLUMN video_metadata_changes.observed_at IS
    'When the change was DETECTED, not when it happened — the edit occurred at '
    'some point since the previous poll, so this carries the poll interval as '
    'uncertainty.';
COMMENT ON COLUMN video_metadata_changes.baseline IS
    'Row written by the initial sweep: the edit predates detection by an '
    'unknown amount.';

CREATE INDEX IF NOT EXISTS video_metadata_changes_observed_idx
    ON video_metadata_changes (observed_at);

-- Fingerprint of the LATEST observed metadata, not the first. Comparing
-- against the first-seen values instead would make a video that was retitled
-- once report a change on every subsequent poll, writing a duplicate row every
-- six hours forever.
ALTER TABLE videos ADD COLUMN IF NOT EXISTS metadata_sha text;

COMMENT ON COLUMN videos.metadata_sha IS
    'SHA-256 of the most recently observed title, description and tags. Null '
    'until first computed; a null is populated without recording a change, so '
    'that enabling detection does not log a phantom edit for every video.';
