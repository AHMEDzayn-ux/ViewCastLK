-- Stop polling channels that have stopped uploading.
--
-- Every rostered channel costs one playlistItems.list call per discovery run
-- whether or not it has published anything. A sweep of all 1,282 channels'
-- most recent upload found 679 with no upload in more than 60 days -- median
-- silence 414 days, 90th percentile 5.4 years -- so better than half the
-- roster was consuming roughly 2,700 quota units a day and returning nothing.
--
-- The rows are flagged, never deleted. They carry the declared-country
-- verification that admitted them to the roster in the first place, and any
-- videos they did contribute hold a foreign key back to them. Only the polling
-- stops; the history stays intact and a channel can be reactivated by setting
-- the flag back.
--
-- last_upload_at records what the sweep saw so the decision is auditable and
-- can be re-run later without guessing which channels were assessed when.

ALTER TABLE channels
    ADD COLUMN IF NOT EXISTS active              boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS last_upload_at      timestamptz,
    ADD COLUMN IF NOT EXISTS activity_checked_at timestamptz;

COMMENT ON COLUMN channels.active IS
    'False once a channel is judged to have stopped uploading. Excluded from '
    'discovery and from the statistics refresh; its rows and collected videos '
    'are retained.';
COMMENT ON COLUMN channels.last_upload_at IS
    'Publication time of the channel''s most recent upload as observed by the '
    'activity sweep. Null when never assessed.';
COMMENT ON COLUMN channels.activity_checked_at IS
    'When the activity sweep last assessed this channel.';

-- Discovery reads only the active rows, so index that subset rather than the
-- whole column.
CREATE INDEX IF NOT EXISTS channels_active_playlist_idx
    ON channels (uploads_playlist_id)
    WHERE active AND uploads_playlist_id IS NOT NULL;
