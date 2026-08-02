-- Record each channel's @handle so the roster file can be matched to stored
-- rows without an API call.
--
-- channel_handles.txt holds handles; channels.list can only look a handle up
-- one at a time (forHandle takes a single value), so every full run spent one
-- quota unit per channel -- 1,282 units twice a day. Looking channels up by id
-- accepts fifty per call for the same unit, but nothing connected a handle in
-- the file to a channel_id in the table.
--
-- snippet.customUrl is that link, and it already arrives free in the response
-- we were paying for anyway. Once stored, a full run resolves the whole known
-- roster in 13 calls instead of 1,282, and only genuinely new handles still
-- need the one-at-a-time path.
--
-- Stored normalised (lower-case, no leading @) so matching is a plain equality
-- test rather than a per-row transformation.

ALTER TABLE channels
    ADD COLUMN IF NOT EXISTS custom_url text;

COMMENT ON COLUMN channels.custom_url IS
    'The channel''s @handle, normalised to lower-case without the leading @. '
    'Links a channel_handles.txt entry to this row so refreshes can be batched '
    'by id. Null for channels that have no custom handle.';

CREATE INDEX IF NOT EXISTS channels_custom_url_idx
    ON channels (custom_url)
    WHERE custom_url IS NOT NULL;
