-- Persist the player dimensions returned by videos.list(part=player).
-- YouTube does not expose a Shorts flag; vertical/square videos up to three
-- minutes are treated as Shorts, with duration <= 60 seconds used only when
-- player dimensions are unavailable.
ALTER TABLE videos
    ADD COLUMN IF NOT EXISTS player_width integer,
    ADD COLUMN IF NOT EXISTS player_height integer;

COMMENT ON COLUMN videos.player_width IS
    'YouTube player.embedWidth captured with the first complete video response.';
COMMENT ON COLUMN videos.player_height IS
    'YouTube player.embedHeight captured with the first complete video response.';
