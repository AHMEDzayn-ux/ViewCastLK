-- category_id=29 (Nonprofits & Activism) is one of YouTube's 15 standard
-- categories but doesn't appear when videoCategories.list is queried with
-- regionCode=LK, so our video_categories lookup table is incomplete for it
-- (and possibly other region-omitted categories). The foreign key was too
-- strict: it rejected otherwise-valid video rows just because the category
-- name lookup happens to be incomplete. category_id stays as a plain
-- reference value; category_name is already denormalized onto videos
-- directly (from flatten_video_identity), so this doesn't lose any info.
ALTER TABLE videos DROP CONSTRAINT IF EXISTS videos_category_id_fkey;
