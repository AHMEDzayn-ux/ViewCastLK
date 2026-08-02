"""Detect post-publication edits to a video's title, description and tags.

videos holds what was seen when a video was first discovered and is never
updated, so an edit made afterwards is currently invisible. That matters:
creators retitle videos that underperform, which makes a late-captured title a
partial reaction to the outcome the model is supposed to predict.

Detection is free. videos.list is already called with the snippet part on every
snapshot pass, so these fields arrive in responses already paid for and are
discarded. This compares them against a fingerprint of the last observation and
records a row only when something actually differs.

Not covered: thumbnails. YouTube serves the current image from a stable URL, so
a replaced thumbnail is byte-identical in the response. It cannot be detected
from the API at any price.
"""
import hashlib

FIELDS = ("title", "description", "tags")


def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", "replace"))
        h.update(b"\x1f")                       # keep fields from running together
    return h.hexdigest()


def fingerprint(title, description, tags):
    return _sha(title, description, tags)


def from_api(item):
    """(title, description, tags, sha) as the API currently reports them."""
    snippet = item.get("snippet", {})
    title = snippet.get("title", "")
    description = snippet.get("description", "")
    tags = "|".join(snippet.get("tags", []))
    return title, description, tags, fingerprint(title, description, tags)


def detect(details, stored, captured_at, baseline=False):
    """Rows for videos whose metadata differs from the last observation.

    `stored` maps video_id -> sha of the last observation, or None where a
    video has never been fingerprinted. A None is treated as "record the
    fingerprint, report no change": enabling detection should not log an edit
    for every video in the warehouse.

    Returns (change_rows, sha_updates). Both are empty when nothing moved,
    which is the normal case.
    """
    changes, shas = [], {}
    for item in details:
        vid = item.get("id")
        if not vid:
            continue
        title, description, tags, sha = from_api(item)
        was = stored.get(vid, "__absent__")
        if was == "__absent__":
            continue                            # not tracked; caller decides
        shas[vid] = sha
        if was is None or was == sha:
            continue                            # first fingerprint, or unchanged
        changes.append({
            "video_id": vid,
            "observed_at": captured_at,
            "title": title,
            "description_len": len(description),
            "description_sha": _sha(description),
            "tags_sha": _sha(tags),
            "baseline": baseline,
        })
    return changes, shas


def summarise(changes, stored_titles):
    """One line for the run log, separating title edits from the rest.

    Descriptions are edited constantly -- creators add links and timestamps --
    so a combined count would drown the signal that actually matters."""
    if not changes:
        return "metadata: no changes"
    retitled = sum(1 for c in changes
                   if stored_titles.get(c["video_id"]) not in (None, c["title"]))
    other = len(changes) - retitled
    return (f"metadata: {len(changes)} changed "
            f"({retitled} retitled, {other} description/tags only)")
