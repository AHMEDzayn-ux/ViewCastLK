"""Behaviour of the metadata change detector.

The fourth case is the one that matters for storage. An unchanged video must
produce no sha update at all: every entry returned becomes an UPDATE against
videos, and Postgres rewrites the entire row even when the value is identical.
Returning all 94,000 hashes every run rewrote the table four times a day.

Run:  python scripts/test_metadata_changes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metadata_changes as mc

AT = "2026-08-30T12:00:00+00:00"


def item(vid, title="T", desc="D", tags=("a", "b")):
    return {"id": vid, "snippet": {"title": title, "description": desc,
                                   "tags": list(tags)}}


def sha_of(it):
    return mc.from_api(it)[3]


def run():
    failures = []

    def check(name, cond):
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            failures.append(name)

    # 1. unchanged -- no change row, and crucially no sha write
    it = item("v1")
    changes, shas = mc.detect([it], {"v1": sha_of(it)}, AT)
    check("unchanged: no change row", changes == [])
    check("unchanged: no sha update (the storage fix)", shas == {})

    # 2. never fingerprinted -- record the hash, report no edit
    it = item("v2")
    changes, shas = mc.detect([it], {"v2": None}, AT)
    check("first fingerprint: no change row", changes == [])
    check("first fingerprint: sha recorded", shas == {"v2": sha_of(it)})

    # 3. genuinely edited -- change row and a new hash
    it = item("v3", title="new title")
    changes, shas = mc.detect([it], {"v3": sha_of(item("v3"))}, AT)
    check("edited: one change row", len(changes) == 1)
    check("edited: change row carries the video", changes[0]["video_id"] == "v3")
    check("edited: sha updated", shas == {"v3": sha_of(it)})

    # 4. not tracked yet -- caller inserts it, detector stays out of the way
    changes, shas = mc.detect([item("v4")], {}, AT)
    check("untracked: ignored entirely", changes == [] and shas == {})

    # 5. a realistic mixed run: one edit among many unchanged videos
    stored, batch = {}, []
    for i in range(500):
        it = item(f"m{i}")
        stored[f"m{i}"] = sha_of(it)
        batch.append(it)
    batch[7] = item("m7", desc="edited description")
    changes, shas = mc.detect(batch, stored, AT)
    check("mixed run: exactly one change row", len(changes) == 1)
    check("mixed run: exactly one row written, not 500", len(shas) == 1)

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
