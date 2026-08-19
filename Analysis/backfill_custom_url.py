"""Populate channels.custom_url for the existing roster.

Batched channel refresh needs a link from a channel_handles.txt entry to a
stored channel_id, and until that link exists every handle looks new and takes
the one-unit-per-channel path -- the exact cost the batching removes.

snippet.customUrl carries it and arrives in the same response already paid
for, so filling it in costs one call per fifty channels: about 26 units for the
whole roster, against the 1,282 a single unbatched refresh costs.

Writes go through the ordinary identity upsert, which only touches the columns
the flattener produces -- active, last_upload_at and activity_checked_at are
left exactly as the activity sweep set them.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

from storage import append_rows, connect  # noqa: E402
from youtube_client import (flatten_channel_identity,  # noqa: E402
                            get_channels_by_ids)

BATCH = 50


def main():
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT channel_id FROM channels ORDER BY channel_id")
        ids = [r[0] for r in cur.fetchall()]
    conn.close()
    print(f"roster: {len(ids):,} channels -> {-(-len(ids)//BATCH)} calls")

    done = missing = 0
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        got = get_channels_by_ids(batch)
        missing += len(batch) - len(got)
        if got:
            append_rows([flatten_channel_identity(c) for c in got], "channels")
            done += len(got)
        print(f"  {done}/{len(ids)}")

    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT active,
                              count(*) AS channels,
                              count(custom_url) AS with_handle
                         FROM channels GROUP BY active ORDER BY active DESC""")
        print(f"\n{'active':>8}{'channels':>10}{'with handle':>13}")
        for a, n, w in cur.fetchall():
            print(f"{str(a):>8}{n:>10,}{w:>13,}")
    conn.close()
    if missing:
        print(f"\n{missing} id(s) not returned by the API — deleted or suspended "
              f"channels; they keep a null custom_url and will fall back to the "
              f"single-lookup path.")


if __name__ == "__main__":
    main()
