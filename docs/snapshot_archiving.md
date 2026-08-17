# Snapshot archiving

`video_snapshots` is the only table that grows without bound: every tracked
video gets four rows a day for as long as it stays inside the tracking window.
The Supabase free tier stops at 500 MB, so Postgres holds a rolling window and
the older data lives on Google Drive as Parquet.

**Nothing is thinned.** Every snapshot the collector takes is preserved at full
fidelity. The exported files are the authoritative long-term dataset — the
thing a future time-series or post-publication model would be trained from.
Postgres is the working set, not the archive.

## How it fits together

| Piece | What it holds | Where |
|---|---|---|
| `video_snapshots` partitions | everything since the last drop | Supabase |
| `video_horizon_labels` | the day 7/14/21/30 observation per video, permanently | Supabase |
| Parquet archives | every snapshot ever taken | `gdrive:ViewCastLK/archives/YYYY/MM` |
| Nightly `pg_dump` | the live database as of last night | `gdrive:ViewCastLK/backups/daily/YYYY/MM` |

Training reads `video_horizon_labels`, never the raw partitions, so the model
pipeline is unaffected by how aggressively partitions are dropped.

## Why daily partitions

Partition granularity sets the smallest unit that can be dropped. With weekly
partitions the database would hold between 7 and 14 days depending on where the
week boundary falls, and the peak exceeds the free tier. Daily partitions can be
retained to the exact day, which is what makes the retention floor a usable
control rather than a coarse one.

That control has had to tighten. Retention was 7 days, then 5, and from
16 August 3. The reason is that the steady state is not stable: the job frees
one partition a night while collection adds a slightly larger one, because the
roster and the tracking window keep growing. At five days the database reached
479 MB of the 500 MB tier — roughly two days from failing writes — and the
archive could not dig itself out, since only one partition was ever eligible.

Holding less costs nothing in data. A partition is on Drive before it can be
dropped, and the day 7/14/21/30 labels the model trains on live in
`video_horizon_labels`, which is never dropped. What it does cost is slack: at
three days the job tolerates fewer consecutive failures before a horizon could
pass unlabelled. It buys that back by keeping the database far enough below the
ceiling to have room to recover in.

Empty partitions — one for each day the collector produced nothing, as on
19 July — hold nothing to preserve, so they are dropped on age alone without
being exported.

Dropping a partition is instant and returns space immediately. The alternative —
`DELETE` then `VACUUM FULL` — rewrites the whole table and needs free space
roughly equal to its size, which is exactly what you do not have when the disk
is nearly full.

## Exporting and dropping are separate decisions

**Export runs every night, unconditionally.** Every completed day goes to Drive
as Parquet whether or not space is short, so a drop is never the first time a
partition has been copied anywhere. Partitions already on the remote are
skipped, which makes re-runs cheap and the job safe to retry.

**Dropping only happens under pressure** — when the database exceeds
`--drop-above-mb` (default 380). Below that nothing is removed: there is no
reason to give up queryable history while there is room, and the full time
series stays available to plain SQL. Once over the threshold the job drops the
oldest partitions, only ones verified present on the remote, only ones older
than the `--retain-days` floor, and only until the database is back under
`--drop-target-mb` (default 300).

On 2 August, at 113 MB of a 500 MB tier, the first live run exported fifteen
partitions (335,771 rows, 3.7 MB) and dropped nothing.

The order within a drop is not negotiable:

```
export -> verify rows -> upload -> verify remote size -> DROP
```

Labels are materialised **every night**, against everything currently in
Postgres — long before any partition becomes a drop candidate. The conflict
rule keeps whichever observation is closest to the mark, so re-running can only
improve a label, never degrade one.

## Schedule

`archive.yml` runs at **20:30 UTC**, an hour after the nightly backup and two
hours after the collector's 18:30 run. Running after the backup means that
night's dump still contains everything the archive is about to remove.

Manual runs accept `retain_days`, `drop_above_mb` and `dry_run` from the
Actions page.

The job runs through the session pooler on port 5432, not the transaction
pooler the collector uses. It holds one connection for the length of the run
doing bulk reads and DDL, which is what the session pooler is for; the
transaction pooler exists for many short-lived connections.

## Restoring

**For analysis — no restore needed.** DuckDB reads Parquet directly and treats a
folder as one table:

```sql
SELECT video_id, captured_at, view_count
FROM 'archives/**/*.parquet'
WHERE video_id = 'abc123'
ORDER BY captured_at;
```

This is the normal way to work with the full history, and why the archives are
Parquet rather than CSV — the schema travels with the file.

**For a full Postgres restore**, restore the nightly dump in the order
`roles.sql`, `schema.sql`, `data.sql`, then bulk-load the archive files into
partitions. You need both sources: the dump has the current window, the archives
have everything already dropped.

## Verifying

Every run that uploads anything writes `manifests/exported_YYYYMMDD.json`
recording each file's day, row count, byte size and SHA-256. That manifest is
the only durable record of the checksums — the Actions log is not one — and it
is what a downloaded copy is checked against.

`Analysis/verify_archives.py` does that check. Point it at a folder of
downloaded archives and it confirms three things: every file's SHA-256 matches
what was recorded at upload, every file holds the row count the job counted, and
a video's trajectory can be reconstructed across file boundaries. Where the
partitions still exist in Postgres it also compares row for row.

Verified on 2 August against all fifteen files: checksums and row counts matched
on every file, 335,771 rows cross-checked against Postgres with zero
disagreements, and a 49-observation trajectory reconstructed across nine files
with a monotonically increasing view count.

Worth re-running monthly, and necessarily before training on archived data —
by then the partitions will be gone from Postgres and the manifest will be the
only thing left to check against.

## The default partition should always be empty

`video_snapshots_default` catches writes whose day has no partition, so a
collection run can never fail for want of one. It is a safety net, not a
destination — a non-empty default means partition creation fell behind, and it
also **blocks that day's real partition from being created** until the rows are
moved out. The archive job reports it loudly.

`ensure_snapshot_partitions(days_ahead)` creates missing partitions and is
called both by the collector at the start of every run and by the archive job,
so two independent things have to fail before the default is used.

## Sizing

Measured on 2 August 2026: 385,930 rows occupied 55 MB in Postgres and 3.0 MB as
zstd Parquet — an 18× reduction. At the current roster the archive grows by
roughly 2–3 MB a day, so Drive's free 15 GB is not a constraint.

The nightly `pg_dump` backups compress just as well: about 10 MB each, not the
size of the live database. At that rate they accumulate roughly 3.6 GB a year,
and even once the database reaches its 380 MB drop threshold a dump would be
around 35 MB. Drive has room for years of them, so no retention policy is
needed for the project's lifetime.
