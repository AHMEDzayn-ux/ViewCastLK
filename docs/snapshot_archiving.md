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
| `video_snapshots` partitions | last ~7 days | Supabase |
| `video_horizon_labels` | the day 7/14/21/30 observation per video, permanently | Supabase |
| Parquet archives | every snapshot ever taken | `gdrive:ViewCastLK/archives/YYYY/MM` |
| Nightly `pg_dump` | the live database as of last night | `gdrive:ViewCastLK/backups/daily/YYYY/MM` |

Training reads `video_horizon_labels`, never the raw partitions, so the model
pipeline is unaffected by how aggressively partitions are dropped.

## Why daily partitions

Partition granularity sets the smallest unit that can be dropped. With weekly
partitions the database holds between 7 and 14 days depending on where the week
boundary falls, and the peak exceeds the free tier. Daily partitions with a
nightly archive sit at 7–8 days steadily and absorb four or five consecutive
failed runs before space becomes a concern.

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

At 113 MB of a 500 MB tier, tonight's run exports sixteen partitions and drops
none.

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

Manual runs accept `retain_days` and `dry_run` from the Actions page.

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

Each run uploads a manifest to `archives/manifests` recording every file's day,
row count, byte size and SHA-256, so completeness can be checked without
downloading anything.

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

The nightly `pg_dump` backups are a different matter: they are full dumps and
are never pruned, so they accumulate at roughly the size of the live database
every night. That needs a retention policy independently of this workflow.
