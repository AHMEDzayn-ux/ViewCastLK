# ViewCastLK Database Backup

The `backup.yml` GitHub Actions workflow creates one logical Supabase backup
every day and uploads it to Google Drive. It also supports a manual run from
the Actions page.

## Schedule

The workflow runs at `19:30 UTC`, which is `01:00` in Sri Lanka. This gives the
collector's midnight run approximately one hour to finish before the backup
starts. GitHub may start scheduled jobs a few minutes late.

## Backup contents

Each archive contains the three SQL files recommended by Supabase for a
logical restore:

- `roles.sql` — custom database roles
- `schema.sql` — tables, indexes, constraints, and other schema objects
- `data.sql` — table records

It also contains a small run manifest and SHA-256 checksums. The archive and a
second checksum file are uploaded to `ViewCastLK/backups/daily/YYYY/MM` by
default. Existing Drive backups are not deleted automatically.

This is a database backup. Supabase Storage objects are not included; the
current ViewCastLK collector stores its records in PostgreSQL tables.

## Required GitHub configuration

Configure these under **Repository Settings → Secrets and variables →
Actions**. Never paste their values into source files, issues, chat messages,
screenshots, or workflow logs.

### Secrets

#### `SUPABASE_BACKUP_DB_URL`

Use the Supabase **Session pooler** connection string from the project's
Connect panel. Supabase currently recommends the session pooler for CLI
backups. It normally uses port `5432`. Replace the password placeholder and
percent-encode special characters in the password. Keep the collector's
transaction-pooler URL in its existing `SUPABASE_DB_URL` secret.

#### `RCLONE_CONFIG`

Create a dedicated rclone configuration outside this repository with a
Google Drive remote named exactly `gdrive`. For a normal personal Google
Drive, authorize rclone using the Drive owner's OAuth login. Copy the full
dedicated configuration into this multiline GitHub Actions secret, then
securely delete the temporary local configuration after testing it.

Configure rclone with a dedicated Google OAuth client ID. Do not rely on
rclone's shared Drive client ID: rclone reports that the shared ID is being
retired during 2026. After changing the local rclone configuration, replace
the `RCLONE_CONFIG` GitHub secret with the updated configuration.

The workflow writes this secret only to a permission-restricted temporary
file on the GitHub runner. It never prints or commits the configuration.

### Optional repository variable

#### `GOOGLE_DRIVE_BACKUP_PATH`

This overrides the default Drive path `ViewCastLK/backups`. It is a folder
path, not a credential.

## First test

1. Add the two secrets above.
2. Open **Actions → Back up Supabase database → Run workflow**.
3. Confirm that the job reports a successful upload and integrity check.
4. In Google Drive, confirm that the new archive and its `.sha256` file exist.
5. Do not share or download the archive to an untrusted computer; it contains
   the project's database records.

The scheduled workflow becomes active only after this workflow file is merged
into the repository's default branch. Do not merge or push it until it has
been reviewed and approved.

## Local pre-deployment test

On 23 July 2026, the complete backup path was tested against the separate test
Supabase project before deployment. Supabase CLI successfully exported
`roles.sql`, `schema.sql`, and `data.sql`. The files were archived with a
SHA-256 checksum and uploaded to
`ViewCastLK/backups/test/2026-07-23` in Google Drive. `rclone check` reported
zero differences and two matching files. The temporary local SQL and archive
files were deleted after verification; the Drive test copies were retained as
evidence.

## Restore testing

A restore test should be performed against a separate temporary Supabase
project, never against the production project. Verify both checksum files,
extract the archive, and restore `roles.sql`, `schema.sql`, and `data.sql` in
that order using the procedure in Supabase's official backup and restore
guide:

https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore

The restore-test result, date, test project, row-count checks, and any errors
should be recorded in the individual project logbook.
