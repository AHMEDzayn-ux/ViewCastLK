# Critical production hardening — 2026-10-08

Base origin/main: `6b921600492791e3db86966e235d6f0a402b73d2`.
OAuth commit `b39df4abbacf22fa37e4ed6f5cdc94625a92dc12` is included.
Branch: `fix/critical-production-hardening`, main checkout fast-forwarded while clean.
No staging, commit, push, merge, deployment or production migration is authorized.

## Baseline before application edits (report items 1–4)

Backend Python 3.11: 239 passed, 7 skipped, one TestClient deprecation warning.
Frontend: 136 passed in 23 files; lint and static production build passed.
Credentials disabled for backend tests; frontend build uses dummy public values.

Public routes: health, accuracy, OAuth GET relay and optional-auth forecast;
FastAPI's default OpenAPI/docs routes are also public. Authenticated routes:
channel-lookup, OAuth start/complete, creator status/disconnect/insights.
Forecast calls YouTube Data API, Gemini and public history DB; completion calls
Google token exchange, YouTube identity, encrypted storage and initial sync.
Weekly creator_refresh_job refreshes credentials and calls Data/Analytics sync;
there is no public refresh/sync HTTP endpoint. Reconnect uses start/complete.
Existing HTTPX calls have 5/10/15 second inactivity timeouts; Gemini already
has 8 second per-model calls and a 15 second fallback budget; Data API discovery
has no explicit transport timeout and retries three times. Postgres connects
in five seconds but has no statement timeout. Lifecycle diagnostics echo
arbitrary exception strings. OAuth Uvicorn query redaction already exists.
Frontend wrappers preserve safe API errors/status codes. Dependencies:
dashboard package.json/package-lock.json and prediction_api/requirements.txt.

## Task 1 — shared admission, reviewed before Task 2 (items 5–10)

Use the existing server-only Auth Postgres connection, not Redis or per-instance
memory. New creator.rate_limits rows store HMAC bucket identities, request count
and expiry epoch. Transaction advisory lock 742193018 serializes admission and
cleanup across service instances and creator refresh jobs. All window checks
inside one scope are in one transaction; rejection rolls them back.
Cleanup deletes at most 1000 expired rows per transaction, indexed by expiry;
new keys fail closed at 10000 rows. No scheduled process is needed: inactive
expired rows are reclaimed on subsequent traffic, and storage stays capped.
The cap assumes all writers use this server-only admission path.

Default limits (count/window seconds), overridable via RATE_LIMITS_JSON:

| Scope | Limits |
| --- | --- |
| forecast_guest | 6/60; 60/86400 |
| forecast_user | 12/60; 120/86400 |
| channel_lookup | 20/60; 300/86400 |
| oauth_start | 3/600; 20/86400 |
| oauth_complete | 6/600; 30/86400 |
| creator_read | 30/60 |
| disconnect | 6/60 |
| creator_sync | 1/1800, rolling cooldown |
| ingress | 60/60 per public identity, before bearer validation |
| forecast_global | 120/60; 1000/86400 across all forecasts/Gemini calls |
| channel_lookup_global | 120/60; 3000/86400 |
| oauth_global | 120/60; 1000/86400 across start/relay/complete |
| creator_global | 300/60 across creator status/disconnect/insights |

User scopes use validated user_id. Guest/public scopes use raw ASGI peer and
only honor XFF when peer is explicitly in RATE_LIMIT_TRUSTED_PROXY_CIDRS.
Walk right-to-left, stopping at nearest untrusted hop; ignore spoofed prefixes.
Unknown private proxy identity shares a conservative bucket. Uvicorn automatic
proxy rewriting is disabled in Docker; any alternative launch must do likewise.
Universal /0 trust configuration falls back to the shared bucket.
Cloud Run's actual ingress hop chain is NOT proven by repo configuration; verify
it in staging and configure only actual trusted hops. Never use universal CIDRs.
Until then, shared fallback protects quota but cannot promise per-client fairness.
All identities are HMACed with RATE_LIMIT_HASH_KEY (>=32 characters), same secret
on every API/job revision. Raw IPs are never persisted. Invalid config, missing
secret/DB/schema, DB failure or capacity exhaustion returns safe 503; over quota
returns safe 429 and Retry-After. OPTIONS/health/accuracy/docs are not throttled.

OAuth successful-state consumption precedes sync cooldown/exchange; a cooldown
rejection requires a new flow later. Denial never starts sync. Job refresh checks
the same rolling user cooldown before token refresh or sync. Fixed request
windows permit boundary bursts; creator cooldown does not. Cooldown is not a
full-duration distributed synchronization lease, so it should exceed sync budget.

Task 1 validation: 249 passed, 7 skipped, one warning. Tests execute production
SQL via a local SQLite adapter and simulate locking with BEGIN IMMEDIATE; shared
Postgres lock semantics, privileges and contention still need staging integration.
Existing auth fixture retained; local regression suites mock hosted limiter only.
Diff reviewed: middleware, identity/counter module, creator lifecycle admission,
server-only migration, proxy startup flag and corresponding tests/docs only.

## Task 2 — input and body limits, reviewed before Task 3 (items 11–14)

Forecast title 1–100 Unicode characters after existing whitespace checks;
duration finite, >0 and <=43200 seconds (YouTube upload limit, not a model
formula). Category 1–100 printable characters: unseen-category warning behavior
is intentionally preserved. Language matches the existing five frontend choices;
day matches seven weekdays; hour is strict integer 0–23; modelEngine keeps v8/v9/v10.
Channel input <=512, exact UC+22 URL-safe ID or validated international handle
(generous 1–100 bound); only actual YouTube channel/handle/legacy URLs accepted.
Malformed URLs/paths/control characters never reach the provider. Lookup and
forecast forbid extra fields; there are no public request list fields. OAuth
request already bounds state/nonce/code and forbids extras.
MAX_REQUEST_BODY_BYTES defaults to 65536 (allowed 1024–1048576); pure ASGI byte
buffering checks both declared and actual bytes, including chunked/missing/lying
Content-Length, before JSON parsing. Oversize returns safe 413; validation stays
400 and preserves the existing channel whitespace error message.
References: https://developers.google.com/youtube/v3/docs/videos and
https://support.google.com/youtube/answer/71673. Frontend design unchanged; safe
API messages already reach existing error UI.
Task 2 suite: 270 passed, 7 skipped, one warning. Added NaN/Infinity/negative/huge
duration, title, category, ID, enum, extra-array, Unicode, size-boundary and
chunked-body tests. Existing unseen-category/model and normal forecast assertions
remain unchanged. Diff reviewed before external-client work.

## Task 3 — external bounds, reviewed before Task 4 (items 15–19)

Shared outbound HTTPX: connect <=5s, write <=5s, pool 2s; read/total budgets:
Supabase user validation 5s/12s, OAuth token/revoke/channel identity 10s/20s,
creator Data/Analytics reads 15s/30s. GET retries at most 3 attempts on transport
failures or 408/429/500/502/503/504; backoff 0.25s, 0.5s, Retry-After respected
up to 2s. A longer Retry-After stops this operation rather than retrying early.
OAuth exchange, refresh/revocation POSTs and all DB writes have ONE attempt.
No retry wraps connection storage or a whole synchronization.
Data discovery/reads: httplib2 transport 8s, discovery retry disabled; channel
GETs <=3 explicit attempts, bounded backoff and same transient status policy.
Channel service runs off the event loop with a 30s request deadline; a cancelled
worker thread cannot be force-killed and remains subject to transport bounds.
Gemini retains 8s per-model bound, now capped to the remaining 15s budget and
three configured models, with bounded backoff and SDK transport closure.
No prompt, feature/model or formula change. Failed optional guidance still
degrades the forecast safely. Creator sync orchestration has a 300s deadline,
shorter than the 1800s cooldown, without changing synchronization mathematics.
Body receive has a 10s total deadline. DB application connections retain 5s
connect bound plus statement 5000ms/lock 1000ms, configurable within bounds;
limiter uses stricter statement 3000ms/lock 1000ms. Migrations are unaffected.
Connection, lock and operation failures cannot lead to repeated sensitive writes.
Existing sync can perform partial idempotent DB updates before failure; this
change adds neither a retry nor an all-or-nothing synchronization transaction.
Task 3 suite: 284 passed, 7 skipped, one warning. Provider success, transient
recovery, exhaustion, non-retryable statuses, POST single attempt, Retry-After,
actual total deadline cancellation, transport settings, safe errors, Gemini
closure/attempt cap and DB configuration are covered. Diff reviewed.

The API workflow now explicitly sets a 420s Cloud Run timeout, leaving room
around the 300s sync budget for authentication, exchange and persistence.
This is an infrastructure ceiling, not a guarantee every request lasts 420s.
Cloud Run can continue processing after its request timeout; application bounds
and staging timing checks remain necessary. See [Cloud Run request timeouts](https://docs.cloud.google.com/run/docs/configuring/request-timeout).
The limiter explicitly closes its Postgres connection on admission/rejection;
owned YouTube transports and Gemini clients close after use. Tests cover real
async sync cancellation without restart and Uvicorn formatter compatibility.

## Task 4 — logs and dependencies (items 20–24)

A central LogRecord factory redacts formatted application and library records
before handlers: callback/query parameters, code/state/binding nonce, bearer
tokens/JWTs, access/refresh tokens, client/API secrets, encryption/hash keys,
database URLs/passwords and configured secret values. Exception diagnostics
retain class and safe endpoint/stage, with arbitrary exception detail omitted.
Uvicorn's five access arguments are preserved for its actual AccessFormatter;
request targets lose queries while endpoint/status remain useful. Refresh jobs
install the same protection and never print arbitrary exception strings.
No new body/header logging was added. Existing feature/model CLI-only prints
are outside served paths and unchanged. Structured custom handler extras and
Google-managed request logs are outside this formatter's protection.

Audits ran against the installed local dependency graph and final lockfile:

| Audit | Before | After |
| --- | --- | --- |
| npm, all dependencies | 1 critical, 8 high | 0 critical, 5 high |
| npm, production dependencies | baseline all-dependency audit above | 0 vulnerabilities |
| pip-audit, Python 3.11 test environment, 84 distributions | 0 known vulnerabilities | 0 known vulnerabilities |

Python requirements and all exact ML dependency pins are unchanged. The Python
audit does not prove versions resolved later in a fresh Linux container: other
requirements use minimum versions. Audit that resolved production environment
before an approved deployment.

| Package | Dependency/path assessment | Small compatible change |
| --- | --- | --- |
| next | Direct; next/og ImageResponse RCE path absent from this static export | 16.3.5 → 16.3.6 |
| eslint-config-next | Direct development dependency; matching Next patch | 16.3.5 → 16.3.6 |
| sharp | Transitive Next image parser; no server image endpoint in static export | 0.35.4 → 0.35.5; matching platform/libvips packages |
| source-map-js | Transitive CSS/source-map tooling; no API accepting untrusted maps | 1.2.1 → 1.2.2 |
| brace-expansion | Transitive development glob expansion, repository-controlled patterns | 1.1.18 → 1.1.21 and 5.0.9 → 5.0.12 |
| braces | Transitive development glob expansion; not imported by application source | 3.0.3 remains; no published fixed version |

The five remaining high findings are ONE underlying braces stack-exhaustion
advisory, propagated through micromatch → fast-glob → @next/eslint-plugin-next
→ eslint-config-next. Local audit --omit=dev excludes that entire path. Builds
still use development tooling: keep build patterns/repository inputs trusted
and track the upstream patch. npm's suggested major downgrade of the Next ESLint
config was rejected; no force upgrade, major change or suppression was used.
[braces advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm),
[Next advisory](https://github.com/advisories/GHSA-vcvr-r3jv-pc5j),
[sharp advisory](https://github.com/advisories/GHSA-wq5f-xc86-pv6w),
[source-map-js advisory](https://github.com/advisories/GHSA-68fv-2mgg-jv7q),
[brace-expansion advisories](https://github.com/advisories/GHSA-q2hr-2g5m-vwhr).

## Final validation and scope (items 25–26)

Backend: **296 passed, 7 skipped, 1 existing warning**. Skips remain one blank
category absent from the served artifact and six training-builder import cases;
warning is the existing Starlette/httpx TestClient deprecation.
Frontend: **136 passed in 23 files**, lint passed, Next **16.3.6** production
build/TypeScript/static export passed, with dummy public configuration.
Clean-directory npm ci dry run passed. Both workflow YAML documents parse;
custom gcloud delimiter mapping preserves every existing env/secret binding.
See [gcloud delimiter syntax](https://docs.cloud.google.com/sdk/gcloud/reference/topic/escaping).
git diff --check passed. Gitleaks scanned all changed/new files with redacted
output: **no leaks found**. This is a scoped scan, not a historical repo scan.

All **34** changed/new files were reviewed; no frontend source/UI changes,
model/artifact changes, forecast/personalization formulas, backup changes or
unrelated refactoring. Historical-feature function ASTs match main except
the database _connect wrapper, which only gains timeout options. Gemini's
prompt, tone schema, model selection configuration and mathematical consumers
are unchanged; the execution budget and fallback attempt count are bounded.
Lockfile changes preserve unrelated versions/platform records and only upgrade
the security targets and their matching binary packages.

| Files (paths relative to repository) | Necessary purpose |
| --- | --- |
| .github/workflows/cloudrun-deployment.yml; .github/workflows/creator-refresh.yml | Shared secret bindings, verified proxy variable, API timeout; existing deployment mappings preserved |
| dashboard/package.json; dashboard/package-lock.json | Compatible security patches |
| prediction_api/.env.example; prediction_api/Dockerfile | New config documentation; raw proxy peer preservation |
| prediction_api/app/rate_limits.py; supabase-auth/supabase/migrations/20261008010000_rate_limits.sql | Shared atomic quota/cooldown storage and admission |
| prediction_api/app/main.py; prediction_api/app/creator_lifecycle.py; prediction_api/app/creator_refresh_job.py | Route/job admission, bounded orchestration, safe diagnostics |
| prediction_api/app/request_limits.py; prediction_api/app/identifiers.py; prediction_api/app/schemas.py | Body and request-field validation |
| prediction_api/app/outbound.py; prediction_api/app/service_limits.py; prediction_api/app/database_limits.py | Reusable transport, orchestration and SQL deadlines |
| prediction_api/app/auth.py; prediction_api/app/youtube_oauth.py; prediction_api/app/youtube.py; prediction_api/app/creator_analytics.py; prediction_api/app/title_analysis.py | Apply finite provider bounds, idempotent-read retries, bounded pagination and transport closure |
| prediction_api/app/config.py | Cap configured sync history at 500; ordinary default stays 200 |
| prediction_api/app/creator_store.py; prediction_api/app/public_roster.py; prediction_api/app/channel_history.py | Database connection timeout options only |
| prediction_api/app/log_safety.py; prediction_api/app/oauth_logging.py | Central redaction and existing OAuth logger integration |
| prediction_api/tests/conftest.py; prediction_api/tests/test_rate_limits.py; prediction_api/tests/test_request_limits.py; prediction_api/tests/test_outbound_limits.py; prediction_api/tests/test_log_safety.py | Local isolation and regression/security coverage |
| docs/critical_production_hardening.md | Review, configuration and release evidence |

## Migration, configuration and release gates (items 27–30)

New migration **20261008010000_rate_limits.sql** applies to the Auth database.
It adds only creator.rate_limits: bucket text primary key (<=128 characters),
positive integer requests and bigint expires_at; an expiry index; RLS;
revocation of PUBLIC/anon/authenticated/service_role table access; a comment.
It changes no existing data/columns. Old backend revisions ignore this new
table, so the additive migration is backward-compatible. It is NOT applied.
The table size cap/cleanup are application-enforced, not a SQL row-count constraint.

Runtime SUPABASE_AUTH_DB_URL must use a server-only Postgres role with USAGE on
creator, SELECT/INSERT/UPDATE/DELETE on creator.rate_limits, and RLS bypass
through table ownership, superuser or an existing authorized BYPASSRLS role.
No policies permit ordinary API roles. Advisory locks must be executable.
The existing OAuth binding also needs USAGE on auth and SELECT of id,user_id
from auth.sessions, plus its existing creator-store privileges. These are raw
server-side psycopg connections, not browser/service-role REST access. Confirm
the actual connected role/privileges; neither hosted permission nor pooler
compatibility with startup statement/lock timeout options was tested here.

Required production configuration, with no secret values committed:

| Variable | Required production value/source |
| --- | --- |
| RATE_LIMIT_HASH_KEY | NEW random secret >=32 characters, identical on every API/job instance and revision; Secret Manager name viewcastlk-rate-limit-hash-key |
| RATE_LIMIT_TRUSTED_PROXY_CIDRS | API: comma-separated verified ingress hop CIDRs via same-named GitHub repository variable; empty/universal/malformed yields shared fallback, not verified per-client fairness |
| SUPABASE_AUTH_DB_URL | Existing server-only Auth DB DSN; same quota database for API/jobs; migration and privileges verified |
| SUPABASE_WAREHOUSE_DB_URL | Existing server-only warehouse DSN |
| YOUTUBE_API_KEY; GEMINI_API_KEY | Existing API provider secrets |
| GOOGLE_OAUTH_CLIENT_ID; GOOGLE_OAUTH_CLIENT_SECRET; TOKEN_ENCRYPTION_KEY | Existing API/job OAuth and encryption secrets |
| GOOGLE_OAUTH_REDIRECT_URI | Existing registered API callback, from same-named GitHub variable |
| SUPABASE_URL; SUPABASE_PUBLISHABLE_KEY | Existing Auth public URL/key, from NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY GitHub variables |
| DASHBOARD_ORIGIN | Existing workflow value https://viewcastlk.pages.dev |
| GEMINI_MODEL; GEMINI_FALLBACK_MODELS | Existing workflow choices; at most three attempted within total budget |

Optional runtime settings need no explicit override for these defaults:
RATE_LIMITS_JSON={}, MAX_REQUEST_BODY_BYTES=65536, DB_STATEMENT_TIMEOUT_MS=5000,
DB_LOCK_TIMEOUT_MS=1000, CREATOR_HISTORY_VIDEO_LIMIT=200 (maximum 500),
YOUTUBE_ANALYTICS_BATCH_SIZE=10. If overrides are adopted, wire them into the
approved deployment configuration; --set-env-vars replaces unspecified values.
API and job must use consistent creator_sync overrides and hash key. Rate limits
count requests/sync starts, not exact provider quota units; read retries and
three Gemini candidates can consume multiple calls per admitted request.

Before an approved release, an operator must:

1. Create the new Secret Manager secret and grant the runtime service account
   viewcastlk-api-runtime@viewcastlk.iam.gserviceaccount.com secretAccessor on it.
   Verify deployer access to configure the binding. Do not expose it as NEXT_PUBLIC.
2. Apply migration with the existing migration process; verify connected role,
   RLS restrictions, index and application timeout options through the actual
   server DSN/pooler. Prove browser anon/authenticated roles cannot access counters.
3. In staging, observe raw peer/XFF from separate real client IPs with automatic
   proxy handling disabled. Configure only verified hops; test spoofed prefixes,
   malformed chains and shared fallback. No live topology was inspected here.
4. With mocked providers, test simultaneous admissions from >=2 API instances
   against real Postgres, API/job cooldown sharing, expiry cleanup, 10000-row cap,
   lock contention and fail-closed DB/config outage. SQLite tests do not prove
   real PostgreSQL locking or performance. The global lock is intentionally small
   and conservative; measure contention at the configured traffic ceiling.
5. Staging-smoke one ordinary guest/auth forecast, valid Unicode, channel lookup,
   OAuth connect/reconnect/denial/disconnect, refresh and creator reads. Verify
   invalid/oversized requests make no provider calls; 429 has Retry-After/CORS;
   503 remains safe; health alone does not prove limiter readiness.
6. Test slow/transient mocks and deadlines, partial-sync recovery, no repeated
   exchanges/writes, and timing within the 420s API / 1800s job ceilings. Finite
   SQL/thread operations may finish after coroutine cancellation; the shared
   cooldown is an admission guard, not a distributed full-duration lease.
7. Verify application AND Google-managed request logs omit callback queries,
   credentials and bodies; application redaction cannot scrub platform requestUrl
   fields. Inspect sanitized evidence without copying real secret values.
8. Audit the freshly resolved Linux/container Python dependencies; verify the
   trusted build-input boundary for the remaining development-only braces issue.

Safe release order after approval: prepare secret/proxy configuration → additive
migration/privileges → backend and refresh-job rollout/verification → patched
frontend. This branch has no coupled frontend API contract change, so simultaneous
frontend/backend rollout is unnecessary. Earlier OAuth binding protocol is already
in the base. Partial rollout preserves that protocol: an old backend lacks these
new guards, whereas a new backend serves the old frontend's valid inputs/errors.
Rollback API/jobs to their preceding revision and keep the additive table/secret;
do not drop data during rollback. Rollback removes hardening, so contain abuse
operationally until the new revision is corrected. Keep the same hash secret
during gradual rollout: rotation changes buckets and resets effective allowances.

**GO FOR COMMIT**: all four tasks are implemented, reviewed and locally verified;
remaining development-only advisory has no compatible published fix and its
exposure is documented. **Production rollout remains blocked** until the migration,
shared secret/IAM, verified proxy topology and real-Postgres integration/logging
checks above pass. No production readiness or hosted controls are asserted.
Nothing staged, committed, pushed, merged, deployed or migrated. Stop for approval.
