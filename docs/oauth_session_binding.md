# YouTube OAuth initiating-session binding

## Root cause and original flow

Base: origin/main c5f86ce9b78ab5999f53609d22e7d0b4e80783c0.

`dashboard/src/app/account/page.tsx` renders `YouTubeConnectionCard`. Its
`connect` calls `dashboard/src/lib/api/youtube-connection.ts::startYouTubeConnection`,
which sends a Supabase bearer token to `GET /auth/youtube/start`.
`prediction_api/app/auth.py::validate_access_token` obtains the user ID from a
successful Supabase `/auth/v1/user` validation, not from browser-supplied user_id.

`main.py::start_youtube_oauth` generates `secrets.token_urlsafe(32)` state through
`youtube_oauth.py::generate_oauth_state`. `CreatorStore.create_oauth_state` stores
only its SHA-256 hash, user_id, ten-minute expires_at and timestamps in the
unexposed `creator.oauth_states` table. `consume_oauth_state` atomically updates
consumed_at only for an unconsumed, unexpired state.

Previously, public `main.py::youtube_oauth_callback` accepted only that state,
exchanged the Google code, looked up the consenting channel, encrypted/stored
the refresh token for the state's user_id, and synchronized creator history.
There was no authenticated user or initiating-browser proof at completion.
Forwarding A's Google authorization URL to B could therefore link B's consent
to A's account. Randomness, expiry and single-use state do not prevent this.

The committed production frontend is https://viewcastlk.pages.dev and the API
is https://viewcastlk-api-932754937666.asia-south1.run.app. Google's registered
redirect URI points to the API `/auth/youtube/callback`, not Pages. They are
different origins and sites; live configured values still require inspection.

## Design

Keep random hashed state, expiry and atomic consumption. Store an independent
random binding nonce's hash and the validated Supabase session_id with state.
Return the nonce only in the authenticated start JSON body (no-store). The
frontend retains it in this tab's sessionStorage, never in the Google URL.

The existing Cloud Run GET callback becomes a side-effect-free relay. It
redirects to `/account/youtube-callback` on the configured Pages origin with
Google's response in a fragment (`youtube_state`, `youtube_code` or a generic
denial marker). Fragments are not sent to Pages/access logs or in HTTP referrers.
The completion page removes the fragment before asynchronous work and never
stores the Google code. It submits a bearer-authenticated POST to
`/auth/youtube/complete` with the code, state and nonce in the JSON body.

Only after remote Supabase token validation do we read its signed session_id
claim and compare its sub to the validated user. This decoding is not a
replacement for signature/expiry validation. OAuth requires a UUID session_id;
other existing authentication flows retain their current behavior.

One SQL UPDATE consumes the state only if state hash, user ID, session ID,
binding hash, expiry and unconsumed status all match, and the session still
exists in auth.sessions for that user. Only then can Google exchange, token
storage, roster handoff and sync happen. Token refresh within the same Supabase
session works; a new login/session or different account fails. Existing states
without binding fields fail closed. Rejection does not consume another flow.

No cookies are used. A Cloud Run cookie set by a Pages fetch would depend on
cross-site cookie acceptance and would not independently prove the current
Supabase account/session. No credentialed CORS or cookie Domain/SameSite change
is needed. The independent nonce is per-tab sessionStorage, not persistent
localStorage; the backend additionally enforces validated identity/session,
active-session existence and one-time state. A fresh tab without the proof or
cleared tab storage fails safely and requires starting again. Browser features
that clone sessionStorage into a duplicate/opener tab can copy this proof within
the same browser session; this is not an unclonable physical-tab identity.

The relay must NOT persist a code retrievable using state alone: otherwise A
could retrieve B's response using A's own state/nonce after B's forwarded flow.
Instead only the browser receiving Google's redirect has the Google response;
B lacks A's tab nonce/session and A cannot retrieve B's code from the backend.
No Supabase JWT, refresh credential or binding nonce enters a URL. Google's
standard callback itself still delivers code/state in its query; application
Uvicorn logging redacts the callback query. Managed ingress/log agents must be
verified separately, since their logs precede application code.

This protects the forwarded-flow threat, not a compromised browser/XSS or an
attacker possessing the bearer token, tab nonce and complete Google response.
Do not share full consent/callback URLs, tokens, bodies or HAR captures.

## Release requirements (not executed)

Apply the additive Auth-project migration for oauth_states binding columns;
publish the static callback page and matching API together. Existing in-flight
flows must restart. No Google redirect URI/domain/CORS change is required.
Verify actual Supabase JWT session claims, DB role access to auth.sessions,
Pages callback routing and managed callback-query logging in staging before
any production rollout. No migration, commit, push or deployment is authorized
by this implementation task.

Technical references: [Supabase sessions](https://supabase.com/docs/guides/auth/sessions)
documents the JWT session_id and auth.sessions relationship;
[URI fragments](https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Fragment)
documents that fragments are not sent to the server;
[Google web-server OAuth](https://developers.google.com/identity/protocols/oauth2/web-server)
documents the existing authorization-code exchange.

## Implementation review, 2026-10-07

Status: **FIXED IN CODE; STILL REQUIRES LIVE VERIFICATION.**

The original checkout was clean on `main` at
`cf606ad040413f29d5b4b3f9dd8ab29167dd00cf`, zero commits ahead and 15 behind the
freshly fetched origin/main. It remains clean and unchanged. Implementation is
uncommitted on `fix/oauth-session-binding` in
`D:\projects\viewcastlk\.worktree-oauth-session-binding`, based exactly on
`c5f86ce9b78ab5999f53609d22e7d0b4e80783c0`.

### Changed files and functions

All paths below are relative to the feature worktree.

| File | Change |
| --- | --- |
| `prediction_api/app/auth.py` | `validate_access_token`, `_validated_session_id`, `AuthenticatedUser`: extract session claim only after remote bearer validation and matching subject. |
| `prediction_api/app/youtube_oauth.py` | `generate_session_binding_nonce`, `hash_session_binding`: independent 256-bit nonce and domain-separated hash; existing state, scopes, token exchange and encryption remain. |
| `prediction_api/app/creator_store.py` | `create_oauth_state`, `oauth_state_is_pending`, `consume_oauth_state`: persist hashes/session and execute atomic owner/session/nonce/expiry/replay/active-session guard. |
| `prediction_api/app/main.py` | `start_youtube_oauth`, `youtube_oauth_callback`, `complete_youtube_oauth`: authenticated start, read-only relay, authenticated completion before existing channel/storage/roster/sync flow. |
| `prediction_api/app/schemas.py` | OAuth start/completion request/response models only. |
| `prediction_api/app/oauth_logging.py` | Redact callback query from Uvicorn access records without changing the actual request. |
| `supabase-auth/supabase/migrations/20261007120000_oauth_session_binding.sql` | Add nullable session_id and binding_hash columns, so pre-release in-flight rows fail closed. Not applied. |
| `dashboard/src/lib/api/youtube-connection.ts` | `startYouTubeConnection`, `completeYouTubeConnection`: tab proof, immediate fragment removal, authenticated POST; existing status/disconnect interfaces remain. |
| `dashboard/src/app/account/youtube-callback/page.tsx` | New static callback page with no-referrer metadata. |
| `dashboard/src/components/account/YouTubeOAuthCompletion.tsx` | Minimal completion/error UI and one submission under StrictMode. |
| `prediction_api/tests/test_oauth_session_binding.py` | New database-predicate and endpoint/security tests. |
| `prediction_api/tests/test_youtube_oauth.py` | Existing success/denial/encryption tests updated for authenticated completion; status/disconnect tests retained. |
| `dashboard/src/lib/api/youtube-connection.test.ts` | New start/proof/completion/storage/denial/reconnect tests. |
| `dashboard/src/components/account/YouTubeOAuthCompletion.test.tsx` | New success, StrictMode, denial and safe-error tests. |
| `docs/oauth_session_binding.md` | Root cause documented before application edits; design, review, limits and manual verification. |

Successful completion preserves `fetch_authenticated_channel`,
`encrypt_refresh_token`, `CreatorStore.upsert_youtube_connection`,
`request_channel_collection_if_available` and `synchronize_creator_history`.
Binding failure returns before any of those or Google token exchange, so it
cannot create a connection, persist a refresh token or create adjustments via
sync. Google exchange failure after valid proof also cannot persist/sync; the
consumed state requires starting again, as with the previous one-use flow.

### Test results

| Command / check | Result |
| --- | --- |
| Python 3.11 `python -m pytest -q -rs -p no:cacheprovider` in `prediction_api` | **239 passed, 7 skipped, 1 warning**, 18.34 seconds. |
| `npm.cmd test` in `dashboard` | **23 files passed, 136 tests passed**, 16.18 seconds. |
| `npm.cmd run lint` | Passed, exit 0. |
| `npm.cmd run build` | Passed, TypeScript and static export; includes `/account/youtube-callback`. |
| `git diff --check` | Passed. |
| Original checkout status/HEAD | Clean, unchanged `main` at `cf606ad040413f29d5b4b3f9dd8ab29167dd00cf`. |

Backend skips: one model fixture has no blank category; six training-builder
parity cases cannot import the training builder from this checkout. These
tests were not changed or forced to pass. One warning concerns Starlette's
deprecated httpx TestClient integration. Tests run with dotenv disabled and
Supabase/Google/data-service credentials cleared; the frontend build used
dummy public configuration. No production OAuth, forecast or sync requests
were issued.

New backend cases exercise the actual production SQL predicates through a
transactional SQLite adapter: correct proof, missing/unknown/expired/replayed
state, wrong user, same user with another session, missing/tampered/other-flow
nonce, missing/revoked/misowned active session, legacy state, missing Google
response, Google exchange failure, unauthenticated completion and reconnect.
Every binding rejection asserts zero exchange, channel lookup, connection
storage, roster handoff and creator synchronization. Tests also verify hashed
persistence, invalid proof not consuming the legitimate flow, safe callback
logging and validated JWT subject/session extraction.

Frontend tests verify tab storage (no localStorage binding), nonce/token
absence from the Google URL, fragment removal before auth/network, refreshed
token use, invalid/missing/expired proof rejection without a completion
request, storage failure, denial, reconnect and safe UI. Existing backend and
frontend suites cover authentication, account, guest/authenticated forecasts,
history, creator status, disconnect, sync and personalization. These are local
regression results, not proof of hosted behavior or visual browser QA.

Diff review found only the 15 listed OAuth-related files. No model/artifact,
forecast calculation, existing account UI, rate-limit, workflow, Docker or
cloud configuration changes. No commit, push, merge, deployment or migration
execution occurred.

### Remaining limits

- SQLite executes the SQL guard logic but does not prove PostgreSQL migration
  syntax, PostgreSQL concurrent-update behavior or hosted database permissions.
  The production UPDATE is atomic; hosted integration still needs verification.
- Check real Supabase access-token session claims and the API DB role's SELECT
  permission on auth.sessions. Missing permission must fail closed, not be
  worked around by removing the session guard.
- The nonce is accessible to this origin's JavaScript, and same-browser tab
  cloning can copy sessionStorage. This protects separate browsers/profiles and
  distinct authenticated sessions; it cannot defeat XSS or copied credentials.
- Navigation refresh, cleared storage, another pending flow in the same tab,
  or a completion network failure requires restarting connection. Old in-flight
  flows fail closed after rollout. These do not delete existing connections.
- Application Uvicorn queries are redacted. Google necessarily returns its
  authorization code/state in the standard callback query. Cloud Run managed
  request logs, proxies, browser extensions and any external telemetry were
  not verified or reconfigured; total infrastructure log safety is not claimed.
- No real Google consent, live Supabase DB, browser-domain integration,
  deployed Cloudflare route or live forecast regression was tested.

## Manual production verification, after an approved release

First perform these checks in staging. Only repeat in production after
separate approval, with dedicated test ViewCastLK accounts and Google channels.
Record status codes, timestamps and safe channel IDs only; do not export HARs,
request bodies, token headers or complete consent/callback URLs.

### Release prerequisites

1. Review/apply the new migration to the **Supabase Auth project**, not the
   optional public warehouse. Confirm both columns exist and the API's server
   DB role can read auth.sessions. Keep creator schema unexposed to browser API.
2. Publish matching API and Pages changes as a coordinated release. A frontend
   mismatch must not be accepted by weakening verification. In-flight flows
   must restart. Keep the Google registered redirect at the existing API
   `/auth/youtube/callback`; no cookie/domain/CORS changes are required.
3. Verify configured DASHBOARD_ORIGIN and GOOGLE_OAUTH_REDIRECT_URI match the
   actual production domains. Open Pages `/account/youtube-callback` without a
   fragment: it should render a safe restart message, not a 404 or success.
4. Inspect application and managed logging using approved access. Confirm
   application callback logs contain method/path/status only. Verify managed
   request URLs and telemetry are also redacted/restricted appropriately before
   considering token/log safety verified. This code cannot redact ingress logs.

### A. Normal flow

1. In Browser A, log into a dedicated ViewCastLK account, open Account and
   choose Connect YouTube. Keep the flow in the initiating tab.
2. In Google consent, choose the dedicated channel and Allow.
3. Observe Cloud Run callback then Pages completion then
   `/account?youtube=connected`. The Pages fragment should disappear immediately.
4. In DevTools Network, confirm start 200, relay 303 and completion POST 200
   with only `{ "connected": true }` as its response. Confirm the connected
   channel/status is correct; allow sync to reach active or inspect its safe
   status. Confirm encrypted credential storage without displaying its value.

### B. Replay

1. Use Firefox for A so Network provides **Edit and Resend**. After successful
   completion, select the completed POST `/auth/youtube/complete`; use Edit and
   Resend and send the unchanged request once within the same private DevTools
   session. Do not copy/export its headers or body.
2. Expect 400 `invalid_oauth_state`, no second exchange/storage/sync/adjustment
   activity and unchanged connection. A replayed GET callback with that state
   should also return 400; do not publish or retain its sensitive URL.
3. Verify the original state's consumed_at remains set once. Use safe counts
   or dedicated-account rows to check that no additional adjustments appeared.

### C. Different browser/session

1. Record A/B's test connection status. In Browser A/profile A, logged in as
   test user A, start Connect and pause at Google consent.
2. Open that authorization navigation only in a separate Browser B/profile B
   under your control, logged into ViewCastLK as test user B. Complete Google
   consent with B's dedicated channel. Do not transfer A's token, tab storage
   or binding nonce, and do not share the URL outside this controlled test.
3. Expect safe rejection at the completion page, no successful completion POST
   and no channel/credential/sync/adjustment changes for A or B. The GET relay
   may return 303: that is deliberately not a successful link.
4. Repeat with the same ViewCastLK account independently logged into B, which
   produces a different Supabase session. Expect the same rejection.
5. To exercise the server's session mismatch with the proof still in the
   initiating tab: start a fresh flow in A; while it is on Google consent, sign
   out and sign back into ViewCastLK in a second tab of that same profile. Then
   finish the original consent in A. The original nonce remains tab-local, but
   the current Supabase session differs. Expect completion 400 and no link.
   Repeat with an account switch in that second tab if desired.

### D. Reconnect

1. Return to Account and start a fresh Connect/Reconnect flow in the same
   signed-in tab. Approve the intended dedicated Google channel.
2. Expect a new state/nonce, successful completion, intended current channel
   and creator sync/status. Retry a previously used flow only in B, not as a
   reconnect mechanism; it must remain rejected.

### E. Disconnect

1. On Account disconnect the dedicated channel through the existing control.
2. Expect authenticated DELETE success, disconnected status and private creator
   connection/data/adjustment cleanup consistent with the existing flow.
3. Reload Account: disconnected status persists. Reconnect using a new flow
   and confirm it succeeds. Never inspect or display plaintext stored tokens.

### F. Forecast regression

1. In a signed-out private browser, submit one valid shared/guest forecast.
   Confirm normal horizons/results and no creator personalization.
2. In the authorized dedicated account after reconnect and sufficient creator
   history/sync, submit one valid forecast for its connected channel. Confirm
   the expected personalized result/status and history entry. If no sufficient
   creator adjustments exist, do not interpret a shared fallback as proof that
   personalization was verified; prepare the proper fixture first.
3. Reload Account and History; verify creator status, forecast history and
   authentication still work. Compare deterministic mocked regression evidence
   for numeric behavior; live data changes are not evidence of a model change.

Stop after collecting safe evidence. Live verification remains outstanding
until these checks and the infrastructure log checks pass. No release action
is authorized by this report.
