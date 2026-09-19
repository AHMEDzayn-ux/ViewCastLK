# ViewCastLK Prediction API

Minimal Python server-side backend providing real YouTube channel analytics for the ViewCastLK creator dashboard.

## Features

- `GET /health` — API health check
- `GET /accuracy` — Honest publication status for approved evaluation results
- `POST /channel-lookup` — Server-side YouTube channel lookup by `@handle`, channel ID (`UC...`), or YouTube URL

- `POST /forecast` — Day 7, 14, 21, and 30 cumulative-view trajectory
- `GET /auth/youtube/start` — Begin a user-bound, server-side Google OAuth flow
- `GET /auth/youtube/callback` — Complete OAuth without returning tokens to the browser
- `GET /creator/youtube-connection` — Return browser-safe connection status

## Authentication and history provenance

`/forecast` and `/channel-lookup` require a current Supabase access token in
the `Authorization: Bearer <token>` header. The API validates that token with
the configured Supabase Auth `/auth/v1/user` endpoint using only
`SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. `/health` and `/accuracy` remain
public. Bearer tokens must never be logged, returned, or placed in URLs.

Forecast-history writes remain browser-to-Supabase operations protected by
RLS. RLS prevents one authenticated user from reading or changing another
user's rows, but it does not prove that values in a user's own row originated
from the prediction API; a user could fabricate values in their own history by
calling the Data API directly. No service-role writer is introduced solely to
remove that provenance limitation.

Creator OAuth state, encrypted refresh credentials, and private Analytics data
use the unexposed `creator` schema through `SUPABASE_AUTH_DB_URL`. Browser roles
have neither schema usage nor table grants. The server requires
`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`,
`GOOGLE_OAUTH_REDIRECT_URI`, and a URL-safe base64 `TOKEN_ENCRYPTION_KEY`
representing exactly 32 random bytes. These values are server-only secrets or
configuration and must never use a `NEXT_PUBLIC_` prefix.

Title guidance uses `GEMINI_MODEL` first and then the semicolon-separated
`GEMINI_FALLBACK_MODELS` list. Retryable quota or availability failures move to
the next model; invalid credentials and invalid requests stop immediately.

## Active model artifact

The API serves `model_artifacts/viewcastlk_monotonic_trajectory_experimental_v1`.
It predicts all four horizons in one call and guarantees a nondecreasing
cumulative trajectory. The previous `viewcastlk_mvp_candidate_v1` directory is
retained only as a rollback artifact and is not loaded by the API.

The active artifact is explicitly experimental: its manifest reports that no
video in the frozen dataset has all four horizon labels, so end-to-end Day 30
accuracy has not been measured. See the artifact's `manifest.json` and
`evaluation/` directory before treating it as an approved production-quality
model.

## Setup & Running Locally

1. Create a `.env` file from `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Configure the values documented in `.env.example`. The Supabase publishable
   key is public project configuration; do not use a service-role or secret key
   as its replacement. Keep all creator database, Google OAuth, and encryption
   values server-only.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the API locally:
   ```bash
   cd prediction_api
   python -m uvicorn app.main:app --reload
   ```

## Running Tests

From the root directory or `prediction_api/`:
```bash
pytest prediction_api/tests
```
