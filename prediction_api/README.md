# ViewCastLK Prediction API

Minimal Python server-side backend providing real YouTube channel analytics for the ViewCastLK creator dashboard.

## Features

- `GET /health` — API health check
- `POST /channel-lookup` — Server-side YouTube channel lookup by `@handle`, channel ID (`UC...`), or YouTube URL

- `POST /forecast` — Day 7, 14, 21, and 30 cumulative-view trajectory

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
2. Configure your `YOUTUBE_API_KEY` in `.env`.
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
