# ViewCastLK Prediction API

Minimal Python server-side backend providing real YouTube channel analytics for the ViewCastLK creator dashboard.

## Features

- `GET /health` — API health check
- `POST /channel-lookup` — Server-side YouTube channel lookup by `@handle`, channel ID (`UC...`), or YouTube URL

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
