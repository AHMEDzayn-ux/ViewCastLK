# ViewCastLK Dashboard

ViewCastLK is a pre-publication forecasting tool for Sri Lankan YouTube
creators. The dashboard lets a creator:

- describe a planned video;
- receive cumulative view forecasts for Day 7, 14, 21, and 30;
- review evidence-backed publishing guidance;
- view published model accuracy; and
- read the methodology and limitations.

The dashboard is the presentation layer. It communicates with the Prediction
API abstraction and does not perform data collection or model execution in the
browser.

## Tech Stack

Versions below match `package.json`:

| Technology | Version / usage |
| --- | --- |
| Next.js | `16.2.12` |
| React / React DOM | `19.2.4` |
| TypeScript | `^5` |
| Tailwind CSS | `^4` |
| Recharts | `^3.10.1` |
| Routing | Next.js App Router |

## Dashboard Routes

- `/forecast` — forecast request form, cumulative forecast result, trajectory,
  and recommendations.
- `/accuracy` — combined model evaluation plus Day 7, 14, 21, and 30 views,
  each comparing the published ViewCastLK model with the naive category
  baseline.
- `/methodology` — creator-friendly methodology and limitations.
- `/about` — redirects to `/methodology`.

## Forecast Inputs

Required inputs:

- video title;
- video category;
- planned duration;
- audio language;
- made-for-kids setting; and
- YouTube channel URL, handle, or ID.

Publishing day and publishing hour are optional. If either is omitted, its
request value remains unknown (`null`) and is not replaced with a default.

Draft form values are stored in `sessionStorage`, so they remain available when
the creator visits another dashboard route and returns during the same browser
session. Clearing the form also clears the stored draft.

## Forecast Output

The dashboard displays cumulative point forecasts for:

- Day 7;
- Day 14;
- Day 21; and
- Day 30.

Prediction intervals, lower/median/upper bounds, confidence intervals, and
confidence bands are **not** part of the current release.

## Recommendations

The UI supports evidence-backed guidance for:

- publishing day and time slot;
- duration;
- format; and
- title framing.

Each returned recommendation includes supporting historical evidence. A
recommendation type can be unavailable when supporting evaluation or ablation
evidence does not justify displaying it. Recommendations describe historical
associations and are not causal guarantees.

## Development / Mock Mode

The isolated development adapter is used while the real Prediction API is
unavailable. Copy `.env.local.example` to `.env.local` and configure only these
public variables:

```dotenv
NEXT_PUBLIC_USE_MOCK_API=true
NEXT_PUBLIC_PREDICTION_API_URL=
```

- `NEXT_PUBLIC_USE_MOCK_API=true` explicitly enables mock mode.
- If no Prediction API URL is configured, the dashboard also falls back to mock
  mode.
- A configured `NEXT_PUBLIC_PREDICTION_API_URL` with mock mode set to `false`
  (or omitted) uses the production API configuration and hides the development
  banner.
- Mock forecasts are illustrative.
- Mock accuracy numbers are not invented. Combined and per-horizon values stay
  **Not published** until approved evaluation results are supplied.

Never put private credentials or real service secrets in `NEXT_PUBLIC_`
variables; they are included in browser JavaScript.

## Prediction API Integration

Browser code communicates only through `src/lib/api/forecast.ts`. It must not
directly access Supabase, the YouTube Data API, Gemini, or model artefacts.

The frontend currently expects these API contracts:

- `POST /forecast`
- `GET /accuracy`

These are expected frontend contracts while the real Prediction API is still
being implemented. Request and response interfaces are defined in
`src/types/forecast.ts`; mock response generation remains isolated in
`src/lib/mock/forecast.ts`.

## Local Development

From the repository root:

```bash
cd dashboard
npm install
npm run dev
```

Open <http://localhost:3000>.

Run checks before handing off changes:

```bash
npm run lint
npm run build
```

On Windows systems where PowerShell blocks `npm.ps1`, use `npm.cmd run dev`,
`npm.cmd run lint`, and `npm.cmd run build`.

## Accessibility and Language Support

The dashboard provides responsive desktop/mobile layouts, keyboard-accessible
native form controls, labelled inputs, visible focus states, and reduced-motion
support. Titles can use Sinhala, Tamil, English, or mixed scripts. WCAG 2.1 AA
guidance is the accessibility target; certified conformance is not claimed.

## Limitations

- Forecasts are estimates, not guarantees.
- No early engagement from the planned video is used.
- Prediction intervals are not included.
- Private YouTube Analytics data is not used.
- Thumbnail or image features are not used.
- Revenue and subscriber growth are not predicted.
- Public data does not identify individual viewer locations.
- Verified Sri Lankan channels are a practical proxy for Sri Lankan content
  and audience patterns; this does not mean every viewer is in Sri Lanka.
- Real forecasts depend on the future Prediction API and a published model.
