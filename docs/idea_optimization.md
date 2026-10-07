# Idea optimization

ViewCastLK's planning guidance is intentionally separate from its view and
breakout prediction models.

## Responsibilities

- The v10 model predicts cumulative views at Days 7, 14, 21 and 30.
- The breakout model estimates the probability and conditional trajectory of
  a channel-relative breakout.
- The idea optimizer compares the submitted publishing plan with a versioned
  historical EDA artifact. It cannot modify either prediction.
- Gemini title analysis provides optional language guidance. It does not claim
  a measured view uplift.

## Evidence source

`idea_optimization_20261001_v1` packages the actionable timing, duration and
format comparisons from the October insights release. Effects are differences
in Day-7 performance relative to each video's own channel normal. Uncertainty
is estimated with a channel-level bootstrap.

The artifact records the training-table and source-insights SHA-256 checksums,
dataset period, usable video count, channel count and release policy.

## Release rules

A comparison must contain at least 300 videos from at least 20 channels. A
suggested alternative must exceed the current option by at least five
percentage points, and its 95% interval must sit above the current option's 95%
interval. Format advice uses a direct Shorts-versus-standard-video contrast
whose interval must exclude zero.

When a rule does not pass, the API explains that no evidence-backed change is
available instead of manufacturing a suggestion.

## API contract

`POST /forecast` returns supported items in `recommendations`, unsupported or
inapplicable comparisons in `unavailableRecommendations`, and provenance in
`guidance`. The provenance contains `isolatedFromForecast: true` so clients can
present the distinction clearly.

Every recommendation includes the evaluated effect, 95% range, video count and
channel count. These are historical associations, not causal promises.

## Rebuilding

After regenerating `dashboard/src/data/insights.json`, run:

```powershell
python scripts/export_idea_optimization_artifact.py
```

Review and test the resulting artifact before changing its released version.
