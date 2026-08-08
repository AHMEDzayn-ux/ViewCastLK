import type {
  AccuracyResponse,
  ForecastEstimate,
  ForecastRequest,
  ForecastResponse,
  Recommendation,
} from "@/types/forecast";

function hashString(value: string): number {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

function createRandom(seed: number): () => number {
  let state = seed;

  return () => {
    state = Math.imul(state, 1664525) + 1013904223;
    return (state >>> 0) / 4294967296;
  };
}

function makeEstimates(request: ForecastRequest, random: () => number) {
  const durationMinutes = request.durationSeconds / 60;
  const durationAdjustment =
    durationMinutes < 3 ? 0.9 : durationMinutes <= 20 ? 1.06 : 0.96;
  const categoryAdjustment =
    0.85 + (hashString(request.category) % 35) / 100;
  const baseDaySeven = Math.round(
    (3600 + random() * 5200) * durationAdjustment * categoryAdjustment,
  );
  const growth = [1, 1.48, 1.78, 2.04] as const;
  let previousCumulativeViews = 0;

  return growth.map((multiplier, index) => {
    const cumulativeViews = Math.max(
      previousCumulativeViews,
      Math.round(baseDaySeven * multiplier * (0.97 + random() * 0.06)),
    );
    previousCumulativeViews = cumulativeViews;

    return {
      horizonDays: [7, 14, 21, 30][index] as 7 | 14 | 21 | 30,
      cumulativeViews,
    };
  }) as [
    ForecastEstimate,
    ForecastEstimate,
    ForecastEstimate,
    ForecastEstimate,
  ];
}

function makeRecommendations(request: ForecastRequest): Recommendation[] {
  const timingKnown =
    request.plannedPublishDay !== null &&
    request.plannedPublishHour !== null;
  const durationMinutes = Math.round(request.durationSeconds / 60);

  return [
    {
      id: "mock-timing",
      type: "timing",
      title: timingKnown
        ? "Compare the planned publishing window"
        : "Publishing time remains flexible",
      guidance: timingKnown
        ? "Historically, similar " +
          request.category +
          " videos may perform differently across publishing windows. Review " +
          request.plannedPublishDay +
          " at " +
          String(request.plannedPublishHour).padStart(2, "0") +
          ":00 SLT alongside the production evidence when it becomes available."
        : "No publishing day or hour was supplied, so timing is treated as unknown rather than assumed.",
      evidence: [
        {
          label: "Development evidence",
          detail:
            "This mock adapter demonstrates the evidence format. The Prediction API will supply the historical category comparison.",
        },
      ],
    },
    {
      id: "mock-duration",
      type: "duration",
      title: "Review the planned duration",
      guidance:
        "Historically, duration can be associated with different viewing patterns. Compare the planned " +
        durationMinutes +
        "-minute format with similar videos before publishing.",
      evidence: [
        {
          label: "Submitted plan",
          detail: durationMinutes + " minutes in " + request.category + ".",
        },
        {
          label: "Development evidence",
          detail:
            "A production response will replace this note with category-specific historical ranges.",
        },
      ],
    },
    {
      id: "mock-title",
      type: "title",
      title: "Keep the title clear and specific",
      guidance:
        "Use accurate wording that helps viewers understand the video. Avoid misleading or provocative framing.",
      evidence: [
        {
          label: "Title supplied",
          detail: request.title,
        },
        {
          label: "Development evidence",
          detail:
            "Hosted title analysis will provide neutral, constructive evidence through the Prediction API.",
        },
      ],
    },
  ];
}

export function createMockForecast(request: ForecastRequest): ForecastResponse {
  const seed = hashString(
    [
      request.title,
      request.category,
      request.durationSeconds,
      request.audioLanguage,
      request.madeForKids,
      request.channelIdentifier,
      request.plannedPublishDay ?? "unknown-day",
      request.plannedPublishHour ?? "unknown-hour",
    ].join("|"),
  );
  const random = createRandom(seed);
  const channelLookupUnavailable = request.channelIdentifier
    .toLowerCase()
    .includes("degraded-demo");

  return {
    forecastId: "mock-" + seed.toString(16),
    estimates: makeEstimates(request, random),
    recommendations: makeRecommendations(request),
    completeness: channelLookupUnavailable
      ? {
          status: "degraded",
          issues: [
            {
              source: "channel_lookup",
              message:
                "Channel statistics were unavailable, so the forecast used the remaining valid inputs.",
            },
          ],
        }
      : { status: "complete", issues: [] },
    titleGuidance: {
      summary:
        "The development adapter keeps title guidance neutral and descriptive.",
      suggestions: [
        "Describe the video accurately.",
        "Keep the most useful context near the beginning.",
        "Avoid claims the video does not support.",
      ],
    },
    model: {
      modelVersion: "development-mock-v1",
      generatedAt: new Date().toISOString(),
      dataSource: "mock",
    },
  };
}

export function createMockAccuracy(): AccuracyResponse {
  return {
    status: "unavailable",
    modelName: "Published ViewCastLK model",
    baselineName: "Naive category baseline",
    evaluatedAt: null,
    dataSource: "mock",
    message:
      "Published evaluation metrics are not available yet. No development values are substituted.",
    metrics: [
      {
        key: "mape",
        label: "MAPE",
        description:
          "Average prediction error expressed as a percentage of actual views.",
        unit: "percent",
        modelValue: null,
        baselineValue: null,
      },
      {
        key: "mae",
        label: "MAE",
        description:
          "Average absolute difference between predicted and actual view counts.",
        unit: "views",
        modelValue: null,
        baselineValue: null,
      },
      {
        key: "rmse",
        label: "RMSE",
        description:
          "A view-count error measure that gives more weight to larger misses.",
        unit: "views",
        modelValue: null,
        baselineValue: null,
      },
      {
        key: "r2",
        label: "R²",
        description:
          "How much of the variation in actual view counts the model explains.",
        unit: "score",
        modelValue: null,
        baselineValue: null,
      },
    ],
  };
}
