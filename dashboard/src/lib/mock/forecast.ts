import type {
  AccuracyResponse,
  ChannelStats,
  ForecastEstimate,
  ForecastRequest,
  ForecastResponse,
  Recommendation,
  UnavailableRecommendation,
} from "@/types/forecast";
import { ACCURACY_SCOPES, PUBLISH_DAYS } from "@/types/forecast";

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

function makeRecommendations(request: ForecastRequest): {
  recommendations: Recommendation[];
  unavailableRecommendations: UnavailableRecommendation[];
} {
  const durationMinutes = Math.round(request.durationSeconds / 60);
  const timingSeed = hashString(
    request.category + "|" + request.channelIdentifier,
  );
  const recommendedDay = PUBLISH_DAYS[timingSeed % PUBLISH_DAYS.length];
  const recommendedStartHours = [10, 12, 16, 18, 20] as const;
  const recommendedStartHour =
    recommendedStartHours[timingSeed % recommendedStartHours.length];
  const recommendedEndHour = recommendedStartHour + 2;

  return {
    recommendations: [
      {
        id: "mock-timing",
        type: "timing",
        title: "Recommended publishing window",
        guidance:
          "Historical publishing patterns can differ across time windows. Consider this window when the production evidence supports the same association.",
        recommendedPublishingWindow: {
          day: recommendedDay,
          startHour: recommendedStartHour,
          endHour: recommendedEndHour,
          timeZone: "Asia/Colombo",
        },
        evidence: [
          {
            label: "Historical timing evidence — development placeholder",
            detail:
              "A production response will provide the observed comparison between this window and alternative publishing times.",
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
            label: "Historical duration evidence — development placeholder",
            detail:
              "A production response will provide the relevant category comparison.",
          },
        ],
      },
      {
        id: "mock-title",
        type: "title",
        title: "Keep the title clear and specific",
        guidance:
          "Historically associated title patterns can support planning. Use accurate wording and avoid misleading or provocative framing.",
        evidence: [
          {
            label: "Title supplied",
            detail: request.title,
          },
          {
            label: "Historical title evidence — development placeholder",
            detail:
              "A production response will provide neutral, constructive title evidence.",
          },
        ],
      },
    ],
    unavailableRecommendations: [
      {
        type: "format",
        reason:
          "Format guidance is unavailable because supporting evaluation evidence is not available for this forecast.",
      },
    ],
  };
}

export function createMockForecast(request: ForecastRequest): ForecastResponse {
  const seed = hashString(
    [
      request.title,
      request.category,
      request.durationSeconds,
      request.audioLanguage,
      request.channelIdentifier,
      request.plannedPublishDay ?? "unknown-day",
      request.plannedPublishHour ?? "unknown-hour",
    ].join("|"),
  );
  const random = createRandom(seed);
  const recommendationResult = makeRecommendations(request);
  const channelLookupUnavailable = request.channelIdentifier
    .toLowerCase()
    .includes("degraded-demo");

  return {
    forecastId: "mock-" + seed.toString(16),
    estimates: makeEstimates(request, random),
    recommendations: recommendationResult.recommendations,
    unavailableRecommendations:
      recommendationResult.unavailableRecommendations,
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
  const createUnpublishedMetrics = () => [
    {
      key: "mape" as const,
      label: "MAPE",
      description:
        "Average prediction error expressed as a percentage of actual views.",
      unit: "percent" as const,
      modelValue: null,
      baselineValue: null,
    },
    {
      key: "mae" as const,
      label: "MAE",
      description:
        "Average absolute difference between predicted and actual view counts.",
      unit: "views" as const,
      modelValue: null,
      baselineValue: null,
    },
    {
      key: "rmse" as const,
      label: "RMSE",
      description:
        "A view-count error measure that gives more weight to larger misses.",
      unit: "views" as const,
      modelValue: null,
      baselineValue: null,
    },
    {
      key: "r2" as const,
      label: "R²",
      description:
        "How much of the variation in actual view counts the model explains.",
      unit: "score" as const,
      modelValue: null,
      baselineValue: null,
    },
  ];

  return {
    status: "unavailable",
    modelName: "Published ViewCastLK model",
    baselineName: "Naive category baseline",
    evaluatedAt: null,
    dataSource: "mock",
    message:
      "Published evaluation metrics are not available yet. No development values are substituted.",
    evaluations: ACCURACY_SCOPES.map((scope) => ({
      scope,
      metrics: createUnpublishedMetrics(),
    })) as AccuracyResponse["evaluations"],
  };
}

export function createMockChannelStats(
  channelIdentifier: string,
): ChannelStats {
  const seed = hashString(channelIdentifier.toLowerCase());
  const random = createRandom(seed);

  return {
    subscriberCount: Math.round(1_000 + random() * 450_000),
    totalViewCount: Math.round(50_000 + random() * 25_000_000),
    videoCount: Math.round(15 + random() * 600),
    createdAt: new Date(
      Date.now() - Math.round((180 + random() * 3000) * 86_400_000),
    ).toISOString(),
  };
}
