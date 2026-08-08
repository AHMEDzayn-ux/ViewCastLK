import type {
  AccuracyResponse,
  ForecastRequest,
  ForecastResponse,
} from "@/types/forecast";
import {
  ACCURACY_SCOPES,
  PUBLISH_DAYS,
  RECOMMENDATION_TYPES,
} from "@/types/forecast";
import {
  createMockAccuracy,
  createMockForecast,
} from "@/lib/mock/forecast";

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim().replace(
  /\/$/,
  "",
);
const MOCK_MODE_REQUESTED =
  process.env.NEXT_PUBLIC_USE_MOCK_API?.trim().toLowerCase() === "true";
const USE_MOCK_API = MOCK_MODE_REQUESTED || !API_BASE_URL;
const EXPECTED_HORIZONS = [7, 14, 21, 30];

export class PredictionApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "PredictionApiError";
  }
}

function wait(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("The request was cancelled.", "AbortError"));
      return;
    }

    const timeoutId = window.setTimeout(resolve, milliseconds);

    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeoutId);
        reject(new DOMException("The request was cancelled.", "AbortError"));
      },
      { once: true },
    );
  });
}

function isForecastResponse(value: unknown): value is ForecastResponse {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<ForecastResponse>;
  if (!Array.isArray(candidate.estimates) || candidate.estimates.length !== 4) {
    return false;
  }

  const validEstimates = candidate.estimates.every((estimate, index) => {
    return (
      estimate.horizonDays === EXPECTED_HORIZONS[index] &&
      Number.isFinite(estimate.cumulativeViews) &&
      estimate.cumulativeViews >= 0
    );
  });

  const validRecommendations =
    Array.isArray(candidate.recommendations) &&
    candidate.recommendations.every((recommendation) => {
      const validEvidence =
        Array.isArray(recommendation.evidence) &&
        recommendation.evidence.length > 0 &&
        recommendation.evidence.every(
          (evidence) =>
            typeof evidence.label === "string" &&
            typeof evidence.detail === "string",
        );
      const validBase =
        RECOMMENDATION_TYPES.includes(recommendation.type) &&
        typeof recommendation.id === "string" &&
        typeof recommendation.title === "string" &&
        typeof recommendation.guidance === "string" &&
        validEvidence;

      if (!validBase || recommendation.type !== "timing") return validBase;

      const window = recommendation.recommendedPublishingWindow;
      return (
        PUBLISH_DAYS.includes(window.day) &&
        Number.isInteger(window.startHour) &&
        window.startHour >= 0 &&
        window.startHour <= 23 &&
        Number.isInteger(window.endHour) &&
        window.endHour >= 0 &&
        window.endHour <= 23 &&
        window.timeZone === "Asia/Colombo"
      );
    });

  const validUnavailableRecommendations =
    Array.isArray(candidate.unavailableRecommendations) &&
    candidate.unavailableRecommendations.every(
      (recommendation) =>
        RECOMMENDATION_TYPES.includes(recommendation.type) &&
        typeof recommendation.reason === "string" &&
        recommendation.reason.length > 0,
    );

  const representedRecommendationTypes = [
    ...(candidate.recommendations ?? []).map(
      (recommendation) => recommendation.type,
    ),
    ...(candidate.unavailableRecommendations ?? []).map(
      (recommendation) => recommendation.type,
    ),
  ];
  const validRecommendationCoverage =
    representedRecommendationTypes.length === RECOMMENDATION_TYPES.length &&
    new Set(representedRecommendationTypes).size ===
      RECOMMENDATION_TYPES.length &&
    RECOMMENDATION_TYPES.every((type) =>
      representedRecommendationTypes.includes(type),
    );

  return (
    typeof candidate.forecastId === "string" &&
    validEstimates &&
    validRecommendations &&
    validUnavailableRecommendations &&
    validRecommendationCoverage &&
    (candidate.completeness?.status === "complete" ||
      candidate.completeness?.status === "degraded") &&
    Array.isArray(candidate.completeness.issues) &&
    typeof candidate.model?.modelVersion === "string" &&
    typeof candidate.model.generatedAt === "string" &&
    (candidate.model.dataSource === "prediction_api" ||
      candidate.model.dataSource === "mock")
  );
}

function isAccuracyResponse(value: unknown): value is AccuracyResponse {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Partial<AccuracyResponse>;
  const validEvaluations =
    Array.isArray(candidate.evaluations) &&
    candidate.evaluations.length === ACCURACY_SCOPES.length &&
    candidate.evaluations.every(
      (evaluation) =>
        ACCURACY_SCOPES.includes(evaluation.scope) &&
        Array.isArray(evaluation.metrics) &&
        evaluation.metrics.some((metric) => metric.key === "mape") &&
        evaluation.metrics.every(
          (metric) =>
            typeof metric.label === "string" &&
            typeof metric.description === "string" &&
            (metric.modelValue === null ||
              Number.isFinite(metric.modelValue)) &&
            (metric.baselineValue === null ||
              Number.isFinite(metric.baselineValue)),
        ),
    );
  const evaluationScopes = (candidate.evaluations ?? []).map(
    (evaluation) => evaluation.scope,
  );

  return (
    (candidate.status === "available" || candidate.status === "unavailable") &&
    typeof candidate.modelName === "string" &&
    typeof candidate.baselineName === "string" &&
    validEvaluations &&
    new Set(evaluationScopes).size === ACCURACY_SCOPES.length &&
    ACCURACY_SCOPES.every((scope) => evaluationScopes.includes(scope))
  );
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  if (!API_BASE_URL) {
    throw new PredictionApiError(
      "The Prediction API URL has not been configured.",
      undefined,
      "api_not_configured",
    );
  }

  const response = await fetch(API_BASE_URL + path, {
    ...init,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  const payload = (await response.json().catch(() => null)) as
    | { message?: string; code?: string }
    | T
    | null;

  if (!response.ok) {
    const errorPayload = payload as { message?: string; code?: string } | null;
    throw new PredictionApiError(
      errorPayload?.message ??
        "The Prediction API could not complete the request.",
      response.status,
      errorPayload?.code,
    );
  }

  return payload as T;
}

export function isDevelopmentMockMode(): boolean {
  return USE_MOCK_API;
}

export async function generateForecast(
  request: ForecastRequest,
  options?: { signal?: AbortSignal },
): Promise<ForecastResponse> {
  if (USE_MOCK_API) {
    await wait(900, options?.signal);
    return createMockForecast(request);
  }

  const response = await requestJson<ForecastResponse>("/forecast", {
    method: "POST",
    body: JSON.stringify(request),
    signal: options?.signal,
  });

  if (!isForecastResponse(response)) {
    throw new PredictionApiError(
      "The Prediction API returned an unsupported forecast response.",
      undefined,
      "invalid_response",
    );
  }

  return response;
}

export async function getAccuracy(options?: {
  signal?: AbortSignal;
}): Promise<AccuracyResponse> {
  if (USE_MOCK_API) {
    await wait(300, options?.signal);
    return createMockAccuracy();
  }

  const response = await requestJson<AccuracyResponse>("/accuracy", {
    method: "GET",
    signal: options?.signal,
  });

  if (!isAccuracyResponse(response)) {
    throw new PredictionApiError(
      "The Prediction API returned unsupported accuracy information.",
      undefined,
      "invalid_response",
    );
  }

  return response;
}
