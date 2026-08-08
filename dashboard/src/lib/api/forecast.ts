import type {
  AccuracyResponse,
  ForecastRequest,
  ForecastResponse,
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

  return (
    typeof candidate.forecastId === "string" &&
    validEstimates &&
    Array.isArray(candidate.recommendations) &&
    candidate.recommendations.every(
      (recommendation) =>
        typeof recommendation.id === "string" &&
        typeof recommendation.title === "string" &&
        typeof recommendation.guidance === "string" &&
        Array.isArray(recommendation.evidence),
    ) &&
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
  return (
    (candidate.status === "available" || candidate.status === "unavailable") &&
    typeof candidate.modelName === "string" &&
    typeof candidate.baselineName === "string" &&
    Array.isArray(candidate.metrics) &&
    candidate.metrics.some((metric) => metric.key === "mape") &&
    candidate.metrics.every(
      (metric) =>
        typeof metric.label === "string" &&
        typeof metric.description === "string" &&
        (metric.modelValue === null || Number.isFinite(metric.modelValue)) &&
        (metric.baselineValue === null || Number.isFinite(metric.baselineValue)),
    )
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
