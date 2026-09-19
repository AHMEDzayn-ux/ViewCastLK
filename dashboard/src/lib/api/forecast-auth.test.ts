import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ForecastRequest, ForecastResponse } from "@/types/forecast";

const mocks = vi.hoisted(() => ({
  getSession: vi.fn(),
}));

vi.mock("@/lib/supabase/client", () => ({
  supabase: { auth: { getSession: mocks.getSession } },
}));

const request: ForecastRequest = {
  title: "Protected forecast",
  category: "Education",
  durationSeconds: 300,
  audioLanguage: "English",
  channelIdentifier: "@creator",
  plannedPublishDay: null,
  plannedPublishHour: null,
};

const response: ForecastResponse = {
  forecastId: "forecast-1",
  estimates: [
    { horizonDays: 7, cumulativeViews: 10 },
    { horizonDays: 14, cumulativeViews: 20 },
    { horizonDays: 21, cumulativeViews: 30 },
    { horizonDays: 30, cumulativeViews: 40 },
  ],
  recommendations: [],
  unavailableRecommendations: [
    { type: "timing", reason: "Unavailable" },
    { type: "duration", reason: "Unavailable" },
    { type: "format", reason: "Unavailable" },
    { type: "title", reason: "Unavailable" },
  ],
  completeness: { status: "complete", issues: [] },
  model: {
    modelVersion: "model-v1",
    generatedAt: "2026-09-20T08:00:00.000Z",
    dataSource: "prediction_api",
  },
};

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.stubEnv("NEXT_PUBLIC_PREDICTION_API_URL", "https://api.example.test");
  vi.stubEnv("NEXT_PUBLIC_USE_MOCK_API", "false");
  mocks.getSession.mockResolvedValue({
    data: { session: { access_token: "test-access-token" } },
    error: null,
  });
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
});

describe("authenticated Prediction API requests", () => {
  it("reads the current session at request time and sends its bearer token", async () => {
    const { generateForecast } = await import("./forecast");

    await expect(generateForecast(request)).resolves.toEqual(response);
    expect(mocks.getSession).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      "https://api.example.test/forecast",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer test-access-token",
        }),
      }),
    );
  });

  it("does not call the protected API when no authenticated session exists", async () => {
    mocks.getSession.mockResolvedValueOnce({
      data: { session: null },
      error: null,
    });
    const { generateForecast } = await import("./forecast");

    await expect(generateForecast(request)).rejects.toMatchObject({
      status: 401,
      code: "session_required",
      message: "Sign in again to continue forecasting.",
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("turns a backend 401 into a safe session-expired error", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          message: "provider-internal-detail",
          code: "private-code",
        }),
        { status: 401, headers: { "Content-Type": "application/json" } },
      ),
    );
    const { generateForecast } = await import("./forecast");

    await expect(generateForecast(request)).rejects.toMatchObject({
      status: 401,
      code: "invalid_session",
      message: "Your session has expired. Sign in again to continue.",
    });
  });
});
