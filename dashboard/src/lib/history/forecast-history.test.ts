import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import {
  deleteForecastHistory,
  listForecastHistory,
  saveForecastHistory,
  toForecastHistoryInsert,
} from "./forecast-history";

const mocks = vi.hoisted(() => ({
  delete: vi.fn(),
  eq: vi.fn(),
  from: vi.fn(),
  insert: vi.fn(),
  order: vi.fn(),
  select: vi.fn(),
}));

vi.mock("@/lib/supabase/client", () => ({
  supabase: { from: mocks.from },
}));

const request: ForecastRequest = {
  title: "A planned video",
  category: "Education",
  durationSeconds: 615,
  isShort: false,
  audioLanguage: "Sinhala",
  channelIdentifier: "@creator",
  plannedPublishDay: "Friday",
  plannedPublishHour: 18,
};

const response: ForecastResponse = {
  forecastId: "forecast-123",
  estimates: [
    { horizonDays: 7, cumulativeViews: 100 },
    { horizonDays: 14, cumulativeViews: 200 },
    { horizonDays: 21, cumulativeViews: 300 },
    { horizonDays: 30, cumulativeViews: 400 },
  ],
  recommendations: [],
  unavailableRecommendations: [],
  completeness: {
    status: "degraded",
    issues: [{ source: "channel_lookup", message: "Unavailable" }],
  },
  model: {
    modelVersion: "model-v1",
    generatedAt: "2026-09-20T08:00:00.000Z",
    dataSource: "prediction_api",
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.insert.mockResolvedValue({ error: null });
  mocks.order.mockResolvedValue({ data: [], error: null });
  mocks.eq.mockResolvedValue({ error: null });
  mocks.select.mockReturnValue({ order: mocks.order });
  mocks.delete.mockReturnValue({ eq: mocks.eq });
  mocks.from.mockReturnValue({
    insert: mocks.insert,
    select: mocks.select,
    delete: mocks.delete,
  });
});

describe("forecast history data access", () => {
  it("maps the authenticated user and real request/response fields", () => {
    expect(toForecastHistoryInsert("authenticated-user", request, response)).toEqual({
      user_id: "authenticated-user",
      forecast_id: "forecast-123",
      title: "A planned video",
      category: "Education",
      duration_seconds: 615,
      audio_language: "Sinhala",
      channel_identifier: "@creator",
      planned_publish_day: "Friday",
      planned_publish_hour: 18,
      day_7_cumulative_views: 100,
      day_14_cumulative_views: 200,
      day_21_cumulative_views: 300,
      day_30_cumulative_views: 400,
      model_version: "model-v1",
      forecast_generated_at: "2026-09-20T08:00:00.000Z",
      model_data_source: "prediction_api",
      completeness_status: "degraded",
      completeness_issues: [
        { source: "channel_lookup", message: "Unavailable" },
      ],
    });
  });

  it("inserts one mapped record for one successful save", async () => {
    await saveForecastHistory("authenticated-user", request, response);

    expect(mocks.from).toHaveBeenCalledWith("forecast_history");
    expect(mocks.insert).toHaveBeenCalledTimes(1);
    expect(mocks.insert.mock.calls[0][0]).toMatchObject({
      user_id: "authenticated-user",
      forecast_id: "forecast-123",
      day_30_cumulative_views: 400,
    });
  });

  it("lists newest first without accepting a user ID", async () => {
    await listForecastHistory();

    expect(mocks.order).toHaveBeenCalledWith("created_at", {
      ascending: false,
    });
  });

  it("deletes only by the internal record ID and exposes no update operation", async () => {
    await deleteForecastHistory("history-id");

    expect(mocks.delete).toHaveBeenCalledTimes(1);
    expect(mocks.eq).toHaveBeenCalledWith("id", "history-id");
  });
});
