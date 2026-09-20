// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import ForecastResults from "./ForecastResults";

vi.mock("./ForecastChart", () => ({ default: () => <div>Forecast chart</div> }));
vi.mock("./HorizonCards", () => ({ default: () => <div>Horizon cards</div> }));
vi.mock("./RecommendationCards", () => ({
  default: () => <div>Recommendations</div>,
}));

const request: ForecastRequest = {
  title: "Creator forecast",
  category: "Education",
  durationSeconds: 45,
  isShort: false,
  audioLanguage: "English",
  channelIdentifier: "@creator",
  plannedPublishDay: null,
  plannedPublishHour: null,
};

const baseResponse: ForecastResponse = {
  forecastId: "forecast-1",
  estimates: [
    { horizonDays: 7, cumulativeViews: 125 },
    { horizonDays: 14, cumulativeViews: 250 },
    { horizonDays: 21, cumulativeViews: 375 },
    { horizonDays: 30, cumulativeViews: 500 },
  ],
  recommendations: [],
  unavailableRecommendations: [],
  completeness: { status: "complete", issues: [] },
  model: {
    modelVersion: "model-v1",
    generatedAt: "2026-09-20T08:00:00.000Z",
    dataSource: "prediction_api",
  },
};

afterEach(cleanup);

describe("ForecastResults personalization", () => {
  it("labels a personalized forecast and exposes the shared comparison", () => {
    render(
      <ForecastResults
        request={request}
        response={{
          ...baseResponse,
          personalization: {
            applied: true,
            format: "short",
            modelVersion: "model-v1",
            sharedEstimates: [
              { horizonDays: 7, cumulativeViews: 100 },
              { horizonDays: 14, cumulativeViews: 200 },
              { horizonDays: 21, cumulativeViews: 300 },
              { horizonDays: 30, cumulativeViews: 400 },
            ],
            adjustments: [
              { horizonDays: 7, format: "short", factor: 1.25, nVideos: 8 },
            ],
          },
        }}
        onChangeInputs={vi.fn()}
      />,
    );

    expect(screen.getByText("Personalised using your channel history")).toBeTruthy();
    expect(screen.getByText("Personalised forecast")).toBeTruthy();
    expect(screen.getByText("Day 7: 100 shared views")).toBeTruthy();
  });

  it("does not label an unchanged shared forecast as personalized", () => {
    render(
      <ForecastResults
        request={request}
        response={{
          ...baseResponse,
          personalization: {
            applied: false,
            format: "short",
            modelVersion: "model-v1",
            sharedEstimates: baseResponse.estimates,
            adjustments: [],
          },
        }}
        onChangeInputs={vi.fn()}
      />,
    );

    expect(screen.queryByText("Personalised forecast")).toBeNull();
    expect(screen.queryByText("Compare with the shared model forecast")).toBeNull();
  });
});
