// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import ForecastResults from "./ForecastResults";

const chartMock = vi.hoisted(() => vi.fn());
vi.mock("./ForecastChart", () => ({ default: (props: unknown) => { chartMock(props); return <div>Forecast chart</div>; } }));
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

afterEach(() => { cleanup(); vi.clearAllMocks(); });

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
    expect(screen.getByText("Channel-adjusted")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Details" }));
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
    expect(screen.getByText("Shared forecast")).toBeTruthy();
  });

  it("explains a connected creator's shared result without implying an adjustment", () => {
    render(<ForecastResults request={request} response={baseResponse} channelConnected onChangeInputs={vi.fn()} />);
    expect(screen.getByText("Connected channel · no personal adjustment applied")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Details" }));
    expect(screen.getByText(/Your channel is connected, but this forecast used the shared model/)).toBeTruthy();
    expect(screen.queryByText("Personalised forecast")).toBeNull();
  });

  it("shows the v8 breakout probability and conditional upside", () => {
    render(
      <ForecastResults
        request={request}
        response={{
          ...baseResponse,
          breakout: {
            probability: 0.146,
            definition: "A channel-relative breakout definition.",
            conditionalUpside: [
              { horizonDays: 7, cumulativeViews: 1_000 },
              { horizonDays: 14, cumulativeViews: 1_400 },
              { horizonDays: 21, cumulativeViews: 1_700 },
              { horizonDays: 30, cumulativeViews: 2_000 },
            ],
          },
        }}
        onChangeInputs={vi.fn()}
      />,
    );

    expect(chartMock).toHaveBeenLastCalledWith({ estimates: baseResponse.estimates });
    expect(screen.getByRole("tabpanel").getAttribute("aria-labelledby")).toMatch(/tab-overview$/);
    fireEvent.click(screen.getByRole("tab", { name: /Breakout scenario/ }));
    expect(chartMock).toHaveBeenLastCalledWith({ estimates: expect.arrayContaining([{ horizonDays: 30, cumulativeViews: 2_000 }]), scenario: "breakout" });
    expect(screen.getByRole("heading", { name: "Breakout potential" })).toBeTruthy();
    expect(screen.getByText("14.6%")).toBeTruthy();
    expect(screen.getByText("2,000")).toBeTruthy();
  });

  it("supports arrow, Home, and End keys without offering an absent breakout", () => {
    render(<ForecastResults request={request} response={baseResponse} onChangeInputs={vi.fn()} />);
    expect(screen.queryByRole("tab", { name: /Breakout scenario/ })).toBeNull();
    const overview = screen.getByRole("tab", { name: "Overview" });
    overview.focus();
    fireEvent.keyDown(overview, { key: "ArrowRight" });
    const guidance = screen.getByRole("tab", { name: "Guidance" });
    expect(document.activeElement).toBe(guidance);
    expect(guidance.getAttribute("aria-selected")).toBe("true");
    fireEvent.keyDown(guidance, { key: "End" });
    const details = screen.getByRole("tab", { name: "Details" });
    expect(document.activeElement).toBe(details);
    fireEvent.keyDown(details, { key: "Home" });
    expect(document.activeElement).toBe(overview);
    fireEvent.keyDown(overview, { key: "ArrowLeft" });
    expect(document.activeElement).toBe(details);
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
  });

  it("identifies historical guidance as separate from the forecast model", () => {
    render(
      <ForecastResults
        request={request}
        response={{
          ...baseResponse,
          guidance: {
            artifactVersion: "idea_optimization_20261001_v1",
            source: "historical_eda",
            isolatedFromForecast: true,
            associationWarning: "Historical associations do not prove causation.",
          },
        }}
        onChangeInputs={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "Details" }));
    expect(screen.getByText("idea_optimization_20261001_v1")).toBeTruthy();
    expect(screen.getByText(/Guidance does not change the forecast/)).toBeTruthy();
  });

  it("keeps missing guidance concise and exposes limited-context issues", () => {
    render(<ForecastResults request={request} response={{ ...baseResponse, unavailableRecommendations: [{ type: "timing", reason: "No supporting evaluation." }], completeness: { status: "degraded", issues: [{ source: "title_analysis", message: "Title analysis unavailable." }] } }} onChangeInputs={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "Guidance" }));
    expect(screen.getByRole("heading", { name: "No evidence-backed changes are suggested." })).toBeTruthy();
    expect(screen.getByText("Why some publishing guidance is unavailable").parentElement?.hasAttribute("open")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: /Limited context/ }));
    expect(screen.getByRole("tab", { name: "Details" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("heading", { name: "Some supporting information was unavailable" })).toBeTruthy();
  });
});
