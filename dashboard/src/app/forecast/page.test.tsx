// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import ForecastPage from "./page";

const request: ForecastRequest = {
  title: "Authenticated forecast",
  category: "Education",
  durationSeconds: 300,
  isShort: false,
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
  unavailableRecommendations: [],
  completeness: { status: "complete", issues: [] },
  model: {
    modelVersion: "model-v1",
    generatedAt: "2026-09-20T08:00:00.000Z",
    dataSource: "prediction_api",
  },
};

const mocks = vi.hoisted(() => ({
  generateForecast: vi.fn(),
  saveForecastHistory: vi.fn(),
  user: { id: "authenticated-user", email: "creator@example.com" } as {
    id: string;
    email: string;
  } | null,
  isLoading: false,
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    isAuthenticated: Boolean(mocks.user),
    isLoading: mocks.isLoading,
    user: mocks.user,
  }),
}));

vi.mock("@/lib/api/forecast", () => ({
  generateForecast: mocks.generateForecast,
}));

vi.mock("@/lib/history/forecast-history", () => ({
  saveForecastHistory: mocks.saveForecastHistory,
}));

vi.mock("@/components/dashboard/ForecastForm", () => ({
  default: ({
    onSubmit,
    canLookupChannel,
  }: {
    onSubmit: (value: ForecastRequest) => void;
    canLookupChannel: boolean;
  }) => (
    <>
      <button type="button" onClick={() => onSubmit(request)}>Run forecast</button>
      {canLookupChannel && <button type="button">Retrieve details</button>}
    </>
  ),
}));

vi.mock("@/components/dashboard/ForecastResults", () => ({
  default: ({
    response: result,
    historySaveNotice,
  }: {
    response: ForecastResponse;
    historySaveNotice?: string;
  }) => (
    <div>
      <span>Result {result.forecastId}</span>
      {historySaveNotice && <span>{historySaveNotice}</span>}
    </div>
  ),
}));

vi.mock("@/components/dashboard/LoadingState", () => ({
  default: () => <span>Loading</span>,
}));

vi.mock("@/components/dashboard/ErrorState", () => ({
  default: ({ message }: { message: string }) => <span>{message}</span>,
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.user = { id: "authenticated-user", email: "creator@example.com" };
  mocks.isLoading = false;
  mocks.generateForecast.mockResolvedValue(response);
  mocks.saveForecastHistory.mockResolvedValue(undefined);
});

afterEach(() => cleanup());

describe("ForecastPage history saving", () => {
  it("lets a guest open and submit the forecast without saving history", async () => {
    mocks.user = null;
    render(<ForecastPage />);

    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Retrieve details" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));

    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(mocks.generateForecast).toHaveBeenCalledWith(request);
    expect(mocks.saveForecastHistory).not.toHaveBeenCalled();
  });

  it("saves one successful forecast with the authenticated user ID", async () => {
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));

    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(mocks.saveForecastHistory).toHaveBeenCalledTimes(1);
    expect(mocks.saveForecastHistory).toHaveBeenCalledWith(
      "authenticated-user",
      request,
      response,
    );
  });

  it("preserves the successful result and shows a safe notice when saving fails", async () => {
    mocks.saveForecastHistory.mockRejectedValueOnce(
      new Error("private database detail"),
    );
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));

    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(
      screen.getByText(
        "Forecast generated, but it could not be saved to your history.",
      ),
    ).toBeTruthy();
    expect(screen.queryByText("private database detail")).toBeNull();
    await waitFor(() => expect(mocks.saveForecastHistory).toHaveBeenCalledTimes(1));
  });
});
