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
  getYouTubeConnection: vi.fn(),
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

vi.mock("@/lib/api/youtube-connection", () => ({
  getYouTubeConnection: mocks.getYouTubeConnection,
}));

vi.mock("@/components/dashboard/ForecastForm", () => ({
  default: ({
    onSubmit,
    canLookupChannel,
    connectedChannel,
  }: {
    onSubmit: (value: ForecastRequest) => void;
    canLookupChannel: boolean;
    connectedChannel: { id: string; title: string | null } | null;
  }) => (
    <>
      <button type="button" onClick={() => onSubmit(request)}>Run forecast</button>
      {canLookupChannel && <button type="button">Retrieve details</button>}
      {connectedChannel && <span>Using {connectedChannel.title} ({connectedChannel.id})</span>}
    </>
  ),
}));

vi.mock("@/components/dashboard/ForecastResults", () => ({
  default: ({
    response: result,
    historySaveNotice,
    detailsContent,
    onChangeInputs,
  }: {
    response: ForecastResponse;
    historySaveNotice?: string;
    detailsContent?: React.ReactNode;
    onChangeInputs: () => void;
  }) => (
    <div>
      <span>Result {result.forecastId}</span>
      {historySaveNotice && <span>{historySaveNotice}</span>}
      <button type="button" onClick={onChangeInputs}>Edit brief</button>
      {detailsContent}
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
  mocks.getYouTubeConnection.mockResolvedValue({ isConnected: false });
});

afterEach(() => cleanup());

describe("ForecastPage history saving", () => {
  it("opens directly on the forecast tools without the landing hero", () => {
    render(<ForecastPage />);
    expect(screen.getByRole("heading", { level: 1, name: "Create a forecast" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /Before you hit publish/ })).toBeNull();
  });

  it("opens and closes an explicitly labelled example without an API call or history write", () => {
    render(<ForecastPage />);
    fireEvent.click(screen.getAllByRole("button", { name: "Try an example" })[0]);

    expect(screen.getByText("Result illustrative-example")).toBeTruthy();
    expect(screen.getByText(/Illustrative numbers, not a real prediction/)).toBeTruthy();
    expect(mocks.generateForecast).not.toHaveBeenCalled();
    expect(mocks.saveForecastHistory).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Close example" }));
    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(screen.queryByText("Result illustrative-example")).toBeNull();
  });

  it("keeps the initial view focused on the brief without a large illustrative canvas", () => {
    render(<ForecastPage />);
    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Your forecast canvas" })).toBeNull();
    expect(mocks.generateForecast).not.toHaveBeenCalled();
  });

  it("disables the example action while a real forecast is running", () => {
    mocks.generateForecast.mockReturnValueOnce(new Promise(() => {}));
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));
    expect((screen.getByRole("button", { name: "Try an example" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("prevents duplicate submissions and swaps between review and the mounted brief", async () => {
    let resolveForecast!: (value: ForecastResponse) => void;
    mocks.generateForecast.mockReturnValueOnce(new Promise<ForecastResponse>((resolve) => { resolveForecast = resolve; }));
    render(<ForecastPage />);
    const submit = screen.getByRole("button", { name: "Run forecast" });
    fireEvent.click(submit);
    fireEvent.click(submit);
    expect(mocks.generateForecast).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Run forecast" })).toBeNull();
    expect(screen.getByText("Loading")).toBeTruthy();
    resolveForecast(response);
    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(screen.getByRole("heading", { level: 1, name: "Your forecast" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Edit brief" }));
    expect(screen.getByRole("button", { name: "Run forecast" })).toBe(submit);
    expect(screen.queryByText("Result forecast-1")).toBeNull();
  });

  it("restores the brief after a failed forecast", async () => {
    mocks.generateForecast.mockRejectedValueOnce(new Error("Service temporarily unavailable."));
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));
    expect(await screen.findByText("Service temporarily unavailable.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(mocks.saveForecastHistory).not.toHaveBeenCalled();
  });

  it("lets a guest open and submit the forecast without saving history", async () => {
    mocks.user = null;
    render(<ForecastPage />);

    expect(screen.getByRole("button", { name: "Run forecast" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Retrieve details" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));

    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(mocks.generateForecast).toHaveBeenCalledWith(request);
    expect(mocks.saveForecastHistory).not.toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Want forecasts tailored to your channel?" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /Create free account/ }).getAttribute("href")).toBe("/signup");
    expect(screen.getByRole("link", { name: "Sign in" }).getAttribute("href")).toBe("/login?next=%2Fforecast");
    expect(screen.getByText("Personalized forecasts")).toBeTruthy();
    expect(mocks.getYouTubeConnection).not.toHaveBeenCalled();
  });

  it("shows no signup or connection prompt while auth is loading", () => {
    mocks.isLoading = true;
    mocks.user = null;
    render(<ForecastPage />);
    expect(screen.getByText("Checking your account")).toBeTruthy();
    expect(screen.queryByText("Want forecasts tailored to your channel?")).toBeNull();
    expect(screen.queryByText("Make forecasts more personal")).toBeNull();
  });

  it("invites an unconnected creator through the existing account flow", async () => {
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));
    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    expect(await screen.findByRole("heading", { name: "Make forecasts more personal" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /Connect YouTube/ }).getAttribute("href")).toBe("/account");
    expect(screen.queryByText("Want forecasts tailored to your channel?")).toBeNull();
  });

  it("hides onboarding promotions for connected creators", async () => {
    mocks.getYouTubeConnection.mockResolvedValue({
      isConnected: true,
      channelId: "UC-connected",
      channelTitle: "My channel",
    });
    render(<ForecastPage />);
    expect(await screen.findByText("Using My channel (UC-connected)")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));
    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    await waitFor(() => expect(mocks.getYouTubeConnection).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("Want forecasts tailored to your channel?")).toBeNull();
    expect(screen.queryByText("Make forecasts more personal")).toBeNull();
  });

  it("does not guess connection state when the lookup fails", async () => {
    mocks.getYouTubeConnection.mockRejectedValue(new Error("connection unavailable"));
    render(<ForecastPage />);
    fireEvent.click(screen.getByRole("button", { name: "Run forecast" }));
    expect(await screen.findByText("Result forecast-1")).toBeTruthy();
    await waitFor(() => expect(mocks.getYouTubeConnection).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("Make forecasts more personal")).toBeNull();
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
