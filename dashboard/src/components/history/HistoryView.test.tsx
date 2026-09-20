// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ForecastHistoryRow } from "@/types/history";
import HistoryView from "./HistoryView";

const records: ForecastHistoryRow[] = [
  {
    id: "history-new",
    user_id: "user-a",
    created_at: "2026-09-20T10:00:00.000Z",
    forecast_id: "forecast-new",
    title: "Newest forecast",
    category: "Education",
    duration_seconds: 600,
    audio_language: "English",
    channel_identifier: "@creator",
    planned_publish_day: "Friday",
    planned_publish_hour: 18,
    day_7_cumulative_views: 100,
    day_14_cumulative_views: 200,
    day_21_cumulative_views: 300,
    day_30_cumulative_views: 400,
    model_version: "model-v1",
    forecast_generated_at: "2026-09-20T09:00:00.000Z",
    model_data_source: "prediction_api",
    completeness_status: "complete",
    completeness_issues: [],
  },
  {
    id: "history-old",
    user_id: "user-a",
    created_at: "2026-09-19T10:00:00.000Z",
    forecast_id: "forecast-old",
    title: "Older forecast",
    category: "Music",
    duration_seconds: 125,
    audio_language: "Sinhala",
    channel_identifier: "@creator",
    planned_publish_day: null,
    planned_publish_hour: null,
    day_7_cumulative_views: 50,
    day_14_cumulative_views: 100,
    day_21_cumulative_views: 150,
    day_30_cumulative_views: 200,
    model_version: "model-v1",
    forecast_generated_at: "2026-09-19T09:00:00.000Z",
    model_data_source: "prediction_api",
    completeness_status: "complete",
    completeness_issues: [],
  },
];

const mocks = vi.hoisted(() => ({
  auth: { isAuthenticated: true, isLoading: false },
  deleteForecastHistory: vi.fn(),
  listForecastHistory: vi.fn(),
  routerReplace: vi.fn(),
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => mocks.auth,
}));

vi.mock("@/lib/history/forecast-history", () => ({
  deleteForecastHistory: mocks.deleteForecastHistory,
  listForecastHistory: mocks.listForecastHistory,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.routerReplace }),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.auth.isAuthenticated = true;
  mocks.auth.isLoading = false;
  mocks.listForecastHistory.mockResolvedValue(records);
  mocks.deleteForecastHistory.mockResolvedValue(undefined);
  vi.spyOn(window, "confirm").mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("HistoryView", () => {
  it("renders history in the newest-first order returned by the data layer", async () => {
    render(<HistoryView />);

    expect(await screen.findByText("Newest forecast")).toBeTruthy();
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.textContent)).toEqual([
      "Newest forecast",
      "Older forecast",
    ]);
    expect(screen.getAllByText("Day 30")).toHaveLength(2);
    expect(screen.getByText("400")).toBeTruthy();
  });

  it("renders an empty history state", async () => {
    mocks.listForecastHistory.mockResolvedValueOnce([]);
    render(<HistoryView />);

    expect(await screen.findByText("Your forecast history will appear here")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Create a forecast" }).getAttribute("href"),
    ).toBe("/forecast");
  });

  it("redirects signed-out users only to the root-relative login return path", async () => {
    mocks.auth.isAuthenticated = false;
    render(<HistoryView />);

    expect(screen.getByText("Your history is private to your account")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Sign in to view history" }).getAttribute("href"),
    ).toBe("/login?next=%2Fhistory");
    await waitFor(() => {
      expect(mocks.routerReplace).toHaveBeenCalledWith(
        "/login?next=%2Fhistory",
      );
    });
    expect(mocks.listForecastHistory).not.toHaveBeenCalled();
  });

  it("removes a record after a successful confirmed deletion", async () => {
    render(<HistoryView />);
    await screen.findByText("Newest forecast");

    fireEvent.click(
      screen.getByRole("button", {
        name: "Delete saved forecast for Newest forecast",
      }),
    );

    await waitFor(() => {
      expect(mocks.deleteForecastHistory).toHaveBeenCalledWith("history-new");
      expect(screen.queryByText("Newest forecast")).toBeNull();
    });
    expect(screen.getByText("Older forecast")).toBeTruthy();
  });

  it("keeps the record and shows a safe error when deletion fails", async () => {
    mocks.deleteForecastHistory.mockRejectedValueOnce(
      new Error("private database detail"),
    );
    render(<HistoryView />);
    await screen.findByText("Newest forecast");

    fireEvent.click(
      screen.getByRole("button", {
        name: "Delete saved forecast for Newest forecast",
      }),
    );

    expect(
      await screen.findByText(
        "The saved forecast could not be deleted. Please try again.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("Newest forecast")).toBeTruthy();
    expect(screen.queryByText("private database detail")).toBeNull();
  });
});
