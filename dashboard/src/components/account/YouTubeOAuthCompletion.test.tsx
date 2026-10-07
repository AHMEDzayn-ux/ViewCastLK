// @vitest-environment jsdom

import { StrictMode } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import YouTubeOAuthCompletion from "./YouTubeOAuthCompletion";

const mocks = vi.hoisted(() => ({ complete: vi.fn(), replace: vi.fn() }));
const router = { replace: mocks.replace };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("@/lib/api/youtube-connection", () => ({ completeYouTubeConnection: mocks.complete }));

beforeEach(() => { vi.clearAllMocks(); });
afterEach(cleanup);

it("completes once under StrictMode and returns to account", async () => {
  mocks.complete.mockResolvedValue(true);
  render(<StrictMode><YouTubeOAuthCompletion /></StrictMode>);
  await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/account?youtube=connected"));
  expect(mocks.complete).toHaveBeenCalledTimes(1);
});

it("returns safely to account on Google denial", async () => {
  mocks.complete.mockResolvedValue(false);
  render(<YouTubeOAuthCompletion />);
  await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/account?youtube=not_connected"));
});

it("shows a safe failure without exposing provider or token details", async () => {
  mocks.complete.mockRejectedValue(new Error("private-provider-detail"));
  render(<YouTubeOAuthCompletion />);
  await screen.findByRole("alert");
  expect(screen.queryByText("private-provider-detail")).toBeNull();
  expect(screen.getByRole("link", { name: "Return to your account" }).getAttribute("href")).toBe("/account");
  expect(mocks.replace).not.toHaveBeenCalled();
});
