// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import YouTubeConnectionCard from "./YouTubeConnectionCard";

const mocks = vi.hoisted(() => ({
  auth: {
    isAuthenticated: true,
    isLoading: false,
  },
  getConnection: vi.fn(),
  startConnection: vi.fn(),
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => mocks.auth,
}));

vi.mock("@/lib/api/youtube-connection", () => ({
  getYouTubeConnection: mocks.getConnection,
  startYouTubeConnection: mocks.startConnection,
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
});

afterEach(cleanup);

describe("YouTubeConnectionCard", () => {
  it("keeps signed-out users on a root-relative sign-in path", () => {
    mocks.auth.isAuthenticated = false;
    render(<YouTubeConnectionCard />);

    expect(screen.getByText("Sign in to connect a channel")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Sign in" }).getAttribute("href")).toBe(
      "/login?next=%2Faccount",
    );
    expect(mocks.getConnection).not.toHaveBeenCalled();
  });

  it("shows the connected channel without exposing credentials", async () => {
    mocks.getConnection.mockResolvedValue({
      isConnected: true,
      channelId: "UC-safe-channel",
      channelTitle: "Creator channel",
      status: "active",
      connectedAt: "2026-09-20T00:00:00Z",
      lastRefreshOkAt: "2026-09-20T01:00:00Z",
    });
    render(<YouTubeConnectionCard />);

    expect(await screen.findByText("Creator channel")).toBeTruthy();
    expect(screen.getByText("UC-safe-channel")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Reconnect channel" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("refresh_token");
  });

  it("starts the server-owned OAuth flow", async () => {
    mocks.getConnection.mockResolvedValue({
      isConnected: false,
      channelId: null,
      channelTitle: null,
      status: null,
      connectedAt: null,
      lastRefreshOkAt: null,
    });
    mocks.startConnection.mockReturnValue(new Promise(() => undefined));
    render(<YouTubeConnectionCard />);

    const button = await screen.findByRole("button", {
      name: "Connect with Google",
    });
    fireEvent.click(button);

    expect(mocks.startConnection).toHaveBeenCalledTimes(1);
    await act(async () => undefined);
    expect(
      (screen.getByRole("button", { name: "Opening Google…" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("shows a safe retry message when connection start fails", async () => {
    mocks.getConnection.mockResolvedValue({
      isConnected: false,
      channelId: null,
      channelTitle: null,
      status: null,
      connectedAt: null,
      lastRefreshOkAt: null,
    });
    mocks.startConnection.mockRejectedValue(new Error("sensitive provider detail"));
    render(<YouTubeConnectionCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Connect with Google" }),
    );

    expect(
      await screen.findByText(
        "We could not start the secure Google connection. Please try again.",
      ),
    ).toBeTruthy();
    expect(document.body.textContent).not.toContain("sensitive provider detail");
  });
});
