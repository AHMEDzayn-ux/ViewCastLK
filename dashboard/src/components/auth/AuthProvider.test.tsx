// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AuthProvider, { useAuth } from "./AuthProvider";
import DashboardHeader from "@/components/dashboard/DashboardHeader";

interface MockUser {
  id: string;
  email: string;
}

interface MockSession {
  user: MockUser;
}

type AuthCallback = (event: string, session: MockSession | null) => void;

const mocks = vi.hoisted(() => ({
  authCallback: undefined as AuthCallback | undefined,
  routerReplace: vi.fn(),
  signOut: vi.fn(),
  subscriptionUnsubscribe: vi.fn(),
}));

vi.mock("@/lib/supabase/client", () => ({
  supabase: {
    auth: {
      onAuthStateChange: vi.fn((callback: AuthCallback) => {
        mocks.authCallback = callback;
        return {
          data: {
            subscription: { unsubscribe: mocks.subscriptionUnsubscribe },
          },
        };
      }),
      signOut: mocks.signOut,
    },
  },
}));

vi.mock("@/lib/api/forecast", () => ({
  isDevelopmentMockMode: () => false,
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/forecast",
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

const USER_A: MockUser = {
  id: "user-a",
  email: "creator@example.com",
};

const USER_B: MockUser = {
  id: "user-b",
  email: "updated@example.com",
};

function renderHeader() {
  return render(
    <AuthProvider>
      <DashboardHeader />
    </AuthProvider>,
  );
}

function AuthStateProbe() {
  const { user, isAuthenticated, isLoading } = useAuth();

  return (
    <output data-testid="auth-state">
      {isLoading
        ? "loading"
        : `${isAuthenticated ? "authenticated" : "anonymous"}:${user?.email ?? "none"}`}
    </output>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.authCallback = undefined;
  mocks.signOut.mockResolvedValue({ error: null });
});

afterEach(() => {
  cleanup();
});

describe("AuthProvider and account navigation", () => {
  it("keeps auth actions stable until INITIAL_SESSION completes, then shows signed-out links", () => {
    renderHeader();

    expect(screen.getByLabelText("Loading account status")).toBeTruthy();
    expect(screen.queryByRole("link", { name: "Sign in" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Create account" })).toBeNull();

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", null);
    });

    expect(screen.queryByLabelText("Loading account status")).toBeNull();
    expect(screen.getByRole("link", { name: "Sign in" }).getAttribute("href")).toBe(
      "/login",
    );
    expect(
      screen.getByRole("link", { name: "Create account" }).getAttribute("href"),
    ).toBe("/signup");
    expect(screen.queryByRole("button", { name: "Sign out" })).toBeNull();
  });

  it("shows authenticated navigation after INITIAL_SESSION and reacts to SIGNED_OUT", () => {
    renderHeader();

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", { user: USER_A });
    });

    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: "Sign in" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Create account" })).toBeNull();

    act(() => {
      mocks.authCallback?.("SIGNED_OUT", null);
    });

    expect(screen.getByRole("link", { name: "Sign in" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Sign out" })).toBeNull();
  });

  it("handles SIGNED_IN, TOKEN_REFRESHED, and USER_UPDATED synchronously", () => {
    render(
      <AuthProvider>
        <AuthStateProbe />
      </AuthProvider>,
    );

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", null);
    });
    expect(screen.getByTestId("auth-state").textContent).toBe("anonymous:none");

    act(() => {
      mocks.authCallback?.("SIGNED_IN", { user: USER_A });
    });
    expect(screen.getByTestId("auth-state").textContent).toBe(
      "authenticated:creator@example.com",
    );

    act(() => {
      mocks.authCallback?.("TOKEN_REFRESHED", { user: USER_A });
      mocks.authCallback?.("USER_UPDATED", { user: USER_B });
    });
    expect(screen.getByTestId("auth-state").textContent).toBe(
      "authenticated:updated@example.com",
    );
  });

  it("invokes local sign out, disables the control, and redirects to root-relative login", async () => {
    let resolveSignOut: ((value: { error: null }) => void) | undefined;
    mocks.signOut.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveSignOut = resolve;
      }),
    );
    renderHeader();

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", { user: USER_A });
    });
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));

    expect(mocks.signOut).toHaveBeenCalledWith({ scope: "local" });
    expect(
      (screen.getByRole("button", { name: "Signing out..." }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await act(async () => {
      resolveSignOut?.({ error: null });
    });

    expect(mocks.routerReplace).toHaveBeenCalledWith("/login");
  });

  it("shows a safe logout error and does not redirect when Supabase rejects sign out", async () => {
    mocks.signOut.mockResolvedValueOnce({
      error: { message: "provider session detail" },
    });
    renderHeader();

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", { user: USER_A });
    });
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));

    expect(
      await screen.findByText("We could not sign you out. Please try again."),
    ).toBeTruthy();
    expect(screen.queryByText("provider session detail")).toBeNull();
    expect(mocks.routerReplace).not.toHaveBeenCalled();
    expect((screen.getByRole("button", { name: "Sign out" }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it("keeps every rendered internal navigation target root-relative", () => {
    renderHeader();

    act(() => {
      mocks.authCallback?.("INITIAL_SESSION", null);
    });

    const hrefs = screen
      .getAllByRole("link")
      .map((link) => link.getAttribute("href"));

    expect(hrefs).toContain("/forecast");
    expect(hrefs).toContain("/login");
    expect(hrefs).toContain("/signup");
    expect(hrefs.every((href) => href?.startsWith("/"))).toBe(true);
    expect(hrefs).not.toContain("/forecast/login");
    expect(hrefs).not.toContain("/forecast/signup");
  });

  it("unsubscribes from Supabase auth changes when unmounted", async () => {
    const { unmount } = renderHeader();

    unmount();

    await waitFor(() => {
      expect(mocks.subscriptionUnsubscribe).toHaveBeenCalledTimes(1);
    });
  });
});
