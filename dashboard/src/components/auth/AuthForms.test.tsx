// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ForgotPasswordForm from "./ForgotPasswordForm";
import LoginForm from "./LoginForm";
import ResetPasswordForm from "./ResetPasswordForm";
import SignupForm from "./SignupForm";

type AuthCallback = (event: string, session: unknown) => void;

interface MockTurnstileProps {
  onSuccess: (token: string) => void;
}

const mocks = vi.hoisted(() => ({
  authCallback: undefined as AuthCallback | undefined,
  resetPasswordForEmail: vi.fn(),
  routerReplace: vi.fn(),
  signInWithPassword: vi.fn(),
  signOut: vi.fn(),
  signUp: vi.fn(),
  subscriptionUnsubscribe: vi.fn(),
  turnstileReset: vi.fn(),
  updateUser: vi.fn(),
  searchParams: new URLSearchParams(),
}));

vi.mock("@/lib/auth/turnstile-config", () => ({
  requireTurnstileSiteKey: () => "test-site-key",
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
      resetPasswordForEmail: mocks.resetPasswordForEmail,
      signInWithPassword: mocks.signInWithPassword,
      signOut: mocks.signOut,
      signUp: mocks.signUp,
      updateUser: mocks.updateUser,
    },
  },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.routerReplace }),
  useSearchParams: () => mocks.searchParams,
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

vi.mock("@marsidev/react-turnstile", async () => {
  const React = await import("react");

  return {
    Turnstile: React.forwardRef<{ reset: () => void }, MockTurnstileProps>(
      function MockTurnstile({ onSuccess }, ref) {
        React.useImperativeHandle(ref, () => ({ reset: mocks.turnstileReset }));

        return (
          <button type="button" onClick={() => onSuccess("captcha-token")}>
            Complete verification
          </button>
        );
      },
    ),
  };
});

function completeVerification() {
  fireEvent.click(
    screen.getByRole("button", { name: "Complete verification" }),
  );
}

function fillSignupForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: "creator@example.com" },
  });
  fireEvent.change(screen.getByLabelText("Password"), {
    target: { value: "StrongPass1!" },
  });
  fireEvent.change(screen.getByLabelText("Confirm password"), {
    target: { value: "StrongPass1!" },
  });
}

function fillLoginForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: "creator@example.com" },
  });
  fireEvent.change(screen.getByLabelText("Password"), {
    target: { value: "StrongPass1!" },
  });
}

function fillForgotPasswordForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: "creator@example.com" },
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.authCallback = undefined;
  mocks.searchParams = new URLSearchParams();
  mocks.resetPasswordForEmail.mockResolvedValue({ data: {}, error: null });
  mocks.signInWithPassword.mockResolvedValue({
    data: { session: { user: { id: "user-a" } } },
    error: null,
  });
  mocks.signOut.mockResolvedValue({ error: null });
  mocks.signUp.mockResolvedValue({ data: {}, error: null });
  mocks.updateUser.mockResolvedValue({ data: {}, error: null });
});

afterEach(() => {
  cleanup();
});

describe("SignupForm", () => {
  it("requires Turnstile and sends its transient token to Supabase", async () => {
    render(<SignupForm />);
    fillSignupForm();

    const submit = screen.getByRole("button", { name: "Create account" });
    expect((submit as HTMLButtonElement).disabled).toBe(true);

    completeVerification();
    expect((submit as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(submit);

    await waitFor(() => {
      expect(mocks.signUp).toHaveBeenCalledWith({
        email: "creator@example.com",
        password: "StrongPass1!",
        options: {
          captchaToken: "captcha-token",
          emailRedirectTo: "http://localhost:3000/login?verified=1",
        },
      });
    });
    expect(await screen.findByText(/Check your email to verify/i)).toBeTruthy();
    expect(mocks.turnstileReset).toHaveBeenCalledTimes(1);
  });
});

describe("LoginForm", () => {
  it("signs in with the Turnstile token, resets it, and redirects", async () => {
    render(<LoginForm />);
    fillLoginForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(mocks.signInWithPassword).toHaveBeenCalledWith({
        email: "creator@example.com",
        password: "StrongPass1!",
        options: { captchaToken: "captcha-token" },
      });
      expect(mocks.routerReplace).toHaveBeenCalledWith("/forecast");
    });
    expect(mocks.turnstileReset).toHaveBeenCalledTimes(1);
  });

  it("returns only to the allowlisted history route after sign in", async () => {
    mocks.searchParams = new URLSearchParams("next=%2Fhistory");
    render(<LoginForm />);
    fillLoginForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(mocks.routerReplace).toHaveBeenCalledWith("/history");
    });
  });

  it("rejects an external post-login destination", async () => {
    mocks.searchParams = new URLSearchParams("next=https%3A%2F%2Fevil.example");
    render(<LoginForm />);
    fillLoginForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(mocks.routerReplace).toHaveBeenCalledWith("/forecast");
    });
  });

  it("shows a safe credential error without exposing the provider message", async () => {
    mocks.signInWithPassword.mockResolvedValueOnce({
      data: { session: null },
      error: { code: "invalid_credentials", message: "provider detail" },
    });

    render(<LoginForm />);
    fillLoginForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Email or password is incorrect.")).toBeTruthy();
    expect(screen.queryByText("provider detail")).toBeNull();
    expect(mocks.routerReplace).not.toHaveBeenCalled();
    expect(mocks.turnstileReset).toHaveBeenCalledTimes(1);
  });
});

describe("ForgotPasswordForm", () => {
  it("keeps account-existence responses generic", async () => {
    mocks.resetPasswordForEmail.mockResolvedValueOnce({
      data: null,
      error: { code: "user_not_found", message: "user does not exist" },
    });

    render(<ForgotPasswordForm />);
    fillForgotPasswordForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(
      await screen.findByText(
        "If an account exists for that email, a password reset link has been sent.",
      ),
    ).toBeTruthy();
    expect(screen.queryByText("user does not exist")).toBeNull();
    expect(mocks.resetPasswordForEmail).toHaveBeenCalledWith(
      "creator@example.com",
      {
        redirectTo: "http://localhost:3000/reset-password",
        captchaToken: "captcha-token",
      },
    );
  });

  it("does not report a Supabase CAPTCHA failure as success", async () => {
    mocks.resetPasswordForEmail.mockResolvedValueOnce({
      data: null,
      error: { code: "captcha_failed", message: "provider detail" },
    });

    render(<ForgotPasswordForm />);
    fillForgotPasswordForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(
      await screen.findByText(/Verification could not be confirmed/i),
    ).toBeTruthy();
    expect(screen.queryByText(/If an account exists/i)).toBeNull();
    expect(screen.queryByText("provider detail")).toBeNull();
    expect(mocks.turnstileReset).toHaveBeenCalledTimes(1);
  });

  it("shows a safe retry message for a thrown transport failure", async () => {
    mocks.resetPasswordForEmail.mockRejectedValueOnce(
      new Error("private network detail"),
    );

    render(<ForgotPasswordForm />);
    fillForgotPasswordForm();
    completeVerification();
    fireEvent.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(
      await screen.findByText("We could not process the request. Please try again."),
    ).toBeTruthy();
    expect(screen.queryByText("private network detail")).toBeNull();
  });
});

describe("ResetPasswordForm", () => {
  it("rejects an ordinary authenticated session", async () => {
    render(<ResetPasswordForm />);

    await act(async () => {
      mocks.authCallback?.("INITIAL_SESSION", { user: { id: "user-a" } });
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    });

    expect(screen.getByText("Request a new reset link")).toBeTruthy();
    expect(screen.queryByLabelText("New password")).toBeNull();
  });

  it("rejects a missing recovery session", () => {
    render(<ResetPasswordForm />);

    act(() => {
      mocks.authCallback?.("PASSWORD_RECOVERY", null);
    });

    expect(screen.getByText("Request a new reset link")).toBeTruthy();
  });

  it("updates the password only after recovery, signs out, and redirects", async () => {
    render(<ResetPasswordForm />);

    act(() => {
      mocks.authCallback?.("PASSWORD_RECOVERY", { user: { id: "user-a" } });
    });

    fireEvent.change(screen.getByLabelText("New password"), {
      target: { value: "NewStrong1!" },
    });
    fireEvent.change(screen.getByLabelText("Confirm new password"), {
      target: { value: "NewStrong1!" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Update password" }));

    await waitFor(() => {
      expect(mocks.updateUser).toHaveBeenCalledWith({
        password: "NewStrong1!",
      });
      expect(mocks.signOut).toHaveBeenCalledTimes(1);
      expect(mocks.routerReplace).toHaveBeenCalledWith(
        "/login?password_reset=1",
      );
    });
  });
});
