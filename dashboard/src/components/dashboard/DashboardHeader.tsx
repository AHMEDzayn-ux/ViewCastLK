"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { isDevelopmentMockMode } from "@/lib/api/forecast";

const NAVIGATION = [
  { href: "/forecast", label: "Forecast" },
  { href: "/accuracy", label: "Accuracy" },
  { href: "/methodology", label: "Methodology & Limitations" },
];

export default function DashboardHeader() {
  const pathname = usePathname();
  const isMockMode = isDevelopmentMockMode();
  const { isAuthenticated, isLoading, isSigningOut, signOut } = useAuth();
  const [signOutError, setSignOutError] = useState<string | null>(null);

  async function handleSignOut() {
    setSignOutError(null);
    const wasSuccessful = await signOut();

    if (!wasSuccessful) {
      setSignOutError("We could not sign you out. Please try again.");
    }
  }

  return (
    <header className="site-header">
      {isMockMode && (
        <div className="development-notice" role="status">
          <span>Development adapter active</span>
          <span>Forecasts are illustrative; evaluation values are not simulated.</span>
        </div>
      )}

      <div className="site-header__inner">
        <Link className="brand" href="/forecast" aria-label="ViewCastLK forecast home">
          <span className="brand__mark" aria-hidden="true">
            <span>VC</span>
            <i />
            <span>LK</span>
          </span>
          <span className="brand__name">
            <strong>ViewCastLK</strong>
            <small>Pre-publication view forecasting</small>
          </span>
        </Link>

        <div className="site-header__actions">
          <nav className="primary-nav" aria-label="Primary navigation">
            {NAVIGATION.map((item) => {
              const isCurrent =
                pathname === item.href ||
                (item.href === "/methodology" && pathname === "/about");

              return (
                <Link
                  href={item.href}
                  key={item.href}
                  aria-current={isCurrent ? "page" : undefined}
                  className={
                    isCurrent
                      ? "primary-nav__link is-current"
                      : "primary-nav__link"
                  }
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <nav className="auth-navigation" aria-label="Account navigation">
            {isLoading ? (
              <span
                className="auth-navigation__placeholder"
                aria-label="Loading account status"
              >
                Account
              </span>
            ) : isAuthenticated ? (
              <button
                className="auth-navigation__button"
                type="button"
                disabled={isSigningOut}
                onClick={handleSignOut}
              >
                {isSigningOut ? "Signing out..." : "Sign out"}
              </button>
            ) : (
              <>
                <Link className="auth-navigation__link" href="/login">
                  Sign in
                </Link>
                <Link
                  className="auth-navigation__link auth-navigation__link--primary"
                  href="/signup"
                >
                  Create account
                </Link>
              </>
            )}

            {signOutError && (
              <p className="auth-navigation__error" role="alert">
                {signOutError}
              </p>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
