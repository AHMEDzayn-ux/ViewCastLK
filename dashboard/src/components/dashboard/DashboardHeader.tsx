"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { isDevelopmentMockMode } from "@/lib/api/forecast";
import StudioIcon, { type StudioIconName } from "./StudioIcon";

const NAVIGATION: { href: string; label: string; icon: StudioIconName; group: string }[] = [
  { href: "/forecast", label: "New forecast", icon: "forecast", group: "Your workspace" },
  { href: "/history", label: "Forecast history", icon: "history", group: "Your workspace" },
  { href: "/account", label: "Your channel", icon: "channel", group: "Your workspace" },
  { href: "/accuracy", label: "Model accuracy", icon: "chart", group: "Behind the forecast" },
  { href: "/methodology", label: "How it works", icon: "book", group: "Behind the forecast" },
];

export default function DashboardHeader() {
  const pathname = usePathname();
  const { user, isAuthenticated, isLoading, isSigningOut, signOut } = useAuth();
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const currentPage = NAVIGATION.find((item) => item.href === pathname)?.label
    ?? ({ "/login": "Welcome back", "/signup": "Create your account", "/privacy": "Privacy", "/forgot-password": "Reset password", "/reset-password": "New password" }[pathname] || "Creator studio");
  const displayName = String(user?.user_metadata?.display_name || user?.email?.split("@")[0] || "Creator");

  async function handleSignOut() {
    setSignOutError(null);
    if (!(await signOut())) setSignOutError("We could not sign you out. Please try again.");
  }

  return (
    <>
      <aside className="studio-sidebar" aria-label="Creator workspace">
        <Link className="studio-brand" href="/forecast" aria-label="ViewCastLK forecast home">
          <span className="studio-brand__symbol" aria-hidden="true"><i /><i /><i /></span>
          <span>ViewCast<span className="studio-brand__lk">LK</span><small>CREATOR STUDIO</small></span>
        </Link>
        <div className="workspace-label"><span className="workspace-label__icon">V</span><span>Your creative space<small>Sri Lanka edition</small></span><span className="workspace-label__dot" /></div>
        <nav className="studio-nav" aria-label="Primary navigation">
          {["Your workspace", "Behind the forecast"].map((group) => (
            <div className="studio-nav__group" key={group}>
              <p>{group}</p>
              {NAVIGATION.filter((item) => item.group === group).map((item) => {
                const current = pathname === item.href || (item.href === "/methodology" && pathname === "/about");
                return <Link key={item.href} href={item.href} className={`studio-nav__link${current ? " is-current" : ""}`} aria-current={current ? "page" : undefined}><StudioIcon name={item.icon} /><span>{item.label}</span>{current && <span className="studio-nav__active-dot" />}</Link>;
              })}
            </div>
          ))}
        </nav>
        <div className="sidebar-note">
          <StudioIcon name="spark" width="26" height="26" />
          <h2>Your next idea starts here.</h2>
          <p>A little more perspective before you press publish.</p>
          <Link href="/methodology">Explore the approach <StudioIcon name="arrow" width="16" height="16" /></Link>
        </div>
        <div className="sidebar-account" aria-label={isLoading ? "Loading account status" : undefined}>
          <span className="account-avatar" aria-hidden="true">{isAuthenticated ? String(displayName).slice(0, 1).toUpperCase() : "G"}</span>
          <span className="sidebar-account__name"><strong>{isLoading ? "Opening studio…" : isAuthenticated ? displayName : "Guest workspace"}</strong><small>{isAuthenticated ? "Your personal workspace" : "Free to explore"}</small></span>
          {isAuthenticated && <button type="button" title="Sign out" aria-label={isSigningOut ? "Signing out..." : "Sign out"} className="icon-button" disabled={isSigningOut} onClick={handleSignOut}><StudioIcon name="logout" width="17" height="17" /></button>}
          {signOutError && <p role="alert" className="sidebar-account__error">{signOutError}</p>}
        </div>
      </aside>
      <header className="studio-topbar">
        <div className="studio-breadcrumb"><span>Workspace</span><span aria-hidden="true">/</span><strong>{currentPage}</strong></div>
        <div className="studio-topbar__actions">
          <Link href="/accuracy" className="model-status"><span />Experimental model</Link>
          {!isLoading && !isAuthenticated && <Link href="/signup" className="studio-signup">Create account</Link>}
          {!isLoading && !isAuthenticated && <Link href="/login" className="studio-signin">Sign in <StudioIcon name="arrow" width="15" height="15" /></Link>}
          {!isLoading && isAuthenticated && <Link href="/account" className="topbar-avatar" aria-label="Your account">{String(displayName).slice(0, 1).toUpperCase()}</Link>}
        </div>
      </header>
      {isDevelopmentMockMode() && <div className="studio-mock-notice" role="status">Example mode · Forecasts are illustrative. Evaluation values are not simulated.</div>}
    </>
  );
}
