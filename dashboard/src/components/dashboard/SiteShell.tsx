"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import DashboardHeader from "./DashboardHeader";

export default function SiteShell({ children }: { children: React.ReactNode }) {
  const isLanding = usePathname() === "/";

  return (
    <div className={isLanding ? "landing-shell" : "studio-shell"}>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <DashboardHeader />
      <div id="main-content" className="site-content" tabIndex={-1}>{children}</div>
      <footer className="site-footer">
        <div>
          <p>Made for the next generation of Sri Lankan creators.</p>
          <p>ViewCastLK · Research project · Not affiliated with or endorsed by YouTube or Google. · <Link href="/privacy">Privacy</Link></p>
        </div>
      </footer>
    </div>
  );
}
