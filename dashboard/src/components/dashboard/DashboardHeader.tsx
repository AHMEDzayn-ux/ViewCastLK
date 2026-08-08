"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { isDevelopmentMockMode } from "@/lib/api/forecast";

const NAVIGATION = [
  { href: "/forecast", label: "Forecast" },
  { href: "/accuracy", label: "Accuracy" },
  { href: "/methodology", label: "Methodology & Limitations" },
];

export default function DashboardHeader() {
  const pathname = usePathname();
  const isMockMode = isDevelopmentMockMode();

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
                className={isCurrent ? "primary-nav__link is-current" : "primary-nav__link"}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
