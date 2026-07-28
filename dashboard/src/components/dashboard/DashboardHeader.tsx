"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function DashboardHeader() {
  const pathname = usePathname();

  return (
    <header className="bg-slate-900 border-b border-slate-700">
      {/* Mock-data notice banner */}
      <div className="bg-amber-500 text-amber-950 text-center text-xs font-semibold py-1.5 px-4">
        ⚠ Demonstration mode — all forecasts use mock prediction data and do
        not represent real model outputs.
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-red-500 to-rose-700 flex items-center justify-center flex-shrink-0">
              <svg
                className="w-5 h-5 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-2.88 2.5 2.89 2.89 0 0 1-2.89-2.89 2.89 2.89 0 0 1 2.89-2.89c.28 0 .54.04.79.1V9.01a6.32 6.32 0 0 0-.79-.05 6.34 6.34 0 0 0-6.34 6.34 6.34 6.34 0 0 0 6.34 6.34 6.34 6.34 0 0 0 6.33-6.34V8.69a8.18 8.18 0 0 0 4.78 1.52V6.77a4.85 4.85 0 0 1-1.01-.08z" />
              </svg>
            </div>
            <div>
              <h1 className="text-white font-bold text-lg leading-tight">
                ViewCastLK
              </h1>
              <p className="text-slate-400 text-xs">
                Pre-publication YouTube view forecasting
              </p>
            </div>
          </div>

          {/* Meta badges */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
              Mock data — Phase 1
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              Sri Lankan YouTube content
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              University project · Group 2
            </span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex gap-6 border-t border-slate-800 pt-3 text-sm font-medium">
          <Link
            href="/forecast"
            className={`pb-2 px-1 transition-colors relative ${
              pathname === "/forecast" || pathname === "/"
                ? "text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Forecast
            {(pathname === "/forecast" || pathname === "/") && (
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-sm" />
            )}
          </Link>
          <Link
            href="/about"
            className={`pb-2 px-1 transition-colors relative ${
              pathname === "/about"
                ? "text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            About
            {pathname === "/about" && (
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-sm" />
            )}
          </Link>
        </nav>
      </div>
    </header>
  );
}
