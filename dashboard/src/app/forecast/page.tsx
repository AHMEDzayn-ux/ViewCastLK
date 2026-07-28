"use client";

import { useState } from "react";
import type { ForecastInput, ForecastResult } from "@/types/forecast";
import { getPrediction } from "@/lib/predictionClient";
import ForecastForm from "@/components/dashboard/ForecastForm";
import ForecastResults from "@/components/dashboard/ForecastResults";

type PageState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: ForecastResult; input: ForecastInput }
  | { status: "error"; message: string };

export default function ForecastPage() {
  const [state, setState] = useState<PageState>({ status: "idle" });

  async function handleSubmit(input: ForecastInput) {
    setState({ status: "loading" });
    try {
      const result = await getPrediction(input);
      setState({ status: "success", result, input });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setState({ status: "error", message });
    }
  }

  const isLoading = state.status === "loading";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8 items-start">

        {/* ── Left column: form ── */}
        <div className="xl:sticky xl:top-6">
          <div className="mb-4">
            <h2 className="text-slate-100 font-semibold text-xl">
              Enter video metadata
            </h2>
            <p className="text-slate-400 text-sm mt-1 leading-relaxed">
              Fill in the details of a video you are planning to publish. The
              forecasting system will estimate the expected view count at days
              7, 14, 21, and 30. All results are currently mock demonstration
              data.
            </p>
          </div>
          <ForecastForm onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {/* ── Right column: results ── */}
        <div className="min-w-0">
          {state.status === "idle" && (
            <div className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-800/20 py-20 px-8 text-center">
              <div
                className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center mb-4"
                aria-hidden="true"
              >
                <svg
                  className="w-7 h-7 text-slate-500"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"
                  />
                </svg>
              </div>
              <p className="text-slate-300 font-medium text-base">
                Your forecast will appear here
              </p>
              <p className="text-slate-500 text-sm mt-2 max-w-xs leading-relaxed">
                Enter the planned video information and select{" "}
                <span className="text-slate-400 font-medium">
                  Generate forecast
                </span>.
              </p>
            </div>
          )}

          {state.status === "loading" && (
            <div
              className="flex flex-col items-center justify-center rounded-2xl border border-slate-700 bg-slate-800/20 py-24 px-8 text-center"
              role="status"
              aria-label="Generating forecast"
            >
              <div className="w-12 h-12 rounded-full border-4 border-slate-700 border-t-blue-500 animate-spin mb-4" />
              <p className="text-slate-300 font-medium">
                Generating forecast…
              </p>
              <p className="text-slate-500 text-sm mt-1">
                Running mock prediction pipeline
              </p>
            </div>
          )}

          {state.status === "error" && (
            <div
              className="rounded-2xl border border-red-800 bg-red-950/30 p-6"
              role="alert"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-9 h-9 rounded-lg bg-red-900/40 flex items-center justify-center flex-shrink-0">
                  <svg
                    className="w-5 h-5 text-red-400"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
                    />
                  </svg>
                </div>
                <h3 className="text-red-300 font-semibold">Forecast error</h3>
              </div>
              <p className="text-red-400 text-sm">{state.message}</p>
              <button
                onClick={() => setState({ status: "idle" })}
                className="mt-4 text-sm text-red-400 hover:text-red-300 underline underline-offset-2 focus:outline-none focus:ring-2 focus:ring-red-400 rounded"
              >
                Try again
              </button>
            </div>
          )}

          {state.status === "success" && (
            <ForecastResults result={state.result} input={state.input} />
          )}
        </div>
      </div>
    </div>
  );
}
