"use client";

import { useState } from "react";
import ForecastForm from "@/components/dashboard/ForecastForm";
import ForecastResults from "@/components/dashboard/ForecastResults";
import ErrorState from "@/components/dashboard/ErrorState";
import LoadingState from "@/components/dashboard/LoadingState";
import { generateForecast } from "@/lib/api/forecast";
import type { ForecastRequest, ForecastResponse } from "@/types/forecast";

type PageState =
  | { status: "idle" }
  | { status: "loading"; request: ForecastRequest }
  | { status: "success"; request: ForecastRequest; response: ForecastResponse }
  | { status: "error"; request: ForecastRequest; message: string };

export default function ForecastPage() {
  const [state, setState] = useState<PageState>({ status: "idle" });

  async function runForecast(request: ForecastRequest) {
    setState({ status: "loading", request });

    try {
      const response = await generateForecast(request);
      setState({ status: "success", request, response });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "An unexpected error prevented the forecast.";
      setState({ status: "error", request, message });
    }
  }

  function focusForm() {
    document.getElementById("forecast-form-title")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    window.setTimeout(() => document.getElementById("title")?.focus(), 250);
  }

  const isLoading = state.status === "loading";

  return (
    <main className="page-shell forecast-page">
      <header className="page-intro">
        <p className="section-kicker">Pre-publication planning</p>
        <h1>Forecast a planned video before it goes live</h1>
        <p>
          Estimate cumulative views at Day 7, 14, 21, and 30 using the details
          available during planning. No early engagement figures are needed.
        </p>
      </header>

      <div className="forecast-workspace">
        <div className="forecast-workspace__form">
          <ForecastForm
            onSubmit={runForecast}
            onReset={() => setState({ status: "idle" })}
            isLoading={isLoading}
          />
        </div>

        <div className="forecast-workspace__output" aria-live="polite">
          {state.status === "idle" && (
            <section className="result-state result-state--idle">
              <p className="result-state__eyebrow">Your result workspace</p>
              <h2>Four useful checkpoints, one clear forecast</h2>
              <p>
                Complete the required video and channel details. Results will
                appear here without replacing your form entries.
              </p>
              <ol className="idle-horizons" aria-label="Forecast horizons">
                {[7, 14, 21, 30].map((day) => (
                  <li key={day}>Day {day}</li>
                ))}
              </ol>
            </section>
          )}

          {state.status === "loading" && <LoadingState />}

          {state.status === "error" && (
            <ErrorState
              message={state.message}
              onRetry={() => runForecast(state.request)}
            />
          )}

          {state.status === "success" && (
            <ForecastResults
              request={state.request}
              response={state.response}
              onChangeInputs={focusForm}
            />
          )}
        </div>
      </div>
    </main>
  );
}
