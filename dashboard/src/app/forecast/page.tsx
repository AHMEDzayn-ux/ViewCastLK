"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/components/auth/AuthProvider";
import ForecastForm from "@/components/dashboard/ForecastForm";
import ForecastResults from "@/components/dashboard/ForecastResults";
import ErrorState from "@/components/dashboard/ErrorState";
import LoadingState from "@/components/dashboard/LoadingState";
import ForecastPreview from "@/components/dashboard/ForecastPreview";
import StudioIcon from "@/components/dashboard/StudioIcon";
import { generateForecast } from "@/lib/api/forecast";
import { saveForecastHistory } from "@/lib/history/forecast-history";
import type { ForecastRequest, ForecastResponse } from "@/types/forecast";

type PageState =
  | { status: "idle" }
  | { status: "loading"; request: ForecastRequest }
  | {
      status: "success";
      request: ForecastRequest;
      response: ForecastResponse;
      historySaveNotice?: string;
      isExample?: boolean;
    }
  | { status: "error"; request: ForecastRequest; message: string };

export default function ForecastPage() {
  const { isAuthenticated, isLoading: isAuthLoading, user } = useAuth();
  const [state, setState] = useState<PageState>({ status: "idle" });
  const userRef = useRef(user);
  const outputRef = useRef<HTMLDivElement>(null);
  const isShowingExample = state.status === "success" && Boolean(state.isExample);

  useEffect(() => {
    userRef.current = user;
  }, [user]);

  useEffect(() => {
    if (isShowingExample && window.innerWidth <= 960) {
      outputRef.current?.scrollIntoView({ block: "start" });
    }
  }, [isShowingExample]);

  async function runForecast(request: ForecastRequest) {
    const forecastUserId = userRef.current?.id;
    setState({ status: "loading", request });

    try {
      const response = await generateForecast(request);
      let historySaveNotice: string | undefined;
      if (forecastUserId && userRef.current?.id === forecastUserId) {
        try {
          await saveForecastHistory(forecastUserId, request, response);
        } catch {
          historySaveNotice =
            "Forecast generated, but it could not be saved to your history.";
        }
      }

      setState({
        status: "success",
        request,
        response,
        historySaveNotice,
      });
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

  function showExample() {
    const request: ForecastRequest = {
      title: "A slow weekend in Ella | A Sri Lankan travel diary",
      category: "Travel & Events",
      durationSeconds: 480,
      isShort: false,
      audioLanguage: "English",
      channelIdentifier: "@example-creator",
      plannedPublishDay: "Saturday",
      plannedPublishHour: 10,
    };
    const estimates: ForecastResponse["estimates"] = [
      { horizonDays: 7, cumulativeViews: 2840 },
      { horizonDays: 14, cumulativeViews: 4320 },
      { horizonDays: 21, cumulativeViews: 5790 },
      { horizonDays: 30, cumulativeViews: 7240 },
    ];
    setState({
      status: "success", request, isExample: true,
      response: {
        forecastId: "illustrative-example", estimates,
        personalization: { applied: false, format: "long", modelVersion: "illustrative-example", sharedEstimates: estimates, adjustments: [] },
        recommendations: [], unavailableRecommendations: [],
        completeness: { status: "complete", issues: [] },
        model: { modelVersion: "illustrative-example", generatedAt: new Date().toISOString(), dataSource: "mock" },
      },
    });
  }

  if (isAuthLoading) {
    return (
      <main className="page-shell forecast-page">
        <section className="result-state" aria-busy="true" role="status">
          <p className="result-state__eyebrow">Secure forecast access</p>
          <h1>Checking your account</h1>
          <p>Please wait a moment.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="page-shell forecast-page">
      <header className="studio-intro">
        <div><p className="section-kicker"><span /> FROM IDEA TO WHAT’S NEXT</p>
          <h1>Before you hit publish,<br />see the <em>possibilities.</em></h1>
          <p>Give your next video a little foresight. Explore its first 30 days<br className="desktop-break" /> with forecasts made for Sri Lankan creators.</p>
          <div className="studio-intro__links"><a href="#forecast-form">Start with your idea <StudioIcon name="arrow" width="16" height="16" /></a><button type="button" disabled={isLoading} onClick={showExample}><StudioIcon name="play" width="14" height="14" /> Try an example</button></div>
        </div>
        <div className="studio-art" aria-hidden="true">
          <div className="studio-art__orbit studio-art__orbit--one" /><div className="studio-art__orbit studio-art__orbit--two" />
          <span className="studio-art__star">✳</span>
          <div className="studio-art__tile"><StudioIcon name="play" width="52" height="52" /></div>
          <span className="studio-art__label studio-art__label--idea">YOUR NEXT BIG IDEA</span>
          <span className="studio-art__label studio-art__label--days"><span>30</span> days of possibility <StudioIcon name="arrow" width="16" height="16" /></span>
          <span className="studio-art__dot" />
        </div>
      </header>

      <div className="workspace-heading"><div><span className="workspace-heading__number">01</span><h2>The forecast workspace</h2></div><Link href="/methodology">How does this work? <StudioIcon name="arrow" width="15" height="15" /></Link></div>

      <div className="forecast-workspace">
        <div className="forecast-workspace__form">
          <ForecastForm
            onSubmit={runForecast}
            onReset={() => setState({ status: "idle" })}
            isLoading={isLoading}
            canLookupChannel={isAuthenticated}
          />
        </div>

        <div className="forecast-workspace__output" ref={outputRef} aria-live="polite">
          {state.status === "idle" && (
            <ForecastPreview onExample={showExample} isAuthenticated={isAuthenticated} />
          )}

          {state.status === "loading" && <LoadingState />}

          {state.status === "error" && (
            <ErrorState
              message={state.message}
              onRetry={() => runForecast(state.request)}
            />
          )}

          {state.status === "success" && (
            <>
            {state.isExample && <div className="example-notice" role="status"><div><strong>You’re exploring an example</strong><p>Illustrative numbers, not a real prediction. This example is not saved to your history.</p></div><button type="button" onClick={() => setState({ status: "idle" })}>Close example <span aria-hidden="true">×</span></button></div>}
            <ForecastResults
              request={state.request}
              response={state.response}
              historySaveNotice={state.historySaveNotice}
              onChangeInputs={focusForm}
            />
            </>
          )}
        </div>
      </div>
    </main>
  );
}
