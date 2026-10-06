"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/components/auth/AuthProvider";
import ForecastForm from "@/components/dashboard/ForecastForm";
import ForecastResults from "@/components/dashboard/ForecastResults";
import ForecastOnboarding, { type CreatorAudience } from "@/components/dashboard/ForecastOnboarding";
import ErrorState from "@/components/dashboard/ErrorState";
import LoadingState from "@/components/dashboard/LoadingState";
import StudioIcon from "@/components/dashboard/StudioIcon";
import { generateForecast } from "@/lib/api/forecast";
import { getYouTubeConnection } from "@/lib/api/youtube-connection";
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
  const [isEditing, setIsEditing] = useState(true);
  const isRunningRef = useRef(false);
  const focusEditorRef = useRef(false);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [connection, setConnection] = useState<{
    userId: string;
    audience: CreatorAudience;
    channelId: string | null;
    channelTitle: string | null;
  } | null>(null);
  const audience: CreatorAudience = !user
    ? "guest"
    : connection?.userId === user.id ? connection.audience : "checking";

  const userId = user?.id;
  useEffect(() => {
    if (!userId) return;
    let active = true;
    getYouTubeConnection()
      .then((status) => {
        if (active) setConnection({
          userId,
          audience: status.isConnected && status.channelId ? "connected" : "unconnected",
          channelId: status.isConnected ? status.channelId : null,
          channelTitle: status.isConnected ? status.channelTitle : null,
        });
      })
      .catch(() => {
        if (active) setConnection({ userId, audience: "checking", channelId: null, channelTitle: null });
      });
    return () => { active = false; };
  }, [userId]);
  const userRef = useRef(user);

  useEffect(() => {
    userRef.current = user;
  }, [user]);

  useEffect(() => {
    if (isEditing) {
      if (focusEditorRef.current) {
        focusEditorRef.current = false;
        document.getElementById("title")?.focus();
      }
    } else {
      headingRef.current?.focus({ preventScroll: true });
      const bounds = headingRef.current?.getBoundingClientRect();
      if (bounds && (bounds.top < 0 || bounds.bottom > window.innerHeight)) {
        headingRef.current?.scrollIntoView?.({ block: "start" });
      }
    }
  }, [isEditing, state.status]);

  async function runForecast(request: ForecastRequest) {
    if (isRunningRef.current) return;
    isRunningRef.current = true;
    const forecastUserId = userRef.current?.id;
    setIsEditing(false);
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
      setIsEditing(true);
    } finally {
      isRunningRef.current = false;
    }
  }

  function focusForm() {
    focusEditorRef.current = true;
    setIsEditing(true);
  }

  function closeExample() {
    focusForm();
    setState({ status: "idle" });
  }

  const isLoading = state.status === "loading";

  function showExample() {
    if (isRunningRef.current) return;
    setIsEditing(false);
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
        breakout: {
          probability: 0.04,
          definition: "In this illustrative example, a breakout represents an unusually strong outcome relative to the channel’s usual performance.",
          conditionalUpside: [
            { horizonDays: 7, cumulativeViews: 38000 },
            { horizonDays: 14, cumulativeViews: 42500 },
            { horizonDays: 21, cumulativeViews: 44800 },
            { horizonDays: 30, cumulativeViews: 45894 },
          ],
        },
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
    <main className={`page-shell forecast-page compact-forecast${isEditing ? " compact-forecast--editing" : " compact-forecast--review"}`}>
      <header className="forecast-heading">
        <div><p className="section-kicker">YOUR CREATOR WORKSPACE</p><h1 ref={headingRef} tabIndex={-1}>{isEditing ? "Create a forecast" : isLoading ? "Preparing your forecast" : "Your forecast"}</h1></div>
        <div className="forecast-heading__actions">
          <button type="button" disabled={isLoading} onClick={showExample}><StudioIcon name="play" width="14" height="14" /> Try an example</button>
          <Link href="/methodology">How does this work? <StudioIcon name="arrow" width="15" height="15" /></Link>
        </div>
      </header>

      <div className="forecast-brief-editor" hidden={!isEditing}>
          {state.status === "error" && <ErrorState message={state.message} onRetry={() => runForecast(state.request)} />}
          <ForecastForm
            onSubmit={runForecast}
            onReset={() => setState({ status: "idle" })}
            isLoading={isLoading}
            canLookupChannel={isAuthenticated}
            connectedChannel={audience === "connected" && connection?.channelId
              ? { id: connection.channelId, title: connection.channelTitle }
              : null}
            isCheckingChannel={Boolean(user && connection?.userId !== user.id)}
          />
      </div>

      {!isEditing && <div className="forecast-review">
          {state.status === "loading" && <LoadingState />}

          {state.status === "success" && (
            <>
            {state.isExample && <div className="example-notice" role="status"><div><strong>You’re exploring an example</strong><p>Illustrative numbers, not a real prediction. This example is not saved to your history.</p></div><button type="button" onClick={closeExample}>Close example <span aria-hidden="true">×</span></button></div>}
            <ForecastResults
              request={state.request}
              response={state.response}
              historySaveNotice={state.historySaveNotice}
              onChangeInputs={focusForm}
              channelConnected={audience === "connected"}
              detailsContent={!state.isExample ? <ForecastOnboarding audience={audience} /> : undefined}
            />
            </>
          )}
      </div>}
      <p className="sr-only" role="status">{!isEditing && state.status === "success" ? "Forecast ready. Review your four view estimates in Overview." : ""}</p>
    </main>
  );
}
