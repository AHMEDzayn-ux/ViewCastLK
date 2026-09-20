"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import {
  deleteForecastHistory,
  listForecastHistory,
} from "@/lib/history/forecast-history";
import type { ForecastHistoryRow } from "@/types/history";

type HistoryState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; records: ForecastHistoryRow[] }
  | { status: "error" };

function formatViews(value: number) {
  return value.toLocaleString("en-US");
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function formatDate(value: string) {
  return new Date(value).toLocaleString("en-LK", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Colombo",
  });
}

export default function HistoryView() {
  const { replace } = useRouter();
  const { isAuthenticated, isLoading } = useAuth();
  const [state, setState] = useState<HistoryState>({ status: "idle" });
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (isLoading) return;

    if (!isAuthenticated) {
      // Client-side route state is UX only. Supabase RLS is the real data
      // authorization boundary for every history read and deletion.
      replace("/login?next=%2Fhistory");
      return;
    }

    let isActive = true;

    void Promise.resolve()
      .then(() => {
        if (isActive) setState({ status: "loading" });
        return listForecastHistory();
      })
      .then((records) => {
        if (isActive) setState({ status: "success", records });
      })
      .catch(() => {
        if (isActive) setState({ status: "error" });
      });

    return () => {
      isActive = false;
    };
  }, [isAuthenticated, isLoading, replace]);

  async function handleDelete(record: ForecastHistoryRow) {
    const confirmed = window.confirm(
      `Delete the saved forecast for “${record.title}”?`,
    );
    if (!confirmed || deletingId) return;

    setDeletingId(record.id);
    setDeleteError(null);

    try {
      await deleteForecastHistory(record.id);
      setState((current) =>
        current.status === "success"
          ? {
              status: "success",
              records: current.records.filter((item) => item.id !== record.id),
            }
          : current,
      );
    } catch {
      setDeleteError(
        "The saved forecast could not be deleted. Please try again.",
      );
    } finally {
      setDeletingId(null);
    }
  }

  if (isLoading) {
    return (
      <section className="history-state" aria-busy="true" role="status">
        <p className="section-kicker">Your forecast memory</p>
        <h2>Checking your account</h2>
        <p>Please wait a moment.</p>
      </section>
    );
  }

  if (!isAuthenticated) {
    return (
      <section className="history-state">
        <p className="section-kicker">Sign in required</p>
        <h2>Your history is private to your account</h2>
        <p>Sign in to view forecasts saved for your account.</p>
        <Link className="primary-button history-state__action" href="/login?next=%2Fhistory">
          Sign in to view history
        </Link>
      </section>
    );
  }

  if (state.status === "idle" || state.status === "loading") {
    return (
      <section className="history-state" aria-busy="true" role="status">
        <p className="section-kicker">Your forecast memory</p>
        <h2>Loading saved forecasts</h2>
        <p>Retrieving your newest forecasts first.</p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="history-state" role="alert">
        <p className="section-kicker">History unavailable</p>
        <h2>We could not load your saved forecasts</h2>
        <p>Please refresh the page and try again.</p>
      </section>
    );
  }

  if (state.records.length === 0) {
    return (
      <section className="history-state">
        <p className="section-kicker">No saved forecasts yet</p>
        <h2>Your forecast history will appear here</h2>
        <p>Generate a forecast while signed in to save it automatically.</p>
        <Link className="primary-button history-state__action" href="/forecast">
          Create a forecast
        </Link>
      </section>
    );
  }

  return (
    <div className="history-list">
      {deleteError && (
        <div className="auth-message auth-message--error" role="alert">
          {deleteError}
        </div>
      )}

      {state.records.map((record) => (
        <article className="history-card" key={record.id}>
          <header className="history-card__header">
            <div>
              <p className="section-kicker">{record.category}</p>
              <h2 dir="auto">{record.title}</h2>
              <p>
                Forecast generated {formatDate(record.forecast_generated_at)} SLT
              </p>
            </div>
            <button
              className="history-card__delete"
              type="button"
              disabled={deletingId !== null}
              onClick={() => void handleDelete(record)}
              aria-label={`Delete saved forecast for ${record.title}`}
            >
              {deletingId === record.id ? "Deleting..." : "Delete"}
            </button>
          </header>

          <dl className="history-card__summary">
            <div>
              <dt>Channel</dt>
              <dd>{record.channel_identifier}</dd>
            </div>
            <div>
              <dt>Audio</dt>
              <dd>{record.audio_language}</dd>
            </div>
            <div>
              <dt>Duration</dt>
              <dd>{formatDuration(record.duration_seconds)}</dd>
            </div>
            <div>
              <dt>Publishing plan</dt>
              <dd>
                {record.planned_publish_day ?? "Not decided"}
                {record.planned_publish_hour === null
                  ? ""
                  : ` at ${String(record.planned_publish_hour).padStart(2, "0")}:00 SLT`}
              </dd>
            </div>
          </dl>

          <dl className="history-card__horizons" aria-label="Forecast checkpoints">
            {[
              [7, record.day_7_cumulative_views],
              [14, record.day_14_cumulative_views],
              [21, record.day_21_cumulative_views],
              [30, record.day_30_cumulative_views],
            ].map(([day, views]) => (
              <div key={day}>
                <dt>Day {day}</dt>
                <dd>{formatViews(views)}</dd>
              </div>
            ))}
          </dl>

          <footer className="history-card__footer">
            <span>Saved {formatDate(record.created_at)} SLT</span>
            <span>Model {record.model_version}</span>
          </footer>
        </article>
      ))}
    </div>
  );
}
