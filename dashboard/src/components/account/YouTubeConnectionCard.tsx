"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import {
  getYouTubeConnection,
  startYouTubeConnection,
  type YouTubeConnectionStatus,
} from "@/lib/api/youtube-connection";

type ViewState =
  | { kind: "idle" | "loading" }
  | { kind: "loaded"; connection: YouTubeConnectionStatus }
  | { kind: "error"; message: string };

const STATUS_COPY: Record<string, string> = {
  pending_sync: "Connected. Your private channel history is waiting for its first sync.",
  active: "Connected and ready for personalised forecasts.",
  reauth_required: "Google access needs to be renewed. Reconnect your channel.",
  error: "The last channel sync did not complete. Reconnect to restore access.",
};

export default function YouTubeConnectionCard() {
  const { isAuthenticated, isLoading } = useAuth();
  const [state, setState] = useState<ViewState>({ kind: "idle" });
  const [isConnecting, setIsConnecting] = useState(false);

  useEffect(() => {
    if (isLoading || !isAuthenticated) return;
    let isActive = true;
    getYouTubeConnection()
      .then((connection) => {
        if (isActive) setState({ kind: "loaded", connection });
      })
      .catch(() => {
        if (isActive) {
          setState({
            kind: "error",
            message: "We could not load your YouTube connection. Please try again.",
          });
        }
      });
    return () => {
      isActive = false;
    };
  }, [isAuthenticated, isLoading]);

  async function connect() {
    setIsConnecting(true);
    setState((current) =>
      current.kind === "error" ? { kind: "loading" } : current,
    );
    try {
      const authorizationUrl = await startYouTubeConnection();
      window.location.assign(authorizationUrl);
    } catch {
      setState({
        kind: "error",
        message: "We could not start the secure Google connection. Please try again.",
      });
      setIsConnecting(false);
    }
  }

  if (isLoading) {
    return (
      <section className="connection-card" aria-busy="true" role="status">
        <p>Checking your account…</p>
      </section>
    );
  }

  if (!isAuthenticated) {
    return (
      <section className="connection-card">
        <h2>Sign in to connect a channel</h2>
        <p>Your YouTube connection is private to your ViewCastLK account.</p>
        <Link className="primary-button" href="/login?next=%2Faccount">
          Sign in
        </Link>
      </section>
    );
  }

  if (state.kind === "idle" || state.kind === "loading") {
    return (
      <section className="connection-card" aria-busy="true" role="status">
        <p>Checking your YouTube connection…</p>
      </section>
    );
  }

  const connection = state.kind === "loaded" ? state.connection : null;
  const isConnected = connection?.isConnected === true;
  const needsReconnect =
    connection?.status === "reauth_required" || connection?.status === "error";

  return (
    <section className="connection-card" aria-labelledby="youtube-connection-title">
      <div>
        <p className="section-kicker">Private creator data</p>
        <h2 id="youtube-connection-title">
          {isConnected
            ? connection.channelTitle || "Connected YouTube channel"
            : "Connect your YouTube channel"}
        </h2>
        {isConnected ? (
          <>
            <p>{STATUS_COPY[connection.status ?? "pending_sync"]}</p>
            <p className="connection-card__identity">
              Channel ID: <span>{connection.channelId}</span>
            </p>
          </>
        ) : (
          <p>
            Allow read-only access so ViewCastLK can prepare private,
            creator-specific forecast adjustments. Your Analytics are not used
            to train the shared model.
          </p>
        )}
        {state.kind === "error" && (
          <p className="connection-card__error" role="alert">
            {state.message}
          </p>
        )}
      </div>

      <div className="connection-card__actions">
        <button
          className="primary-button"
          type="button"
          disabled={isConnecting}
          onClick={connect}
        >
          {isConnecting
            ? "Opening Google…"
            : isConnected || needsReconnect
              ? "Reconnect channel"
              : "Connect with Google"}
        </button>
        <Link href="/privacy">Review the privacy policy</Link>
      </div>
    </section>
  );
}
