"use client";

import { useEffect, useState } from "react";
import AccuracySummary from "./AccuracySummary";
import ErrorState from "./ErrorState";
import LoadingState from "./LoadingState";
import { getAccuracy } from "@/lib/api/forecast";
import type { AccuracyResponse } from "@/types/forecast";

type AccuracyState =
  | { status: "loading" }
  | { status: "success"; response: AccuracyResponse }
  | { status: "error"; message: string };

export default function AccuracyView() {
  const [state, setState] = useState<AccuracyState>({ status: "loading" });
  const [requestKey, setRequestKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    getAccuracy({ signal: controller.signal })
      .then((response) => setState({ status: "success", response }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;

        setState({
          status: "error",
          message:
            error instanceof Error
              ? error.message
              : "Accuracy information could not be loaded.",
        });
      });

    return () => controller.abort();
  }, [requestKey]);

  function retry() {
    setState({ status: "loading" });
    setRequestKey((current) => current + 1);
  }

  if (state.status === "loading") {
    return (
      <LoadingState
        eyebrow="Loading evaluation"
        title="Checking the published accuracy status"
        message="ViewCastLK is requesting approved metrics and baseline comparisons from the configured adapter."
      />
    );
  }
  if (state.status === "error") {
    return (
      <ErrorState
        eyebrow="Evaluation unavailable"
        title="We could not load the accuracy status"
        message={state.message}
        onRetry={retry}
      />
    );
  }

  return <AccuracySummary accuracy={state.response} />;
}
