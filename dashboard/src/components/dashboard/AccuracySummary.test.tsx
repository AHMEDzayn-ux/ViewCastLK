import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import AccuracySummary from "./AccuracySummary";
import type {
  AvailableAccuracyResponse,
  UnavailableAccuracyResponse,
} from "@/types/forecast";


describe("AccuracySummary", () => {
  it("renders a neutral unavailable state without metric or baseline UI", () => {
    const accuracy: UnavailableAccuracyResponse = {
      status: "unavailable",
      modelName: "viewcastlk_monotonic_trajectory_experimental_v1",
      evaluatedAt: null,
      evaluations: [],
      dataSource: "prediction_api",
      message: "Evaluation results are not available yet.",
    };

    const html = renderToStaticMarkup(<AccuracySummary accuracy={accuracy} />);

    expect(html).toContain("Evaluation results are not available yet");
    expect(html).toContain("Evaluation pending");
    expect(html).not.toContain("We could not load");
    expect(html).not.toContain("Accuracy view");
    expect(html).not.toContain("MAPE");
    expect(html).not.toContain("Baseline");
  });
});

describe("AccuracySummary with measured results", () => {
  const accuracy: AvailableAccuracyResponse = {
    status: "available",
    modelName: "viewcastlk_reconciled_latest_clean_20260918_v9",
    evaluatedAt: "2026-10-01T22:12:00Z",
    periodStart: "2026-09-18",
    periodEnd: "2026-09-24",
    method: "Scored on videos published after the model's training data ended.",
    dataSource: "prediction_api",
    evaluations: [
      {
        scope: "day_7",
        segment: "tracked_channel",
        segmentLabel: "Channels ViewCastLK already tracks",
        videos: 7616,
        baselineName: "The channel's usual views",
        metrics: [
          { key: "within_2x", label: "Forecasts within a factor of two", description: "Share within half to double.", unit: "percent", betterWhen: "higher", modelValue: 44.7, baselineValue: 43.5 },
          { key: "typical_factor", label: "Typical error", description: "A typical multiple off.", unit: "factor", betterWhen: "lower", modelValue: 2.2, baselineValue: 2.31 },
          { key: "rank_correlation", label: "Ranking accuracy", description: "Ordering videos.", unit: "score", betterWhen: "higher", modelValue: 0.828, baselineValue: null },
        ],
      },
      {
        scope: "day_7",
        segment: "new_channel",
        segmentLabel: "Channels new to ViewCastLK",
        videos: 106,
        baselineName: "Typical views across all channels",
        metrics: [
          { key: "within_2x", label: "Forecasts within a factor of two", description: "Share within half to double.", unit: "percent", betterWhen: "higher", modelValue: 25.5, baselineValue: 19.8 },
        ],
      },
    ],
    notYetMeasured: [{ scope: "day_30", measurableFrom: "2026-10-18" }],
  };

  const html = renderToStaticMarkup(<AccuracySummary accuracy={accuracy} />);

  it("leads with the first measured figure against the creator's own baseline", () => {
    expect(html).toContain("Forecasts within a factor of two");
    expect(html).toContain("44.7%");
    expect(html).toContain("43.5%");
    expect(html).toContain("The channel&#x27;s usual views");
    expect(html).toContain("A higher value is better.");
  });

  it("shows the error as a multiple and says lower is better for it", () => {
    expect(html).toContain("2.2\u00d7");
    expect(html).toContain("A lower value is better.");
  });

  it("offers both channel groups and states the sample and period", () => {
    expect(html).toContain("Channels ViewCastLK already tracks");
    expect(html).toContain("Channels new to ViewCastLK");
    expect(html).toContain("7,616");
    expect(html).toContain("Not applicable");
  });

  it("says when the remaining horizons can be measured", () => {
    expect(html).toContain("Day 30 accuracy can be measured from");
  });
});
