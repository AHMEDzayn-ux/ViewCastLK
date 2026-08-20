import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import AccuracySummary from "./AccuracySummary";
import type { UnavailableAccuracyResponse } from "@/types/forecast";


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
