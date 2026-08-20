import { describe, expect, it } from "vitest";

import { isAccuracyResponse } from "./forecast";


const unavailableResponse = {
  status: "unavailable",
  modelName: "viewcastlk_monotonic_trajectory_experimental_v1",
  evaluatedAt: null,
  evaluations: [],
  dataSource: "prediction_api",
  message: "Evaluation results are not available yet.",
};


describe("isAccuracyResponse", () => {
  it("accepts the honest unavailable response", () => {
    expect(isAccuracyResponse(unavailableResponse)).toBe(true);
  });

  it("rejects unavailable responses containing invented evaluation content", () => {
    expect(
      isAccuracyResponse({
        ...unavailableResponse,
        baselineName: "Unapproved baseline",
      }),
    ).toBe(false);

    expect(
      isAccuracyResponse({
        ...unavailableResponse,
        evaluations: [{ scope: "combined", metrics: [] }],
      }),
    ).toBe(false);
  });
});
