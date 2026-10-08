// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import RecommendationCards from "./RecommendationCards";

afterEach(cleanup);

describe("RecommendationCards", () => {
  it("distinguishes supported changes, aligned choices, and benchmarks", () => {
    render(
      <RecommendationCards
        recommendations={[
          {
            id: "timing",
            type: "timing",
            status: "change",
            title: "Test a stronger window",
            guidance: "The comparison passed the release threshold.",
            recommendedPublishingWindow: {
              day: "Saturday",
              startHour: 18,
              endHour: 21,
              timeZone: "Asia/Colombo",
            },
            evidence: [{ label: "Timing", detail: "Supported timing evidence." }],
          },
          {
            id: "duration",
            type: "duration",
            status: "benchmark",
            title: "Compare duration benchmarks",
            guidance: "This is exploratory rather than a proven improvement.",
            evidence: [{ label: "Duration", detail: "Supported duration evidence." }],
          },
          {
            id: "format",
            type: "format",
            status: "aligned",
            title: "Keep the current format",
            guidance: "The current choice aligns with the evidence.",
            evidence: [{ label: "Format", detail: "Supported format evidence." }],
          },
        ]}
        unavailableRecommendations={[
          { type: "title", reason: "Title evidence is unavailable." },
        ]}
      />,
    );

    expect(screen.getByText("Change supported")).toBeTruthy();
    expect(screen.getByText("Benchmark to consider")).toBeTruthy();
    expect(screen.getByText("Already aligned")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Remaining data limitations" })).toBeTruthy();
    expect(screen.getByText(/Title evidence is unavailable/)).toBeTruthy();
  });
});
