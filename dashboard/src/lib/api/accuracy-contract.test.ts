import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/supabase/client", () => ({
  supabase: { auth: { getSession: vi.fn() } },
}));

import { isAccuracyResponse } from "./forecast";
import type { AvailableAccuracyResponse } from "@/types/forecast";

// The API publishes measured accuracy from a file stored beside each model.
// Reading those real files here means a change on either side that breaks the
// other fails in this suite, rather than as "unsupported accuracy information"
// on the live accuracy page.
const ARTIFACTS = resolve(__dirname, "../../../../prediction_api/model_artifacts");

function publishedFiles(): string[] {
  if (!existsSync(ARTIFACTS)) return [];
  return readdirSync(ARTIFACTS)
    .map((name) => join(ARTIFACTS, name, "published_accuracy.json"))
    .filter((path) => existsSync(path));
}

function asServed(path: string): AvailableAccuracyResponse {
  // The endpoint adds the status and data source around the stored figures.
  return {
    status: "available",
    dataSource: "prediction_api",
    ...JSON.parse(readFileSync(path, "utf-8")),
  };
}

describe("published accuracy contract", () => {
  it("finds at least one published accuracy file", () => {
    expect(publishedFiles().length).toBeGreaterThan(0);
  });

  it.each(publishedFiles())("the dashboard accepts %s", (path) => {
    expect(isAccuracyResponse(asServed(path))).toBe(true);
  });
});

describe("isAccuracyResponse for measured results", () => {
  const valid = asServed(publishedFiles()[0]);

  it("rejects an evaluation without a channel segment", () => {
    const [first, ...rest] = valid.evaluations;
    const broken = {
      ...valid,
      evaluations: [{ ...first, segment: undefined }, ...rest],
    };
    expect(isAccuracyResponse(broken)).toBe(false);
  });

  it("rejects the same horizon and segment reported twice", () => {
    const broken = { ...valid, evaluations: [valid.evaluations[0], valid.evaluations[0]] };
    expect(isAccuracyResponse(broken)).toBe(false);
  });

  it("rejects a metric that does not say which direction is better", () => {
    const [first, ...rest] = valid.evaluations;
    const broken = {
      ...valid,
      evaluations: [
        {
          ...first,
          metrics: first.metrics.map((metric) => ({ ...metric, betterWhen: "sideways" })),
        },
        ...rest,
      ],
    };
    expect(isAccuracyResponse(broken)).toBe(false);
  });

  it("rejects an available response with no evaluations", () => {
    expect(isAccuracyResponse({ ...valid, evaluations: [] })).toBe(false);
  });
});
