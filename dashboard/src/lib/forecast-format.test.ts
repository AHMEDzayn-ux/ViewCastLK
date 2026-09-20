import { describe, expect, it } from "vitest";

import { formatForecastViews } from "@/lib/forecast-format";

describe("formatForecastViews", () => {
  it("never tells a creator their video will get zero views", () => {
    expect(formatForecastViews(0)).toBe("fewer than 10");
  });

  it("groups every value below the threshold, because the model cannot separate them", () => {
    expect(formatForecastViews(1)).toBe("fewer than 10");
    expect(formatForecastViews(9)).toBe("fewer than 10");
  });

  it("shows the number once it is meaningful", () => {
    expect(formatForecastViews(10)).toBe("10");
    expect(formatForecastViews(1250)).toBe("1,250");
  });

  it("rounds rather than printing a fraction", () => {
    expect(formatForecastViews(1250.6)).toBe("1,251");
  });

  it("does not print NaN or a negative count", () => {
    expect(formatForecastViews(Number.NaN)).toBe("unavailable");
    expect(formatForecastViews(-5)).toBe("unavailable");
  });
});
