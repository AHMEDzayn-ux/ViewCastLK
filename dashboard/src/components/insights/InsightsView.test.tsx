import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import insights from "@/data/insights.json";
import type { EffectCell, InsightsData } from "@/types/insights";
import { effectTone, formatEffect } from "./EffectChart";
import InsightsView from "./InsightsView";

// The section fetches the signed-in creator's data; neither the session nor
// the Supabase client exists in a unit test.
vi.mock("@/lib/supabase/client", () => ({
  supabase: { auth: { getSession: async () => ({ data: { session: null } }) } },
}));
vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({ isAuthenticated: false, isLoading: true }),
}));

const data = insights as InsightsData;

function allCells(d: InsightsData): EffectCell[] {
  return [
    ...d.spacing.byGapToNext,
    ...d.spacing.byGapSincePrevious,
    ...d.spacing.bySize,
    ...d.spacing.byCategory,
    ...d.timing.byHour,
    ...d.timing.byHourBlock,
    ...d.timing.byDay,
    ...d.format.byCategory,
    ...d.format.bySize,
    ...d.format.durationByCategory.flatMap((c) => c.bands),
    ...d.thumbnails.byCategory,
  ];
}

describe("published insights data", () => {
  it("only publishes groups that meet the stated minimums", () => {
    const { minVideos, minChannels } = data.dataset;
    for (const cell of allCells(data)) {
      // Two-group contrasts need half the minimum in each group, so the
      // combined count is at least the minimum.
      expect(cell.videos, cell.label).toBeGreaterThanOrEqual(minVideos);
      expect(cell.channels, cell.label).toBeGreaterThanOrEqual(minChannels);
    }
  });

  it("keeps every estimate inside its own 95% range", () => {
    for (const cell of allCells(data)) {
      expect(cell.lowPct, cell.label).toBeLessThanOrEqual(cell.effectPct);
      expect(cell.highPct, cell.label).toBeGreaterThanOrEqual(cell.effectPct);
    }
  });
});

describe("effect helpers", () => {
  it("calls a difference only when the range excludes zero", () => {
    expect(effectTone({ lowPct: 1.7, highPct: 15.5 })).toBe("up");
    expect(effectTone({ lowPct: -9.3, highPct: -1.8 })).toBe("down");
    expect(effectTone({ lowPct: -2.9, highPct: 9.8 })).toBe("unclear");
  });

  it("formats signed percentages with a true minus sign", () => {
    expect(formatEffect(8.4)).toBe("+8%");
    expect(formatEffect(-4.2)).toBe("−4%");
    expect(formatEffect(0.2)).toBe("0%");
  });
});

describe("InsightsView", () => {
  const html = renderToStaticMarkup(<InsightsView data={data} />);

  it("renders every section with its takeaway", () => {
    for (const heading of [
      "Give each upload some room",
      "Evening uploads do best",
      "Under four minutes? Make it a Short, or make it longer",
      "Shorts help smaller channels most",
      "Know when a video has had its run",
      "A face on the thumbnail is not a rule",
      "How these were measured",
    ]) {
      expect(html).toContain(heading.replace("'", "&#x27;"));
    }
    expect(html.match(/insight-takeaway/g)?.length).toBe(6);
  });

  it("states the caveats a creator needs", () => {
    expect(html).toContain("not proof of cause");
    expect(html).toContain("Shorts view each time a Short starts or replays");
    expect(html).toContain("changed thumbnail from the same address");
  });

  it("leads with the creator's own channel section", () => {
    expect(html).toContain("How these patterns look on your own uploads");
  });

  it("offers each chart's numbers as a table", () => {
    expect(html.match(/Show the numbers/g)?.length).toBeGreaterThanOrEqual(8);
  });
});
