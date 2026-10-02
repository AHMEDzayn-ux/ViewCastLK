import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import insights from "@/data/insights.json";
import { EXAMPLE_CREATOR_INSIGHTS } from "@/lib/api/creator-insights";
import type { CreatorInsights } from "@/types/creator-insights";
import type { InsightsData } from "@/types/insights";
import { ChannelPattern } from "./YourChannelInsights";

// The section fetches the signed-in creator's data; neither the session nor
// the Supabase client exists in a unit test.
vi.mock("@/lib/supabase/client", () => ({
  supabase: { auth: { getSession: async () => ({ data: { session: null } }) } },
}));
vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({ isAuthenticated: false, isLoading: true }),
}));

const benchmarks = insights as InsightsData;

describe("ChannelPattern", () => {
  it("compares the creator's spacing with the population", () => {
    const html = renderToStaticMarkup(
      <ChannelPattern insights={EXAMPLE_CREATOR_INSIGHTS} benchmarks={benchmarks} />,
    );
    expect(html).toContain("39%");
    expect(html).toContain(`${Math.round(benchmarks.spacing.shareWithinHour)}%`);
    expect(html).toContain("Evening");
  });

  it("hedges an effect whose range includes zero", () => {
    const html = renderToStaticMarkup(
      <ChannelPattern insights={EXAMPLE_CREATOR_INSIGHTS} benchmarks={benchmarks} />,
    );
    // The example Shorts effect runs from -3% to +54%.
    expect(html).toContain("includes no difference");
  });

  it("does not claim a best time when none is clear", () => {
    const flat: CreatorInsights = {
      ...EXAMPLE_CREATOR_INSIGHTS,
      timing: EXAMPLE_CREATOR_INSIGHTS.timing.map((t) => ({ ...t, lowPct: -10, highPct: 10 })),
      format: { shorts: 2, regular: 50, shortsVsRegular: null },
      growth: null,
    };
    const html = renderToStaticMarkup(<ChannelPattern insights={flat} benchmarks={benchmarks} />);
    expect(html).toContain("No clear winner");
    expect(html).toContain("you have 2 and 50");
    expect(html).toContain("Not enough of your videos are 30 days old yet");
  });
});
