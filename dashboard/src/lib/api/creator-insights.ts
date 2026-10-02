import { supabase } from "@/lib/supabase/client";
import { isDevelopmentMockMode } from "@/lib/api/forecast";
import type { CreatorInsights } from "@/types/creator-insights";

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim().replace(/\/$/, "");

export type CreatorInsightsResult =
  | { status: "ready"; insights: CreatorInsights; example?: boolean }
  | { status: "signed_out" }
  | { status: "not_connected" }
  | { status: "error"; message: string };

/** Illustrative data for mock mode, so the section can be previewed offline. */
export const EXAMPLE_CREATOR_INSIGHTS: CreatorInsights = {
  channelTitle: "Example channel",
  videosSynced: 120,
  videosMeasured: 112,
  periodStart: "2026-05-02",
  periodEnd: "2026-09-28",
  mainCategory: "Education",
  spacing: {
    uploadsCounted: 119,
    shareWithinHour: 38.7,
    medianGapHours: 9.5,
    buckets: [
      { key: "under_1h", label: "Within an hour of the next upload", videos: 44, effectPct: -18.2, lowPct: -31.0, highPct: -4.1 },
      { key: "1h_to_6h", label: "1 to 6 hours before the next upload", videos: 21, effectPct: 2.4, lowPct: -15.8, highPct: 24.0 },
      { key: "6h_plus", label: "6 hours or more before the next upload", videos: 47, effectPct: 14.9, lowPct: 1.2, highPct: 30.3 },
    ],
  },
  timing: [
    { key: "morning", label: "Morning (06:00–12:00)", videos: 31, effectPct: -9.6, lowPct: -24.5, highPct: 7.8 },
    { key: "afternoon", label: "Afternoon (12:00–18:00)", videos: 38, effectPct: -3.1, lowPct: -16.0, highPct: 11.9 },
    { key: "evening", label: "Evening (18:00–24:00)", videos: 39, effectPct: 16.4, lowPct: 2.0, highPct: 32.7 },
  ],
  format: {
    shorts: 40,
    regular: 72,
    shortsVsRegular: { videos: 112, effectPct: 22.5, lowPct: -3.0, highPct: 54.1 },
  },
  growth: { videos: 64, medianGrowthPct: 9.8 },
};

export async function getCreatorInsights(): Promise<CreatorInsightsResult> {
  if (isDevelopmentMockMode()) {
    return { status: "ready", insights: EXAMPLE_CREATOR_INSIGHTS, example: true };
  }
  if (!API_BASE_URL) {
    return { status: "error", message: "Channel insights are not configured in this environment." };
  }
  const { data } = await supabase.auth.getSession();
  const accessToken = data.session?.access_token;
  if (!accessToken) return { status: "signed_out" };

  try {
    const response = await fetch(`${API_BASE_URL}/creator/insights`, {
      headers: { Accept: "application/json", Authorization: `Bearer ${accessToken}` },
    });
    const payload = await response.json().catch(() => null);
    if (response.status === 401) return { status: "signed_out" };
    if (response.status === 404) return { status: "not_connected" };
    if (!response.ok || !payload) {
      return {
        status: "error",
        message: payload?.message ?? "Your channel insights are temporarily unavailable.",
      };
    }
    return { status: "ready", insights: payload as CreatorInsights };
  } catch {
    return { status: "error", message: "Your channel insights are temporarily unavailable." };
  }
}
