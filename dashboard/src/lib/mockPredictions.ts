import type {
  ForecastInput,
  ForecastResult,
  HorizonResult,
  Recommendation,
} from "@/types/forecast";

// ─── Deterministic seeding ────────────────────────────────────────────────────
// Produces the same output for the same inputs so repeated submissions of the
// same form values always show the same chart, while different inputs give
// meaningfully different curves.

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return h;
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ─── Category base multipliers ────────────────────────────────────────────────
// Rough category-level scale factors — larger audiences = higher base views.
const CATEGORY_MULTIPLIERS: Record<string, number> = {
  "Film & Animation": 1.4,
  "Autos & Vehicles": 0.8,
  Music: 2.2,
  "Pets & Animals": 0.9,
  Sports: 1.6,
  "Travel & Events": 1.0,
  Gaming: 1.8,
  "People & Blogs": 0.9,
  Comedy: 1.5,
  Entertainment: 1.7,
  "News & Politics": 1.3,
  "Howto & Style": 1.1,
  Education: 1.0,
  "Science & Technology": 1.1,
  "Nonprofits & Activism": 0.6,
};

// ─── Publish-day engagement patterns ─────────────────────────────────────────
const DAY_MULTIPLIERS: Record<string, number> = {
  Monday: 1.0,
  Tuesday: 1.05,
  Wednesday: 1.08,
  Thursday: 1.03,
  Friday: 1.12,
  Saturday: 1.15,
  Sunday: 1.10,
};

// ─── Duration sweet-spot bonus ────────────────────────────────────────────────
// Videos in the 6–15 min band tend to perform slightly better for most categories.
function durationBonus(minutes: number): number {
  if (minutes < 1) return 0.75;
  if (minutes < 3) return 0.85;
  if (minutes < 6) return 0.95;
  if (minutes <= 15) return 1.0;
  if (minutes <= 30) return 0.92;
  return 0.82;
}

// ─── Core mock prediction generator ──────────────────────────────────────────

export function generateMockPrediction(input: ForecastInput): ForecastResult {
  const seedStr = `${input.category}|${input.publishDay}|${input.durationMinutes}|${input.publishHour}|${input.channelHandle.toLowerCase()}`;
  const rand = seededRandom(hashString(seedStr));

  const catMult = CATEGORY_MULTIPLIERS[input.category] ?? 1.0;
  const dayMult = DAY_MULTIPLIERS[input.publishDay] ?? 1.0;
  const durBonus = durationBonus(input.durationMinutes);

  // Base median at day 7 (views)
  const base7 = Math.round(
    4000 * catMult * dayMult * durBonus * (0.8 + rand() * 0.4)
  );

  // Growth curve: log-shaped plateau
  const growth = [1.0, 1.52 + rand() * 0.2, 1.82 + rand() * 0.2, 2.0 + rand() * 0.2];
  const jitter = () => 0.92 + rand() * 0.16;

  const makeHorizon = (
    day: 7 | 14 | 21 | 30,
    multiplier: number
  ): HorizonResult => {
    const median = Math.round(base7 * multiplier * jitter());
    const spread = 0.28 + rand() * 0.12; // 28–40% spread
    return {
      day,
      low: Math.round(median * (1 - spread)),
      median,
      high: Math.round(median * (1 + spread)),
    };
  };

  const horizons: [HorizonResult, HorizonResult, HorizonResult, HorizonResult] =
    [
      makeHorizon(7, growth[0]),
      makeHorizon(14, growth[1]),
      makeHorizon(21, growth[2]),
      makeHorizon(30, growth[3]),
    ];

  const recommendations = buildRecommendations(input, rand);

  return {
    horizons,
    recommendations,
    generatedAt: new Date().toISOString(),
    isMock: true,
  };
}

// ─── Recommendation builder ───────────────────────────────────────────────────
// All headlines and body text below are illustrative mock wording.
// They will be replaced with approved EDA-based recommendations after
// the team's exploratory analysis is complete.

function buildRecommendations(
  input: ForecastInput,
  rand: () => number
): Recommendation[] {
  const recs: Recommendation[] = [];

  // Timing — illustrative example only
  const hour = input.publishHour;
  if (hour < 8 || hour >= 22) {
    recs.push({
      type: "timing",
      headline: "Outside the most common publish window in this dataset",
      body: "This is illustrative mock wording. Your selected time falls outside 08:00–22:00 SLT. Real timing recommendations will be derived from approved exploratory analysis of the collected dataset.",
    });
  } else if (hour >= 17 && hour <= 21) {
    recs.push({
      type: "timing",
      headline: "Within an evening publish window",
      body: "This is illustrative mock wording. Your planned time is in the 17:00–21:00 SLT range. Real timing recommendations will be derived from approved exploratory analysis of the collected dataset.",
    });
  } else {
    recs.push({
      type: "timing",
      headline: "Within a daytime publish window",
      body: "This is illustrative mock wording. Your planned time falls in standard daytime hours. Real timing recommendations will be derived from approved exploratory analysis of the collected dataset.",
    });
  }

  // Duration — illustrative example only
  const totalMin =
    input.durationMinutes + input.durationSeconds / 60;
  if (totalMin < 3) {
    recs.push({
      type: "duration",
      headline: "Short planned duration",
      body: `This is illustrative mock wording. Your planned duration is under 3 minutes. Real duration guidance will come from approved analysis of the collected dataset, not from this placeholder.`,
    });
  } else if (totalMin > 30) {
    recs.push({
      type: "duration",
      headline: "Long planned duration",
      body: `This is illustrative mock wording. Your planned duration exceeds 30 minutes. Real duration guidance will come from approved analysis of the collected dataset, not from this placeholder.`,
    });
  } else {
    recs.push({
      type: "duration",
      headline: "Duration within a common range",
      body: `This is illustrative mock wording. Your planned duration (${input.durationMinutes}m ${input.durationSeconds}s) is within a typical range for ${input.category || "this category"}. Real guidance will come from approved analysis of the collected dataset.`,
    });
  }

  // Publication day — illustrative example only
  const weekendOrFriday = ["Friday", "Saturday", "Sunday"];
  if (weekendOrFriday.includes(input.publishDay)) {
    recs.push({
      type: "general",
      headline: `${input.publishDay} is a weekend or Friday publication`,
      body: "This is illustrative mock wording. Real publication-day guidance will be derived from approved exploratory analysis of the collected dataset, not from this placeholder.",
    });
  } else if (rand() > 0.5) {
    recs.push({
      type: "general",
      headline: "Weekday publication selected",
      body: "This is illustrative mock wording. Real publication-day guidance will be derived from approved exploratory analysis of the collected dataset, not from this placeholder.",
    });
  }

  return recs.slice(0, 3);
}
