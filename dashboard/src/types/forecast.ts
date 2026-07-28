// ─── Forecast domain types ────────────────────────────────────────────────────

export const YOUTUBE_CATEGORIES = [
  "Film & Animation",
  "Autos & Vehicles",
  "Music",
  "Pets & Animals",
  "Sports",
  "Travel & Events",
  "Gaming",
  "People & Blogs",
  "Comedy",
  "Entertainment",
  "News & Politics",
  "Howto & Style",
  "Education",
  "Science & Technology",
  "Nonprofits & Activism",
] as const;

export type YoutubeCategory = (typeof YOUTUBE_CATEGORIES)[number];

export const PUBLISH_DAYS = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
] as const;

export type PublishDay = (typeof PUBLISH_DAYS)[number];

export interface ForecastInput {
  /** Planned video title */
  title: string;
  /** YouTube category */
  category: YoutubeCategory | "";
  /** Planned duration – minutes component */
  durationMinutes: number;
  /** Planned duration – seconds component (0-59) */
  durationSeconds: number;
  /** Planned publish day of week */
  publishDay: PublishDay | "";
  /** Planned publish hour in Sri Lanka Time (0-23) */
  publishHour: number;
  /** Planned publish minute (0 or 30) */
  publishMinute: number;
  /** YouTube channel @handle or channel ID */
  channelHandle: string;
}

export interface HorizonResult {
  day: 7 | 14 | 21 | 30;
  /** Lower-bound estimate */
  low: number;
  /** Median (central) estimate */
  median: number;
  /** Upper-bound estimate */
  high: number;
}

export type RecommendationType = "timing" | "duration" | "category" | "general";

export interface Recommendation {
  type: RecommendationType;
  headline: string;
  body: string;
}

export interface ForecastResult {
  horizons: [HorizonResult, HorizonResult, HorizonResult, HorizonResult];
  recommendations: Recommendation[];
  /** ISO 8601 timestamp of when this result was generated */
  generatedAt: string;
  /** Always true for Phase 1 – mock data flag */
  isMock: true;
}

export interface ValidationErrors {
  title?: string;
  category?: string;
  duration?: string;
  publishDay?: string;
  channelHandle?: string;
}
