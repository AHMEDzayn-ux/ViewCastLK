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

export const AUDIO_LANGUAGES = [
  "Sinhala",
  "Tamil",
  "English",
  "Mixed / multilingual",
  "Other",
] as const;

export type AudioLanguage = (typeof AUDIO_LANGUAGES)[number];

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
export type ForecastHorizon = 7 | 14 | 21 | 30;

export interface ForecastRequest {
  title: string;
  category: YoutubeCategory;
  durationSeconds: number;
  audioLanguage: AudioLanguage;
  madeForKids: boolean;
  channelIdentifier: string;
  plannedPublishDay: PublishDay | null;
  plannedPublishHour: number | null;
}

export type MadeForKidsSelection = "" | "yes" | "no";

export interface ForecastFormValues {
  title: string;
  category: YoutubeCategory | "";
  durationMinutes: string;
  durationSeconds: string;
  audioLanguage: AudioLanguage | "";
  madeForKids: MadeForKidsSelection;
  channelIdentifier: string;
  plannedPublishDay: PublishDay | "";
  plannedPublishHour: string;
}

export interface ForecastEstimate {
  horizonDays: ForecastHorizon;
  cumulativeViews: number;
}

export type RecommendationType = "timing" | "duration" | "format" | "title";

export interface RecommendationEvidence {
  label: string;
  detail: string;
}

export interface Recommendation {
  id: string;
  type: RecommendationType;
  title: string;
  guidance: string;
  evidence: RecommendationEvidence[];
}

export type SupportingSource =
  | "channel_lookup"
  | "title_analysis"
  | "historical_recommendations";

export interface DataCompletenessIssue {
  source: SupportingSource;
  message: string;
}

export interface DataCompleteness {
  status: "complete" | "degraded";
  issues: DataCompletenessIssue[];
}

export interface TitleGuidance {
  summary: string;
  suggestions: string[];
}

export interface ModelMetadata {
  modelVersion: string;
  generatedAt: string;
  dataSource: "prediction_api" | "mock";
}

export interface ForecastResponse {
  forecastId: string;
  estimates: [
    ForecastEstimate,
    ForecastEstimate,
    ForecastEstimate,
    ForecastEstimate,
  ];
  recommendations: Recommendation[];
  completeness: DataCompleteness;
  titleGuidance?: TitleGuidance;
  model: ModelMetadata;
}

export type AccuracyMetricKey = "mape" | "mae" | "rmse" | "r2";

export interface AccuracyMetric {
  key: AccuracyMetricKey;
  label: string;
  description: string;
  unit: "percent" | "views" | "score";
  modelValue: number | null;
  baselineValue: number | null;
}

export interface AccuracyResponse {
  status: "available" | "unavailable";
  modelName: string;
  baselineName: string;
  evaluatedAt: string | null;
  metrics: AccuracyMetric[];
  dataSource: "prediction_api" | "mock";
  message?: string;
}

export interface ForecastValidationErrors {
  title?: string;
  category?: string;
  duration?: string;
  audioLanguage?: string;
  madeForKids?: string;
  channelIdentifier?: string;
  plannedPublishDay?: string;
  plannedPublishHour?: string;
}
