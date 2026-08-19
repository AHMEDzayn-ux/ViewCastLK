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

export interface ChannelStats {
  subscriberCount: number | null;
  totalViewCount: number | null;
  videoCount: number | null;
  createdAt: string | null;
}

export interface ForecastRequest {
  title: string;
  category: YoutubeCategory;
  durationSeconds: number;
  audioLanguage: AudioLanguage;
  channelIdentifier: string;
  plannedPublishDay: PublishDay | null;
  plannedPublishHour: number | null;
}

export interface ForecastFormValues {
  title: string;
  category: YoutubeCategory | "";
  durationMinutes: string;
  durationSeconds: string;
  audioLanguage: AudioLanguage | "";
  channelIdentifier: string;
  plannedPublishDay: PublishDay | "";
  plannedPublishHour: string;
}

export interface ForecastEstimate {
  horizonDays: ForecastHorizon;
  cumulativeViews: number;
}

export const RECOMMENDATION_TYPES = [
  "timing",
  "duration",
  "format",
  "title",
] as const;

export type RecommendationType = (typeof RECOMMENDATION_TYPES)[number];

export interface RecommendationEvidence {
  label: string;
  detail: string;
}

interface RecommendationBase {
  id: string;
  title: string;
  guidance: string;
  evidence: [RecommendationEvidence, ...RecommendationEvidence[]];
}

export interface RecommendedPublishingWindow {
  day: PublishDay;
  startHour: number;
  endHour: number;
  timeZone: "Asia/Colombo";
}

export interface TimingRecommendation extends RecommendationBase {
  type: "timing";
  recommendedPublishingWindow: RecommendedPublishingWindow;
}

export interface ContentRecommendation extends RecommendationBase {
  type: "duration" | "format" | "title";
}

export type Recommendation = TimingRecommendation | ContentRecommendation;

export interface UnavailableRecommendation {
  type: RecommendationType;
  reason: string;
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
  unavailableRecommendations: UnavailableRecommendation[];
  completeness: DataCompleteness;
  titleGuidance?: TitleGuidance;
  model: ModelMetadata;
}

export type AccuracyMetricKey = "mape" | "mae" | "rmse" | "r2";

export const ACCURACY_SCOPES = [
  "combined",
  "day_7",
  "day_14",
  "day_21",
  "day_30",
] as const;

export type AccuracyScope = (typeof ACCURACY_SCOPES)[number];

export interface AccuracyMetric {
  key: AccuracyMetricKey;
  label: string;
  description: string;
  unit: "percent" | "views" | "score";
  modelValue: number | null;
  baselineValue: number | null;
}

export interface AccuracyEvaluation {
  scope: AccuracyScope;
  metrics: AccuracyMetric[];
}

export interface AccuracyResponse {
  status: "available" | "unavailable";
  modelName: string;
  baselineName: string;
  evaluatedAt: string | null;
  evaluations: [
    AccuracyEvaluation,
    AccuracyEvaluation,
    AccuracyEvaluation,
    AccuracyEvaluation,
    AccuracyEvaluation,
  ];
  dataSource: "prediction_api" | "mock";
  message?: string;
}

export interface ForecastValidationErrors {
  title?: string;
  category?: string;
  duration?: string;
  audioLanguage?: string;
  channelIdentifier?: string;
  plannedPublishDay?: string;
  plannedPublishHour?: string;
}
