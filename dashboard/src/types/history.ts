import type {
  AudioLanguage,
  DataCompletenessIssue,
  PublishDay,
  YoutubeCategory,
} from "@/types/forecast";

export interface ForecastHistoryRow {
  id: string;
  user_id: string;
  created_at: string;
  forecast_id: string;
  title: string;
  category: YoutubeCategory;
  duration_seconds: number;
  audio_language: AudioLanguage;
  channel_identifier: string;
  planned_publish_day: PublishDay | null;
  planned_publish_hour: number | null;
  day_7_cumulative_views: number;
  day_14_cumulative_views: number;
  day_21_cumulative_views: number;
  day_30_cumulative_views: number;
  model_version: string;
  forecast_generated_at: string;
  model_data_source: "prediction_api" | "mock";
  completeness_status: "complete" | "degraded";
  completeness_issues: DataCompletenessIssue[];
}

export type ForecastHistoryInsert = Omit<
  ForecastHistoryRow,
  "id" | "created_at"
>;
