import { supabase } from "@/lib/supabase/client";
import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import type {
  ForecastHistoryInsert,
  ForecastHistoryRow,
} from "@/types/history";

const HISTORY_COLUMNS = [
  "id",
  "user_id",
  "created_at",
  "forecast_id",
  "title",
  "category",
  "duration_seconds",
  "audio_language",
  "channel_identifier",
  "planned_publish_day",
  "planned_publish_hour",
  "day_7_cumulative_views",
  "day_14_cumulative_views",
  "day_21_cumulative_views",
  "day_30_cumulative_views",
  "model_version",
  "forecast_generated_at",
  "model_data_source",
  "completeness_status",
  "completeness_issues",
].join(",");

export class ForecastHistoryError extends Error {
  constructor(operation: "save" | "list" | "delete") {
    super(`Forecast history ${operation} failed.`);
    this.name = "ForecastHistoryError";
  }
}

function cumulativeViews(response: ForecastResponse, horizonDays: number) {
  const estimate = response.estimates.find(
    (candidate) => candidate.horizonDays === horizonDays,
  );

  if (!estimate) {
    throw new ForecastHistoryError("save");
  }

  return estimate.cumulativeViews;
}

export function toForecastHistoryInsert(
  userId: string,
  request: ForecastRequest,
  response: ForecastResponse,
): ForecastHistoryInsert {
  return {
    user_id: userId,
    forecast_id: response.forecastId,
    title: request.title,
    category: request.category,
    duration_seconds: request.durationSeconds,
    audio_language: request.audioLanguage,
    channel_identifier: request.channelIdentifier,
    planned_publish_day: request.plannedPublishDay,
    planned_publish_hour: request.plannedPublishHour,
    day_7_cumulative_views: cumulativeViews(response, 7),
    day_14_cumulative_views: cumulativeViews(response, 14),
    day_21_cumulative_views: cumulativeViews(response, 21),
    day_30_cumulative_views: cumulativeViews(response, 30),
    model_version: response.model.modelVersion,
    forecast_generated_at: response.model.generatedAt,
    model_data_source: response.model.dataSource,
    completeness_status: response.completeness.status,
    completeness_issues: response.completeness.issues,
  };
}

export async function saveForecastHistory(
  userId: string,
  request: ForecastRequest,
  response: ForecastResponse,
): Promise<void> {
  const record = toForecastHistoryInsert(userId, request, response);
  const { error } = await supabase.from("forecast_history").insert(record);

  if (error) {
    throw new ForecastHistoryError("save");
  }
}

export async function listForecastHistory(): Promise<ForecastHistoryRow[]> {
  const { data, error } = await supabase
    .from("forecast_history")
    .select(HISTORY_COLUMNS)
    .order("created_at", { ascending: false });

  if (error || !data) {
    throw new ForecastHistoryError("list");
  }

  return data as unknown as ForecastHistoryRow[];
}

export async function deleteForecastHistory(historyId: string): Promise<void> {
  const { error } = await supabase
    .from("forecast_history")
    .delete()
    .eq("id", historyId);

  if (error) {
    throw new ForecastHistoryError("delete");
  }
}
