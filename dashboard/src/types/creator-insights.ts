/** Response of GET /creator/insights (prediction_api/app/creator_insights.py). */

export interface CreatorEffect {
  videos: number;
  effectPct: number;
  lowPct: number;
  highPct: number;
}

export interface CreatorGroupEffect extends CreatorEffect {
  key: string;
  label: string;
}

export interface CreatorInsights {
  channelTitle: string | null;
  videosSynced: number;
  videosMeasured: number;
  periodStart: string | null;
  periodEnd: string | null;
  mainCategory: string | null;
  spacing: {
    uploadsCounted: number;
    shareWithinHour: number | null;
    medianGapHours: number | null;
    buckets: CreatorGroupEffect[];
  } | null;
  timing: CreatorGroupEffect[];
  format: {
    shorts: number;
    regular: number;
    shortsVsRegular: CreatorEffect | null;
  } | null;
  growth: { videos: number; medianGrowthPct: number } | null;
}
