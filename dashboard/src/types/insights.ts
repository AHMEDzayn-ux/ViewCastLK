/**
 * Shape of src/data/insights.json, written by
 * Analysis/deep_dives/creator_insights/build_insights.py.
 *
 * Effects are percentage differences in day-7 views against the same
 * channel's normal, with a 95% range from a channel-level bootstrap.
 */

export interface EffectCell {
  label: string;
  videos: number;
  channels: number;
  effectPct: number;
  lowPct: number;
  highPct: number;
}

export interface GrowthRow {
  label: string;
  videos: number;
  channels: number;
  medianGrowthPct: number;
  upperQuartileGrowthPct: number;
  shareGrowing10Pct: number;
}

export interface InsightsData {
  generatedAt: string;
  dataset: {
    uploads: number;
    videos: number;
    channels: number;
    periodStart: string;
    periodEnd: string;
    minChannelVideos: number;
    minVideos: number;
    minChannels: number;
  };
  spacing: {
    byGapToNext: EffectCell[];
    byGapSincePrevious: EffectCell[];
    bySize: EffectCell[];
    byCategory: EffectCell[];
    shareWithinHour: number;
    medianGapHours: number;
    medianUploadsPrev24h: number;
  };
  timing: {
    byHour: EffectCell[];
    byHourBlock: EffectCell[];
    byDay: EffectCell[];
  };
  format: {
    overall: EffectCell | Omit<EffectCell, "label"> | null;
    byCategory: EffectCell[];
    bySize: EffectCell[];
    durationByCategory: { category: string; bands: EffectCell[] }[];
  };
  growth: {
    videos: number;
    medianShareByDay7: number;
    byCategory: GrowthRow[];
    bySize: GrowthRow[];
    byFormat: GrowthRow[];
  };
  thumbnails: {
    videos: number;
    channels: number;
    overall: Omit<EffectCell, "label"> | null;
    byCategory: EffectCell[];
  };
}
