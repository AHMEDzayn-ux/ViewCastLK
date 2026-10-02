/**
 * Shape of src/data/dataset.json, written by scripts/build_dataset_release.py.
 */

export type ColumnSource = "youtube" | "collected" | "computed";

export interface DatasetColumn {
  name: string;
  type: string;
  group: string;
  meaning: string;
  source: ColumnSource;
  how: string;
  emptyPct: number;
}

export interface DatasetFile {
  name: string;
  format: string;
  bytes: number;
  sha256: string;
  url: string;
}

export interface DatasetRelease {
  generatedAt: string;
  releaseTag: string;
  videos: number;
  channels: number;
  periodStart: string;
  periodEnd: string;
  toleranceHours: number;
  labelled: Record<string, number>;
  files: DatasetFile[];
  columns: DatasetColumn[];
}
