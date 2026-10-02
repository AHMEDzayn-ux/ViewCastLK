import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import release from "@/data/dataset.json";
import type { DatasetRelease } from "@/types/dataset";
import DatasetView from "./DatasetView";

const data = release as DatasetRelease;

describe("published dataset description", () => {
  it("describes every column once, with a known source", () => {
    const names = data.columns.map((c) => c.name);
    expect(new Set(names).size).toBe(names.length);
    for (const column of data.columns) {
      expect(["youtube", "collected", "computed"]).toContain(column.source);
      expect(column.meaning.length, column.name).toBeGreaterThan(5);
      expect(column.how.length, column.name).toBeGreaterThan(5);
    }
  });

  it("leaves out columns that can be rebuilt from the ones released", () => {
    const names = new Set(data.columns.map((c) => c.name));
    for (const dropped of ["tag_count", "title_length", "publish_hour_slt", "d7_usable",
      "eligible", "channel_stats_backfilled", "thumbnail_url"]) {
      expect(names.has(dropped), dropped).toBe(false);
    }
    expect(names.has("is_short")).toBe(true);
    expect(names.has("channel_title")).toBe(true);
  });

  it("links every file to the release named in the data", () => {
    for (const file of data.files) {
      expect(file.url).toContain(`/releases/download/${data.releaseTag}/${file.name}`);
      expect(file.sha256).toMatch(/^[0-9a-f]{64}$/);
    }
  });
});

describe("DatasetView", () => {
  const html = renderToStaticMarkup(<DatasetView release={data} />);

  it("renders the dictionary and download links", () => {
    for (const column of data.columns) expect(html).toContain(column.name);
    for (const file of data.files) expect(html).toContain(file.url);
  });

  it("explains how to rebuild the dropped flags", () => {
    expect(html).toContain("ch_stats_as_of &gt; published_at");
    expect(html).toContain(`abs(dN_hours_off) ≤ ${data.toleranceHours}`);
  });
});
