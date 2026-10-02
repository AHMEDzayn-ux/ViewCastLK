"use client";

import { useState } from "react";
import type { EffectCell, GrowthRow, InsightsData } from "@/types/insights";
import EffectChart, { effectTone, formatEffect, formatRange } from "./EffectChart";

interface InsightsViewProps {
  data: InsightsData;
}

function cell(cells: EffectCell[], label: string): EffectCell | undefined {
  return cells.find((c) => c.label === label);
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

/** "Under 1K" reads as "under 1K" mid-sentence. */
function sizeName(label: string): string {
  return label.replace(/^Under /, "under ");
}

function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

function Takeaway({ children }: { children: React.ReactNode }) {
  return (
    <p className="insight-takeaway">
      <strong>Try this</strong>
      <span>{children}</span>
    </p>
  );
}

function InsightSection({
  index,
  kicker,
  title,
  id,
  children,
}: {
  index: string;
  kicker: string;
  title: string;
  id: string;
  children: React.ReactNode;
}) {
  return (
    <section className="insight-section" aria-labelledby={id}>
      <p className="section-index">{index}</p>
      <div>
        <p className="section-kicker">{kicker}</p>
        <h2 id={id}>{title}</h2>
        {children}
      </div>
    </section>
  );
}

function SpacingSection({ spacing }: { spacing: InsightsData["spacing"] }) {
  const crowded = cell(spacing.byGapToNext, "Under 1 hour");
  const best = [...spacing.byGapToNext].sort((a, b) => b.effectPct - a.effectPct)[0];
  const clearSizes = spacing.bySize.filter((c) => effectTone(c) === "up");
  const smallest = spacing.bySize[0];

  return (
    <InsightSection index="01" kicker="Upload spacing" title="Give each upload some room" id="spacing-title">
      <p>
        {spacing.shareWithinHour}% of uploads go up within an hour of another video from the same
        channel.
        {crowded && best && (
          <>
            {" "}Those videos got {formatEffect(crowded.effectPct)} views against their channel&apos;s
            normal. Videos with {best.label.toLowerCase()} before the channel&apos;s next upload got{" "}
            {formatEffect(best.effectPct)}.
          </>
        )}
      </p>
      <EffectChart
        cells={spacing.byGapToNext}
        label="Day-7 views against the channel's normal, by time until the same channel's next upload"
      />
      <p className="insight-caption">
        Time until the same channel&apos;s next upload. A later upload cannot have affected when
        the earlier video was published, so this gap is the fairer measure. The gap since the
        previous upload shows the same pattern.
      </p>

      <h3>By channel size</h3>
      <p>
        Uploads followed by a day or more of quiet, compared with uploads followed by another within
        three hours.
        {clearSizes.length > 0 && (
          <>
            {" "}The difference is clearest for channels with{" "}
            {listNames(clearSizes.map((c) => sizeName(c.label)))} subscribers.
          </>
        )}
        {smallest && effectTone(smallest) === "unclear" && (
          <> There is no clear effect for channels under {smallest.label.replace(/^Under /, "")} subscribers.</>
        )}
      </p>
      <EffectChart
        cells={spacing.bySize}
        layout="rows"
        label="Spaced uploads against crowded uploads, by subscriber count"
      />
      <Takeaway>
        Leave at least six hours between uploads. If you have several clips ready, schedule them
        across the day instead of releasing them together.
      </Takeaway>
    </InsightSection>
  );
}

function TimingSection({ timing }: { timing: InsightsData["timing"] }) {
  const blocks = [...timing.byHourBlock].sort((a, b) => b.effectPct - a.effectPct);
  const best = blocks[0];
  const clearlyDown = timing.byHourBlock.filter((c) => effectTone(c) === "down");
  const dayRange = Math.max(...timing.byDay.map((c) => Math.abs(c.effectPct)));

  return (
    <InsightSection index="02" kicker="Publishing time" title="Evening uploads do best" id="timing-title">
      {best && (
        <p>
          Videos published between {best.label} Sri Lanka time got {formatEffect(best.effectPct)}{" "}
          against their channel&apos;s normal (95% range {formatRange(best)}).
          {clearlyDown.length > 0 && (
            <>
              {" "}The only {clearlyDown.length === 1 ? "window" : "windows"} clearly below normal:{" "}
              {listNames(clearlyDown.map((c) => `${c.label} (${formatEffect(c.effectPct)})`))}.
            </>
          )}{" "}
          Fewer videos go up overnight, so those bars are less certain.
        </p>
      )}
      <EffectChart
        cells={timing.byHourBlock}
        label="Day-7 views against the channel's normal, by three-hour publishing window in Sri Lanka time"
      />
      <h3>Day of the week</h3>
      <p>
        The day matters much less than the hour: every day is within {Math.round(dayRange)}% of the
        channel&apos;s normal.
      </p>
      <EffectChart
        cells={timing.byDay}
        height={200}
        label="Day-7 views against the channel's normal, by day of the week"
      />
      <Takeaway>
        Aim for {best ? best.label : "the evening"}. Don&apos;t hold a finished video back just to
        hit a particular day.
      </Takeaway>
    </InsightSection>
  );
}

function DurationSection({ format }: { format: InsightsData["format"] }) {
  const categories = format.durationByCategory;
  const [selected, setSelected] = useState(categories[0]?.category ?? "");
  const current = categories.find((c) => c.category === selected) ?? categories[0];
  const short = categories
    .map((c) => ({ category: c.category, band: cell(c.bands, "Under 4 min") }))
    .filter((c): c is { category: string; band: EffectCell } => c.band !== undefined);
  const clearlyWorse = short.filter((c) => effectTone(c.band) === "down");
  const effects = clearlyWorse.map((c) => c.band.effectPct);
  const weakest = categories.filter((c) => {
    const band = cell(c.bands, "Under 4 min");
    return band !== undefined && band.effectPct === Math.min(...c.bands.map((b) => b.effectPct));
  }).length;

  if (!current) return null;

  return (
    <InsightSection
      index="03"
      kicker="Duration"
      title="Under four minutes? Make it a Short, or make it longer"
      id="duration-title"
    >
      {clearlyWorse.length > 0 && (
        <p>
          In {clearlyWorse.length} of the {short.length} categories with enough data, regular
          (non-Shorts) videos under four minutes did clearly worse than their channel&apos;s normal,
          by {Math.round(Math.abs(Math.max(...effects)))}% to{" "}
          {Math.round(Math.abs(Math.min(...effects)))}%. They are the weakest length of all in{" "}
          {weakest} of them.
        </p>
      )}
      <div className="insight-control">
        <label htmlFor="duration-category">Category</label>
        <select
          id="duration-category"
          className="field-control"
          value={current.category}
          onChange={(event) => setSelected(event.target.value)}
        >
          {categories.map((c) => (
            <option key={c.category} value={c.category}>
              {c.category}
            </option>
          ))}
        </select>
      </div>
      <EffectChart
        cells={current.bands}
        label={`Day-7 views against the channel's normal by length, ${current.category}`}
      />
      <p className="insight-caption">
        Lengths with fewer than the minimum videos or channels are left out rather than guessed.
        Grey bars have a 95% range that includes zero, so the data can&apos;t tell them apart from
        normal.
      </p>
      <Takeaway>
        If a regular video is coming in under four minutes, publish it as a Short or build it out
        into a fuller video.
      </Takeaway>
    </InsightSection>
  );
}

function ShortsSection({ format }: { format: InsightsData["format"] }) {
  const clear = format.bySize.filter((c) => effectTone(c) === "up");
  const unclear = format.bySize.filter((c) => effectTone(c) === "unclear");

  return (
    <InsightSection index="04" kicker="Format" title="Shorts help smaller channels most" id="shorts-title">
      <p>
        On the same channel, Shorts compared with regular videos.
        {clear.length > 0 && (
          <> Shorts did clearly better on channels with {listNames(clear.map((c) => sizeName(c.label)))} subscribers.</>
        )}
        {unclear.length > 0 && (
          <> For channels with {listNames(unclear.map((c) => sizeName(c.label)))} subscribers there is no clear difference.</>
        )}
      </p>
      <EffectChart cells={format.bySize} layout="rows" label="Shorts against regular videos, by subscriber count" />
      <h3>By category</h3>
      <EffectChart cells={format.byCategory} layout="rows" label="Shorts against regular videos, by category" />
      <p className="insight-note">
        YouTube counts a Shorts view each time a Short starts or replays, a looser count than for
        regular videos. More Shorts views do not mean more watch time or more revenue.
      </p>
      <Takeaway>
        Under 100K subscribers, cut Shorts from your longer videos as a way to reach new viewers.
      </Takeaway>
    </InsightSection>
  );
}

function GrowthTable({ rows, label }: { rows: GrowthRow[]; label: string }) {
  return (
    <div className="effect-table-wrap">
      <table className="effect-table">
        <caption className="sr-only">{label}</caption>
        <thead>
          <tr>
            <th scope="col">Group</th>
            <th scope="col">Typical growth, day 7 to 30</th>
            <th scope="col">One in four grows more than</th>
            <th scope="col">Videos still growing (10%+)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>+{row.medianGrowthPct.toFixed(1)}%</td>
              <td>+{Math.round(row.upperQuartileGrowthPct)}%</td>
              <td>{Math.round(row.shareGrowing10Pct)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GrowthSection({ growth }: { growth: InsightsData["growth"] }) {
  const sorted = [...growth.byCategory].sort((a, b) => b.medianGrowthPct - a.medianGrowthPct);
  const slow = sorted.slice(0, 3);
  const done = sorted[sorted.length - 1];
  const longRun = sorted.slice(0, 4).map((r) => r.label);

  return (
    <InsightSection index="05" kicker="After the first week" title="Know when a video has had its run" id="growth-title">
      <p>
        A typical video already has {growth.medianShareByDay7}% of its day-30 views by day 7. That
        varies a lot by category.
        {done && slow.length > 0 && (
          <>
            {" "}{done.label} is finished within a week (+{done.medianGrowthPct.toFixed(1)}%
            afterwards), while a typical {listNames(slow.map((r) => r.label))} video gains another{" "}
            {Math.round(slow[slow.length - 1].medianGrowthPct)}–{Math.round(slow[0].medianGrowthPct)}%.
          </>
        )}
      </p>
      <GrowthTable rows={sorted} label="Growth in views between day 7 and day 30, by category" />
      <h3>By channel size</h3>
      <GrowthTable rows={growth.bySize} label="Growth in views between day 7 and day 30, by subscriber count" />
      <p className="insight-caption">
        Measured on {growth.videos.toLocaleString("en-LK")} videos tracked to day 30. The biggest
        channels collect almost everything in the first week. Mid-sized channels keep growing
        longest.
      </p>
      <Takeaway>
        {done ? `Judge a ${done.label} video at day 7, but give` : "Give"}{" "}
        {listNames(longRun)} videos the full month before deciding whether they worked.
      </Takeaway>
    </InsightSection>
  );
}

function ThumbnailSection({ thumbnails }: { thumbnails: InsightsData["thumbnails"] }) {
  const helps = thumbnails.byCategory.filter((c) => effectTone(c) === "up").map((c) => c.label);
  const hurts = thumbnails.byCategory.filter((c) => effectTone(c) === "down").map((c) => c.label);

  return (
    <InsightSection
      index="06"
      kicker="Thumbnails"
      title="A face on the thumbnail is not a rule"
      id="thumbnails-title"
    >
      <p>
        On channels that sometimes show a face and sometimes don&apos;t, thumbnails with a face got{" "}
        {thumbnails.overall ? `${formatEffect(thumbnails.overall.effectPct)} (95% range ${formatRange(thumbnails.overall)})` : "no clear difference"}.
        The direction depends on the category.
        {helps.length > 0 && <> Faces did clearly better in {listNames(helps)}.</>}
        {hurts.length > 0 && <> They did clearly worse in {listNames(hurts)}.</>}
      </p>
      <EffectChart
        cells={thumbnails.byCategory}
        layout="rows"
        label="Thumbnails with a face against thumbnails without, by category"
      />
      <p className="insight-note">
        Based on {thumbnails.videos.toLocaleString("en-LK")} thumbnails collected for our analysis.
        YouTube serves a changed thumbnail from the same address, so some images may not be the
        ones the videos launched with. The face detector also misses small and side-on faces.
        Brightness, contrast, colour and text on the thumbnail showed no link with which of a
        channel&apos;s videos did better.
      </p>
      <Takeaway>
        Don&apos;t add a face just because thumbnail guides say so. Look at what works in your
        category, and compare both styles on your own channel.
      </Takeaway>
    </InsightSection>
  );
}

export default function InsightsView({ data }: InsightsViewProps) {
  const { dataset } = data;

  return (
    <>
      <dl className="insight-stats">
        <div>
          <dt>Videos measured</dt>
          <dd>{dataset.videos.toLocaleString("en-LK")}</dd>
        </div>
        <div>
          <dt>Sri Lankan channels</dt>
          <dd>{dataset.channels.toLocaleString("en-LK")}</dd>
        </div>
        <div>
          <dt>Published</dt>
          <dd>
            {formatDate(dataset.periodStart)} to {formatDate(dataset.periodEnd)}
          </dd>
        </div>
      </dl>

      <aside className="insight-reading" aria-label="How to read this page">
        <p>
          Every number compares a video with <strong>its own channel&apos;s normal</strong>, so a
          big channel posting at 8 pm can&apos;t make 8 pm look good on its own. Bars show the
          typical difference, the line through each bar shows the 95% range, and grey means the
          range includes zero.
        </p>
      </aside>

      <div className="insight-grid">
        <SpacingSection spacing={data.spacing} />
        <TimingSection timing={data.timing} />
        <DurationSection format={data.format} />
        <ShortsSection format={data.format} />
        <GrowthSection growth={data.growth} />
        <ThumbnailSection thumbnails={data.thumbnails} />

        <section className="insight-section" aria-labelledby="method-title">
          <p className="section-index">07</p>
          <div>
            <p className="section-kicker">Method</p>
            <h2 id="method-title">How these were measured</h2>
            <ul className="limitations-list">
              <li>
                Each video&apos;s day-7 views are compared with the average for its own channel,
                using only channels with at least {dataset.minChannelVideos} measured videos.
              </li>
              <li>
                The 95% ranges come from resampling whole channels, because videos from one
                channel are not independent of each other.
              </li>
              <li>
                A group is shown only with at least {dataset.minVideos} videos from{" "}
                {dataset.minChannels} channels.
              </li>
              <li>
                These are patterns in past videos, not proof of cause. A channel that posts many
                clips at once may also be posting lighter content, for example.
              </li>
              <li>
                Only Sri Lankan channels published between {formatDate(dataset.periodStart)} and{" "}
                {formatDate(dataset.periodEnd)} are included, so seasonal effects such as
                festivals are not covered.
              </li>
            </ul>
            <p className="insight-caption">Figures computed {formatDate(data.generatedAt)}.</p>
          </div>
        </section>
      </div>
    </>
  );
}
