"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { getCreatorInsights, type CreatorInsightsResult } from "@/lib/api/creator-insights";
import type { CreatorEffect, CreatorInsights } from "@/types/creator-insights";
import type { InsightsData } from "@/types/insights";
import { effectTone, formatEffect, formatRange } from "./EffectChart";

interface YourChannelInsightsProps {
  benchmarks: InsightsData;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

/** "−18% (range −31% to −4%)", or a hedge when the range includes zero. */
function describe(effect: CreatorEffect): string {
  const tone = effectTone(effect);
  if (tone === "unclear") {
    return `about ${formatEffect(effect.effectPct)}, but the range (${formatRange(effect)}) includes no difference`;
  }
  return `${formatEffect(effect.effectPct)} (range ${formatRange(effect)})`;
}

function Tile({ label, headline, children }: { label: string; headline: string; children: React.ReactNode }) {
  return (
    <article className="channel-tile">
      <p className="channel-tile__label">{label}</p>
      <p className="channel-tile__headline">{headline}</p>
      <div className="channel-tile__body">{children}</div>
    </article>
  );
}

export function ChannelPattern({ insights, benchmarks }: { insights: CreatorInsights; benchmarks: InsightsData }) {
  const spacing = insights.spacing;
  const crowded = spacing?.buckets.find((b) => b.key === "under_1h");
  const spaced = spacing?.buckets.find((b) => b.key === "6h_plus");
  const bestTime = [...insights.timing]
    .filter((t) => effectTone(t) === "up")
    .sort((a, b) => b.effectPct - a.effectPct)[0];
  const shorts = insights.format?.shortsVsRegular ?? null;
  const categoryGrowth = benchmarks.growth.byCategory.find((r) => r.label === insights.mainCategory);
  const populationEvening = benchmarks.timing.byHourBlock.find((c) => c.label === "18:00–21:00");

  return (
    <div className="channel-grid">
      <Tile
        label="Upload spacing"
        headline={spacing?.shareWithinHour != null ? `${Math.round(spacing.shareWithinHour)}%` : "—"}
      >
        <p>
          of your uploads went up within an hour of another. Across Sri Lankan channels it is{" "}
          {Math.round(benchmarks.spacing.shareWithinHour)}%.
        </p>
        {crowded ? (
          <p>
            Those crowded uploads got {describe(crowded)} against your normal
            {spaced ? <>, and uploads with 6 hours or more before the next got {describe(spaced)}</> : null}.
          </p>
        ) : (
          <p>Too few crowded uploads to measure their effect on your channel.</p>
        )}
      </Tile>

      <Tile label="Time of day" headline={bestTime ? bestTime.label.split(" (")[0] : "No clear winner"}>
        {bestTime ? (
          <p>
            Your {bestTime.label.toLowerCase()} uploads got {describe(bestTime)} against your normal.
          </p>
        ) : (
          <p>No time of day clearly beats your normal yet. Across all channels, 18:00–21:00 does best
            {populationEvening ? ` (${formatEffect(populationEvening.effectPct)})` : ""}.</p>
        )}
        {insights.timing.length > 0 && (
          <ul className="channel-tile__list">
            {insights.timing.map((t) => (
              <li key={t.key}>
                <span>{t.label}</span>
                <span className={`channel-tone channel-tone--${effectTone(t)}`}>{formatEffect(t.effectPct)}</span>
              </li>
            ))}
          </ul>
        )}
      </Tile>

      <Tile label="Shorts" headline={shorts ? formatEffect(shorts.effectPct) : "—"}>
        {shorts ? (
          <p>
            Your Shorts against your regular videos: {describe(shorts)}. Shorts views are counted
            each time a Short plays, so they run higher.
          </p>
        ) : (
          <p>
            You need at least 5 Shorts and 5 regular videos with day-7 views to compare them
            (you have {insights.format?.shorts ?? 0} and {insights.format?.regular ?? 0}).
          </p>
        )}
      </Tile>

      <Tile
        label="After the first week"
        headline={insights.growth ? `+${insights.growth.medianGrowthPct.toFixed(1)}%` : "—"}
      >
        {insights.growth ? (
          <p>
            A typical video of yours gains this between day 7 and day 30
            {categoryGrowth
              ? `, against +${categoryGrowth.medianGrowthPct.toFixed(1)}% for ${insights.mainCategory} on other channels`
              : ""}
            .
          </p>
        ) : (
          <p>Not enough of your videos are 30 days old yet.</p>
        )}
      </Tile>
    </div>
  );
}

export default function YourChannelInsights({ benchmarks }: YourChannelInsightsProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const [result, setResult] = useState<CreatorInsightsResult | null>(null);

  useEffect(() => {
    if (isLoading) return;
    let active = true;
    getCreatorInsights().then((next) => {
      if (active) setResult(next);
    });
    return () => {
      active = false;
    };
  }, [isLoading, isAuthenticated]);

  let body: React.ReactNode;
  if (!result) {
    body = <p className="channel-insights__status">Loading your channel…</p>;
  } else if (result.status === "signed_out" || result.status === "not_connected") {
    body = (
      <div className="channel-insights__cta">
        <p>
          Connect your YouTube channel to see these patterns for your own uploads: how your
          crowded uploads do, which time of day works for you, and how your Shorts compare.
        </p>
        <Link className="secondary-button" href={result.status === "signed_out" ? "/login" : "/account"}>
          {result.status === "signed_out" ? "Sign in to connect" : "Connect your channel"}
        </Link>
      </div>
    );
  } else if (result.status === "error") {
    body = <p className="channel-insights__status" role="alert">{result.message}</p>;
  } else if (result.insights.videosMeasured < 10) {
    body = (
      <p className="channel-insights__status">
        We have day-7 views for {result.insights.videosMeasured} of your videos so far. Your
        pattern appears once at least 10 are measured.
      </p>
    );
  } else {
    const { insights } = result;
    body = (
      <>
        <p className="channel-insights__meta">
          {result.example && <strong>Example data. </strong>}
          {insights.channelTitle ? <strong>{insights.channelTitle}</strong> : "Your channel"}: your
          last {insights.videosSynced} uploads
          {insights.periodStart && insights.periodEnd
            ? `, ${formatDate(insights.periodStart)} to ${formatDate(insights.periodEnd)}`
            : ""}
          , each compared with your own normal.
        </p>
        <ChannelPattern insights={insights} benchmarks={benchmarks} />
      </>
    );
  }

  return (
    <section className="channel-insights" aria-labelledby="your-channel-title">
      <p className="section-kicker">Your channel</p>
      <h2 id="your-channel-title">How these patterns look on your own uploads</h2>
      {body}
    </section>
  );
}
