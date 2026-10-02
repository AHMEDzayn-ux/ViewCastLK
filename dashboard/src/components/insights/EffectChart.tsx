"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EffectCell } from "@/types/insights";

interface EffectChartProps {
  cells: EffectCell[];
  /** Accessible name for the chart and its table. */
  label: string;
  /** "columns" for ordered bands (hours, gaps); "rows" for named groups. */
  layout?: "columns" | "rows";
  height?: number;
}

export function formatEffect(value: number): string {
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : rounded < 0 ? "−" : ""}${Math.abs(rounded)}%`;
}

export function formatRange(cell: Pick<EffectCell, "lowPct" | "highPct">): string {
  return `${formatEffect(cell.lowPct)} to ${formatEffect(cell.highPct)}`;
}

/** A difference is only called a difference when its 95% range excludes zero. */
export function effectTone(cell: Pick<EffectCell, "lowPct" | "highPct">): "up" | "down" | "unclear" {
  if (cell.lowPct > 0) return "up";
  if (cell.highPct < 0) return "down";
  return "unclear";
}

const TONE_COLOURS = {
  up: "var(--success)",
  down: "var(--accent)",
  unclear: "var(--border-strong)",
} as const;

function EffectTooltip({ active, payload }: { active?: boolean; payload?: { payload: EffectCell }[] }) {
  if (!active || !payload?.length) return null;
  const cell = payload[0].payload;
  return (
    <div className="effect-tooltip">
      <strong>{cell.label}</strong>
      <span>{formatEffect(cell.effectPct)} against the channel&apos;s normal</span>
      <span>95% range {formatRange(cell)}</span>
      <small>
        {cell.videos.toLocaleString("en-LK")} videos from {cell.channels.toLocaleString("en-LK")} channels
      </small>
    </div>
  );
}

// Bars are drawn on a ratio scale: a doubling (+100%) and a halving (-50%)
// sit the same distance from zero, so one large effect can't flatten the rest.
const toScale = (pct: number) => Math.log1p(pct / 100);
const fromScale = (value: number) => Math.expm1(value) * 100;
const TICK_CANDIDATES = [-90, -75, -50, -25, -10, 0, 10, 25, 50, 100, 200, 400, 900];

function scaleTicks(min: number, max: number): number[] {
  const inRange = TICK_CANDIDATES.filter((t) => t >= min && t <= max);
  const span = toScale(max) - toScale(min);
  // Keep ticks at least ~12% of the axis apart so labels never collide.
  const ticks: number[] = [];
  for (const t of inRange) {
    if (t === 0 || ticks.every((u) => Math.abs(toScale(t) - toScale(u)) > span * 0.12)) ticks.push(t);
  }
  return ticks.sort((a, b) => a - b).map(toScale);
}

/** Column labels collide on a phone, so every chart turns into rows there. */
function useNarrowScreen(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 680px)");
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return narrow;
}

export default function EffectChart({ cells, label, layout = "columns", height }: EffectChartProps) {
  const narrow = useNarrowScreen();
  const data = cells.map((cell) => ({
    ...cell,
    scaled: toScale(cell.effectPct),
    range: [toScale(cell.effectPct) - toScale(cell.lowPct), toScale(cell.highPct) - toScale(cell.effectPct)],
  }));
  const lowest = Math.min(0, ...cells.map((c) => c.lowPct));
  const highest = Math.max(0, ...cells.map((c) => c.highPct));
  const ticks = scaleTicks(
    TICK_CANDIDATES.filter((t) => t <= lowest).at(-1) ?? lowest,
    TICK_CANDIDATES.find((t) => t >= highest) ?? highest,
  );
  const rows = layout === "rows" || narrow;
  const chartHeight = rows ? Math.max(180, cells.length * 34 + 40) : height ?? 260;
  const valueAxis = {
    type: "number" as const,
    tickFormatter: (v: number) => formatEffect(fromScale(v)),
    domain: [ticks[0], ticks[ticks.length - 1]] as [number, number],
    ticks,
    allowDataOverflow: true,
    tick: { fill: "var(--text-muted)", fontSize: 12 },
    axisLine: false,
    tickLine: false,
  };
  const categoryAxis = {
    type: "category" as const,
    dataKey: "label",
    tick: { fill: "var(--text-secondary)", fontSize: 12 },
    axisLine: { stroke: "var(--border-strong)" },
    tickLine: false,
    interval: 0,
  };

  return (
    <figure className="effect-chart">
      <div role="img" aria-label={`${label}. The numbers are in the table below the chart.`}>
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart
            data={data}
            layout={rows ? "vertical" : "horizontal"}
            margin={{ top: 8, right: 16, left: rows ? 8 : 0, bottom: 4 }}
          >
            <CartesianGrid stroke="var(--border)" strokeDasharray="2 5" horizontal={!rows} vertical={rows} />
            {rows ? (
              <>
                <XAxis {...valueAxis} />
                <YAxis {...categoryAxis} width={narrow ? 104 : 150} />
              </>
            ) : (
              <>
                <XAxis {...categoryAxis} />
                <YAxis {...valueAxis} width={48} />
              </>
            )}
            <ReferenceLine {...(rows ? { x: 0 } : { y: 0 })} stroke="var(--text-muted)" />
            <Tooltip content={<EffectTooltip />} cursor={{ fill: "var(--surface-soft)" }} />
            <Bar dataKey="scaled" isAnimationActive={false} maxBarSize={rows ? 18 : 42} radius={2}>
              {data.map((cell) => (
                <Cell key={cell.label} fill={TONE_COLOURS[effectTone(cell)]} />
              ))}
              <ErrorBar
                dataKey="range"
                direction={rows ? "x" : "y"}
                stroke="var(--text-primary)"
                strokeWidth={1.4}
                width={5}
                isAnimationActive={false}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <details className="effect-chart__table">
        <summary>Show the numbers</summary>
        <table className="effect-table">
          <caption className="sr-only">{label}</caption>
          <thead>
            <tr>
              <th scope="col">Group</th>
              <th scope="col">Against normal</th>
              <th scope="col">95% range</th>
              <th scope="col">Videos</th>
              <th scope="col">Channels</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((cell) => (
              <tr key={cell.label}>
                <th scope="row">{cell.label}</th>
                <td>{formatEffect(cell.effectPct)}</td>
                <td>{formatRange(cell)}</td>
                <td>{cell.videos.toLocaleString("en-LK")}</td>
                <td>{cell.channels.toLocaleString("en-LK")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
