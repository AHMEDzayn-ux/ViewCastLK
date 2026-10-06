"use client";

import { formatForecastViews } from "@/lib/forecast-format";
import { useId } from "react";
import {
  CartesianGrid,
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastEstimate } from "@/types/forecast";

interface ForecastChartProps {
  estimates: ForecastEstimate[];
  scenario?: "forecast" | "breakout";
}

function formatAxisViews(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return String(value);
}

export default function ForecastChart({
  estimates,
  scenario = "forecast",
}: ForecastChartProps) {
  const id = useId();
  const isBreakout = scenario === "breakout";
  const color = isBreakout ? "var(--accent-warm)" : "var(--accent)";
  const data = estimates.map((estimate) => ({
    day: estimate.horizonDays,
    cumulativeViews: estimate.cumulativeViews,
  }));

  return (
    <section className="forecast-chart" aria-labelledby={`${id}-trajectory-title`}>
      <div className="section-heading">
        <div>
          <p className="section-kicker">{isBreakout ? "Conditional trajectory" : "Trajectory"}</p>
          <h3 id={`${id}-trajectory-title`}>{isBreakout ? "If a breakout happens" : "The view ahead"}</h3>
        </div>
      </div>

      <div
        className="forecast-chart__canvas"
        role="img"
        aria-label={isBreakout ? "Conditional breakout views on days 7, 14, 21, and 30. This scenario is not the main forecast." : "Cumulative forecast views on days 7, 14, 21, and 30"}
      >
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart
            data={data}
            margin={{ top: 12, right: 16, left: 4, bottom: 4 }}
          >
            <defs><linearGradient id={`${id}-area-fill`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.2} /><stop offset="100%" stopColor={color} stopOpacity={0.01} /></linearGradient></defs>
            <CartesianGrid
              stroke="var(--border)"
              strokeDasharray="2 5"
              vertical={false}
            />
            <XAxis
              dataKey="day"
              tickFormatter={(day) => `Day ${day}`}
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              axisLine={{ stroke: "var(--border-strong)" }}
              tickLine={false}
            />
            <YAxis
              allowDecimals={false}
              tickFormatter={formatAxisViews}
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              width={54}
            />
            <Tooltip
              cursor={{ stroke: "var(--border-strong)", strokeWidth: 1 }}
              contentStyle={{
                background: "var(--surface)",
                border: "1px solid var(--border-strong)",
                borderRadius: "8px",
                boxShadow: "var(--shadow-small)",
              }}
              labelFormatter={(day) => `Day ${day}`}
              formatter={(value, name) => [
                typeof value === "number"
                  ? value.toLocaleString("en-LK")
                  : value,
                name,
              ]}
            />
            <Area
              type="monotone"
              dataKey="cumulativeViews"
              name={isBreakout ? "Conditional breakout views" : "Cumulative views"}
              stroke={color}
              strokeDasharray={isBreakout ? "7 5" : undefined}
              strokeWidth={3}
              fill={`url(#${id}-area-fill)`}
              isAnimationActive={false}
              dot={{ fill: "var(--surface)", strokeWidth: 3, r: 5 }}
              activeDot={{ fill: color, strokeWidth: 0, r: 6 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <table className="sr-only">
        <caption>{isBreakout ? "Conditional breakout values shown in the chart" : "Cumulative forecast values shown in the chart"}</caption>
        <thead>
          <tr>
            <th>Horizon</th>
            <th>{isBreakout ? "Conditional breakout views" : "Cumulative views"}</th>
          </tr>
        </thead>
        <tbody>
          {estimates.map((estimate) => (
            <tr key={estimate.horizonDays}>
              <td>Day {estimate.horizonDays}</td>
              <td>{formatForecastViews(estimate.cumulativeViews)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
