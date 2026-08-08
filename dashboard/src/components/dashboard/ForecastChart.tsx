"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastEstimate } from "@/types/forecast";

interface ForecastChartProps {
  estimates: ForecastEstimate[];
}

function formatAxisViews(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}K`;
  return String(value);
}

export default function ForecastChart({ estimates }: ForecastChartProps) {
  const data = estimates.map((estimate) => ({
    day: estimate.horizonDays,
    cumulativeViews: estimate.cumulativeViews,
  }));

  return (
    <section className="forecast-chart" aria-labelledby="trajectory-title">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Trajectory</p>
          <h3 id="trajectory-title">How the total builds over time</h3>
        </div>
      </div>

      <div
        className="forecast-chart__canvas"
        role="img"
        aria-label="Line chart of cumulative forecast views on days 7, 14, 21, and 30"
      >
        <ResponsiveContainer width="100%" height={280}>
          <LineChart
            data={data}
            margin={{ top: 12, right: 16, left: 4, bottom: 4 }}
          >
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
              formatter={(value) => [
                typeof value === "number"
                  ? value.toLocaleString("en-LK")
                  : value,
                "Cumulative views",
              ]}
            />
            <Line
              type="monotone"
              dataKey="cumulativeViews"
              name="Cumulative views"
              stroke="var(--accent)"
              strokeWidth={3}
              dot={{ fill: "var(--surface)", strokeWidth: 3, r: 5 }}
              activeDot={{ fill: "var(--accent)", strokeWidth: 0, r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <table className="sr-only">
        <caption>Cumulative forecast values shown in the chart</caption>
        <thead>
          <tr>
            <th>Horizon</th>
            <th>Cumulative views</th>
          </tr>
        </thead>
        <tbody>
          {estimates.map((estimate) => (
            <tr key={estimate.horizonDays}>
              <td>Day {estimate.horizonDays}</td>
              <td>{estimate.cumulativeViews.toLocaleString("en-LK")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
