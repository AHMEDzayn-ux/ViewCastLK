"use client";

import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { HorizonResult } from "@/types/forecast";

interface ForecastChartProps {
  horizons: HorizonResult[];
}

function formatViews(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
  return v.toLocaleString();
}

interface TooltipPayloadItem {
  name: string;
  value: number;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const median = payload.find((p) => p.name === "Median");
  const low = payload.find((p) => p.name === "Lower bound");
  const high = payload.find((p) => p.name === "Upper bound");

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="font-semibold text-slate-200 mb-1">Day {label}</p>
      {median && (
        <p className="text-blue-400">
          Median: <span className="font-bold text-white">{median.value.toLocaleString()}</span>
        </p>
      )}
      {low && high && (
        <p className="text-slate-400">
          Range: {low.value.toLocaleString()} – {high.value.toLocaleString()}
        </p>
      )}
    </div>
  );
}

export default function ForecastChart({ horizons }: ForecastChartProps) {
  const data = horizons.map((h) => ({
    day: h.day,
    low: h.low,
    median: h.median,
    high: h.high,
    // Recharts area needs [low, high] so we store the spread
    range: [h.low, h.high] as [number, number],
  }));

  const maxVal = Math.max(...horizons.map((h) => h.high));

  return (
    <div className="bg-slate-800/50 rounded-2xl border border-slate-700 p-5 sm:p-6">
      <div className="mb-4">
        <h3 className="text-slate-100 font-semibold text-base">
          Forecast trajectory
        </h3>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart
          data={data}
          margin={{ top: 8, right: 16, left: 8, bottom: 4 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            vertical={false}
          />
          <XAxis
            dataKey="day"
            tickFormatter={(v) => `Day ${v}`}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#475569" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={formatViews}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            domain={[0, Math.ceil(maxVal * 1.15)]}
            width={52}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px", color: "#94a3b8", paddingTop: "8px" }}
            iconType="circle"
            iconSize={8}
          />

          {/* Confidence band */}
          <Area
            type="monotone"
            dataKey="high"
            name="Upper bound"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.08}
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="low"
            name="Lower bound"
            stroke="none"
            fill="#0f172a"
            fillOpacity={1}
            legendType="none"
          />

          {/* Shaded band label lines */}
          <Line
            type="monotone"
            dataKey="high"
            name="Upper bound"
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="4 2"
            dot={false}
            legendType="line"
          />
          <Line
            type="monotone"
            dataKey="low"
            name="Lower bound"
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="4 2"
            dot={false}
            legendType="line"
          />

          {/* Median line */}
          <Line
            type="monotone"
            dataKey="median"
            name="Median"
            stroke="#60a5fa"
            strokeWidth={2.5}
            dot={{ fill: "#60a5fa", r: 5, strokeWidth: 0 }}
            activeDot={{ r: 7, fill: "#93c5fd" }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="text-xs text-slate-500 mt-3 text-center">
        Shaded band represents the lower–upper prediction range · Demonstration mock data only
      </p>
    </div>
  );
}
