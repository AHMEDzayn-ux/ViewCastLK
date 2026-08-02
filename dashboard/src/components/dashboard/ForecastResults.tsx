"use client";

import type { ForecastResult, ForecastInput } from "@/types/forecast";
import ForecastChart from "./ForecastChart";
import HorizonCards from "./HorizonCards";
import RecommendationCards from "./RecommendationCards";
import AccuracySummary from "./AccuracySummary";

interface ForecastResultsProps {
  result: ForecastResult;
  input: ForecastInput;
}

export default function ForecastResults({ result, input }: ForecastResultsProps) {
  const generatedAt = new Date(result.generatedAt).toLocaleString("en-LK", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Colombo",
  });

  return (
    <div
      className="flex flex-col gap-6 animate-fade-in"
      role="region"
      aria-label="Forecast results"
    >
      {/* Result header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h2 className="text-slate-100 font-semibold text-lg leading-tight">
            Forecast results
          </h2>
          <p className="text-slate-400 text-sm mt-0.5">
            For &ldquo;{input.title}&rdquo;
            {input.category ? ` · ${input.category}` : ""}
            {input.channelHandle ? ` · ${input.channelHandle}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-amber-500/15 text-amber-400 border border-amber-700/40">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
            Mock data
          </span>
          <span className="text-xs text-slate-500">{generatedAt} SLT</span>
        </div>
      </div>

      {/* Horizon summary cards */}
      <HorizonCards horizons={result.horizons} />

      {/* Chart */}
      <ForecastChart horizons={result.horizons} />

      {/* Recommendations */}
      <RecommendationCards recommendations={result.recommendations} />

      {/* Accuracy placeholder */}
      <AccuracySummary />

      {/* Single consolidated transparency note */}
      <p className="text-xs text-slate-500 border-t border-slate-800 pt-4">
        All forecasts, recommendations, and accuracy figures shown above are
        generated from mock demonstration data. They do not represent real model
        outputs or approved research findings. Results will be updated after
        model training, held-out evaluation, and EDA sign-off are complete.
      </p>
    </div>
  );
}
