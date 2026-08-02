import type { HorizonResult } from "@/types/forecast";

interface HorizonCardsProps {
  horizons: HorizonResult[];
}

function formatViews(v: number): string {
  if (v >= 1_000_000) {
    return `${(v / 1_000_000).toFixed(2)}M`;
  }
  if (v >= 1_000) {
    return `${(v / 1_000).toFixed(1)}K`;
  }
  return v.toLocaleString();
}

const DAY_COLOURS: Record<number, { bg: string; ring: string; accent: string }> = {
  7:  { bg: "from-blue-950/60 to-blue-900/40",  ring: "ring-blue-700",  accent: "text-blue-400" },
  14: { bg: "from-indigo-950/60 to-indigo-900/40", ring: "ring-indigo-700", accent: "text-indigo-400" },
  21: { bg: "from-violet-950/60 to-violet-900/40", ring: "ring-violet-700", accent: "text-violet-400" },
  30: { bg: "from-purple-950/60 to-purple-900/40", ring: "ring-purple-700", accent: "text-purple-400" },
};

export default function HorizonCards({ horizons }: HorizonCardsProps) {
  return (
    <div>
      <h3 className="text-slate-100 font-semibold text-base mb-3">
        Forecast horizons
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {horizons.map((h) => {
          const colours = DAY_COLOURS[h.day] ?? DAY_COLOURS[7];
          return (
            <div
              key={h.day}
              className={`rounded-xl bg-gradient-to-br ${colours.bg} ring-1 ${colours.ring} p-4 flex flex-col gap-2`}
            >
              <span className={`text-xs font-bold uppercase tracking-wider ${colours.accent}`}>
                Day {h.day}
              </span>

              {/* Median (large) */}
              <div>
                <p className="text-2xl sm:text-3xl font-bold text-slate-100 leading-tight">
                  {formatViews(h.median)}
                </p>
                <p className="text-xs text-slate-400 mt-0.5">median views</p>
              </div>

              {/* Range */}
              <div className="border-t border-slate-700 pt-2 mt-1">
                <p className="text-xs text-slate-400">
                  <span className="text-slate-300 font-medium">
                    {formatViews(h.low)}
                  </span>
                  {" – "}
                  <span className="text-slate-300 font-medium">
                    {formatViews(h.high)}
                  </span>
                </p>
                <p className="text-xs text-slate-500">lower – upper range</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
