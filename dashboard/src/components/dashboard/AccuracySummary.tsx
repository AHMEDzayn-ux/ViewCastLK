// Model accuracy section.
// Fabricated numbers have been intentionally omitted.
// These fields will be populated with genuine hold-out evaluation results
// after model training and approved evaluation are complete.

const METRICS = [
  {
    label: "MAPE",
    description: "Mean Absolute Percentage Error",
    colour: "text-slate-400",
    bg: "bg-slate-800/50",
    ring: "ring-slate-700",
  },
  {
    label: "R²",
    description: "Variance explained",
    colour: "text-slate-400",
    bg: "bg-slate-800/50",
    ring: "ring-slate-700",
  },
  {
    label: "MAE",
    description: "Mean Absolute Error (views)",
    colour: "text-slate-400",
    bg: "bg-slate-800/50",
    ring: "ring-slate-700",
  },
  {
    label: "Baseline",
    description: "Category-average naive baseline",
    colour: "text-slate-400",
    bg: "bg-slate-800/50",
    ring: "ring-slate-700",
  },
];

export default function AccuracySummary() {
  return (
    <div className="bg-slate-800/30 rounded-2xl border border-slate-700 p-5 sm:p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-slate-100 font-semibold text-base">
          Model accuracy
        </h3>
        <span className="text-xs text-slate-500 font-medium">
          Awaiting evaluation
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {METRICS.map((m) => (
          <div
            key={m.label}
            className={`rounded-xl ${m.bg} ring-1 ${m.ring} p-3 flex flex-col gap-1`}
          >
            <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
              {m.label}
            </span>
            <span className={`text-sm font-semibold ${m.colour}`}>
              Awaiting evaluation
            </span>
            <span className="text-xs text-slate-500 leading-tight">
              {m.description}
            </span>
          </div>
        ))}
      </div>

      <p className="text-xs text-slate-500 mt-3">
        Accuracy metrics will be added here after model training and approved
        held-out evaluation are complete. No placeholder figures are shown.
      </p>
    </div>
  );
}
