import type { Recommendation } from "@/types/forecast";

interface RecommendationCardsProps {
  recommendations: Recommendation[];
}

// Map each recommendation type to a neutral section-title prefix and visual style.
// These titles make clear the cards are illustrative examples, not EDA findings.
const TYPE_META: Record<
  string,
  { sectionTitle: string; icon: string; border: string; iconBg: string }
> = {
  timing:   { sectionTitle: "Timing recommendation example",            icon: "🕐", border: "border-cyan-800",  iconBg: "bg-cyan-900/40"  },
  duration: { sectionTitle: "Duration recommendation example",          icon: "⏱",  border: "border-teal-800",  iconBg: "bg-teal-900/40"  },
  category: { sectionTitle: "Category recommendation example",          icon: "🏷",  border: "border-sky-800",   iconBg: "bg-sky-900/40"   },
  general:  { sectionTitle: "Publication-day recommendation example",   icon: "💡",  border: "border-slate-700", iconBg: "bg-slate-800"    },
};

export default function RecommendationCards({
  recommendations,
}: RecommendationCardsProps) {
  if (recommendations.length === 0) return null;

  return (
    <div>
      <div className="mb-3">
        <h3 className="text-slate-100 font-semibold text-base">
          Publishing recommendations
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Illustrative mock wording — will be replaced with approved
          EDA-based recommendations after analysis is complete.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        {recommendations.map((rec, idx) => {
          const meta = TYPE_META[rec.type] ?? TYPE_META.general;
          return (
            <div
              key={idx}
              className={`rounded-xl border ${meta.border} bg-slate-800/50 p-4 flex gap-3`}
            >
              <div
                className={`w-9 h-9 flex-shrink-0 rounded-lg ${meta.iconBg} flex items-center justify-center text-lg`}
                aria-hidden="true"
              >
                {meta.icon}
              </div>
              <div className="flex-1 min-w-0">
                {/* Neutral section title makes it unambiguous this is illustrative */}
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-1">
                  {meta.sectionTitle}
                </p>
                <p className="text-slate-100 font-semibold text-sm leading-tight mb-1">
                  {rec.headline}
                </p>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {rec.body}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
