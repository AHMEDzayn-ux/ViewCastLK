import type {
  Recommendation,
  RecommendationType,
  UnavailableRecommendation,
} from "@/types/forecast";

interface RecommendationCardsProps {
  recommendations: Recommendation[];
  unavailableRecommendations: UnavailableRecommendation[];
}

const TYPE_LABELS: Record<RecommendationType, string> = {
  timing: "Publishing day and time",
  duration: "Duration",
  format: "Format",
  title: "Title framing",
};

const STATUS_LABELS = {
  change: "Change supported",
  aligned: "Already aligned",
  benchmark: "Benchmark to consider",
} as const;

const TIMING_ACTION_LABELS = {
  change: "Suggested window",
  aligned: "Current strong window",
  benchmark: "Benchmark window",
} as const;

function formatHour(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`;
}

export default function RecommendationCards({
  recommendations,
  unavailableRecommendations,
}: RecommendationCardsProps) {
  if (
    recommendations.length === 0 &&
    unavailableRecommendations.length === 0
  ) {
    return null;
  }

  return (
    <section className="recommendations" aria-labelledby="recommendations-title">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Planning guidance</p>
          <h3 id="recommendations-title">What to review before publishing</h3>
        </div>
        <p>
          This guidance is generated separately from the forecast using released
          historical evidence rules. Associations do not prove what caused
          previous performance.
        </p>
      </div>

      {recommendations.length > 0 && (
        <div className="recommendation-list">
          {recommendations.map((recommendation, index) => {
            const recommendationStatus = recommendation.status ?? "benchmark";
            return <article className="recommendation-item" key={recommendation.id}>
              <div className="recommendation-item__number" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </div>
              <div>
                <div className="recommendation-item__labels">
                  <p className="recommendation-item__type">
                    {TYPE_LABELS[recommendation.type]}
                  </p>
                  <span className={`recommendation-status recommendation-status--${recommendationStatus}`}>
                    {STATUS_LABELS[recommendationStatus]}
                  </span>
                </div>
                <h4>{recommendation.title}</h4>
                {recommendation.type === "timing" && (
                  <p className="recommendation-item__action">
                    <span>{TIMING_ACTION_LABELS[recommendationStatus]}</span>
                    <strong>
                      {recommendation.recommendedPublishingWindow.day},{" "}
                      {formatHour(
                        recommendation.recommendedPublishingWindow.startHour,
                      )}
                      –
                      {formatHour(
                        recommendation.recommendedPublishingWindow.endHour,
                      )}{" "}
                      SLT
                    </strong>
                  </p>
                )}
                <p>{recommendation.guidance}</p>
                <details>
                  <summary>Supporting historical evidence</summary>
                  <dl>
                    {recommendation.evidence.map((evidence) => (
                      <div key={`${evidence.label}-${evidence.detail}`}>
                        <dt>{evidence.label}</dt>
                        <dd dir="auto">{evidence.detail}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              </div>
            </article>;
          })}
        </div>
      )}

      {unavailableRecommendations.length > 0 && (
        <aside className="recommendation-unavailable" role="status">
          <p className="section-kicker">Guidance availability</p>
          <h4>
            {recommendations.length === 0
              ? "Some parts of the plan could not be evaluated"
              : "Remaining data limitations"}
          </h4>
          <ul>
            {unavailableRecommendations.map((recommendation) => (
              <li key={recommendation.type}>
                <strong>{TYPE_LABELS[recommendation.type]}:</strong>{" "}
                {recommendation.reason}
              </li>
            ))}
          </ul>
        </aside>
      )}
    </section>
  );
}
