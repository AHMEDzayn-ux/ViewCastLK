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
          Only guidance supported by model evaluation is returned. Historical
          associations do not prove what caused previous performance.
        </p>
      </div>

      {recommendations.length > 0 && (
        <div className="recommendation-list">
          {recommendations.map((recommendation, index) => (
            <article className="recommendation-item" key={recommendation.id}>
              <div className="recommendation-item__number" aria-hidden="true">
                {String(index + 1).padStart(2, "0")}
              </div>
              <div>
                <p className="recommendation-item__type">
                  {TYPE_LABELS[recommendation.type]}
                </p>
                <h4>{recommendation.title}</h4>
                {recommendation.type === "timing" && (
                  <p className="recommendation-item__action">
                    <span>Recommended window</span>
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
            </article>
          ))}
        </div>
      )}

      {unavailableRecommendations.length > 0 && (
        <aside className="recommendation-unavailable" role="status">
          <p className="section-kicker">Guidance availability</p>
          <h4>
            {recommendations.length === 0
              ? "No evidence-backed recommendations are available"
              : "Some guidance is not available for this forecast"}
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
