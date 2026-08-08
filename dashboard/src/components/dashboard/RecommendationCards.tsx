import type { Recommendation } from "@/types/forecast";

interface RecommendationCardsProps {
  recommendations: Recommendation[];
}

export default function RecommendationCards({
  recommendations,
}: RecommendationCardsProps) {
  if (recommendations.length === 0) return null;

  return (
    <section className="recommendations" aria-labelledby="recommendations-title">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Planning guidance</p>
          <h3 id="recommendations-title">What to review before publishing</h3>
        </div>
        <p>
          Historical associations can inform planning, but they do not prove
          what caused previous performance.
        </p>
      </div>

      <div className="recommendation-list">
        {recommendations.map((recommendation, index) => (
          <article className="recommendation-item" key={recommendation.id}>
            <div className="recommendation-item__number" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div>
              <h4>{recommendation.title}</h4>
              <p>{recommendation.guidance}</p>
              {recommendation.evidence.length > 0 && (
                <details>
                  <summary>Supporting evidence</summary>
                  <dl>
                    {recommendation.evidence.map((evidence) => (
                      <div key={`${evidence.label}-${evidence.detail}`}>
                        <dt>{evidence.label}</dt>
                        <dd dir="auto">{evidence.detail}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
