import type { ForecastEstimate } from "@/types/forecast";

interface HorizonCardsProps {
  estimates: ForecastEstimate[];
}

function formatViews(value: number): string {
  return new Intl.NumberFormat("en-LK").format(value);
}

export default function HorizonCards({ estimates }: HorizonCardsProps) {
  return (
    <section className="forecast-section" aria-labelledby="horizon-title">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Cumulative forecast</p>
          <h3 id="horizon-title">Estimated total views</h3>
        </div>
        <p>Each value is the expected cumulative total by that day.</p>
      </div>

      <ol className="horizon-list">
        {estimates.map((estimate) => (
          <li className="horizon-item" key={estimate.horizonDays}>
            <span className="horizon-item__day">
              Day {estimate.horizonDays}
            </span>
            <strong>{formatViews(estimate.cumulativeViews)}</strong>
            <span>cumulative views</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
