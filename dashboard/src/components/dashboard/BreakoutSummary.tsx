import { formatForecastViews } from "@/lib/forecast-format";
import type { BreakoutForecast } from "@/types/forecast";

interface BreakoutSummaryProps {
  breakout: BreakoutForecast;
}

export default function BreakoutSummary({ breakout }: BreakoutSummaryProps) {
  const dayThirty = breakout.conditionalUpside.find(
    (estimate) => estimate.horizonDays === 30,
  );

  return (
    <section className="breakout-summary" aria-labelledby="breakout-title">
      <div>
        <p className="section-kicker">V8 experimental scenario</p>
        <h3 id="breakout-title">Breakout potential</h3>
        <p>{breakout.definition}</p>
        <p className="breakout-summary__caveat">
          This probability is estimated before publication. The upside path is
          conditional on a breakout and is not a second equally likely forecast.
        </p>
      </div>
      <div className="breakout-summary__metrics">
        <div>
          <span>Estimated breakout probability</span>
          <strong>
            {new Intl.NumberFormat("en-LK", {
              style: "percent",
              maximumFractionDigits: 1,
            }).format(breakout.probability)}
          </strong>
        </div>
        {dayThirty && (
          <div>
            <span>Conditional Day-30 upside</span>
            <strong>{formatForecastViews(dayThirty.cumulativeViews)}</strong>
          </div>
        )}
      </div>
    </section>
  );
}
