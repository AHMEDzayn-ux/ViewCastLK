import type {
  AccuracyMetric,
  AccuracyResponse,
} from "@/types/forecast";

interface AccuracySummaryProps {
  accuracy: AccuracyResponse;
}

function formatMetric(metric: AccuracyMetric, value: number | null): string {
  if (value === null) return "Not published";
  if (metric.unit === "percent") return `${value.toFixed(1)}%`;
  if (metric.unit === "views") return value.toLocaleString("en-LK");
  return value.toFixed(3);
}

export default function AccuracySummary({ accuracy }: AccuracySummaryProps) {
  const primaryMetric = accuracy.metrics.find((metric) => metric.key === "mape");
  const supportingMetrics = accuracy.metrics.filter(
    (metric) => metric.key !== "mape",
  );

  return (
    <div className="accuracy-summary">
      {accuracy.status === "unavailable" && (
        <section className="accuracy-unavailable" role="status">
          <p className="section-kicker">Evaluation pending</p>
          <h2>No accuracy figures are published yet</h2>
          <p>
            {accuracy.message ??
              "Held-out model evaluation has not been approved for publication."}
          </p>
          <p>
            ViewCastLK does not substitute demonstration values when real
            evaluation results are unavailable.
          </p>
        </section>
      )}

      {primaryMetric && (
        <section className="primary-metric" aria-labelledby="primary-metric-title">
          <div>
            <p className="section-kicker">Primary metric</p>
            <h2 id="primary-metric-title">{primaryMetric.label}</h2>
            <p>{primaryMetric.description}</p>
          </div>
          <dl className="metric-comparison">
            <div>
              <dt>{accuracy.modelName}</dt>
              <dd>{formatMetric(primaryMetric, primaryMetric.modelValue)}</dd>
            </div>
            <div>
              <dt>{accuracy.baselineName}</dt>
              <dd>{formatMetric(primaryMetric, primaryMetric.baselineValue)}</dd>
            </div>
          </dl>
          <p className="metric-direction">For MAPE, a lower value is better.</p>
        </section>
      )}

      <section className="supporting-metrics" aria-labelledby="supporting-title">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Supporting measures</p>
            <h2 id="supporting-title">A fuller evaluation view</h2>
          </div>
          <p>These measures provide context; none should be read in isolation.</p>
        </div>

        <div className="metric-table-wrap">
          <table className="metric-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Meaning</th>
                <th scope="col">Model</th>
                <th scope="col">Baseline</th>
              </tr>
            </thead>
            <tbody>
              {supportingMetrics.map((metric) => (
                <tr key={metric.key}>
                  <th scope="row">{metric.label}</th>
                  <td>{metric.description}</td>
                  <td>{formatMetric(metric, metric.modelValue)}</td>
                  <td>{formatMetric(metric, metric.baselineValue)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="accuracy-notes" aria-labelledby="accuracy-notes-title">
        <h2 id="accuracy-notes-title">How to read this page</h2>
        <ul>
          <li>
            The baseline is a simple benchmark. Comparing against it shows
            whether the forecasting model adds useful predictive value.
          </li>
          <li>
            Evaluation results should come from held-out videos that were not
            used to fit the model.
          </li>
          <li>
            A strong average result does not guarantee an accurate forecast for
            every individual video.
          </li>
        </ul>
        {accuracy.evaluatedAt && (
          <p>
            Evaluation last updated {" "}
            {new Date(accuracy.evaluatedAt).toLocaleDateString("en-LK", {
              dateStyle: "long",
            })}
            .
          </p>
        )}
      </section>
    </div>
  );
}
