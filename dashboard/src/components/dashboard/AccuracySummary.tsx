"use client";

import { useState } from "react";
import type {
  AccuracyEvaluation,
  AccuracyMetric,
  AccuracyResponse,
  AccuracyScope,
  AvailableAccuracyResponse,
  UnavailableAccuracyResponse,
} from "@/types/forecast";

interface AccuracySummaryProps {
  accuracy: AccuracyResponse;
}

const SCOPE_LABELS: Record<AccuracyScope, string> = {
  day_7: "Day 7",
  day_14: "Day 14",
  day_21: "Day 21",
  day_30: "Day 30",
};

function evaluationKey(evaluation: AccuracyEvaluation): string {
  return `${evaluation.scope}/${evaluation.segment}`;
}

function formatMetric(metric: AccuracyMetric, value: number | null): string {
  if (value === null) return "Not applicable";
  if (metric.unit === "percent") return `${value.toFixed(1)}%`;
  if (metric.unit === "factor") return `${value.toFixed(1)}\u00d7`;
  return value.toFixed(2);
}

function directionNote(metric: AccuracyMetric): string {
  return metric.betterWhen === "higher"
    ? "A higher value is better."
    : "A lower value is better.";
}

function metricByKey(
  evaluation: AccuracyEvaluation,
  key: AccuracyMetric["key"],
): AccuracyMetric | undefined {
  return evaluation.metrics.find((metric) => metric.key === key);
}

function improvementLabel(metric: AccuracyMetric): string | null {
  if (metric.modelValue === null || metric.baselineValue === null) return null;

  const difference = metric.modelValue - metric.baselineValue;
  const improves =
    metric.betterWhen === "higher" ? difference > 0 : difference < 0;
  if (!improves) return null;

  if (metric.unit === "percent") {
    return `${Math.abs(difference).toFixed(1)} percentage points ahead of the baseline`;
  }

  if (metric.unit === "factor" && metric.baselineValue > 0) {
    const reduction = Math.abs(difference / metric.baselineValue) * 100;
    return `${reduction.toFixed(1)}% lower typical error than the baseline`;
  }

  return `${Math.abs(difference).toFixed(2)} stronger than the baseline`;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString("en-LK", { dateStyle: "long" });
}

function UnavailableAccuracySummary({
  accuracy,
}: {
  accuracy: UnavailableAccuracyResponse;
}) {
  return (
    <div className="accuracy-summary">
      <section className="accuracy-unavailable" role="status">
        <p className="section-kicker">Evaluation pending</p>
        <h2>Evaluation results are not available yet</h2>
        <p>{accuracy.message}</p>
        <p>
          ViewCastLK does not display demonstration or development values as
          approved accuracy results.
        </p>
      </section>
    </div>
  );
}

function AvailableAccuracySummary({
  accuracy,
}: {
  accuracy: AvailableAccuracyResponse;
}) {
  const [selectedKey, setSelectedKey] = useState<string>(
    evaluationKey(accuracy.evaluations[0]),
  );
  const selectedEvaluation =
    accuracy.evaluations.find(
      (evaluation) => evaluationKey(evaluation) === selectedKey,
    ) ?? accuracy.evaluations[0];
  const [primaryMetric, ...supportingMetrics] = selectedEvaluation.metrics;
  const trackedEvaluation =
    accuracy.evaluations.find(
      (evaluation) => evaluation.segment === "tracked_channel",
    ) ?? accuracy.evaluations[0];
  const trackedWithinTwo = metricByKey(trackedEvaluation, "within_2x");
  const trackedTypicalError = metricByKey(trackedEvaluation, "typical_factor");
  const trackedRanking = metricByKey(trackedEvaluation, "rank_correlation");
  const totalEvaluatedVideos = accuracy.evaluations.reduce(
    (total, evaluation) => total + evaluation.videos,
    0,
  );
  const primaryImprovement = primaryMetric
    ? improvementLabel(primaryMetric)
    : null;

  return (
    <div className="accuracy-summary">
      <section className="accuracy-overview" aria-labelledby="accuracy-overview-title">
        <div className="accuracy-overview__intro">
          <p className="section-kicker">Evaluation at a glance</p>
          <h2 id="accuracy-overview-title">Strong ranking with measurable baseline gains</h2>
          <p>
            The model was tested on {totalEvaluatedVideos.toLocaleString("en-LK")} unseen
            videos. For channels with history, it improves every reported measure over
            using the channel&apos;s usual views alone.
          </p>
        </div>
        <dl className="accuracy-highlights">
          {trackedWithinTwo && (
            <div>
              <dt>Within a factor of two</dt>
              <dd>{formatMetric(trackedWithinTwo, trackedWithinTwo.modelValue)}</dd>
              <p>{improvementLabel(trackedWithinTwo)}</p>
            </div>
          )}
          {trackedTypicalError && (
            <div>
              <dt>Typical error</dt>
              <dd>{formatMetric(trackedTypicalError, trackedTypicalError.modelValue)}</dd>
              <p>{improvementLabel(trackedTypicalError)}</p>
            </div>
          )}
          {trackedRanking && (
            <div>
              <dt>Ranking accuracy</dt>
              <dd>{formatMetric(trackedRanking, trackedRanking.modelValue)}</dd>
              <p>{improvementLabel(trackedRanking)}</p>
            </div>
          )}
        </dl>
      </section>

      <div className="accuracy-scope-control">
        <div>
          <label htmlFor="accuracy-scope">Accuracy view</label>
          <p>
            Accuracy depends most on whether we already track the channel, so
            the two are shown separately.
          </p>
        </div>
        <select
          id="accuracy-scope"
          className="field-control"
          value={evaluationKey(selectedEvaluation)}
          onChange={(event) => setSelectedKey(event.target.value)}
        >
          {accuracy.evaluations.map((evaluation) => (
            <option value={evaluationKey(evaluation)} key={evaluationKey(evaluation)}>
              {SCOPE_LABELS[evaluation.scope]}: {evaluation.segmentLabel}
            </option>
          ))}
        </select>
      </div>

      {primaryMetric && (
        <section className="primary-metric" aria-labelledby="primary-metric-title">
          <div>
            <p className="section-kicker">
              {SCOPE_LABELS[selectedEvaluation.scope]} accuracy,{" "}
              {selectedEvaluation.segmentLabel.toLowerCase()}
            </p>
            <h2 id="primary-metric-title">{primaryMetric.label}</h2>
            <p>{primaryMetric.description}</p>
            <p>
              Measured on {selectedEvaluation.videos.toLocaleString("en-LK")}{" "}
              videos published {formatDate(accuracy.periodStart)} to{" "}
              {formatDate(accuracy.periodEnd)}.
            </p>
          </div>
          <dl className="metric-comparison">
            <div>
              <dt>ViewCastLK forecast</dt>
              <dd>{formatMetric(primaryMetric, primaryMetric.modelValue)}</dd>
            </div>
            <div>
              <dt>{selectedEvaluation.baselineName}</dt>
              <dd>{formatMetric(primaryMetric, primaryMetric.baselineValue)}</dd>
            </div>
          </dl>
          <p className="metric-direction">
            {primaryImprovement ?? directionNote(primaryMetric)}
          </p>
        </section>
      )}

      <section className="supporting-metrics" aria-labelledby="supporting-title">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Supporting measures</p>
            <h2 id="supporting-title">
              Detailed {SCOPE_LABELS[selectedEvaluation.scope]} evaluation metrics
            </h2>
          </div>
          <p>These measures provide context; none should be read in isolation.</p>
        </div>

        <div className="metric-table-wrap">
          <table className="metric-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Meaning</th>
                <th scope="col">ViewCastLK forecast</th>
                <th scope="col">{selectedEvaluation.baselineName}</th>
              </tr>
            </thead>
            <tbody>
              {supportingMetrics.map((metric) => (
                <tr key={metric.key}>
                  <th scope="row">{metric.label}</th>
                  <td>
                    {metric.description} {directionNote(metric)}
                    {improvementLabel(metric) && (
                      <strong className="metric-improvement">
                        {improvementLabel(metric)}
                      </strong>
                    )}
                  </td>
                  <td>{formatMetric(metric, metric.modelValue)}</td>
                  <td>{formatMetric(metric, metric.baselineValue)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="accuracy-notes" aria-labelledby="accuracy-notes-title">
        <h2 id="accuracy-notes-title">A transparent, real-world evaluation</h2>
        <ul>
          <li>
            The baseline is the simple estimate available from a channel&apos;s
            usual performance. The comparison shows the additional signal
            contributed by ViewCastLK.
          </li>
          <li>{accuracy.method}</li>
          {accuracy.notYetMeasured.map((pending) => (
            <li key={pending.scope}>
              {SCOPE_LABELS[pending.scope]} accuracy can be measured from{" "}
              {formatDate(pending.measurableFrom)}, once those videos are old
              enough.
            </li>
          ))}
          <li>
            Results summarize performance across the full evaluation sample;
            individual videos can naturally vary around these measured results.
          </li>
        </ul>
        <p>
          Evaluation last updated {formatDate(accuracy.evaluatedAt)}. Model:{" "}
          {accuracy.modelName}.
        </p>
      </section>
    </div>
  );
}

export default function AccuracySummary({ accuracy }: AccuracySummaryProps) {
  if (accuracy.status === "unavailable") {
    return <UnavailableAccuracySummary accuracy={accuracy} />;
  }

  return <AvailableAccuracySummary accuracy={accuracy} />;
}
