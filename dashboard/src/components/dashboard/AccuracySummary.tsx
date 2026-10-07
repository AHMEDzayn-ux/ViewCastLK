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

  return (
    <div className="accuracy-summary">
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
          <p className="metric-direction">{directionNote(primaryMetric)}</p>
        </section>
      )}

      <section className="supporting-metrics" aria-labelledby="supporting-title">
        <div className="section-heading">
          <div>
            <p className="section-kicker">Supporting measures</p>
            <h2 id="supporting-title">
              {SCOPE_LABELS[selectedEvaluation.scope]} evaluation details
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
        <h2 id="accuracy-notes-title">How to read this page</h2>
        <ul>
          <li>
            The comparison is the simple guess a creator could make without
            ViewCastLK. Beating it shows the forecast adds something.
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
            A strong average result does not guarantee an accurate forecast for
            every individual video.
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
