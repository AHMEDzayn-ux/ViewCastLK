import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import DegradedNotice from "./DegradedNotice";
import ForecastChart from "./ForecastChart";
import HorizonCards from "./HorizonCards";
import RecommendationCards from "./RecommendationCards";

interface ForecastResultsProps {
  response: ForecastResponse;
  request: ForecastRequest;
  onChangeInputs: () => void;
  historySaveNotice?: string;
}

export default function ForecastResults({
  response,
  request,
  onChangeInputs,
  historySaveNotice,
}: ForecastResultsProps) {
  const generatedAt = new Date(response.model.generatedAt).toLocaleString(
    "en-LK",
    {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "Asia/Colombo",
    },
  );

  return (
    <div className="forecast-results" aria-labelledby="forecast-results-title">
      <header className="forecast-results__header">
        <div>
          <p className="section-kicker">Forecast ready</p>
          <h2 id="forecast-results-title">Four planning checkpoints</h2>
          <p className="forecast-results__subject" dir="auto">
            {request.title}
          </p>
        </div>
        <div className="forecast-results__meta">
          <span
            className={
              response.model.dataSource === "mock"
                ? "status-tag status-tag--development"
                : "status-tag"
            }
          >
            {response.model.dataSource === "mock"
              ? "Development forecast"
              : "Prediction API"}
          </span>
          <span>{generatedAt} SLT</span>
        </div>
      </header>

      <DegradedNotice completeness={response.completeness} />
      {response.personalization?.applied && (
        <section className="personalization-notice" role="status">
          <div>
            <p className="section-kicker">Creator-specific result</p>
            <h3>Personalised using your channel history</h3>
            <p>
              The shared model forecast was adjusted with mature videos that
              this model did not train on.
            </p>
          </div>
          <span className="status-tag status-tag--personalized">
            Personalised forecast
          </span>
        </section>
      )}
      {historySaveNotice && (
        <div className="history-save-notice" role="status">
          {historySaveNotice}
        </div>
      )}
      <HorizonCards estimates={response.estimates} />
      <ForecastChart estimates={response.estimates} />
      {response.personalization?.applied && (
        <details className="personalization-details">
          <summary>Compare with the shared model forecast</summary>
          <ul>
            {response.personalization.sharedEstimates.map((estimate) => (
              <li key={estimate.horizonDays}>
                Day {estimate.horizonDays}:{" "}
                {estimate.cumulativeViews.toLocaleString("en-US")} shared views
              </li>
            ))}
          </ul>
        </details>
      )}
      <RecommendationCards
        recommendations={response.recommendations}
        unavailableRecommendations={response.unavailableRecommendations}
      />

      {response.titleGuidance && (
        <section className="title-guidance" aria-labelledby="title-guidance-title">
          <div>
            <p className="section-kicker">Title review</p>
            <h3 id="title-guidance-title">Clear, accurate wording</h3>
            <p>{response.titleGuidance.summary}</p>
          </div>
          <ul>
            {response.titleGuidance.suggestions.map((suggestion) => (
              <li key={suggestion}>{suggestion}</li>
            ))}
          </ul>
        </section>
      )}

      <footer className="forecast-results__footer">
        <p>
          Forecast ID <code>{response.forecastId}</code> · Model {" "}
          <code>{response.model.modelVersion}</code>. Forecasts are estimates,
          not guaranteed outcomes.
        </p>
        <button
          className="secondary-button"
          type="button"
          onClick={onChangeInputs}
        >
          Change inputs
        </button>
      </footer>
    </div>
  );
}
