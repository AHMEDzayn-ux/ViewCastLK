"use client";

import { useId, useRef, useState } from "react";
import { formatForecastViews } from "@/lib/forecast-format";
import type { ForecastRequest, ForecastResponse } from "@/types/forecast";
import DegradedNotice from "./DegradedNotice";
import BreakoutSummary from "./BreakoutSummary";
import ForecastChart from "./ForecastChart";
import HorizonCards from "./HorizonCards";
import RecommendationCards from "./RecommendationCards";
import StudioIcon from "./StudioIcon";

interface ForecastResultsProps {
  response: ForecastResponse;
  request: ForecastRequest;
  onChangeInputs: () => void;
  historySaveNotice?: string;
  channelConnected?: boolean;
  detailsContent?: React.ReactNode;
}

type ResultTab = "overview" | "breakout" | "guidance" | "details";

export default function ForecastResults({
  response, request, onChangeInputs, historySaveNotice,
  channelConnected = false, detailsContent,
}: ForecastResultsProps) {
  const [activeTab, setActiveTab] = useState<ResultTab>("overview");
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const id = useId();
  const tabs: { key: ResultTab; label: string }[] = [
    { key: "overview", label: "Overview" },
    ...(response.breakout ? [{ key: "breakout" as const, label: "Breakout scenario" }] : []),
    { key: "guidance", label: "Guidance" },
    { key: "details", label: "Details" },
  ];
  const generatedAt = new Date(response.model.generatedAt).toLocaleString("en-LK", {
    dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Colombo",
  });
  const isExample = response.model.dataSource === "mock";
  const isPersonalized = Boolean(response.personalization?.applied);
  const duration = `${Math.floor(request.durationSeconds / 60)}m${request.durationSeconds % 60 ? ` ${request.durationSeconds % 60}s` : ""}`;

  function navigateTabs(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    setActiveTab(tabs[next].key);
    tabRefs.current[next]?.focus();
  }

  return (
    <div className="forecast-results focused-results" aria-labelledby={`${id}-title`}>
      <header className="result-brief-summary">
        <span className="panel-icon"><StudioIcon name="play" /></span>
        <div className="result-brief-summary__copy">
          <h2 id={`${id}-title`} dir="auto">{request.title}</h2>
          <p>{request.category}<span>·</span>{request.isShort ? "YouTube Short" : "Standard video"}<span>·</span>{duration}<span>·</span>{request.audioLanguage}</p>
        </div>
        <button className="secondary-button" type="button" onClick={onChangeInputs}>Edit brief <StudioIcon name="arrow" width="15" height="15" /></button>
      </header>

      <div className="result-status-row">
        <span className={`status-tag${isExample ? " status-tag--development" : ""}`}>{isExample ? "Illustrative example" : isPersonalized ? "Channel-adjusted" : "Shared forecast"}</span>
        <p>{isExample ? "Example numbers, not a real prediction." : isPersonalized ? "Personalised using your channel history" : channelConnected ? "Connected channel · no personal adjustment applied" : "No personal adjustment applied"}</p>
        {response.completeness.status === "degraded" && <button type="button" className="result-context-link" onClick={() => setActiveTab("details")}>Limited context · View details</button>}
      </div>
      {historySaveNotice && <div className="history-save-notice" role="status">{historySaveNotice}</div>}

      <div className="result-tabs" role="tablist" aria-label="Forecast sections">
        {tabs.map((tab, index) => (
          <button key={tab.key} type="button" role="tab" id={`${id}-tab-${tab.key}`} aria-controls={`${id}-panel-${tab.key}`} aria-selected={activeTab === tab.key} tabIndex={activeTab === tab.key ? 0 : -1} ref={(element) => { tabRefs.current[index] = element; }} onClick={() => setActiveTab(tab.key)} onKeyDown={(event) => navigateTabs(event, index)}>
            {tab.label}{tab.key === "breakout" && <span className="result-tab-experimental">Experimental</span>}
          </button>
        ))}
      </div>

      <section className="result-tab-panel" role="tabpanel" id={`${id}-panel-overview`} aria-labelledby={`${id}-tab-overview`} tabIndex={0} hidden={activeTab !== "overview"}>
        <HorizonCards estimates={response.estimates} />
        {activeTab === "overview" && <ForecastChart estimates={response.estimates} />}
        <p className="result-planning-note"><StudioIcon name="shield" width="14" height="14" /> A planning perspective, never a promise of views.</p>
      </section>

      {response.breakout && <section className="result-tab-panel" role="tabpanel" id={`${id}-panel-breakout`} aria-labelledby={`${id}-tab-breakout`} tabIndex={0} hidden={activeTab !== "breakout"}>
        <BreakoutSummary breakout={response.breakout} />
        {activeTab === "breakout" && <ForecastChart estimates={response.breakout.conditionalUpside} scenario="breakout" />}
      </section>}

      <section className="result-tab-panel" role="tabpanel" id={`${id}-panel-guidance`} aria-labelledby={`${id}-tab-guidance`} tabIndex={0} hidden={activeTab !== "guidance"}>
        {response.titleGuidance && <section className="title-guidance" aria-labelledby={`${id}-title-guidance`}>
          <div><p className="section-kicker">Title review</p><h3 id={`${id}-title-guidance`}>Clear, accurate wording</h3><p>{response.titleGuidance.summary}</p></div>
          <ul>{response.titleGuidance.suggestions.map((suggestion) => <li key={suggestion}>{suggestion}</li>)}</ul>
        </section>}
        {response.recommendations.length > 0 ? <RecommendationCards recommendations={response.recommendations} unavailableRecommendations={[]} /> : <div className="result-guidance-empty"><StudioIcon name="book" width="24" height="24" /><h3>No evidence-backed changes are suggested.</h3><p>The submitted plan did not have a clearly stronger alternative under the released evidence rules, or the relevant comparison was unavailable.</p></div>}
        {response.unavailableRecommendations.length > 0 && <details className="result-detail-disclosure"><summary>Why some publishing guidance is unavailable</summary><RecommendationCards recommendations={[]} unavailableRecommendations={response.unavailableRecommendations} /></details>}
      </section>

      <section className="result-tab-panel result-details-panel" role="tabpanel" id={`${id}-panel-details`} aria-labelledby={`${id}-tab-details`} tabIndex={0} hidden={activeTab !== "details"}>
        <div className="result-model-details">
          <p className="section-kicker">Behind this forecast</p><h3>Context & model details</h3>
          <p>{isExample ? "This is an illustrative example. It was not generated by the prediction model." : isPersonalized ? "The shared model forecast was adjusted with mature videos that this model did not train on." : channelConnected ? "Your channel is connected, but this forecast used the shared model. An adjustment requires eligible channel history for this request." : "Built with the shared model. No personal adjustment was applied to this forecast."}</p>
          <dl><div><dt>Channel</dt><dd dir="auto">{request.channelIdentifier}</dd></div><div><dt>Generated</dt><dd>{generatedAt} SLT</dd></div><div><dt>Model</dt><dd><code>{response.model.modelVersion}</code></dd></div><div><dt>Forecast ID</dt><dd><code>{response.forecastId}</code></dd></div>
          {response.guidance && <div><dt>Planning guidance</dt><dd><code>{response.guidance.artifactVersion}</code> · separate historical EDA</dd></div>}
          {(request.plannedPublishDay || request.plannedPublishHour !== null) && <div><dt>Publishing plan</dt><dd>{request.plannedPublishDay ?? "Day not decided"} · {request.plannedPublishHour === null ? "Time not decided" : `${String(request.plannedPublishHour).padStart(2, "0")}:00 SLT`}</dd></div>}
          </dl>
        </div>
        {response.guidance && <p className="result-planning-note"><StudioIcon name="book" width="14" height="14" /> {response.guidance.associationWarning} Guidance does not change the forecast.</p>}
        <DegradedNotice completeness={response.completeness} />
        {response.personalization?.applied && <details className="personalization-details"><summary>Compare with the shared model forecast</summary><ul>{response.personalization.sharedEstimates.map((estimate) => <li key={estimate.horizonDays}>Day {estimate.horizonDays}: {formatForecastViews(estimate.cumulativeViews)} shared views</li>)}</ul></details>}
        {detailsContent}
        <p className="result-planning-note"><StudioIcon name="shield" width="14" height="14" /> Forecasts are estimates, not guaranteed outcomes.</p>
      </section>
    </div>
  );
}
