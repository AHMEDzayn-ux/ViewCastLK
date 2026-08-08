import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Methodology & Limitations",
  description:
    "A creator-friendly explanation of how ViewCastLK forecasts should be used and where their limits apply.",
};

const REQUIRED_INPUTS = [
  "Planned video title",
  "YouTube category",
  "Expected duration",
  "Main audio language",
  "Made-for-kids setting",
  "Channel URL, handle, or ID",
];

const LIMITATIONS = [
  "Forecasts are estimates, not guarantees of future performance.",
  "Recommendations describe historical associations, not causal effects.",
  "No early engagement from the planned video is used.",
  "Private YouTube Analytics data is not used.",
  "Thumbnail and image features are not used.",
  "Revenue is not predicted.",
  "Subscriber growth is not predicted.",
  "ViewCastLK does not determine the location of individual viewers.",
  "Prediction intervals and confidence ranges are not part of the current release.",
  "Unexpected news, collaborations, external promotion, platform changes, and other events may cause actual performance to differ from the forecast.",
];

export default function MethodologyPage() {
  return (
    <main className="page-shell information-page methodology-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Methodology &amp; limitations</p>
        <h1>Use the forecast as a planning aid</h1>
        <p>
          ViewCastLK estimates how many total views a planned video may have by
          Day 7, 14, 21, and 30. It is designed to support a creator&apos;s judgment,
          not replace it.
        </p>
      </header>

      <div className="methodology-grid">
        <section className="methodology-section" aria-labelledby="inputs-title">
          <p className="section-index">01</p>
          <div>
            <h2 id="inputs-title">What you provide</h2>
            <p>
              The forecast starts with information available before publication.
              The following details are required:
            </p>
            <ul className="check-list">
              {REQUIRED_INPUTS.map((input) => (
                <li key={input}>{input}</li>
              ))}
            </ul>
            <p>
              Publishing day and hour are optional. If omitted, they remain
              unknown and are not replaced with assumed defaults.
            </p>
          </div>
        </section>

        <section className="methodology-section" aria-labelledby="process-title">
          <p className="section-index">02</p>
          <div>
            <h2 id="process-title">How the estimate is prepared</h2>
            <ol className="process-list">
              <li>ViewCastLK checks that the required planned details are present.</li>
              <li>
                Relevant channel information and title context are added where
                available.
              </li>
              <li>
                This information is compared with historical performance
                patterns.
              </li>
              <li>
                The result provides four cumulative forecasts and may include
                neutral planning guidance with supporting evidence.
              </li>
            </ol>
          </div>
        </section>

        <section className="methodology-section" aria-labelledby="information-title">
          <p className="section-index">03</p>
          <div>
            <h2 id="information-title">How your information is used</h2>
            <p>
              You provide the details of the video you are planning, and
              relevant channel information is retrieved automatically. The
              title may be analysed to offer neutral planning guidance. The
              forecasting service combines this information with historical
              patterns to prepare the forecast.
            </p>
          </div>
        </section>

        <section className="methodology-section" aria-labelledby="limits-title">
          <p className="section-index">04</p>
          <div>
            <h2 id="limits-title">Important limitations</h2>
            <ul className="limitations-list">
              {LIMITATIONS.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </div>
        </section>
      </div>

      <aside className="methodology-cta" aria-labelledby="evaluation-title">
        <div>
          <p className="section-kicker">Evaluation transparency</p>
          <h2 id="evaluation-title">Check evidence before relying on a model</h2>
          <p>
            ViewCastLK publishes approved model metrics separately and does not
            invent values while evaluation is pending.
          </p>
        </div>
        <Link className="secondary-button" href="/accuracy">
          View accuracy status
        </Link>
      </aside>
    </main>
  );
}
