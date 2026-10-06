import type { Metadata } from "next";
import Link from "next/link";
import LandingForecastPreview from "@/components/landing/LandingForecastPreview";
import StudioIcon from "@/components/dashboard/StudioIcon";

export const metadata: Metadata = {
  title: "ViewCastLK — A little foresight for your next video",
  description: "Explore your video's first 30 days before you publish. A forecasting workspace made for Sri Lankan YouTube creators.",
};

const steps = [
  { number: "01", icon: "play" as const, title: "Start with your idea.", text: "Add your title, category, format, language, and channel. A few details are all it takes to begin.", note: "YOUR VIDEO BRIEF" },
  { number: "02", icon: "chart" as const, title: "See the possibilities.", text: "Explore cumulative view estimates at Day 7, 14, 21, and 30, with your trajectory in one clear view.", note: "YOUR FIRST 30 DAYS" },
  { number: "03", icon: "spark" as const, title: "Make your next move.", text: "Review the available publishing guidance and use the forecast as another perspective on your plan.", note: "A MORE INFORMED START" },
];

export default function LandingPage() {
  return (
    <main className="landing-page">
      <section className="landing-hero" aria-labelledby="landing-title">
        <div className="landing-hero__copy">
          <p className="landing-eyebrow"><span /> A LITTLE FORESIGHT. MADE IN SRI LANKA.</p>
          <h1 id="landing-title">Before you<br />hit publish,<br /><em>see the possibilities.</em></h1>
          <p className="landing-hero__description">Big ideas deserve a little perspective. Explore your next video’s first 30 days with forecasts made for Sri Lankan creators.</p>
          <div className="landing-hero__actions">
            <Link className="landing-button" href="/forecast">Create a forecast <StudioIcon name="arrow" width="19" height="19" /></Link>
            <a className="landing-text-link" href="#how-it-works"><StudioIcon name="play" width="15" height="15" /> See how it works</a>
          </div>
          <p className="landing-hero__note"><StudioIcon name="check" width="15" height="15" /> Start as a guest. Bring your next idea.</p>
        </div>
        <LandingForecastPreview />
      </section>

      <div className="landing-context" aria-label="Forecast overview">
        <p>A clearer view of<br /><strong>what could come next.</strong></p>
        <div><strong>4 checkpoints</strong><span>One unfolding story</span></div>
        <div><strong>30 days ahead</strong><span>From first week to first month</span></div>
        <div><strong>Your language. Your ideas.</strong><span>Sinhala, Tamil, English & mixed titles</span></div>
      </div>

      <section id="how-it-works" className="landing-how" aria-labelledby="landing-how-title">
        <div className="landing-section-heading">
          <div><p className="landing-eyebrow">FROM AN IDEA TO A LITTLE MORE CLARITY</p><h2 id="landing-how-title">Your next chapter starts here.</h2></div>
          <p>You bring the creativity.<br />We bring another way to look ahead.</p>
        </div>
        <div className="landing-steps">
          {steps.map((step) => (
            <article key={step.number} className="landing-step">
              <div className="landing-step__top"><span>{step.number}</span><StudioIcon name={step.icon} width="27" height="27" /></div>
              <h3>{step.title}</h3><p>{step.text}</p><span className="landing-step__note">{step.note}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-channel" aria-labelledby="landing-channel-title">
        <div className="landing-channel__art" aria-hidden="true">
          <div className="landing-channel__orbit" />
          <div className="landing-channel__tile"><StudioIcon name="channel" width="52" height="52" /></div>
          <span className="landing-channel__tag"><StudioIcon name="check" width="15" height="15" /> A perspective of your own</span>
          <StudioIcon className="landing-channel__spark" name="spark" width="46" height="46" />
        </div>
        <div className="landing-channel__copy">
          <p className="landing-eyebrow">EVERY CHANNEL HAS ITS OWN STORY</p>
          <h2 id="landing-channel-title">Make room for <em>your perspective.</em></h2>
          <p>Connect your YouTube channel to personalize forecasts using eligible channel history. Save your forecasts and return to them as your next idea takes shape.</p>
          <Link className="landing-text-link" href="/account">Explore your channel workspace <StudioIcon name="arrow" width="17" height="17" /></Link>
          <span className="landing-channel__privacy"><StudioIcon name="shield" width="15" height="15" /> Your private Analytics never train the shared model.</span>
        </div>
      </section>

      <section className="landing-evidence" aria-labelledby="landing-evidence-title">
        <div><p className="landing-eyebrow">A LITTLE FORESIGHT, WITH OPEN EYES</p><h2 id="landing-evidence-title">Built on research.<br /><em>Grounded in possibility.</em></h2><p>A forecast is a planning perspective, never a promise of views. Get to know the data, the model, and the limits behind your results.</p></div>
        <div className="landing-evidence__links">
          <Link href="/insights"><StudioIcon name="spark" /><span><strong>Explore what works</strong><small>Patterns from historical videos</small></span><StudioIcon name="arrow" /></Link>
          <Link href="/accuracy"><StudioIcon name="chart" /><span><strong>See the model accuracy</strong><small>Published evaluation and comparisons</small></span><StudioIcon name="arrow" /></Link>
          <Link href="/dataset"><StudioIcon name="data" /><span><strong>Meet the dataset</strong><small>The public data behind the forecast</small></span><StudioIcon name="arrow" /></Link>
        </div>
      </section>

      <section className="landing-final" aria-labelledby="landing-final-title">
        <StudioIcon name="spark" width="33" height="33" />
        <p className="landing-eyebrow">FOR THE IDEA YOU CAN’T WAIT TO SHARE</p>
        <h2 id="landing-final-title">What will you <em>create next?</em></h2>
        <p>Your next video starts with an idea. Give it a little foresight.</p>
        <Link className="landing-button" href="/forecast">Create a forecast <StudioIcon name="arrow" width="18" height="18" /></Link>
      </section>
    </main>
  );
}
