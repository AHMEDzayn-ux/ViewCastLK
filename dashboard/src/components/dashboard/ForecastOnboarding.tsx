import Link from "next/link";
import StudioIcon from "./StudioIcon";

export type CreatorAudience = "guest" | "unconnected" | "connected" | "checking";

export default function ForecastOnboarding({ audience }: { audience: CreatorAudience }) {
  if (audience === "guest") {
    return (
      <section className="creator-cta" aria-labelledby="creator-cta-title">
        <div className="creator-cta__intro">
          <span className="creator-cta__icon"><StudioIcon name="spark" width="27" height="27" /></span>
          <p className="section-kicker">Your creator workspace</p>
          <h2 id="creator-cta-title">Want forecasts tailored to your channel?</h2>
          <p>Sign in and connect your YouTube channel to personalize forecasts using eligible channel history, save predictions, and revisit them whenever you need.</p>
          <div className="creator-cta__actions">
            <Link className="primary-button" href="/signup">Create free account <StudioIcon name="arrow" width="16" height="16" /></Link>
            <Link className="secondary-button" href="/login?next=%2Fforecast">Sign in</Link>
          </div>
        </div>
        <ul className="creator-cta__benefits">
          <li><StudioIcon name="chart" width="19" height="19" /><div><strong>Personalized forecasts</strong><span>Tailored using eligible channel history.</span></div></li>
          <li><StudioIcon name="history" width="19" height="19" /><div><strong>Forecast history</strong><span>Save and revisit predictions.</span></div></li>
          <li><StudioIcon name="shield" width="19" height="19" /><div><strong>YouTube connection</strong><span>Securely connect with read-only access.</span></div></li>
        </ul>
      </section>
    );
  }

  if (audience === "unconnected") {
    return (
      <section className="creator-cta creator-cta--compact" aria-labelledby="creator-cta-title">
        <span className="creator-cta__icon"><StudioIcon name="channel" width="25" height="25" /></span>
        <div>
          <p className="section-kicker">Optional channel connection</p>
          <h2 id="creator-cta-title">Make forecasts more personal</h2>
          <p>Connect your YouTube channel so ViewCastLK can use your eligible historical performance to tailor future forecasts. Google access is read-only, and connecting is optional.</p>
        </div>
        <Link className="primary-button" href="/account">Connect YouTube <StudioIcon name="arrow" width="16" height="16" /></Link>
      </section>
    );
  }

  return null;
}
