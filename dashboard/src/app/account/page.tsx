import type { Metadata } from "next";

import YouTubeConnectionCard from "@/components/account/YouTubeConnectionCard";
import StudioIcon from "@/components/dashboard/StudioIcon";

export const metadata: Metadata = {
  title: "Channel connection",
  description: "Manage the YouTube channel connected to your ViewCastLK account.",
};

export default function AccountPage() {
  return (
    <main className="page-shell account-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Account &amp; channel</p>
        <h1>Your creator connection</h1>
        <p>
          Connect the YouTube channel you own to prepare private channel-history
          adjustments for future forecasts.
        </p>
      </header>
      <YouTubeConnectionCard />
      <div className="connection-benefits">
        <article><StudioIcon name="shield" /><h2>Private by design</h2><p>Your private Analytics stay in your account and never train the shared model.</p></article>
        <article><StudioIcon name="chart" /><h2>Grounded in your history</h2><p>Eligible mature videos help adjust the shared forecast to your channel.</p></article>
        <article><StudioIcon name="channel" /><h2>Always in your control</h2><p>Read-only YouTube access. Disconnect any time to remove your private creator data.</p></article>
      </div>
    </main>
  );
}
