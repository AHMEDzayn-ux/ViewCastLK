import type { Metadata } from "next";

import YouTubeConnectionCard from "@/components/account/YouTubeConnectionCard";

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
    </main>
  );
}
