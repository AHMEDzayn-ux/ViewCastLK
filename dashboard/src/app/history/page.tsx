import type { Metadata } from "next";

import HistoryView from "@/components/history/HistoryView";

export const metadata: Metadata = {
  title: "Forecast history",
  description: "Review and manage forecasts saved to your ViewCastLK account.",
};

export default function HistoryPage() {
  return (
    <main className="page-shell history-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Forecast memory</p>
        <h1>Your saved forecasts</h1>
        <p>
          Revisit completed forecasts and compare their Day 7, 14, 21, and 30
          planning checkpoints.
        </p>
      </header>

      <HistoryView />
    </main>
  );
}
