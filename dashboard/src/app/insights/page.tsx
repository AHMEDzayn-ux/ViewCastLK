import type { Metadata } from "next";
import InsightsView from "@/components/insights/InsightsView";
import insights from "@/data/insights.json";
import type { InsightsData } from "@/types/insights";

export const metadata: Metadata = {
  title: "Publishing insights",
  description:
    "What Sri Lankan YouTube data says about upload spacing, timing, length, format and thumbnails.",
};

export default function InsightsPage() {
  return (
    <main className="page-shell information-page insights-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Publishing insights</p>
        <h1>What works on Sri Lankan YouTube</h1>
        <p>
          Patterns from tens of thousands of videos, each measured against its own
          channel&apos;s normal, so the advice applies whatever the size of your channel.
        </p>
      </header>
      <InsightsView data={insights as InsightsData} />
    </main>
  );
}
