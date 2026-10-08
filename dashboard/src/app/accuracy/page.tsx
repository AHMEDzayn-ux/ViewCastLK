import type { Metadata } from "next";
import AccuracyView from "@/components/dashboard/AccuracyView";

export const metadata: Metadata = {
  title: "Model Accuracy",
  description:
    "Explore ViewCastLK's measured performance on unseen videos and compare it with simple channel-history baselines.",
};

export default function AccuracyPage() {
  return (
    <main className="page-shell information-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Model evaluation</p>
        <h1>Measured performance on unseen videos</h1>
        <p>
          Explore approved held-out results, understand each metric, and see
          the additional predictive signal ViewCastLK provides over a simple
          channel-history baseline.
        </p>
      </header>
      <AccuracyView />
    </main>
  );
}
