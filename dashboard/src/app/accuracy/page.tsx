import type { Metadata } from "next";
import AccuracyView from "@/components/dashboard/AccuracyView";

export const metadata: Metadata = {
  title: "Accuracy",
  description:
    "Understand ViewCastLK evaluation metrics and baseline comparisons.",
};

export default function AccuracyPage() {
  return (
    <main className="page-shell information-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Model evaluation</p>
        <h1>Accuracy, without placeholder numbers</h1>
        <p>
          This page reports approved held-out evaluation results when they are
          available and compares the model with a simple baseline.
        </p>
      </header>
      <AccuracyView />
    </main>
  );
}
