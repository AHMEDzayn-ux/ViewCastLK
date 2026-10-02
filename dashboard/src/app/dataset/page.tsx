import type { Metadata } from "next";
import DatasetView from "@/components/dataset/DatasetView";
import release from "@/data/dataset.json";
import type { DatasetRelease } from "@/types/dataset";

export const metadata: Metadata = {
  title: "Dataset",
  description:
    "Download the ViewCastLK dataset of Sri Lankan YouTube videos, with views, likes and comments at days 7, 14, 21 and 30.",
};

export default function DatasetPage() {
  return (
    <main className="page-shell information-page dataset-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Open data</p>
        <h1>The ViewCastLK dataset</h1>
        <p>
          Sri Lankan YouTube videos with their channel&apos;s figures at the moment of publishing
          and their views, likes and comments at days 7, 14, 21 and 30. Every column is described
          below, with where it came from.
        </p>
      </header>
      <DatasetView release={release as DatasetRelease} />
    </main>
  );
}
