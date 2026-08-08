import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Forecast",
  description:
    "Estimate cumulative YouTube views at Day 7, 14, 21, and 30 before publication.",
};

export default function ForecastLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return children;
}
