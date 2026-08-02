import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import DashboardHeader from "@/components/dashboard/DashboardHeader";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "ViewCastLK — YouTube View Forecasting",
  description:
    "Pre-publication YouTube view forecasting for Sri Lankan content creators. Predict expected view counts at day 7, 14, 21, and 30.",
  robots: "noindex, nofollow",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-slate-950 text-slate-100 antialiased min-h-screen flex flex-col">
        <DashboardHeader />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-slate-800 py-4 px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-xs text-slate-600">
            ViewCastLK · University of Moratuwa · Group 2 · CS3501 Data Science
            and Engineering Project ·{" "}
            <span className="text-amber-700">Phase 1 — demonstration mock data only</span>
          </p>
        </footer>
      </body>
    </html>
  );
}
