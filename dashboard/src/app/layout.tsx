import type { Metadata } from "next";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "ViewCastLK",
    template: "%s | ViewCastLK",
  },
  description:
    "Pre-publication YouTube view forecasting for Sri Lankan content creators.",
  robots: "noindex, nofollow",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <DashboardHeader />
        <div id="main-content" className="site-content" tabIndex={-1}>
          {children}
        </div>
        <footer className="site-footer">
          <div>
            <p>
              <strong>ViewCastLK</strong> · University project for Sri Lankan
              creator forecasting
            </p>
            <p>Not affiliated with or endorsed by YouTube or Google.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
