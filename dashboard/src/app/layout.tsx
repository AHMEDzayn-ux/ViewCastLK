import type { Metadata } from "next";
import AuthProvider from "@/components/auth/AuthProvider";
import SiteShell from "@/components/dashboard/SiteShell";
import "./globals.css";
import "./studio.css";
import "./landing.css";

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
        <AuthProvider>
          <SiteShell>{children}</SiteShell>
        </AuthProvider>
      </body>
    </html>
  );
}
