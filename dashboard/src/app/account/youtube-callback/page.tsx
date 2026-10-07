import type { Metadata } from "next";
import YouTubeOAuthCompletion from "@/components/account/YouTubeOAuthCompletion";

export const metadata: Metadata = {
  title: "Verify YouTube connection",
  referrer: "no-referrer",
  robots: "noindex, nofollow",
};

export default function YouTubeCallbackPage() {
  return <YouTubeOAuthCompletion />;
}
