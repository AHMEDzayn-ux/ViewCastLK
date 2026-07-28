import { redirect } from "next/navigation";

/**
 * Root route — immediately redirects to /forecast.
 * No content is rendered here; the dashboard's single page is /forecast.
 */
export default function RootPage() {
  redirect("/forecast");
}
