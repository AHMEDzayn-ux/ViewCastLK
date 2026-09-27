import Link from "next/link";
import StudioIcon from "@/components/dashboard/StudioIcon";

const CONTENT = {
  history: {
    title: "Save and revisit your forecasts.",
    description: "Create an account or sign in to keep your predictions in one place.",
    destination: "/login?next=%2Fhistory",
  },
  account: {
    title: "Your creator workspace starts here.",
    description: "Sign in to manage your account and optionally connect your YouTube channel.",
    destination: "/login?next=%2Faccount",
  },
} as const;

export default function GuestAccessPrompt({ feature }: { feature: keyof typeof CONTENT }) {
  const content = CONTENT[feature];
  return (
    <section className="guest-access" aria-labelledby={`${feature}-access-title`}>
      <span className="guest-access__icon"><StudioIcon name={feature === "history" ? "history" : "channel"} width="28" height="28" /></span>
      <p className="section-kicker">Your private creator space</p>
      <h2 id={`${feature}-access-title`}>{content.title}</h2>
      <p>{content.description}</p>
      <div className="guest-access__actions">
        <Link className="primary-button" href="/signup">Create account</Link>
        <Link className="secondary-button" href={content.destination}>Sign in</Link>
      </div>
    </section>
  );
}
