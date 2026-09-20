import type { Metadata } from "next";

import SignupForm from "@/components/auth/SignupForm";

export const metadata: Metadata = {
  title: "Create account",
  description: "Create a ViewCastLK account using a verified email address.",
};

export default function SignupPage() {
  return (
    <main className="page-shell auth-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">ViewCastLK account</p>
        <h1>Create an account</h1>
        <p>
          Set up secure access with your email address. You will need to verify
          your email before signing in.
        </p>
      </header>

      <div className="auth-shell">
        <SignupForm />
      </div>
    </main>
  );
}
