import type { Metadata } from "next";
import { Suspense } from "react";

import LoginForm from "@/components/auth/LoginForm";

export const metadata: Metadata = {
  title: "Sign in",
  description: "Sign in securely to your ViewCastLK account.",
};

function LoginFormFallback() {
  return (
    <section className="auth-success" aria-busy="true">
      <p className="section-kicker">Secure sign in</p>
      <h2>Loading sign in</h2>
      <p>Please wait a moment.</p>
    </section>
  );
}

export default function LoginPage() {
  return (
    <main className="page-shell auth-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">ViewCastLK account</p>
        <h1>Sign in</h1>
        <p>
          Access your ViewCastLK account with your verified email address.
        </p>
      </header>

      <div className="auth-shell">
        <Suspense fallback={<LoginFormFallback />}>
          <LoginForm />
        </Suspense>
      </div>
    </main>
  );
}
