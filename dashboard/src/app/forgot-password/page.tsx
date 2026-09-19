import type { Metadata } from "next";

import ForgotPasswordForm from "@/components/auth/ForgotPasswordForm";

export const metadata: Metadata = {
  title: "Forgot password",
  description: "Request a secure password reset link for ViewCastLK.",
};

export default function ForgotPasswordPage() {
  return (
    <main className="page-shell auth-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">ViewCastLK account</p>
        <h1>Forgot your password?</h1>
        <p>
          Enter your email address and we will send password reset instructions
          if an account is associated with it.
        </p>
      </header>

      <div className="auth-shell">
        <ForgotPasswordForm />
      </div>
    </main>
  );
}
