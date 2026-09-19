import type { Metadata } from "next";

import ResetPasswordForm from "@/components/auth/ResetPasswordForm";

export const metadata: Metadata = {
  title: "Reset password",
  description: "Choose a new password for your ViewCastLK account.",
};

export default function ResetPasswordPage() {
  return (
    <main className="page-shell auth-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">ViewCastLK account</p>
        <h1>Reset your password</h1>
        <p>
          Choose a strong new password after your recovery link has been
          verified.
        </p>
      </header>

      <div className="auth-shell">
        <ResetPasswordForm />
      </div>
    </main>
  );
}
