"use client";

import Link from "next/link";
import { useCallback, useRef, useState } from "react";
import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";

import AuthField, { authFieldDescribedBy } from "@/components/auth/AuthField";
import {
  isForgotPasswordFormValid,
  type ForgotPasswordFormValues,
  validateForgotPasswordForm,
} from "@/lib/auth/forgot-password-validation";
import { requireTurnstileSiteKey } from "@/lib/auth/turnstile-config";
import { supabase } from "@/lib/supabase/client";

const TURNSTILE_SITE_KEY = requireTurnstileSiteKey();
const NEUTRAL_SUCCESS_MESSAGE =
  "If an account exists for that email, a password reset link has been sent.";

const ACCOUNT_ENUMERATION_SAFE_CODES = new Set([
  "email_not_found",
  "identity_not_found",
  "user_not_found",
]);

type SubmissionState =
  | { status: "idle" }
  | { status: "error"; message: string }
  | { status: "success" };

function readErrorCode(error: unknown): string | undefined {
  if (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof error.code === "string"
  ) {
    return error.code;
  }

  return undefined;
}

function safeForgotPasswordErrorMessage(error: unknown): string {
  switch (readErrorCode(error)) {
    case "captcha_failed":
      return "Verification could not be confirmed. Complete a new verification and try again.";
    case "over_email_send_rate_limit":
    case "over_request_rate_limit":
      return "We could not process the request right now. Please wait a moment and try again.";
    default:
      return "We could not process the request. Please try again.";
  }
}

export default function ForgotPasswordForm() {
  const [values, setValues] = useState<ForgotPasswordFormValues>({ email: "" });
  const [emailTouched, setEmailTouched] = useState(false);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaMessage, setCaptchaMessage] = useState<string | null>(null);
  const [submission, setSubmission] = useState<SubmissionState>({
    status: "idle",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const turnstileRef = useRef<TurnstileInstance | null>(null);

  const validationErrors = validateForgotPasswordForm(values);
  const emailError = emailTouched ? validationErrors.email : undefined;
  const canSubmit =
    isForgotPasswordFormValid(values) &&
    Boolean(captchaToken) &&
    !isSubmitting;

  const handleCaptchaSuccess = useCallback((token: string) => {
    setCaptchaToken(token);
    setCaptchaMessage(null);
  }, []);

  const handleCaptchaExpire = useCallback(() => {
    setCaptchaToken(null);
    setCaptchaMessage(
      "Verification expired. Complete the refreshed check before continuing.",
    );
  }, []);

  const handleCaptchaError = useCallback(() => {
    setCaptchaToken(null);
    setCaptchaMessage(
      "Verification could not be completed. Please try the check again.",
    );
  }, []);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const errors = validateForgotPasswordForm(values);
    const currentCaptchaToken = captchaToken;

    if (Object.keys(errors).length > 0 || !currentCaptchaToken) {
      setEmailTouched(true);
      if (!currentCaptchaToken) {
        setCaptchaMessage("Complete the verification before continuing.");
      }
      return;
    }

    setIsSubmitting(true);
    setSubmission({ status: "idle" });

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(
        values.email.trim(),
        {
          redirectTo: `${window.location.origin}/reset-password`,
          captchaToken: currentCaptchaToken,
        },
      );

      const errorCode = readErrorCode(error);
      if (error && (!errorCode || !ACCOUNT_ENUMERATION_SAFE_CODES.has(errorCode))) {
        setSubmission({
          status: "error",
          message: safeForgotPasswordErrorMessage(error),
        });
        return;
      }

      setValues({ email: "" });
      setEmailTouched(false);
      setSubmission({ status: "success" });
    } catch (error) {
      setSubmission({
        status: "error",
        message: safeForgotPasswordErrorMessage(error),
      });
    } finally {
      setCaptchaToken(null);
      turnstileRef.current?.reset();
      setIsSubmitting(false);
    }
  }

  if (submission.status === "success") {
    return (
      <section className="auth-success" role="status" aria-live="polite">
        <p className="section-kicker">Check your inbox</p>
        <h2>Reset request received</h2>
        <p>{NEUTRAL_SUCCESS_MESSAGE}</p>
        <Link className="primary-button" href="/login">
          Back to sign in
        </Link>
      </section>
    );
  }

  return (
    <form
      className="forecast-form auth-form"
      onSubmit={handleSubmit}
      noValidate
      aria-labelledby="forgot-password-form-title"
      aria-busy={isSubmitting}
    >
      <div className="form-section__header">
        <div>
          <p className="section-kicker">Account recovery</p>
          <h2 id="forgot-password-form-title">Request a reset link</h2>
        </div>
        <p>Email is required.</p>
      </div>

      <div className="form-grid">
        <AuthField
          id="email"
          label="Email"
          hint="Enter the email address associated with your account."
          error={emailError}
        >
          <input
            id="email"
            name="email"
            type="email"
            inputMode="email"
            autoComplete="email"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            value={values.email}
            disabled={isSubmitting}
            className={`field-control${
              emailError ? " field-control--invalid" : ""
            }`}
            placeholder="you@example.com"
            aria-invalid={Boolean(emailError)}
            aria-describedby={authFieldDescribedBy("email", true, emailError)}
            onBlur={() => setEmailTouched(true)}
            onChange={(event) => {
              setValues({ email: event.target.value });
              if (submission.status === "error") {
                setSubmission({ status: "idle" });
              }
            }}
          />
        </AuthField>
      </div>

      <div className="auth-turnstile">
        <div className="auth-turnstile__heading">
          <span className="field__label">Security verification</span>
          <span className="field__requirement field__requirement--required">
            Required
          </span>
        </div>
        <Turnstile
          ref={turnstileRef}
          siteKey={TURNSTILE_SITE_KEY}
          onSuccess={handleCaptchaSuccess}
          onExpire={handleCaptchaExpire}
          onError={handleCaptchaError}
          onTimeout={handleCaptchaError}
          options={{
            action: "forgot_password",
            responseField: false,
            size: "flexible",
            theme: "light",
          }}
        />
        {captchaMessage && (
          <p className="field__error" role="alert">
            {captchaMessage}
          </p>
        )}
      </div>

      {submission.status === "error" && (
        <div className="auth-message auth-message--error" role="alert">
          <strong>Request was not completed</strong>
          <p>{submission.message}</p>
        </div>
      )}

      <div className="form-actions auth-form__actions">
        <button className="primary-button" type="submit" disabled={!canSubmit}>
          {isSubmitting ? "Sending reset link..." : "Send reset link"}
        </button>
      </div>

      <p className="auth-switch">
        <Link href="/login">Back to sign in</Link>
      </p>
    </form>
  );
}
