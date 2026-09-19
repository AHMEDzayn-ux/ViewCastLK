"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";

import {
  isLoginFormValid,
  type LoginFormValues,
  type LoginValidationErrors,
  validateLoginForm,
} from "@/lib/auth/login-validation";
import { requireTurnstileSiteKey } from "@/lib/auth/turnstile-config";
import { supabase } from "@/lib/supabase/client";

const TURNSTILE_SITE_KEY = requireTurnstileSiteKey();

const INITIAL_VALUES: LoginFormValues = {
  email: "",
  password: "",
};

type FieldName = keyof LoginFormValues;
type TouchedFields = Partial<Record<FieldName, boolean>>;
type SubmissionState =
  | { status: "idle" }
  | { status: "error"; message: string };

interface FieldProps {
  id: FieldName;
  label: string;
  hint?: string;
  error?: string;
  children: React.ReactNode;
}

function Field({ id, label, hint, error, children }: FieldProps) {
  return (
    <div className="field">
      <div className="field__heading">
        <label className="field__label" htmlFor={id}>
          {label}
        </label>
        <span className="field__requirement field__requirement--required">
          Required
        </span>
      </div>
      {hint && (
        <p className="field__hint" id={`${id}-hint`}>
          {hint}
        </p>
      )}
      {children}
      {error && (
        <p className="field__error" id={`${id}-error`} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

function describedBy(id: FieldName, hasHint: boolean, error?: string) {
  return [hasHint ? `${id}-hint` : null, error ? `${id}-error` : null]
    .filter(Boolean)
    .join(" ") || undefined;
}

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

function safeLoginErrorMessage(error: unknown): string {
  switch (readErrorCode(error)) {
    case "email_not_confirmed":
      return "Please verify your email before signing in.";
    case "invalid_credentials":
    case "user_not_found":
      return "Email or password is incorrect.";
    case "captcha_failed":
      return "Verification could not be confirmed. Complete a new verification and try again.";
    case "over_request_rate_limit":
      return "We could not sign you in right now. Please wait a moment and try again.";
    default:
      return "We could not sign you in. Please try again.";
  }
}

export default function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [values, setValues] = useState<LoginFormValues>(INITIAL_VALUES);
  const [touched, setTouched] = useState<TouchedFields>({});
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaMessage, setCaptchaMessage] = useState<string | null>(null);
  const [submission, setSubmission] = useState<SubmissionState>({
    status: "idle",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const turnstileRef = useRef<TurnstileInstance | null>(null);

  const validationErrors = validateLoginForm(values);
  const canSubmit =
    isLoginFormValid(values) && Boolean(captchaToken) && !isSubmitting;
  const statusMessage =
    searchParams.get("password_reset") === "1"
      ? "Your password has been reset. Sign in with your new password."
      : searchParams.get("verified") === "1"
        ? "Email verified successfully. You can now sign in."
        : null;

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

  function setValue(field: FieldName, value: string) {
    setValues((current) => ({ ...current, [field]: value }));
    if (submission.status === "error") {
      setSubmission({ status: "idle" });
    }
  }

  function fieldError(field: keyof LoginValidationErrors) {
    return touched[field] ? validationErrors[field] : undefined;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const errors = validateLoginForm(values);
    const currentCaptchaToken = captchaToken;

    if (Object.keys(errors).length > 0 || !currentCaptchaToken) {
      setTouched({ email: true, password: true });
      if (!currentCaptchaToken) {
        setCaptchaMessage("Complete the verification before signing in.");
      }
      return;
    }

    setIsSubmitting(true);
    setSubmission({ status: "idle" });

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email: values.email.trim(),
        password: values.password,
        options: {
          captchaToken: currentCaptchaToken,
        },
      });

      if (error) {
        setSubmission({ status: "error", message: safeLoginErrorMessage(error) });
        return;
      }

      if (!data.session) {
        setSubmission({
          status: "error",
          message: "We could not sign you in. Please try again.",
        });
        return;
      }

      router.replace("/forecast");
    } catch (error) {
      setSubmission({ status: "error", message: safeLoginErrorMessage(error) });
    } finally {
      setCaptchaToken(null);
      turnstileRef.current?.reset();
      setIsSubmitting(false);
    }
  }

  const emailError = fieldError("email");
  const passwordError = fieldError("password");
  const inputClass = (error?: string) =>
    `field-control${error ? " field-control--invalid" : ""}`;

  return (
    <form
      className="forecast-form auth-form"
      onSubmit={handleSubmit}
      noValidate
      aria-labelledby="login-form-title"
      aria-busy={isSubmitting}
    >
      <div className="form-section__header">
        <div>
          <p className="section-kicker">Secure sign in</p>
          <h2 id="login-form-title">Access your account</h2>
        </div>
        <p>All fields are required.</p>
      </div>

      {statusMessage && (
        <div className="auth-message auth-message--success" role="status">
          {statusMessage}
        </div>
      )}

      <div className="form-grid">
        <Field id="email" label="Email" error={emailError}>
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
            className={inputClass(emailError)}
            placeholder="you@example.com"
            aria-invalid={Boolean(emailError)}
            aria-describedby={describedBy("email", false, emailError)}
            onBlur={() => setTouched((current) => ({ ...current, email: true }))}
            onChange={(event) => setValue("email", event.target.value)}
          />
        </Field>

        <Field id="password" label="Password" error={passwordError}>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={values.password}
            disabled={isSubmitting}
            className={inputClass(passwordError)}
            aria-invalid={Boolean(passwordError)}
            aria-describedby={describedBy("password", false, passwordError)}
            onBlur={() =>
              setTouched((current) => ({ ...current, password: true }))
            }
            onChange={(event) => setValue("password", event.target.value)}
          />
        </Field>
      </div>

      <p className="auth-recovery-link">
        <Link href="/forgot-password">Forgot password?</Link>
      </p>

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
            action: "login",
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
          <strong>Sign in was not completed</strong>
          <p>{submission.message}</p>
        </div>
      )}

      <div className="form-actions auth-form__actions">
        <button className="primary-button" type="submit" disabled={!canSubmit}>
          {isSubmitting ? "Signing in..." : "Sign in"}
        </button>
      </div>

      <p className="auth-switch">
        Don&apos;t have an account? <Link href="/signup">Create account</Link>
      </p>
    </form>
  );
}
