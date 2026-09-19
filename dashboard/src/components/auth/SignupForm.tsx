"use client";

import Link from "next/link";
import { useCallback, useRef, useState } from "react";
import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";

import {
  getPasswordRequirementStatus,
  isSignupFormValid,
  type SignupFormValues,
  type SignupValidationErrors,
  validateSignupForm,
} from "@/lib/auth/signup-validation";
import { requireTurnstileSiteKey } from "@/lib/auth/turnstile-config";
import { supabase } from "@/lib/supabase/client";

const TURNSTILE_SITE_KEY = requireTurnstileSiteKey();

const INITIAL_VALUES: SignupFormValues = {
  email: "",
  password: "",
  confirmPassword: "",
};

const GENERIC_SUCCESS_MESSAGE =
  "Account created. Check your email to verify your account before signing in.";

const ACCOUNT_ENUMERATION_SAFE_CODES = new Set([
  "email_exists",
  "identity_already_exists",
  "user_already_exists",
]);

type FieldName = keyof SignupFormValues;
type TouchedFields = Partial<Record<FieldName, boolean>>;
type SubmissionState =
  | { status: "idle" }
  | { status: "error"; message: string }
  | { status: "success"; message: string };

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

function safeSignupErrorMessage(error: unknown): string {
  switch (readErrorCode(error)) {
    case "captcha_failed":
      return "Verification could not be confirmed. Complete a new verification and try again.";
    case "weak_password":
      return "Choose a password that meets all of the listed requirements.";
    case "email_address_invalid":
      return "Enter a valid email address.";
    case "over_email_send_rate_limit":
    case "over_request_rate_limit":
      return "We could not complete signup right now. Please wait a moment and try again.";
    case "signup_disabled":
      return "Account creation is temporarily unavailable.";
    default:
      return "We could not complete signup. Please try again.";
  }
}

export default function SignupForm() {
  const [values, setValues] = useState<SignupFormValues>(INITIAL_VALUES);
  const [touched, setTouched] = useState<TouchedFields>({});
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaMessage, setCaptchaMessage] = useState<string | null>(null);
  const [submission, setSubmission] = useState<SubmissionState>({
    status: "idle",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const turnstileRef = useRef<TurnstileInstance | null>(null);

  const validationErrors = validateSignupForm(values);
  const passwordRequirements = getPasswordRequirementStatus(values.password);
  const canSubmit =
    isSignupFormValid(values) && Boolean(captchaToken) && !isSubmitting;

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

  function fieldError(field: keyof SignupValidationErrors) {
    return touched[field] ? validationErrors[field] : undefined;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) return;

    const errors = validateSignupForm(values);
    const currentCaptchaToken = captchaToken;

    if (Object.keys(errors).length > 0 || !currentCaptchaToken) {
      setTouched({ email: true, password: true, confirmPassword: true });
      if (!currentCaptchaToken) {
        setCaptchaMessage("Complete the verification before creating an account.");
      }
      return;
    }

    setIsSubmitting(true);
    setSubmission({ status: "idle" });

    try {
      const { error } = await supabase.auth.signUp({
        email: values.email.trim(),
        password: values.password,
        options: {
          captchaToken: currentCaptchaToken,
          emailRedirectTo: `${window.location.origin}/login?verified=1`,
        },
      });

      const errorCode = readErrorCode(error);
      if (error && !errorCode) {
        setSubmission({ status: "error", message: safeSignupErrorMessage(error) });
        return;
      }

      if (errorCode && !ACCOUNT_ENUMERATION_SAFE_CODES.has(errorCode)) {
        setSubmission({ status: "error", message: safeSignupErrorMessage(error) });
        return;
      }

      setValues(INITIAL_VALUES);
      setTouched({});
      setSubmission({ status: "success", message: GENERIC_SUCCESS_MESSAGE });
    } catch (error) {
      setSubmission({ status: "error", message: safeSignupErrorMessage(error) });
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
        <h2>Verify your email address</h2>
        <p>{submission.message}</p>
        <Link className="primary-button" href="/login">
          Continue to sign in
        </Link>
      </section>
    );
  }

  const emailError = fieldError("email");
  const passwordError = fieldError("password");
  const confirmPasswordError = fieldError("confirmPassword");
  const inputClass = (error?: string) =>
    `field-control${error ? " field-control--invalid" : ""}`;

  return (
    <form
      className="forecast-form auth-form"
      onSubmit={handleSubmit}
      noValidate
      aria-labelledby="signup-form-title"
      aria-busy={isSubmitting}
    >
      <div className="form-section__header">
        <div>
          <p className="section-kicker">Secure signup</p>
          <h2 id="signup-form-title">Create your account</h2>
        </div>
        <p>All fields are required.</p>
      </div>

      <div className="form-grid">
        <Field
          id="email"
          label="Email"
          hint="Use an address you can access for account verification."
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
            className={inputClass(emailError)}
            placeholder="you@example.com"
            aria-invalid={Boolean(emailError)}
            aria-describedby={describedBy("email", true, emailError)}
            onBlur={() => setTouched((current) => ({ ...current, email: true }))}
            onChange={(event) => setValue("email", event.target.value)}
          />
        </Field>

        <Field
          id="password"
          label="Password"
          hint="Use a unique password that you do not use elsewhere."
          error={passwordError}
        >
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="new-password"
            value={values.password}
            disabled={isSubmitting}
            className={inputClass(passwordError)}
            aria-invalid={Boolean(passwordError)}
            aria-describedby={[
              "password-hint",
              "password-requirements",
              passwordError ? "password-error" : null,
            ]
              .filter(Boolean)
              .join(" ")}
            onBlur={() =>
              setTouched((current) => ({ ...current, password: true }))
            }
            onChange={(event) => setValue("password", event.target.value)}
          />

          <ul
            className="password-requirements"
            id="password-requirements"
            aria-label="Password requirements"
          >
            {passwordRequirements.map((requirement) => (
              <li
                className={requirement.met ? "is-met" : undefined}
                key={requirement.key}
              >
                <span className="sr-only">
                  {requirement.met ? "Requirement met: " : "Requirement not met: "}
                </span>
                {requirement.label}
              </li>
            ))}
          </ul>
        </Field>

        <Field
          id="confirmPassword"
          label="Confirm password"
          error={confirmPasswordError}
        >
          <input
            id="confirmPassword"
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={values.confirmPassword}
            disabled={isSubmitting}
            className={inputClass(confirmPasswordError)}
            aria-invalid={Boolean(confirmPasswordError)}
            aria-describedby={describedBy(
              "confirmPassword",
              false,
              confirmPasswordError,
            )}
            onBlur={() =>
              setTouched((current) => ({
                ...current,
                confirmPassword: true,
              }))
            }
            onChange={(event) =>
              setValue("confirmPassword", event.target.value)
            }
          />
        </Field>
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
          onTimeout={handleCaptchaExpire}
          options={{
            action: "signup",
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
          <strong>Signup was not completed</strong>
          <p>{submission.message}</p>
        </div>
      )}

      <div className="form-actions auth-form__actions">
        <button className="primary-button" type="submit" disabled={!canSubmit}>
          {isSubmitting ? "Creating account…" : "Create account"}
        </button>
      </div>

      <p className="auth-switch">
        Already have an account? <Link href="/login">Sign in</Link>
      </p>
    </form>
  );
}
