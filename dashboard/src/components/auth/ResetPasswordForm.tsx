"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import AuthField, { authFieldDescribedBy } from "@/components/auth/AuthField";
import {
  getPasswordRequirementStatus,
  isNewPasswordFormValid,
  type NewPasswordFormValues,
  type PasswordValidationErrors,
  validateNewPasswordForm,
} from "@/lib/auth/password-validation";
import { supabase } from "@/lib/supabase/client";

const INITIAL_VALUES: NewPasswordFormValues = {
  password: "",
  confirmPassword: "",
};

const INVALID_RECOVERY_MESSAGE =
  "This password reset link is invalid or has expired. Request a new one.";

const INVALID_SESSION_ERROR_CODES = new Set([
  "bad_jwt",
  "invalid_jwt",
  "jwt_expired",
  "refresh_token_already_used",
  "refresh_token_not_found",
  "session_not_found",
]);

type FieldName = keyof NewPasswordFormValues;
type TouchedFields = Partial<Record<FieldName, boolean>>;
type RecoveryState = "checking" | "valid" | "invalid" | "success";
type SubmissionState =
  | { status: "idle" }
  | { status: "error"; message: string };

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

function safePasswordUpdateErrorMessage(error: unknown): string {
  switch (readErrorCode(error)) {
    case "weak_password":
      return "Choose a password that meets all of the listed requirements.";
    case "same_password":
      return "Choose a new password that is different from your current password.";
    case "over_request_rate_limit":
      return "We could not update your password right now. Please wait a moment and try again.";
    default:
      return "We could not update your password. Please try again.";
  }
}

export default function ResetPasswordForm() {
  const router = useRouter();
  const [recoveryState, setRecoveryState] =
    useState<RecoveryState>("checking");
  const [values, setValues] =
    useState<NewPasswordFormValues>(INITIAL_VALUES);
  const [touched, setTouched] = useState<TouchedFields>({});
  const [submission, setSubmission] = useState<SubmissionState>({
    status: "idle",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const recoveryObserved = useRef(false);

  useEffect(() => {
    let isActive = true;
    let invalidSessionTimer: number | null = null;

    function markInvalid() {
      setRecoveryState((current) =>
        current === "success" ? current : "invalid",
      );
    }

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (!isActive) return;

      if (event === "PASSWORD_RECOVERY") {
        recoveryObserved.current = Boolean(session);
        setRecoveryState(session ? "valid" : "invalid");
        return;
      }

      if (event === "SIGNED_OUT") {
        recoveryObserved.current = false;
        markInvalid();
        return;
      }

      if (event === "INITIAL_SESSION") {
        if (!session) {
          markInvalid();
          return;
        }

        // Supabase emits INITIAL_SESSION before its queued PASSWORD_RECOVERY
        // event. Give that recovery event one task to arrive, while ensuring an
        // ordinary stored session never authorizes this form.
        invalidSessionTimer = window.setTimeout(() => {
          if (isActive && !recoveryObserved.current) {
            markInvalid();
          }
        }, 0);
        return;
      }

      if (
        !recoveryObserved.current &&
        (event === "SIGNED_IN" || event === "TOKEN_REFRESHED")
      ) {
        markInvalid();
      }
    });

    return () => {
      isActive = false;
      if (invalidSessionTimer !== null) {
        window.clearTimeout(invalidSessionTimer);
      }
      subscription.unsubscribe();
    };
  }, []);

  const validationErrors = validateNewPasswordForm(values);
  const passwordRequirements = getPasswordRequirementStatus(values.password);
  const canSubmit =
    recoveryState === "valid" &&
    isNewPasswordFormValid(values) &&
    !isSubmitting;

  function setValue(field: FieldName, value: string) {
    setValues((current) => ({ ...current, [field]: value }));
    if (submission.status === "error") {
      setSubmission({ status: "idle" });
    }
  }

  function fieldError(field: keyof PasswordValidationErrors) {
    return touched[field] ? validationErrors[field] : undefined;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting || recoveryState !== "valid") return;

    const errors = validateNewPasswordForm(values);
    if (Object.keys(errors).length > 0) {
      setTouched({ password: true, confirmPassword: true });
      return;
    }

    setIsSubmitting(true);
    setSubmission({ status: "idle" });

    try {
      const { error } = await supabase.auth.updateUser({
        password: values.password,
      });

      if (error) {
        const errorCode = readErrorCode(error);
        if (errorCode && INVALID_SESSION_ERROR_CODES.has(errorCode)) {
          setRecoveryState("invalid");
          return;
        }

        setSubmission({
          status: "error",
          message: safePasswordUpdateErrorMessage(error),
        });
        return;
      }

      const { error: signOutError } = await supabase.auth.signOut();
      if (signOutError) {
        setValues(INITIAL_VALUES);
        setTouched({});
        setSubmission({
          status: "error",
          message:
            "Your password was updated, but automatic sign out was not completed. Please close this page before signing in again.",
        });
        return;
      }

      setValues(INITIAL_VALUES);
      setTouched({});
      setRecoveryState("success");
      router.replace("/login?password_reset=1");
    } catch (error) {
      const errorCode = readErrorCode(error);
      if (errorCode && INVALID_SESSION_ERROR_CODES.has(errorCode)) {
        setRecoveryState("invalid");
        return;
      }

      setSubmission({
        status: "error",
        message: safePasswordUpdateErrorMessage(error),
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  if (recoveryState === "checking") {
    return (
      <section className="auth-success" aria-busy="true" aria-live="polite">
        <p className="section-kicker">Secure recovery</p>
        <h2>Checking your reset link</h2>
        <p>Please wait while we verify your recovery session.</p>
      </section>
    );
  }

  if (recoveryState === "invalid") {
    return (
      <section className="auth-success" role="alert">
        <p className="section-kicker">Reset link unavailable</p>
        <h2>Request a new reset link</h2>
        <p>{INVALID_RECOVERY_MESSAGE}</p>
        <Link className="primary-button" href="/forgot-password">
          Request a new link
        </Link>
      </section>
    );
  }

  if (recoveryState === "success") {
    return (
      <section className="auth-success" role="status" aria-live="polite">
        <p className="section-kicker">Password updated</p>
        <h2>Your password has been reset</h2>
        <p>You will be redirected to sign in with your new password.</p>
      </section>
    );
  }

  const passwordError = fieldError("password");
  const confirmPasswordError = fieldError("confirmPassword");
  const inputClass = (error?: string) =>
    `field-control${error ? " field-control--invalid" : ""}`;

  return (
    <form
      className="forecast-form auth-form"
      onSubmit={handleSubmit}
      noValidate
      aria-labelledby="reset-password-form-title"
      aria-busy={isSubmitting}
    >
      <div className="form-section__header">
        <div>
          <p className="section-kicker">Secure recovery</p>
          <h2 id="reset-password-form-title">Choose a new password</h2>
        </div>
        <p>All fields are required.</p>
      </div>

      <div className="form-grid">
        <AuthField
          id="password"
          label="New password"
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
              "reset-password-requirements",
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
            id="reset-password-requirements"
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
        </AuthField>

        <AuthField
          id="confirmPassword"
          label="Confirm new password"
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
            aria-describedby={authFieldDescribedBy(
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
        </AuthField>
      </div>

      {submission.status === "error" && (
        <div className="auth-message auth-message--error" role="alert">
          <strong>Password update was not completed</strong>
          <p>{submission.message}</p>
        </div>
      )}

      <div className="form-actions auth-form__actions">
        <button className="primary-button" type="submit" disabled={!canSubmit}>
          {isSubmitting ? "Updating password..." : "Update password"}
        </button>
      </div>
    </form>
  );
}
