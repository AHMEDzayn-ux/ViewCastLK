import { validateEmail } from "./email-validation";

export interface ForgotPasswordFormValues {
  email: string;
}

export interface ForgotPasswordValidationErrors {
  email?: string;
}

export function validateForgotPasswordForm(
  values: ForgotPasswordFormValues,
): ForgotPasswordValidationErrors {
  const emailError = validateEmail(values.email);
  return emailError ? { email: emailError } : {};
}

export function isForgotPasswordFormValid(
  values: ForgotPasswordFormValues,
): boolean {
  return Object.keys(validateForgotPasswordForm(values)).length === 0;
}
