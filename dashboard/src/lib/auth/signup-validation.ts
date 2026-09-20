import { validateEmail } from "./email-validation";
import {
  type PasswordValidationErrors,
  validateNewPasswordForm,
} from "./password-validation";

export { getPasswordRequirementStatus } from "./password-validation";

export interface SignupFormValues {
  email: string;
  password: string;
  confirmPassword: string;
}

export interface SignupValidationErrors extends PasswordValidationErrors {
  email?: string;
}

export function validateSignupForm(
  values: SignupFormValues,
): SignupValidationErrors {
  const errors: SignupValidationErrors = validateNewPasswordForm(values);
  const emailError = validateEmail(values.email);

  if (emailError) {
    errors.email = emailError;
  }

  return errors;
}

export function isSignupFormValid(values: SignupFormValues): boolean {
  return Object.keys(validateSignupForm(values)).length === 0;
}
