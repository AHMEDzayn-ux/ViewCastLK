import { validateEmail } from "./email-validation";

export interface LoginFormValues {
  email: string;
  password: string;
}

export interface LoginValidationErrors {
  email?: string;
  password?: string;
}

export function validateLoginForm(
  values: LoginFormValues,
): LoginValidationErrors {
  const errors: LoginValidationErrors = {};
  const emailError = validateEmail(values.email);

  if (emailError) {
    errors.email = emailError;
  }

  if (!values.password) {
    errors.password = "Enter your password.";
  }

  return errors;
}

export function isLoginFormValid(values: LoginFormValues): boolean {
  return Object.keys(validateLoginForm(values)).length === 0;
}
