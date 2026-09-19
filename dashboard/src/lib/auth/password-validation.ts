export type PasswordRequirementKey =
  | "length"
  | "lowercase"
  | "uppercase"
  | "digit"
  | "symbol";

interface PasswordRequirement {
  key: PasswordRequirementKey;
  label: string;
  test: (password: string) => boolean;
}

export interface NewPasswordFormValues {
  password: string;
  confirmPassword: string;
}

export interface PasswordValidationErrors {
  password?: string;
  confirmPassword?: string;
}

const SUPABASE_PASSWORD_SYMBOLS =
  "!@#$%^&*()_+-=[]{};'\\:\"|<>?,./`~";

export const PASSWORD_REQUIREMENTS: readonly PasswordRequirement[] = [
  {
    key: "length",
    label: "10 or more characters",
    test: (password) => password.length >= 10,
  },
  {
    key: "lowercase",
    label: "One lowercase letter",
    test: (password) => /[a-z]/.test(password),
  },
  {
    key: "uppercase",
    label: "One uppercase letter",
    test: (password) => /[A-Z]/.test(password),
  },
  {
    key: "digit",
    label: "One digit",
    test: (password) => /[0-9]/.test(password),
  },
  {
    key: "symbol",
    label: "One symbol",
    test: (password) =>
      [...password].some((character) =>
        SUPABASE_PASSWORD_SYMBOLS.includes(character),
      ),
  },
];

export function getPasswordRequirementStatus(password: string) {
  return PASSWORD_REQUIREMENTS.map((requirement) => ({
    key: requirement.key,
    label: requirement.label,
    met: requirement.test(password),
  }));
}

export function validateNewPasswordForm(
  values: NewPasswordFormValues,
): PasswordValidationErrors {
  const errors: PasswordValidationErrors = {};

  if (!values.password) {
    errors.password = "Create a password.";
  } else if (
    getPasswordRequirementStatus(values.password).some(
      (requirement) => !requirement.met,
    )
  ) {
    errors.password = "Your password must meet every requirement below.";
  }

  if (!values.confirmPassword) {
    errors.confirmPassword = "Confirm your password.";
  } else if (values.password !== values.confirmPassword) {
    errors.confirmPassword = "The passwords do not match.";
  }

  return errors;
}

export function isNewPasswordFormValid(values: NewPasswordFormValues): boolean {
  return Object.keys(validateNewPasswordForm(values)).length === 0;
}
