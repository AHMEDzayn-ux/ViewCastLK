import type { ReactNode } from "react";

interface AuthFieldProps {
  id: string;
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
}

export default function AuthField({
  id,
  label,
  hint,
  error,
  children,
}: AuthFieldProps) {
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

export function authFieldDescribedBy(
  id: string,
  hasHint: boolean,
  error?: string,
) {
  return [hasHint ? `${id}-hint` : null, error ? `${id}-error` : null]
    .filter(Boolean)
    .join(" ") || undefined;
}
