import { describe, expect, it } from "vitest";

import {
  isForgotPasswordFormValid,
  validateForgotPasswordForm,
} from "./forgot-password-validation";

describe("forgot-password email validation", () => {
  it("accepts a structurally valid email", () => {
    expect(
      isForgotPasswordFormValid({ email: "creator@example.com" }),
    ).toBe(true);
  });

  it("rejects empty and malformed email addresses", () => {
    expect(validateForgotPasswordForm({ email: "" }).email).toBe(
      "Enter your email address.",
    );
    expect(
      validateForgotPasswordForm({ email: "creator@example" }).email,
    ).toBe("Enter a valid email address.");
  });
});
