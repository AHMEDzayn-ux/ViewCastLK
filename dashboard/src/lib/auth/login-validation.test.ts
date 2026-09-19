import { describe, expect, it } from "vitest";

import { isLoginFormValid, validateLoginForm } from "./login-validation";

describe("login validation", () => {
  it("accepts a structurally valid email and any non-empty password", () => {
    expect(
      isLoginFormValid({
        email: "creator@example.com",
        password: "x",
      }),
    ).toBe(true);
  });

  it("rejects an empty or malformed email", () => {
    expect(validateLoginForm({ email: "", password: "password" }).email).toBe(
      "Enter your email address.",
    );
    expect(
      validateLoginForm({ email: "creator@example", password: "password" })
        .email,
    ).toBe("Enter a valid email address.");
  });

  it("rejects an empty password without applying signup complexity rules", () => {
    expect(
      validateLoginForm({ email: "creator@example.com", password: "" })
        .password,
    ).toBe("Enter your password.");
  });
});
