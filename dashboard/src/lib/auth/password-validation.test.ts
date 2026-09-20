import { describe, expect, it } from "vitest";

import {
  getPasswordRequirementStatus,
  isNewPasswordFormValid,
  validateNewPasswordForm,
} from "./password-validation";

describe("shared new-password validation", () => {
  it("accepts a matching password that meets every configured requirement", () => {
    const values = {
      password: "StrongPass1!",
      confirmPassword: "StrongPass1!",
    };

    expect(isNewPasswordFormValid(values)).toBe(true);
    expect(
      getPasswordRequirementStatus(values.password).every(
        (requirement) => requirement.met,
      ),
    ).toBe(true);
  });

  it.each([
    ["length", "Aa1!short"],
    ["lowercase", "UPPERCASE1!"],
    ["uppercase", "lowercase1!"],
    ["digit", "NoDigitsHere!"],
    ["symbol", "NoSymbols123"],
  ])("rejects a password missing the %s requirement", (key, password) => {
    const requirement = getPasswordRequirementStatus(password).find(
      (candidate) => candidate.key === key,
    );

    expect(requirement?.met).toBe(false);
    expect(
      validateNewPasswordForm({ password, confirmPassword: password }).password,
    ).toBeDefined();
  });

  it("rejects passwords that do not match", () => {
    expect(
      validateNewPasswordForm({
        password: "StrongPass1!",
        confirmPassword: "DifferentPass1!",
      }).confirmPassword,
    ).toBe("The passwords do not match.");
  });
});
