import { describe, expect, it } from "vitest";

import {
  getPasswordRequirementStatus,
  isSignupFormValid,
  validateSignupForm,
} from "./signup-validation";

const VALID_VALUES = {
  email: "creator@example.com",
  password: "StrongPass1!",
  confirmPassword: "StrongPass1!",
};

describe("signup validation", () => {
  it("accepts a valid email, matching passwords, and every password requirement", () => {
    expect(validateSignupForm(VALID_VALUES)).toEqual({});
    expect(isSignupFormValid(VALID_VALUES)).toBe(true);
    expect(
      getPasswordRequirementStatus(VALID_VALUES.password).every(
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
      validateSignupForm({
        ...VALID_VALUES,
        password,
        confirmPassword: password,
      }).password,
    ).toBeDefined();
  });

  it("rejects an invalid email address", () => {
    expect(
      validateSignupForm({ ...VALID_VALUES, email: "creator@example" }).email,
    ).toBe("Enter a valid email address.");
  });

  it("rejects passwords that do not match", () => {
    expect(
      validateSignupForm({
        ...VALID_VALUES,
        confirmPassword: "DifferentPass1!",
      }).confirmPassword,
    ).toBe("The passwords do not match.");
  });
});
