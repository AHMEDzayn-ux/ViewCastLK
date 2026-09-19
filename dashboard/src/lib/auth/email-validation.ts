const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function validateEmail(emailValue: string): string | undefined {
  const email = emailValue.trim();

  if (!email) {
    return "Enter your email address.";
  }

  if (!EMAIL_PATTERN.test(email)) {
    return "Enter a valid email address.";
  }

  return undefined;
}
