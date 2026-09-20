export function requireTurnstileSiteKey(): string {
  const siteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY?.trim();

  if (!siteKey) {
    throw new Error(
      "Missing required public environment variable: NEXT_PUBLIC_TURNSTILE_SITE_KEY",
    );
  }

  return siteKey;
}
