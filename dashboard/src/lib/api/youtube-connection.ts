import { supabase } from "@/lib/supabase/client";

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim().replace(
  /\/$/,
  "",
);

const BINDING_STORAGE_KEY = "viewcastlk.youtube.oauth.binding";
const BINDING_LIFETIME_MS = 10 * 60 * 1000;
const NONCE_PATTERN = /^[A-Za-z0-9_-]{43}$/;

interface BrowserBinding {
  state: string;
  bindingNonce: string;
  expiresAt: number;
}

function invalidCompletion(): YouTubeConnectionError {
  return new YouTubeConnectionError(
    "This connection could not be verified. Start again from your account in this browser.",
    400,
  );
}

export interface YouTubeConnectionStatus {
  isConnected: boolean;
  channelId: string | null;
  channelTitle: string | null;
  status: "pending_sync" | "active" | "reauth_required" | "error" | null;
  connectedAt: string | null;
  lastRefreshOkAt: string | null;
}

export class YouTubeConnectionError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "YouTubeConnectionError";
  }
}

async function authenticatedRequest<T>(
  path: string,
  method: "GET" | "DELETE" | "POST" = "GET",
  body?: Record<string, unknown>,
): Promise<T> {
  if (!API_BASE_URL) {
    throw new YouTubeConnectionError(
      "YouTube channel connection is not configured in this environment.",
    );
  }

  const { data, error } = await supabase.auth.getSession();
  const accessToken = data.session?.access_token;
  if (error || !accessToken) {
    throw new YouTubeConnectionError("Sign in again to continue.", 401);
  }

  const response = await fetch(API_BASE_URL + path, {
    method,
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${accessToken}`,
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
    cache: "no-store",
  });
  const payload = (await response.json().catch(() => null)) as
    | T
    | { message?: string }
    | null;

  if (!response.ok) {
    throw new YouTubeConnectionError(
      (payload as { message?: string } | null)?.message ??
        "YouTube channel connection is temporarily unavailable.",
      response.status,
    );
  }
  return payload as T;
}

export async function getYouTubeConnection(): Promise<YouTubeConnectionStatus> {
  return authenticatedRequest<YouTubeConnectionStatus>(
    "/creator/youtube-connection",
  );
}

export async function startYouTubeConnection(): Promise<string> {
  // Fail before navigation when storage is unavailable. Never fall back to a
  // flow that can complete without independent tab proof.
  window.sessionStorage.removeItem(BINDING_STORAGE_KEY);
  const response = await authenticatedRequest<{
    authorizationUrl: string;
    state: string;
    bindingNonce: string;
  }>(
    "/auth/youtube/start",
  );
  const authorizationUrl = new URL(response.authorizationUrl);
  if (
    authorizationUrl.protocol !== "https:" ||
    authorizationUrl.hostname !== "accounts.google.com" ||
    authorizationUrl.pathname !== "/o/oauth2/v2/auth" ||
    typeof response.state !== "string" || !response.state || response.state.length > 512 ||
    authorizationUrl.searchParams.get("state") !== response.state ||
    typeof response.bindingNonce !== "string" || !NONCE_PATTERN.test(response.bindingNonce)
  ) {
    throw new YouTubeConnectionError(
      "The authorization destination was not recognized.",
    );
  }
  window.sessionStorage.setItem(BINDING_STORAGE_KEY, JSON.stringify({
    state: response.state,
    bindingNonce: response.bindingNonce,
    expiresAt: Date.now() + BINDING_LIFETIME_MS,
  } satisfies BrowserBinding));
  return authorizationUrl.toString();
}

export async function completeYouTubeConnection(): Promise<boolean> {
  const fragment = new URLSearchParams(window.location.hash.slice(1));
  // Remove Google's response before awaiting auth/network work. It is never
  // written to storage, logs, page content, query parameters or a referrer.
  window.history.replaceState(null, "", window.location.pathname);
  const state = fragment.get("youtube_state");
  const code = fragment.get("youtube_code");
  const denied = fragment.get("youtube_denied") === "1";
  const serialized = window.sessionStorage.getItem(BINDING_STORAGE_KEY);
  window.sessionStorage.removeItem(BINDING_STORAGE_KEY);
  let binding: BrowserBinding;
  try {
    binding = JSON.parse(serialized ?? "null") as BrowserBinding;
  } catch {
    throw invalidCompletion();
  }
  if (
    !binding || !state || state.length > 512 || binding.state !== state ||
    typeof binding.bindingNonce !== "string" || !NONCE_PATTERN.test(binding.bindingNonce) ||
    !Number.isFinite(binding.expiresAt) || binding.expiresAt <= Date.now() ||
    fragment.getAll("youtube_state").length !== 1 ||
    fragment.getAll("youtube_code").length > 1 ||
    fragment.getAll("youtube_denied").length > 1 ||
    (denied ? Boolean(code) : !code || code.length > 4096)
  ) {
    throw invalidCompletion();
  }
  const result = await authenticatedRequest<{ connected: boolean }>(
    "/auth/youtube/complete", "POST", {
      state, bindingNonce: binding.bindingNonce,
      ...(denied ? { denied: true } : { code }),
    },
  );
  if (typeof result?.connected !== "boolean") throw invalidCompletion();
  return result.connected;
}

export async function disconnectYouTubeConnection(): Promise<void> {
  await authenticatedRequest<{ disconnected: boolean }>(
    "/creator/youtube-connection",
    "DELETE",
  );
}
