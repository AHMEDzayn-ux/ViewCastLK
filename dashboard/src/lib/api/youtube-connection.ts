import { supabase } from "@/lib/supabase/client";

const API_BASE_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL?.trim().replace(
  /\/$/,
  "",
);

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

async function authenticatedGet<T>(path: string): Promise<T> {
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
    method: "GET",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
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
  return authenticatedGet<YouTubeConnectionStatus>(
    "/creator/youtube-connection",
  );
}

export async function startYouTubeConnection(): Promise<string> {
  const response = await authenticatedGet<{ authorizationUrl: string }>(
    "/auth/youtube/start",
  );
  const authorizationUrl = new URL(response.authorizationUrl);
  if (
    authorizationUrl.protocol !== "https:" ||
    authorizationUrl.hostname !== "accounts.google.com" ||
    authorizationUrl.pathname !== "/o/oauth2/v2/auth"
  ) {
    throw new YouTubeConnectionError(
      "The authorization destination was not recognized.",
    );
  }
  return authorizationUrl.toString();
}
