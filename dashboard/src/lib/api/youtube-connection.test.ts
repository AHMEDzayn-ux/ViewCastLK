// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ getSession: vi.fn() }));
vi.mock("@/lib/supabase/client", () => ({
  supabase: { auth: { getSession: mocks.getSession } },
}));

const STORAGE_KEY = "viewcastlk.youtube.oauth.binding";
const nonce = "a".repeat(43);
const authorizationUrl = "https://accounts.google.com/o/oauth2/v2/auth?state=test-state";

function response(payload: unknown, status = 200) {
  return { ok: status === 200, status, json: async () => payload };
}

async function start() {
  const api = await import("./youtube-connection");
  await api.startYouTubeConnection();
  return api;
}

function callback(fragment = "youtube_state=test-state&youtube_code=google-test-code") {
  window.history.replaceState(null, "", `/account/youtube-callback#${fragment}`);
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.stubEnv("NEXT_PUBLIC_PREDICTION_API_URL", "https://api.example.test");
  window.sessionStorage.clear();
  window.history.replaceState(null, "", "/account");
  mocks.getSession.mockResolvedValue({ data: { session: { access_token: "test-bearer" } }, error: null });
  vi.stubGlobal("fetch", vi.fn()
    .mockResolvedValueOnce(response({ authorizationUrl, state: "test-state", bindingNonce: nonce }))
    .mockResolvedValue(response({ connected: true })));
});

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("YouTube OAuth initiating tab proof", () => {
  it("keeps nonce in per-tab storage and never sends it or bearer in the Google URL", async () => {
    const api = await import("./youtube-connection");
    const url = await api.startYouTubeConnection();
    expect(url).toBe(authorizationUrl);
    expect(url).not.toContain(nonce);
    expect(url).not.toContain("test-bearer");
    const proof = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY)!);
    expect(proof).toMatchObject({ state: "test-state", bindingNonce: nonce });
    expect(proof.expiresAt).toBeGreaterThan(Date.now());
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(fetch).toHaveBeenCalledWith("https://api.example.test/auth/youtube/start", expect.objectContaining({
      cache: "no-store", headers: { Accept: "application/json", Authorization: "Bearer test-bearer" },
    }));
  });

  it("removes Google response from address bar before auth/network and posts all proof in body", async () => {
    const api = await start();
    callback();
    mocks.getSession.mockImplementation(async () => {
      expect(window.location.hash).toBe("");
      return { data: { session: { access_token: "fresh-same-session-bearer" } }, error: null };
    });
    expect(await api.completeYouTubeConnection()).toBe(true);
    expect(window.location.search).toBe("");
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(fetch).toHaveBeenLastCalledWith("https://api.example.test/auth/youtube/complete", expect.objectContaining({
      method: "POST", body: JSON.stringify({ state: "test-state", bindingNonce: nonce, code: "google-test-code" }),
      headers: expect.objectContaining({ Authorization: "Bearer fresh-same-session-bearer", "Content-Type": "application/json" }),
    }));
    expect(JSON.stringify(window.sessionStorage)).not.toContain("google-test-code");
  });

  it.each(["different_browser", "wrong_state", "expired_proof", "tampered_proof", "duplicate_state", "missing_code"])(
    "rejects %s without any completion request", async (problem) => {
      const api = await start();
      callback();
      if (problem === "different_browser") window.sessionStorage.clear();
      if (problem === "wrong_state") callback("youtube_state=other&youtube_code=test-code");
      if (problem === "duplicate_state") callback("youtube_state=test-state&youtube_state=other&youtube_code=test-code");
      if (problem === "missing_code") callback("youtube_state=test-state");
      if (problem === "expired_proof" || problem === "tampered_proof") {
        const proof = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY)!);
        if (problem === "expired_proof") proof.expiresAt = Date.now() - 1;
        else proof.bindingNonce = "invalid";
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(proof));
      }
      await expect(api.completeYouTubeConnection()).rejects.toThrow("could not be verified");
      expect(fetch).toHaveBeenCalledTimes(1); // start only, never exchange/link
      expect(window.location.hash).toBe("");
      expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    },
  );

  it("requires a current authenticated session even when nonce is present", async () => {
    const api = await start();
    callback();
    mocks.getSession.mockResolvedValue({ data: { session: null }, error: null });
    await expect(api.completeYouTubeConnection()).rejects.toThrow("Sign in again");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("consumes Google denial with proof and does not send a code", async () => {
    const api = await start();
    callback("youtube_state=test-state&youtube_denied=1");
    vi.mocked(fetch).mockResolvedValueOnce(response({ connected: false }) as Response);
    expect(await api.completeYouTubeConnection()).toBe(false);
    expect(fetch).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({
      body: JSON.stringify({ state: "test-state", bindingNonce: nonce, denied: true }),
    }));
  });

  it("fails closed when nonce storage is unavailable", async () => {
    const api = await import("./youtube-connection");
    const storage = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("unavailable"); });
    await expect(api.startYouTubeConnection()).rejects.toThrow("unavailable");
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
    storage.mockRestore();
  });

  it("reconnect replaces old per-flow proof", async () => {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ state: "old", bindingNonce: "b".repeat(43), expiresAt: Date.now()+600000 }));
    await start();
    expect(JSON.parse(window.sessionStorage.getItem(STORAGE_KEY)!)).toMatchObject({ state: "test-state", bindingNonce: nonce });
  });
});
