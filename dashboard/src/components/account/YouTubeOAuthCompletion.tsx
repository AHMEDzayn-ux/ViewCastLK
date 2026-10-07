"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { completeYouTubeConnection } from "@/lib/api/youtube-connection";

export default function YouTubeOAuthCompletion() {
  const router = useRouter();
  const attempt = useRef<Promise<boolean> | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    // Reuse the one attempt across React StrictMode effect replay: a consumed
    // state must never be submitted twice by this component.
    attempt.current ??= completeYouTubeConnection();
    attempt.current.then(
      (connected) => {
        if (active) router.replace(`/account?youtube=${connected ? "connected" : "not_connected"}`);
      },
      () => { if (active) setFailed(true); },
    );
    return () => { active = false; };
  }, [router]);

  return (
    <main className="page-shell account-page">
      <h1>{failed ? "Connection could not be verified" : "Verifying your YouTube connection"}</h1>
      {failed ? (
        <>
          <p role="alert">Start again from your account in the browser where you signed in.</p>
          <Link href="/account">Return to your account</Link>
        </>
      ) : <p role="status">Please wait while we securely complete your connection.</p>}
    </main>
  );
}
