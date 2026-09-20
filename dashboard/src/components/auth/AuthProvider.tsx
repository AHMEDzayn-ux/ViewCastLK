"use client";

import { useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { User } from "@supabase/supabase-js";

import { supabase } from "@/lib/supabase/client";

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isSigningOut: boolean;
  signOut: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export default function AuthProvider({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const signOutInFlight = useRef(false);

  useEffect(() => {
    let isActive = true;

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (!isActive) return;

      // This callback stays synchronous: Supabase state is copied into UI state
      // without starting another auth or database operation from inside it.
      switch (event) {
        case "INITIAL_SESSION":
        case "SIGNED_IN":
        case "TOKEN_REFRESHED":
        case "USER_UPDATED":
          setUser(session?.user ?? null);
          setIsLoading(false);
          break;
        case "SIGNED_OUT":
          setUser(null);
          setIsLoading(false);
          break;
      }
    });

    return () => {
      isActive = false;
      subscription.unsubscribe();
    };
  }, []);

  const signOut = useCallback(async () => {
    if (signOutInFlight.current) return false;

    signOutInFlight.current = true;
    setIsSigningOut(true);

    try {
      const { error } = await supabase.auth.signOut({ scope: "local" });
      if (error) return false;

      router.replace("/login");
      return true;
    } catch {
      return false;
    } finally {
      signOutInFlight.current = false;
      setIsSigningOut(false);
    }
  }, [router]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      isLoading,
      isSigningOut,
      signOut,
    }),
    [isLoading, isSigningOut, signOut, user],
  );

  // Client-side auth state controls presentation only. Forecast-history access
  // must be enforced by Supabase RLS and backend authenticated-user/JWT checks.
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used within AuthProvider.");
  }

  return context;
}
