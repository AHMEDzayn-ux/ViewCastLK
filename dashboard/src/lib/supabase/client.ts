import "client-only";

import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
const supabasePublishableKey =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY?.trim();

if (!supabaseUrl || !supabasePublishableKey) {
  const missingVariables = [
    !supabaseUrl && "NEXT_PUBLIC_SUPABASE_URL",
    !supabasePublishableKey && "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
  ].filter(Boolean);

  throw new Error(
    `Missing required public Supabase environment variable${
      missingVariables.length === 1 ? "" : "s"
    }: ${missingVariables.join(", ")}`,
  );
}

export const supabase = createClient(supabaseUrl, supabasePublishableKey);
