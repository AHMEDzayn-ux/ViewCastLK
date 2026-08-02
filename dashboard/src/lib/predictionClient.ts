/**
 * predictionClient.ts
 *
 * The single seam between the UI and the prediction back-end.
 *
 * Phase 1: returns mock data from mockPredictions.ts.
 * Phase 2: replace the body of getPrediction() with a fetch() call to the
 *          real prediction API endpoint — no other file needs to change.
 */

import type { ForecastInput, ForecastResult } from "@/types/forecast";
import { generateMockPrediction } from "@/lib/mockPredictions";

/**
 * Returns a view-count forecast for the given video metadata.
 * In Phase 1 this is entirely client-side mock data.
 */
export async function getPrediction(
  input: ForecastInput
): Promise<ForecastResult> {
  // Simulate a short network round-trip so loading states are exercisable.
  await new Promise((resolve) => setTimeout(resolve, 900));

  return generateMockPrediction(input);
}
