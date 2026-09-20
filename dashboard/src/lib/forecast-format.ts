/**
 * How a forecast number is written for the creator.
 *
 * The model works on a log scale and is typically out by a factor of about
 * two, so at the bottom of the range the exact integer is meaningless: a
 * forecast of 0 and one of 9 are the same claim. Showing a bare "0" also tells
 * a creator their video will get no views at all, which the model cannot know
 * and which is what a connected small channel saw on 20 September 2026.
 */
export const LOW_FORECAST_THRESHOLD = 10;

export function formatForecastViews(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "unavailable";
  if (value < LOW_FORECAST_THRESHOLD) return `fewer than ${LOW_FORECAST_THRESHOLD}`;
  return new Intl.NumberFormat("en-LK").format(Math.round(value));
}
