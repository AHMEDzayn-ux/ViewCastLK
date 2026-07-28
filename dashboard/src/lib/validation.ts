import type { ForecastInput, ValidationErrors } from "@/types/forecast";
import { YOUTUBE_CATEGORIES, PUBLISH_DAYS } from "@/types/forecast";

/**
 * Validates a ForecastInput and returns a map of field-level error messages.
 * Returns an empty object if the input is valid.
 */
export function validateForecastInput(
  input: ForecastInput
): ValidationErrors {
  const errors: ValidationErrors = {};

  // Title
  if (!input.title.trim()) {
    errors.title = "Please enter a planned video title.";
  } else if (input.title.trim().length > 200) {
    errors.title = "Title must be 200 characters or fewer.";
  }

  // Category
  if (!input.category) {
    errors.category = "Please select a YouTube category.";
  } else if (!(YOUTUBE_CATEGORIES as readonly string[]).includes(input.category)) {
    errors.category = "Please select a valid YouTube category.";
  }

  // Duration
  const totalSeconds =
    input.durationMinutes * 60 + input.durationSeconds;
  if (
    isNaN(input.durationMinutes) ||
    isNaN(input.durationSeconds) ||
    input.durationMinutes < 0 ||
    input.durationSeconds < 0 ||
    input.durationSeconds > 59
  ) {
    errors.duration = "Enter a valid duration (minutes 0–999, seconds 0–59).";
  } else if (totalSeconds === 0) {
    errors.duration = "Duration must be greater than zero.";
  } else if (totalSeconds > 43200) {
    // 12 hours max
    errors.duration = "Duration must be 12 hours or less.";
  }

  // Publish day
  if (!input.publishDay) {
    errors.publishDay = "Please select a planned publish day.";
  } else if (!(PUBLISH_DAYS as readonly string[]).includes(input.publishDay)) {
    errors.publishDay = "Please select a valid day of the week.";
  }

  // Channel handle — optional but must not contain spaces if provided
  if (input.channelHandle.trim() && /\s/.test(input.channelHandle.trim())) {
    errors.channelHandle =
      "Channel handle should not contain spaces (e.g. @mychannel or UCxxxxx).";
  }

  return errors;
}

export function hasErrors(errors: ValidationErrors): boolean {
  return Object.keys(errors).length > 0;
}
