import type {
  ForecastFormValues,
  ForecastRequest,
  ForecastValidationErrors,
} from "@/types/forecast";
import {
  AUDIO_LANGUAGES,
  PUBLISH_DAYS,
  YOUTUBE_CATEGORIES,
} from "@/types/forecast";

export function validateForecastForm(
  values: ForecastFormValues,
): ForecastValidationErrors {
  const errors: ForecastValidationErrors = {};
  const minutes = Number(values.durationMinutes);
  const seconds = Number(values.durationSeconds || "0");

  if (!values.title.trim()) {
    errors.title = "Enter the planned video title.";
  } else if (values.title.trim().length > 200) {
    errors.title = "Keep the title to 200 characters or fewer.";
  }

  if (!values.category) {
    errors.category = "Choose a video category.";
  } else if (
    !(YOUTUBE_CATEGORIES as readonly string[]).includes(values.category)
  ) {
    errors.category = "Choose a valid video category.";
  }

  if (
    values.durationMinutes.trim() === "" ||
    !Number.isInteger(minutes) ||
    !Number.isInteger(seconds) ||
    minutes < 0 ||
    seconds < 0 ||
    seconds > 59
  ) {
    errors.duration = "Enter whole minutes and seconds from 0 to 59.";
  } else {
    const totalSeconds = minutes * 60 + seconds;

    if (totalSeconds === 0) {
      errors.duration = "Planned duration must be longer than zero.";
    } else if (totalSeconds > 43_200) {
      errors.duration = "Planned duration must be 12 hours or less.";
    }
  }

  if (!values.audioLanguage) {
    errors.audioLanguage = "Choose the main audio language.";
  } else if (
    !(AUDIO_LANGUAGES as readonly string[]).includes(values.audioLanguage)
  ) {
    errors.audioLanguage = "Choose a valid audio language.";
  }

  const channelIdentifier = values.channelIdentifier.trim();
  if (!channelIdentifier) {
    errors.channelIdentifier =
      "Enter the YouTube channel URL, @handle, or channel ID.";
  } else if (/\s/.test(channelIdentifier)) {
    errors.channelIdentifier =
      "The channel URL or identifier should not contain spaces.";
  }

  if (
    values.plannedPublishDay &&
    !(PUBLISH_DAYS as readonly string[]).includes(
      values.plannedPublishDay,
    )
  ) {
    errors.plannedPublishDay = "Choose a valid publishing day.";
  }

  if (values.plannedPublishHour !== "") {
    const hour = Number(values.plannedPublishHour);
    if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
      errors.plannedPublishHour = "Choose an hour from 00:00 to 23:00.";
    }
  }

  return errors;
}

export function hasValidationErrors(
  errors: ForecastValidationErrors,
): boolean {
  return Object.keys(errors).length > 0;
}

export function toForecastRequest(
  values: ForecastFormValues,
): ForecastRequest {
  return {
    title: values.title.trim(),
    category: values.category as ForecastRequest["category"],
    durationSeconds:
      Number(values.durationMinutes) * 60 +
      Number(values.durationSeconds || "0"),
    audioLanguage:
      values.audioLanguage as ForecastRequest["audioLanguage"],
    channelIdentifier: values.channelIdentifier.trim(),
    plannedPublishDay: values.plannedPublishDay || null,
    plannedPublishHour:
      values.plannedPublishHour === ""
        ? null
        : Number(values.plannedPublishHour),
  };
}
