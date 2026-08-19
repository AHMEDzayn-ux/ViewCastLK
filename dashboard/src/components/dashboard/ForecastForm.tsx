"use client";

import { useEffect, useRef, useState } from "react";
import type {
  AudioLanguage,
  ChannelStats,
  ForecastFormValues,
  ForecastRequest,
  ForecastValidationErrors,
  PublishDay,
  YoutubeCategory,
} from "@/types/forecast";
import {
  AUDIO_LANGUAGES,
  PUBLISH_DAYS,
  YOUTUBE_CATEGORIES,
} from "@/types/forecast";
import {
  isChannelLookupMockMode,
  lookupChannelStats,
  PredictionApiError,
} from "@/lib/api/forecast";
import {
  hasValidationErrors,
  toForecastRequest,
  validateForecastForm,
} from "@/lib/validation";

function formatCount(value: number | null): string {
  if (value === null || value === undefined) return "Unavailable";
  if (value < 1000) return value.toLocaleString("en-US");
  if (value < 1_000_000) {
    const k = value / 1000;
    return (k % 1 === 0 ? k.toFixed(0) : k.toFixed(1)) + "K";
  }
  if (value < 1_000_000_000) {
    const m = value / 1_000_000;
    return (m % 1 === 0 ? m.toFixed(0) : m.toFixed(1)) + "M";
  }
  const b = value / 1_000_000_000;
  return (b % 1 === 0 ? b.toFixed(0) : b.toFixed(2)) + "B";
}

function formatChannelAge(createdAtIso: string | null): string {
  if (!createdAtIso) return "Unavailable";
  try {
    const created = new Date(createdAtIso);
    if (isNaN(created.getTime())) return "Unavailable";
    const diffMs = Date.now() - created.getTime();
    const days = Math.max(0, Math.floor(diffMs / (1000 * 60 * 60 * 24)));
    const years = Math.floor(days / 365.25);
    if (years >= 1) {
      return `${years} ${years === 1 ? "year" : "years"}`;
    }
    const months = Math.floor(days / 30.44);
    if (months >= 1) {
      return `${months} ${months === 1 ? "month" : "months"}`;
    }
    return `${days} ${days === 1 ? "day" : "days"}`;
  } catch {
    return "Unavailable";
  }
}


interface ForecastFormProps {
  onSubmit: (request: ForecastRequest) => void;
  onReset: () => void;
  isLoading: boolean;
}

const INITIAL_VALUES: ForecastFormValues = {
  title: "",
  category: "",
  durationMinutes: "",
  durationSeconds: "",
  audioLanguage: "",
  channelIdentifier: "",
  plannedPublishDay: "",
  plannedPublishHour: "",
};

const DRAFT_STORAGE_KEY = "viewcastlk.forecast-draft.v1";
const DRAFT_FIELDS = [
  "title",
  "category",
  "durationMinutes",
  "durationSeconds",
  "audioLanguage",
  "channelIdentifier",
  "plannedPublishDay",
  "plannedPublishHour",
] as const satisfies readonly (keyof ForecastFormValues)[];

function readForecastDraft(): ForecastFormValues | null {
  try {
    const serialized = window.sessionStorage.getItem(DRAFT_STORAGE_KEY);
    if (!serialized) return null;

    const candidate = JSON.parse(serialized) as unknown;
    if (!candidate || typeof candidate !== "object") return null;

    const values = candidate as Record<string, unknown>;
    if (!DRAFT_FIELDS.every((field) => typeof values[field] === "string")) {
      return null;
    }

    return candidate as ForecastFormValues;
  } catch {
    return null;
  }
}

function storeForecastDraft(values: ForecastFormValues) {
  try {
    window.sessionStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(values));
  } catch {
    return;
  }
}

function removeForecastDraft() {
  try {
    window.sessionStorage.removeItem(DRAFT_STORAGE_KEY);
  } catch {
    return;
  }
}

const ERROR_FOCUS_TARGETS: Record<
  keyof ForecastValidationErrors,
  string
> = {
  title: "title",
  category: "category",
  duration: "durationMinutes",
  audioLanguage: "audioLanguage",
  channelIdentifier: "channelIdentifier",
  plannedPublishDay: "plannedPublishDay",
  plannedPublishHour: "plannedPublishHour",
};

interface FieldProps {
  id: string;
  label: string;
  requirement: "required" | "optional";
  error?: string;
  hint?: string;
  children: React.ReactNode;
}

function Field({
  id,
  label,
  requirement,
  error,
  hint,
  children,
}: FieldProps) {
  return (
    <div className="field">
      <div className="field__heading">
        <label className="field__label" htmlFor={id}>
          {label}
        </label>
        <span className={"field__requirement field__requirement--" + requirement}>
          {requirement === "required" ? "Required" : "Optional"}
        </span>
      </div>
      {hint && (
        <p className="field__hint" id={id + "-hint"}>
          {hint}
        </p>
      )}
      {children}
      {error && (
        <p className="field__error" id={id + "-error"} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

function describedBy(id: string, hasHint: boolean, error?: string) {
  return [
    hasHint ? id + "-hint" : null,
    error ? id + "-error" : null,
  ]
    .filter(Boolean)
    .join(" ") || undefined;
}

export default function ForecastForm({
  onSubmit,
  onReset,
  isLoading,
}: ForecastFormProps) {
  const [values, setValues] = useState<ForecastFormValues>(INITIAL_VALUES);
  const [errors, setErrors] = useState<ForecastValidationErrors>({});
  const [channelStats, setChannelStats] = useState<ChannelStats | null>(null);
  const [isLookupLoading, setIsLookupLoading] = useState(false);
  const [lookupError, setLookupError] = useState<string | null>(null);
  const draftRestored = useRef(false);
  const draftChangedByUser = useRef(false);

  useEffect(() => {
    const savedDraft = readForecastDraft();
    const frameId = window.requestAnimationFrame(() => {
      draftRestored.current = true;
      if (savedDraft && !draftChangedByUser.current) {
        setValues(savedDraft);
      }
    });

    return () => window.cancelAnimationFrame(frameId);
  }, []);

  useEffect(() => {
    if (draftRestored.current) {
      storeForecastDraft(values);
    }
  }, [values]);

  function setValue<K extends keyof ForecastFormValues>(
    key: K,
    value: ForecastFormValues[K],
    errorKey: keyof ForecastValidationErrors,
  ) {
    draftChangedByUser.current = true;
    setValues((current) => ({ ...current, [key]: value }));
    setErrors((current) => {
      const next = { ...current };
      delete next[errorKey];
      return next;
    });
    if (key === "channelIdentifier") {
      setChannelStats(null);
      setLookupError(null);
    }
  }

  async function handleChannelLookup() {
    const identifier = values.channelIdentifier.trim();
    if (!identifier || isLookupLoading || isLoading) return;

    setIsLookupLoading(true);
    setLookupError(null);
    setChannelStats(null);

    try {
      const stats = await lookupChannelStats(identifier);
      setChannelStats(stats);
    } catch (error) {
      if (error instanceof PredictionApiError && error.message) {
        setLookupError(error.message);
      } else {
        setLookupError("Channel details could not be retrieved.");
      }
    } finally {
      setIsLookupLoading(false);
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isLoading) return;

    const nextErrors = validateForecastForm(values);
    if (hasValidationErrors(nextErrors)) {
      setErrors(nextErrors);
      const firstError = Object.keys(nextErrors)[0] as
        | keyof ForecastValidationErrors
        | undefined;

      if (firstError) {
        document.getElementById(ERROR_FOCUS_TARGETS[firstError])?.focus();
      }
      return;
    }

    setErrors({});
    onSubmit(toForecastRequest(values));
  }

  function handleReset() {
    draftChangedByUser.current = true;
    removeForecastDraft();
    setValues(INITIAL_VALUES);
    setErrors({});
    setChannelStats(null);
    setLookupError(null);
    setIsLookupLoading(false);
    onReset();
    requestAnimationFrame(() => document.getElementById("title")?.focus());
  }


  const inputClass = (error?: string) =>
    "field-control" + (error ? " field-control--invalid" : "");

  return (
    <form
      id="forecast-form"
      className="forecast-form"
      onSubmit={handleSubmit}
      noValidate
      aria-labelledby="forecast-form-title"
      aria-busy={isLoading}
    >
      <div className="form-section__header">
        <div>
          <p className="section-kicker">Forecast request</p>
          <h2 id="forecast-form-title">Tell us about the planned video</h2>
        </div>
        <p>Required details are marked clearly.</p>
      </div>

      <div className="form-grid">
        <Field
          id="title"
          label="Video title"
          requirement="required"
          error={errors.title}
          hint="Sinhala, Tamil, English, and mixed-script titles are supported."
        >
          <textarea
            id="title"
            name="title"
            rows={3}
            dir="auto"
            value={values.title}
            maxLength={200}
            disabled={isLoading}
            className={inputClass(errors.title)}
            placeholder="Enter the title you plan to publish"
            aria-invalid={Boolean(errors.title)}
            aria-describedby={describedBy("title", true, errors.title)}
            onChange={(event) =>
              setValue("title", event.target.value, "title")
            }
          />
        </Field>

        <Field
          id="category"
          label="Video category"
          requirement="required"
          error={errors.category}
          hint="Choose the category you plan to use when publishing on YouTube."
        >
          <select
            id="category"
            name="category"
            value={values.category}
            disabled={isLoading}
            className={inputClass(errors.category)}
            aria-invalid={Boolean(errors.category)}
            aria-describedby={describedBy("category", true, errors.category)}
            onChange={(event) =>
              setValue(
                "category",
                event.target.value as YoutubeCategory | "",
                "category",
              )
            }
          >
            <option value="">Choose a category</option>
            {YOUTUBE_CATEGORIES.map((category) => (
              <option value={category} key={category}>
                {category}
              </option>
            ))}
          </select>
        </Field>

        <Field
          id="durationMinutes"
          label="Planned duration"
          requirement="required"
          error={errors.duration}
          hint="Enter the expected finished length."
        >
          <div className="duration-fields">
            <label>
              <span>Minutes</span>
              <input
                id="durationMinutes"
                name="durationMinutes"
                type="number"
                inputMode="numeric"
                min="0"
                max="720"
                value={values.durationMinutes}
                disabled={isLoading}
                className={inputClass(errors.duration)}
                aria-invalid={Boolean(errors.duration)}
                aria-describedby={describedBy(
                  "durationMinutes",
                  true,
                  errors.duration,
                )}
                onChange={(event) =>
                  setValue(
                    "durationMinutes",
                    event.target.value,
                    "duration",
                  )
                }
              />
            </label>
            <label>
              <span>Seconds</span>
              <input
                id="durationSeconds"
                name="durationSeconds"
                type="number"
                inputMode="numeric"
                min="0"
                max="59"
                value={values.durationSeconds}
                disabled={isLoading}
                className={inputClass(errors.duration)}
                aria-invalid={Boolean(errors.duration)}
                aria-describedby={[
                  "durationMinutes-hint",
                  errors.duration ? "durationMinutes-error" : null,
                ]
                  .filter(Boolean)
                  .join(" ")}
                onChange={(event) =>
                  setValue(
                    "durationSeconds",
                    event.target.value,
                    "duration",
                  )
                }
              />
            </label>
          </div>
        </Field>

        <Field
          id="audioLanguage"
          label="Audio language"
          requirement="required"
          error={errors.audioLanguage}
          hint="Choose the main spoken or sung language; use Mixed / multilingual when several are used."
        >
          <select
            id="audioLanguage"
            name="audioLanguage"
            value={values.audioLanguage}
            disabled={isLoading}
            className={inputClass(errors.audioLanguage)}
            aria-invalid={Boolean(errors.audioLanguage)}
            aria-describedby={describedBy(
              "audioLanguage",
              true,
              errors.audioLanguage,
            )}
            onChange={(event) =>
              setValue(
                "audioLanguage",
                event.target.value as AudioLanguage | "",
                "audioLanguage",
              )
            }
          >
            <option value="">Choose the main audio language</option>
            {AUDIO_LANGUAGES.map((language) => (
              <option value={language} key={language}>
                {language}
              </option>
            ))}
          </select>
        </Field>


        <Field
          id="channelIdentifier"
          label="YouTube channel"
          requirement="required"
          error={errors.channelIdentifier}
          hint="Use a channel URL, @handle, or channel ID. Channel statistics are retrieved automatically."
        >
          <div className="channel-lookup-control">
            <input
              id="channelIdentifier"
              name="channelIdentifier"
              type="text"
              value={values.channelIdentifier}
              disabled={isLoading || isLookupLoading}
              className={inputClass(errors.channelIdentifier)}
              placeholder="https://youtube.com/@yourchannel"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck="false"
              aria-invalid={Boolean(errors.channelIdentifier)}
              aria-describedby={describedBy(
                "channelIdentifier",
                true,
                errors.channelIdentifier,
              )}
              onChange={(event) =>
                setValue(
                  "channelIdentifier",
                  event.target.value,
                  "channelIdentifier",
                )
              }
            />
            <button
              type="button"
              className="secondary-button channel-lookup-button"
              disabled={
                isLoading ||
                isLookupLoading ||
                !values.channelIdentifier.trim()
              }
              onClick={handleChannelLookup}
            >
              {isLookupLoading ? "Retrieving…" : "Retrieve details"}
            </button>
          </div>

          {isLookupLoading && (
            <p className="channel-details__state" role="status">
              Retrieving channel details…
            </p>
          )}

          {lookupError && (
            <p
              className="channel-details__state channel-details__state--error"
              role="alert"
            >
              {lookupError}
            </p>
          )}

          {channelStats && (
            <div className="channel-details">
              <div className="channel-details__header">
                <div>
                  <p className="section-kicker">Channel details</p>
                  <h3>Retrieved automatically</h3>
                </div>
                {isChannelLookupMockMode() && (
                  <p>Illustrative development data</p>
                )}
              </div>

              <dl className="channel-details__grid">
                <div>
                  <dt>Subscribers</dt>
                  <dd>{formatCount(channelStats.subscriberCount)}</dd>
                </div>
                <div>
                  <dt>Total views</dt>
                  <dd>{formatCount(channelStats.totalViewCount)}</dd>
                </div>
                <div>
                  <dt>Videos</dt>
                  <dd>{formatCount(channelStats.videoCount)}</dd>
                </div>
                <div>
                  <dt>Channel age</dt>
                  <dd>{formatChannelAge(channelStats.createdAt)}</dd>
                </div>
              </dl>
            </div>
          )}
        </Field>
      </div>

      <section className="optional-section" aria-labelledby="timing-title">
        <div>
          <p className="section-kicker">Optional</p>
          <h3 id="timing-title">Publishing plan</h3>
          <p>
            Leave either field blank if the publishing schedule is not decided.
          </p>
        </div>

        <div className="form-grid form-grid--timing">
          <Field
            id="plannedPublishDay"
            label="Publishing day"
            requirement="optional"
            error={errors.plannedPublishDay}
            hint="Choose a planned day only if you have decided one."
          >
            <select
              id="plannedPublishDay"
              name="plannedPublishDay"
              value={values.plannedPublishDay}
              disabled={isLoading}
              className={inputClass(errors.plannedPublishDay)}
              aria-invalid={Boolean(errors.plannedPublishDay)}
              aria-describedby={describedBy(
                "plannedPublishDay",
                true,
                errors.plannedPublishDay,
              )}
              onChange={(event) =>
                setValue(
                  "plannedPublishDay",
                  event.target.value as PublishDay | "",
                  "plannedPublishDay",
                )
              }
            >
              <option value="">Not decided</option>
              {PUBLISH_DAYS.map((day) => (
                <option value={day} key={day}>
                  {day}
                </option>
              ))}
            </select>
          </Field>

          <Field
            id="plannedPublishHour"
            label="Publishing hour (SLT)"
            requirement="optional"
            error={errors.plannedPublishHour}
            hint="Choose the planned Sri Lanka time only if it is decided."
          >
            <select
              id="plannedPublishHour"
              name="plannedPublishHour"
              value={values.plannedPublishHour}
              disabled={isLoading}
              className={inputClass(errors.plannedPublishHour)}
              aria-invalid={Boolean(errors.plannedPublishHour)}
              aria-describedby={describedBy(
                "plannedPublishHour",
                true,
                errors.plannedPublishHour,
              )}
              onChange={(event) =>
                setValue(
                  "plannedPublishHour",
                  event.target.value,
                  "plannedPublishHour",
                )
              }
            >
              <option value="">Not decided</option>
              {Array.from({ length: 24 }, (_, hour) => (
                <option value={String(hour)} key={hour}>
                  {String(hour).padStart(2, "0")}:00
                </option>
              ))}
            </select>
          </Field>
        </div>
      </section>

      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={isLoading}>
          {isLoading ? "Generating forecast…" : "Generate forecast"}
        </button>
        <button
          className="secondary-button"
          type="button"
          disabled={isLoading}
          onClick={handleReset}
        >
          Clear form
        </button>
      </div>

      {hasValidationErrors(errors) && (
        <p className="sr-only" role="alert" aria-live="assertive">
          Please correct the highlighted fields before generating a forecast.
        </p>
      )}
    </form>
  );
}
