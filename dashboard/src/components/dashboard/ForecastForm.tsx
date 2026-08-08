"use client";

import { useState } from "react";
import type {
  AudioLanguage,
  ForecastFormValues,
  ForecastRequest,
  ForecastValidationErrors,
  MadeForKidsSelection,
  PublishDay,
  YoutubeCategory,
} from "@/types/forecast";
import {
  AUDIO_LANGUAGES,
  PUBLISH_DAYS,
  YOUTUBE_CATEGORIES,
} from "@/types/forecast";
import {
  hasValidationErrors,
  toForecastRequest,
  validateForecastForm,
} from "@/lib/validation";

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
  madeForKids: "",
  channelIdentifier: "",
  plannedPublishDay: "",
  plannedPublishHour: "",
};

const ERROR_FOCUS_TARGETS: Record<
  keyof ForecastValidationErrors,
  string
> = {
  title: "title",
  category: "category",
  duration: "durationMinutes",
  audioLanguage: "audioLanguage",
  madeForKids: "madeForKids-yes",
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

  function setValue<K extends keyof ForecastFormValues>(
    key: K,
    value: ForecastFormValues[K],
    errorKey: keyof ForecastValidationErrors,
  ) {
    setValues((current) => ({ ...current, [key]: value }));
    setErrors((current) => {
      const next = { ...current };
      delete next[errorKey];
      return next;
    });
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
    setValues(INITIAL_VALUES);
    setErrors({});
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
        >
          <select
            id="category"
            name="category"
            value={values.category}
            disabled={isLoading}
            className={inputClass(errors.category)}
            aria-invalid={Boolean(errors.category)}
            aria-describedby={describedBy("category", false, errors.category)}
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
                aria-describedby={
                  errors.duration ? "durationMinutes-error" : undefined
                }
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
              false,
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

        <fieldset
          className="field"
          aria-describedby={
            errors.madeForKids ? "madeForKids-error" : undefined
          }
        >
          <div className="field__heading">
            <legend className="field__label">Made for kids</legend>
            <span className="field__requirement field__requirement--required">
              Required
            </span>
          </div>
          <div className="choice-group">
            {[
              { value: "yes", label: "Yes" },
              { value: "no", label: "No" },
            ].map((option) => (
              <label className="choice-control" key={option.value}>
                <input
                  id={"madeForKids-" + option.value}
                  type="radio"
                  name="madeForKids"
                  value={option.value}
                  checked={values.madeForKids === option.value}
                  disabled={isLoading}
                  onChange={(event) =>
                    setValue(
                      "madeForKids",
                      event.target.value as MadeForKidsSelection,
                      "madeForKids",
                    )
                  }
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
          {errors.madeForKids && (
            <p className="field__error" id="madeForKids-error" role="alert">
              {errors.madeForKids}
            </p>
          )}
        </fieldset>

        <Field
          id="channelIdentifier"
          label="YouTube channel"
          requirement="required"
          error={errors.channelIdentifier}
          hint="Use a channel URL, @handle, or channel ID. Channel statistics are retrieved automatically."
        >
          <input
            id="channelIdentifier"
            name="channelIdentifier"
            type="text"
            value={values.channelIdentifier}
            disabled={isLoading}
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
                false,
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
                false,
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
