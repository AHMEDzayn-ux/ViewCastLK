"use client";

import { useState } from "react";
import type { ForecastInput, YoutubeCategory, PublishDay, ValidationErrors } from "@/types/forecast";
import { YOUTUBE_CATEGORIES, PUBLISH_DAYS } from "@/types/forecast";

interface ForecastFormProps {
  onSubmit: (input: ForecastInput) => void;
  isLoading: boolean;
}


const INITIAL_INPUT: ForecastInput = {
  title: "",
  category: "",
  durationMinutes: 0,
  durationSeconds: 0,
  publishDay: "",
  publishHour: 18,
  publishMinute: 0,
  channelHandle: "",
};

interface FieldProps {
  id: string;
  label: string;
  error?: string;
  required?: boolean;
  children: React.ReactNode;
  hint?: string;
}

function Field({ id, label, error, required, children, hint }: FieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium text-slate-300">
        {label}
        {required && <span className="text-red-400 ml-1" aria-hidden="true">*</span>}
      </label>
      {hint && <p className="text-xs text-slate-500">{hint}</p>}
      {children}
      {error && (
        <p id={`${id}-error`} role="alert" className="text-xs text-red-400 mt-0.5">
          {error}
        </p>
      )}
    </div>
  );
}

const inputClass =
  "w-full rounded-lg bg-slate-800 border border-slate-700 text-slate-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-slate-500 disabled:opacity-50";

const errorInputClass =
  "w-full rounded-lg bg-slate-800 border border-red-500 text-slate-100 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent placeholder:text-slate-500";

export default function ForecastForm({ onSubmit, isLoading }: ForecastFormProps) {
  const [input, setInput] = useState<ForecastInput>(INITIAL_INPUT);
  const [errors, setErrors] = useState<ValidationErrors>({});

  function set<K extends keyof ForecastInput>(key: K, value: ForecastInput[K]) {
    setInput((prev) => ({ ...prev, [key]: value }));
    // Clear error for this field when the user edits it
    setErrors((prev) => {
      const next = { ...prev };
      delete next[key as keyof ValidationErrors];
      return next;
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    // Import validation lazily to avoid a circular dep warning in some setups
    const { validateForecastInput, hasErrors } = await import("@/lib/validation");
    const errs = validateForecastInput(input);

    if (hasErrors(errs)) {
      setErrors(errs);
      // Focus first error field
      const firstKey = Object.keys(errs)[0];
      document.getElementById(firstKey)?.focus();
      return;
    }

    setErrors({});
    onSubmit(input);
  }

  function handleReset() {
    setInput(INITIAL_INPUT);
    setErrors({});
  }

  return (
    <form
      onSubmit={handleSubmit}
      noValidate
      aria-label="Video forecast form"
      className="bg-slate-800/50 rounded-2xl border border-slate-700 p-5 sm:p-6 space-y-5"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-slate-100 font-semibold text-base">Video metadata</h2>
        <span className="text-xs text-slate-500">
          <span className="text-red-400">*</span> required
        </span>
      </div>

      {/* Title */}
      <Field id="title" label="Planned video title" error={errors.title} required>
        <input
          id="title"
          type="text"
          value={input.title}
          onChange={(e) => set("title", e.target.value)}
          placeholder="My upcoming video"
          maxLength={200}
          disabled={isLoading}
          className={errors.title ? errorInputClass : inputClass}
          aria-describedby={errors.title ? "title-error" : undefined}
          aria-invalid={!!errors.title}
        />
      </Field>

      {/* Category */}
      <Field id="category" label="YouTube category" error={errors.category} required>
        <select
          id="category"
          value={input.category}
          onChange={(e) => set("category", e.target.value as YoutubeCategory | "")}
          disabled={isLoading}
          className={errors.category ? errorInputClass : inputClass}
          aria-describedby={errors.category ? "category-error" : undefined}
          aria-invalid={!!errors.category}
        >
          <option value="">— Select a category —</option>
          {YOUTUBE_CATEGORIES.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </Field>

      {/* Duration */}
      <Field
        id="durationMinutes"
        label="Planned duration"
        error={errors.duration}
        required
        hint="Enter the total planned length of your video."
      >
        <div className="flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              id="durationMinutes"
              type="number"
              min={0}
              max={999}
              value={input.durationMinutes === 0 ? "" : input.durationMinutes}
              onChange={(e) =>
                set("durationMinutes", Math.max(0, parseInt(e.target.value) || 0))
              }
              placeholder="0"
              disabled={isLoading}
              className={errors.duration ? errorInputClass : inputClass}
              aria-label="Duration minutes"
              aria-describedby={errors.duration ? "durationMinutes-error" : undefined}
              aria-invalid={!!errors.duration}
            />
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">
              min
            </span>
          </div>
          <span className="text-slate-500 font-bold">:</span>
          <div className="w-24 relative">
            <input
              id="durationSeconds"
              type="number"
              min={0}
              max={59}
              value={input.durationSeconds === 0 ? "" : input.durationSeconds}
              onChange={(e) =>
                set(
                  "durationSeconds",
                  Math.min(59, Math.max(0, parseInt(e.target.value) || 0))
                )
              }
              placeholder="0"
              disabled={isLoading}
              className={errors.duration ? errorInputClass : inputClass}
              aria-label="Duration seconds"
              aria-invalid={!!errors.duration}
            />
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">
              sec
            </span>
          </div>
        </div>
      </Field>

      {/* Publish day + time row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Field id="publishDay" label="Planned publish day" error={errors.publishDay} required>
          <select
            id="publishDay"
            value={input.publishDay}
            onChange={(e) => set("publishDay", e.target.value as PublishDay | "")}
            disabled={isLoading}
            className={errors.publishDay ? errorInputClass : inputClass}
            aria-describedby={errors.publishDay ? "publishDay-error" : undefined}
            aria-invalid={!!errors.publishDay}
          >
            <option value="">— Select day —</option>
            {PUBLISH_DAYS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>

        <Field id="publishTime" label="Planned publish time — Sri Lanka time (SLT)">
          <input
            id="publishTime"
            type="time"
            defaultValue={`${String(INITIAL_INPUT.publishHour).padStart(2, "0")}:${String(INITIAL_INPUT.publishMinute).padStart(2, "0")}`}
            onChange={(e) => {
              const [hStr, mStr] = e.target.value.split(":");
              set("publishHour", parseInt(hStr) || 0);
              set("publishMinute", parseInt(mStr) || 0);
            }}
            disabled={isLoading}
            className={inputClass}
            aria-label="Planned publish time in Sri Lanka Time"
          />
        </Field>
      </div>

      {/* Channel handle */}
      <Field
        id="channelHandle"
        label="Channel handle or channel ID"
        error={errors.channelHandle}
        hint="Optional. e.g. @mychannel or UCxxxxxxxxxxxxxxxxxx"
      >
        <input
          id="channelHandle"
          type="text"
          value={input.channelHandle}
          onChange={(e) => set("channelHandle", e.target.value)}
          placeholder="@mychannel"
          disabled={isLoading}
          className={errors.channelHandle ? errorInputClass : inputClass}
          aria-describedby={errors.channelHandle ? "channelHandle-error" : undefined}
          aria-invalid={!!errors.channelHandle}
        />
      </Field>

      {/* Actions */}
      <div className="flex gap-3 pt-1">
        <button
          type="submit"
          disabled={isLoading}
          className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-slate-900 text-white font-semibold text-sm disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <svg
                className="w-4 h-4 animate-spin"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 100 16v-4l-3 3 3 3v-4a8 8 0 01-8-8z"
                />
              </svg>
              Generating forecast…
            </>
          ) : (
            "Generate forecast"
          )}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={isLoading}
          className="px-4 py-2.5 rounded-lg bg-slate-700 hover:bg-slate-600 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-900 text-slate-300 text-sm font-medium disabled:opacity-50 transition-colors"
        >
          Reset
        </button>
      </div>

      {/* Validation summary for screen readers */}
      {Object.keys(errors).length > 0 && (
        <div role="alert" className="sr-only">
          {Object.values(errors).join(" ")}
        </div>
      )}
    </form>
  );
}
