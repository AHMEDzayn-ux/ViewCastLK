-- User-owned completed forecasts. The table is independent of the legacy
-- collection warehouse so it can be applied to the hosted Auth project alone.
CREATE TABLE public.forecast_history (
    id                       uuid             PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                  uuid             NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at               timestamptz      NOT NULL DEFAULT now(),

    forecast_id              text             NOT NULL,
    title                    text             NOT NULL,
    category                 text             NOT NULL,
    duration_seconds         double precision NOT NULL,
    audio_language           text             NOT NULL,
    channel_identifier       text             NOT NULL,
    planned_publish_day      text,
    planned_publish_hour     smallint,

    day_7_cumulative_views   bigint           NOT NULL,
    day_14_cumulative_views  bigint           NOT NULL,
    day_21_cumulative_views  bigint           NOT NULL,
    day_30_cumulative_views  bigint           NOT NULL,

    model_version            text             NOT NULL,
    forecast_generated_at    timestamptz      NOT NULL,
    model_data_source        text             NOT NULL,
    completeness_status      text             NOT NULL,
    completeness_issues      jsonb            NOT NULL DEFAULT '[]'::jsonb,

    CONSTRAINT forecast_history_forecast_id_not_blank
        CHECK (length(btrim(forecast_id)) > 0),
    CONSTRAINT forecast_history_title_not_blank
        CHECK (length(btrim(title)) > 0),
    CONSTRAINT forecast_history_category_not_blank
        CHECK (length(btrim(category)) > 0),
    CONSTRAINT forecast_history_duration_positive
        CHECK (duration_seconds > 0),
    CONSTRAINT forecast_history_audio_language_not_blank
        CHECK (length(btrim(audio_language)) > 0),
    CONSTRAINT forecast_history_channel_identifier_not_blank
        CHECK (length(btrim(channel_identifier)) > 0),
    CONSTRAINT forecast_history_publish_day_valid
        CHECK (
            planned_publish_day IS NULL OR
            planned_publish_day IN (
                'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday'
            )
        ),
    CONSTRAINT forecast_history_publish_hour_valid
        CHECK (
            planned_publish_hour IS NULL OR
            planned_publish_hour BETWEEN 0 AND 23
        ),
    CONSTRAINT forecast_history_day_7_nonnegative
        CHECK (day_7_cumulative_views >= 0),
    CONSTRAINT forecast_history_day_14_nonnegative
        CHECK (day_14_cumulative_views >= 0),
    CONSTRAINT forecast_history_day_21_nonnegative
        CHECK (day_21_cumulative_views >= 0),
    CONSTRAINT forecast_history_day_30_nonnegative
        CHECK (day_30_cumulative_views >= 0),
    CONSTRAINT forecast_history_model_version_not_blank
        CHECK (length(btrim(model_version)) > 0),
    CONSTRAINT forecast_history_model_data_source_valid
        CHECK (model_data_source IN ('prediction_api', 'mock')),
    CONSTRAINT forecast_history_completeness_status_valid
        CHECK (completeness_status IN ('complete', 'degraded')),
    CONSTRAINT forecast_history_completeness_issues_array
        CHECK (jsonb_typeof(completeness_issues) = 'array')
);

COMMENT ON TABLE public.forecast_history IS
    'User-owned completed forecast records. Rows are immutable except for owner-initiated deletion; UPDATE is intentionally unsupported.';
COMMENT ON COLUMN public.forecast_history.user_id IS
    'Owning Supabase Auth user. History is removed automatically when the auth.users row is deleted.';
COMMENT ON COLUMN public.forecast_history.completeness_issues IS
    'Variable-length, non-secret completeness issue objects returned with the forecast.';

CREATE INDEX forecast_history_user_created_at_idx
    ON public.forecast_history (user_id, created_at DESC);

-- Enable RLS explicitly so this migration enforces the authorization boundary
-- even when the hosted automatic RLS event trigger is absent or changes.
ALTER TABLE public.forecast_history ENABLE ROW LEVEL SECURITY;

-- Supabase projects may automatically grant new public-schema tables. Remove
-- those defaults, then expose only the operations the signed-in UI requires.
REVOKE ALL ON TABLE public.forecast_history FROM PUBLIC;
REVOKE ALL ON TABLE public.forecast_history FROM anon;
REVOKE ALL ON TABLE public.forecast_history FROM authenticated;
GRANT SELECT, INSERT, DELETE ON TABLE public.forecast_history TO authenticated;

CREATE POLICY "Users can read their own forecast history"
    ON public.forecast_history
    FOR SELECT
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

CREATE POLICY "Users can insert their own forecast history"
    ON public.forecast_history
    FOR INSERT
    TO authenticated
    WITH CHECK ((SELECT auth.uid()) = user_id);

CREATE POLICY "Users can delete their own forecast history"
    ON public.forecast_history
    FOR DELETE
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

-- There is deliberately no UPDATE grant or policy: completed forecasts are
-- historical records, not mutable drafts.
