-- Additive, server-only flow binding. Existing unbound states fail closed.
-- No policy/grant or existing connection/Analytics changes.
alter table creator.oauth_states
    add column session_id uuid,
    add column binding_hash text
        check (binding_hash is null or binding_hash ~ '^[0-9a-f]{64}$');

comment on column creator.oauth_states.session_id is
    'Validated Supabase initiating session. Completion also requires an active auth.sessions row.';
comment on column creator.oauth_states.binding_hash is
    'SHA-256 of an independent per-flow browser nonce. Never persist the plaintext nonce.';
