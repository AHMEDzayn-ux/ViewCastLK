create schema if not exists creator;

revoke all on schema creator from public, anon, authenticated, service_role;

create table creator.youtube_connections (
  user_id uuid primary key references auth.users(id) on delete cascade,
  channel_id text not null,
  channel_title text,
  encrypted_refresh_token text not null,
  scopes text[] not null,
  connected_at timestamptz not null default now(),
  last_refresh_ok_at timestamptz,
  status text not null default 'pending_sync'
    check (status in ('pending_sync', 'active', 'reauth_required', 'error'))
);

create table creator.video_history (
  user_id uuid not null references auth.users(id) on delete cascade,
  video_id text not null,
  title text,
  category text,
  duration_seconds integer,
  published_at timestamptz not null,
  is_short boolean not null,
  d7 bigint check (d7 is null or d7 >= 0),
  d14 bigint check (d14 is null or d14 >= 0),
  d21 bigint check (d21 is null or d21 >= 0),
  d30 bigint check (d30 is null or d30 >= 0),
  pred7 bigint check (pred7 is null or pred7 >= 0),
  pred14 bigint check (pred14 is null or pred14 >= 0),
  pred21 bigint check (pred21 is null or pred21 >= 0),
  pred30 bigint check (pred30 is null or pred30 >= 0),
  model_version text,
  updated_at timestamptz not null default now(),
  primary key (user_id, video_id)
);

create index video_history_user_published_at_idx
  on creator.video_history (user_id, published_at desc);

create table creator.adjustments (
  user_id uuid not null references auth.users(id) on delete cascade,
  horizon smallint not null check (horizon in (7, 14, 21, 30)),
  format text not null check (format in ('all', 'short', 'long')),
  factor double precision not null check (factor > 0),
  n_videos integer not null check (n_videos >= 0),
  model_version text not null,
  updated_at timestamptz not null default now(),
  primary key (user_id, horizon, format, model_version)
);

create table creator.insights (
  user_id uuid primary key references auth.users(id) on delete cascade,
  payload jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table creator.oauth_states (
  state_hash text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  expires_at timestamptz not null,
  consumed_at timestamptz,
  created_at timestamptz not null default now()
);

create index oauth_states_expires_at_idx
  on creator.oauth_states (expires_at);

alter table creator.youtube_connections enable row level security;
alter table creator.video_history enable row level security;
alter table creator.adjustments enable row level security;
alter table creator.insights enable row level security;
alter table creator.oauth_states enable row level security;

revoke all on all tables in schema creator from public, anon, authenticated, service_role;
alter default privileges in schema creator
  revoke all on tables from public, anon, authenticated, service_role;

comment on schema creator is
  'Server-only creator connection and private Analytics data; not exposed through the browser Data API.';
comment on column creator.youtube_connections.encrypted_refresh_token is
  'AES-256-GCM ciphertext. Never return this value to a browser client.';
comment on table creator.video_history is
  'Private creator Analytics. This table must never feed shared-model training.';
