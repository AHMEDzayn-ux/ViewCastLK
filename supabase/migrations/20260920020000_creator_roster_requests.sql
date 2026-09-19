-- Durable, server-only handoff from creator OAuth to the public collector.
-- This table contains only public channel IDs; no Auth user identifiers,
-- OAuth credentials, private Analytics, or personalization values belong here.

create table if not exists public.roster_requests (
  channel_id text primary key,
  source text not null default 'creator_connection'
    check (source = 'creator_connection'),
  status text not null default 'pending'
    check (status in ('pending', 'fulfilled')),
  requested_at timestamptz not null default now(),
  fulfilled_at timestamptz
);

alter table public.roster_requests enable row level security;
revoke all on table public.roster_requests from public, anon, authenticated, service_role;

comment on table public.roster_requests is
  'Server-only channel roster handoff. Contains no creator-private data.';
