-- Server-only shared fixed-window admission counters. Never store raw IPs.
create table creator.rate_limits (
    bucket text primary key check (length(bucket) <= 128),
    requests integer not null check (requests > 0),
    expires_at bigint not null
);
create index rate_limits_expiry_idx on creator.rate_limits(expires_at);
alter table creator.rate_limits enable row level security;
revoke all on creator.rate_limits from public, anon, authenticated, service_role;
comment on table creator.rate_limits is
    'Server-only HMAC identities. Admission cleans expired rows and caps table at 10000 rows.';
