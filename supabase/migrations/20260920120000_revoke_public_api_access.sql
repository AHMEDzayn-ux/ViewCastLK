-- Close the warehouse to Supabase's public REST API.
--
-- WHY
-- Security testing on 20 September 2026 found that every table in this
-- database had row level security disabled, no policies, and full
-- SELECT/INSERT/UPDATE/DELETE/TRUNCATE granted to the anon and authenticated
-- roles, which are the roles Supabase's PostgREST endpoint uses. Anyone
-- holding this project's publishable (anon) key could therefore have read or
-- truncated the entire collection: 123,000 videos and four months of
-- snapshots, none of which can be re-collected because past view counts are
-- not retrievable.
--
-- WHY IT IS SAFE
-- Nothing in this system reaches the warehouse through PostgREST. The
-- collection scripts, the archive job and the prediction API all connect
-- directly with a PostgreSQL connection string as the table owner, and a
-- table owner is not subject to row level security unless FORCE is set. The
-- dashboard never touches this database; it uses the separate Auth project.
--
-- WHAT THIS DOES
-- 1. Revokes API-role privileges on every existing table and sequence.
-- 2. Stops future tables from being granted to those roles.
-- 3. Enables row level security with no policies, as defence in depth, so a
--    future accidental GRANT still yields no rows.

revoke all on all tables in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke all on all functions in schema public from anon, authenticated;
revoke usage on schema public from anon, authenticated;

alter default privileges in schema public
    revoke all on tables from anon, authenticated;
alter default privileges in schema public
    revoke all on sequences from anon, authenticated;

do $$
declare
    target record;
begin
    for target in
        select c.oid::regclass as ident
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = 'public'
          and c.relkind in ('r', 'p')
          and c.relrowsecurity = false
    loop
        execute format('alter table %s enable row level security', target.ident);
    end loop;
end
$$;

comment on schema public is
    'Collection warehouse. Reached only by direct PostgreSQL connections as '
    'the owner; the anon and authenticated API roles hold no privileges here.';
