-- Execution-level proof for the clone/restore path of an ASYNC ivfflat index
-- (the idxcron / CDC-backed maintenance path re-registered by Scope.RestoreTable).
--
-- This goes beyond metadata/wiring checks and drives the real surface:
--   (a) CREATE TABLE ... CLONE of a table carrying an ASYNC ivfflat index
--       succeeds;
--   (b) the CLONED index still answers vector search (correct neighbors);
--   (c) the background maintenance path is actually RE-ARMED on the clone:
--       rows inserted AFTER the clone are incorporated into the cloned index
--       and become searchable.
--
-- Why a successful search proves the index is live: ivfflat has no brute-force
-- fallback — searching an index whose model is missing errors with
-- "internal error: version not found". So every search that returns a row
-- below is served by a working index, not a full scan. With probe_limit >=
-- lists, every centroid list is probed, so the returned nearest neighbor is
-- exact and deterministic regardless of kmeans sampling.

SET probe_limit=10;

drop database if exists ivf_restore;
create database ivf_restore;
use ivf_restore;

-- Source: ASYNC ivfflat over three well-separated clusters (unambiguous NN).
create table src(a int primary key, b vecf32(3));
insert into src values
  (1,'[1,1,1]'),(2,'[2,2,2]'),
  (10,'[100,100,100]'),(11,'[101,101,101]'),
  (20,'[500,500,500]'),(21,'[501,501,501]');
create index idx using ivfflat on src(b) lists=3 op_type 'vector_l2_ops' ASYNC;

-- Wait for the ASYNC build to finish BEFORE the first search, and wait on the
-- CENTROIDS table rather than by retrying the search itself.
--
-- ASYNC returns as soon as the DDL commits; the centroids are written later by
-- the background build. A search issued in that window loads the index from an
-- empty centroids table, and the result is cached under the same key/version the
-- finished build then writes -- so it is never reloaded and every later search on
-- this index answers from the empty copy. Retrying the SEARCH cannot recover
-- from that: the first attempt is what caches it. Polling a plain table read
-- touches no index and leaves nothing behind.
-- Readiness is the ENTRIES table, not the centroids. The build writes them in two
-- stages: kmeans publishes the centroids first and the per-vector list assignment
-- lands about ten seconds later. Waiting on centroids alone returns while entries
-- is still empty, and a search there finds nothing.
set @entr = (select index_table_name from mo_catalog.mo_indexes
    where name = 'idx' and algo = 'ivfflat' and algo_table_type = 'entries'
      and table_id in (select rel_id from mo_catalog.mo_tables
                       where reldatabase = database() and relname = 'src')
    limit 1);
set @wait_sql = concat('select count(*) >= (select count(*) from src) as ready from `',
                       database(), '`.`', @entr, '`');
prepare wait_build from @wait_sql;
-- @wait_expect(1, 60)
execute wait_build;
deallocate prepare wait_build;

-- Now the index is complete, so this is a real index search.
select a from src order by l2_distance(b,'[1,1,1]') limit 1;

-- (a) clone succeeds and copies the rows + index definition
create table dst clone src;
select count(*) from dst;
show create table dst;

-- (b) the cloned index answers search (no "version not found")
-- @wait_expect(2, 30)
select a from dst order by l2_distance(b,'[1,1,1]') limit 1;
select a from dst order by l2_distance(b,'[500,500,500]') limit 1;

-- (c) maintenance re-armed: rows inserted AFTER the clone get indexed
insert into dst values (30,'[1000,1000,1000]'),(31,'[1001,1001,1001]');
-- the post-clone row is the nearest neighbor for its own vector, served by the
-- cloned index -> the CDC maintenance path on the clone is live
-- @wait_expect(2, 45)
select a from dst order by l2_distance(b,'[1000,1000,1000]') limit 1;
-- pre-existing cloned rows remain searchable too
select a from dst order by l2_distance(b,'[100,100,100]') limit 1;

drop database ivf_restore;
