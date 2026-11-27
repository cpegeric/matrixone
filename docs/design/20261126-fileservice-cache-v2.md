# File-service cache v2: range fills and the object storage server

Version: v2, 2026-11-27 (v1, 2026-11-26). Replaces the design of PR #32812 (closed). Builds on
[index-file I/O](20261024-index-file-io.md),
[thrash-aware prefetch](20261102-thrash-aware-prefetch.md) and
[file-service cache ownership](20261106-fileservice-cache-ownership.md).

## Review status

Approved by the developer and implemented on branch `disk_cache_range_fill`. The decisions
made during the implementation are under "Decisions and known limits".

## Problem

1. **Scans download whole objects for a few columns.** A scan that reads some columns of
   an object makes the disk cache download the whole object. On wide tables most of those
   bytes are never read, and a cache smaller than the data thrashes.
2. **IVF-Flat entries objects are not kept.** A search reads the same entries objects
   again and again; they compete with table data in the disk cache and are evicted.
3. **Object-storage connections have no owner.** Every file service of a process shares one
   HTTP pool; a semaphore is held across response bodies, writes and lists; direct reads,
   background prefetches and writes take connections in no order, so background work can
   starve a query.
4. **Memory has no single bound.** Download buffers, stream buffers and write parts are
   bounded apart, or not at all.
5. **One mechanism is written several times.** Whole-object downloads and range reads,
   synchronous and asynchronous publication, and path owners each have their own
   registration, join, failure and delete paths, which must be kept in step.

## Goals and non-goals

Goals (requirements):
- **Column-range read and prefetch.** A scan prefetches, and a read fetches, only the byte
  ranges of the columns it reads; local and remote scans alike.
- **Memory prefetch objects.** A prefetched range up to a memory limit is held in memory
  while it downloads, so reads of it do not wait for a file.
- **The index-file cache stays a separate cache,** with its own capacity, directory,
  eviction and idle TTL. IVF-Flat entries objects are cached in it.
- One kind of fill, one registry, one join rule, one publisher, one object storage server.
- The same bytes and the same errors for every read as on main.
- No deadlock, by construction; no work dropped to escape a stall; no read failed by a
  file-service timeout.

Non-goals:
- Merging the index-file cache into the disk cache.
- Changing the memory cache, the remote cache or any persisted data format.

## Architecture

Four owners, each with one job:

| Owner | Job |
|---|---|
| **Fills** | Download a range of an object into one cache. One registry of every fill. |
| **Object storage server** | Make every object-storage request of its file service; own its connections and memory. |
| **Publisher** | Make each finished file durable, rename it in, index it, evict for room, order deletes. |
| **Caches** | The disk cache (table data) and the index-file cache (index objects), each a set of immutable range files. |

Callers ask the fills for bytes: a scan prefetches the column ranges of the objects ahead;
a read looks in the memory cache, then the cache's range index, then joins or makes fills.

## Contracts

### Range fills (decision)

- A fill downloads the bytes `[start, end)` of one object into one cache and is identified
  by the object, the cache and the range. A whole object is the range `[0, size)`.
- Its backing is a memory buffer when the range is at most `disk-prefetch-memory-max` and
  its memory fits now (see Memory), else a temporary file in its cache.
- Only a whole-object fill splits into parts read at once; a range fill is one part.
- A fill is waiting, running or ended. All fills, in every state, are in one registry. There
  is no separate prefetch queue, live queue or worker pool; a waiting fill holds no
  goroutine and no resource.

### Prefetch

- A scan's prefetch makes waiting background fills for the column ranges it will read, with
  its due time and share. A request for bytes a waiting fill of the same cache holds merges
  into it; bytes a running fill holds need nothing; other bytes get fills at once.
- Waiting background fills are bounded by bytes. A waiting fill whose scans all ended and
  that no read joined is dropped, never started.
- A started fill first checks its cache, and ends without downloading when its bytes are
  cached.

### Reads join fills (decision)

- A read that misses the cache joins every fill of its own cache that holds bytes it needs,
  waiting or running, each for the bytes it holds, and makes foreground fills only for the
  gaps. A fill that ended holds only the bytes it wrote.
- Joining a waiting fill makes it foreground and puts it before every background fill.
- A read waits until its own context ends or the fill stops. For bytes a stopped fill did not
  write, it makes its own fill. There is no estimate of arrival time and no choice of the
  fastest fill.
- The two caches never join or cover each other.
- At most one fill of the same bytes into the same cache runs at a time when one covers the
  other; partly overlapping fills may each download their overlap.

### The object storage server (decision)

- **One per file service.** It owns that file service's HTTP connection pool and memory
  budget. Nothing is shared with another file service or CN, also in one process. A server
  never calls another server.
- **Every request goes through it.** `Do(ctx, request)`: the request is data — read a range
  into a buffer or a fill's file, write bytes or a part, multipart create/complete/abort,
  list a page, stat, delete, copy — never a function. No other code reaches object storage.
- **Nothing else runs while a request holds room.** No caller code, file, registry, path,
  eviction or publisher work runs inside `Do`; only bytes move and a fill's progress is
  noted.
- **Room.** At most min(`max-conns-per-host`, `concurrency`) requests run at once; they
  default to 128 and 1024, so the room is 128 when neither is set. When one process runs
  several services (a launch with CN, TN and log services), the room is divided among
  them. Never fewer than 64 (a smaller value is raised, with a warning); no upper bound.
  The pool holds as many connections per host as the room.
- **One order for room.** Foreground before background, then larger share, sooner due
  time, arrival. Background holds at most half the room. A running request is never
  stopped.
- **Retries.** The file service retries; the SDK makes one attempt where the file service
  retries. A retry waits for room again, holding only its own buffer.
- **Close** closes the server and the pool's idle connections at once.

### Memory (decision)

- One budget per server: `object-storage-memory`; by default 5% of available memory,
  re-measured about every second, or 256 MiB when it cannot be measured.
- **Nobody waits for memory.** Every take is "fits now, otherwise a file": foreground up to
  the whole budget, background up to half.

| Memory | When it does not fit now |
|---|---|
| A fill's buffer | the fill is file-backed |
| A stream's buffer (`ReadCloserForRead`) | the stream reads into a temporary file |
| A write part read from the caller's reader | the part is read into a temporary file |
| A write of bytes in memory (`IOEntry.Data`) | takes no memory: sent without a copy |

- Part buffers are reused.

### Caller code never runs inside a request

| Caller code | How it runs |
|---|---|
| reads a stream | one request reads the range into memory or a temporary file; the caller reads as bytes arrive |
| receives bytes (`WriterForRead`, `ToCacheData`) | runs after its request ends |
| supplies a write (`ReaderForWrite`) | each part is read from the caller first, then one request writes it |
| handles listed entries (`List`) | one request per page; entries go to the caller between pages |

### Publication and files (decision)

- A cache file is written once under a temporary name and published by renaming it into
  place; it is never changed in place, only unlinked. A read that opened a file reads that
  file's bytes.
- Every finished fill and cached entry is published by the one publisher of its cache.
  Its queue is bounded; a full queue makes submitters wait, ended by their own context
  (the entry is then not cached). A waiting submitter holds nothing the publisher needs.
- Only the publisher writes, renames or removes a published path. There are no path owners.
- A whole object is cached as the range `[0, size)`; there is no separate whole-file name.

### Deletes, eviction and close

- A delete stops every fill a prefetch made of the object before its object-storage
  request, so it never waits for room held by fills it stops. It then stops the fills
  reads made, and has each publisher remove the object's ranges, ordered after every
  publication of that object submitted before the delete. A delete whose context ends
  still has its ranges removed.
- A read never returns bytes of two versions of an object: when a delete of the object
  is counted while the read runs, the read reads the object again.
- Eviction removes ranges. A range whose file cannot be removed is never read again and
  still counts against the capacity; eviction skips it. A later delete of the object, or a
  publication over its path, removes it. An index-file entry whose file cannot be removed
  is never read again either.
- Close stops fills in every state before any of them frees its room, and wakes their
  readers.

## Deadlock freedom

- A request holding room waits only for object storage (bytes, an error, its context, or
  the idle timeout).
- A request waiting for room holds nothing but, on a retry, its own buffer.
- A read waits only for a fill or the server; neither waits for a read or caller code.
- A submitter waiting for the publisher holds nothing the publisher needs.
- Nothing waits for memory. No server calls another.

So nothing waits in a cycle, whatever callers do.

## Alternatives

- **Incremental on the existing mechanisms (PR #32812).** It grew the file service by over
  6,000 non-test lines while keeping the old paths; rejected for this design built from main.
- **Make every fill file-backed.** Rejected: memory prefetch objects are required.
- **A shared server or pool per process.** Rejected: one file service's work must not
  starve another's.

## Compatibility and operations

- No persisted data format changes. Whole objects cached by main under the whole-file name
  are removed at startup and refilled on use.
- New setting: `object-storage-memory`. Settings made pointless (e.g. the extra-wait
  setting) are accepted and ignored, with a log.
- Metrics: the server's room and memory by class; the publisher's queue depth and waits;
  fill and join counts.
- Upgrade is by full shutdown.

## Validation and acceptance

- Each contract above has a probe on a public entry point (`S3FS.Read`, `PrefetchObject`,
  `Delete`, `Close`) with gated storage, a control, and red/green evidence.
- `-race` with `-cpu 1,2,8`; dependent packages (objectio, disttae, readutil); the static check.
- TPC-H SF100 on two CNs, cold and warm Q1–Q22, at least as fast as the 2026-11-25
  baseline (cold 357 s, warm 220 s), with object-storage requests and bytes per CN.
- Net non-test growth of `pkg/fileservice` well below PR #32812's.

## Decisions and known limits

- **Room is not a multiple of the CPU count.** The requests in flight needed are about
  bandwidth × latency ÷ request size, which does not depend on the CPU count. The
  defaults stay as on main; the room is measured on TPC-H SF100 cold at 64, 128 and 256.
- **Removals of one path are queued once.** A removal of a path already queued for the
  same reason (evict, reindex, repair, unload, delete) is not queued again; the newest
  request's work runs.
- **A file that cannot be removed is indexed again after a restart.** Which files could not
  be removed is held only in memory, so a restart reads such a file again as cached. Main
  has the same limit.
- **Ignored settings** (`disk-download-max-extra-wait`,
  `disk-prefetch-object-split-downloads`) are logged once per file service.
- **A read racing a delete** may return "not found", as on main.

## Open questions

- The order of commits for review.
