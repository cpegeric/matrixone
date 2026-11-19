# Disk cache meta layer

Version: v1, 2026-11-19. Builds on the [object prefetcher](20261110-object-prefetcher.md).

## Problem

A read of an object reads its meta first, and of its key a bloom filter, before any of
its data: a miss of either is a request to object storage that no prefetch hides. The
disk cache orders all its files by their last use alone, so the data a scan reads evicts
the metas every query of the object reads next.

Measured on TPC-H SF100, two CNs, 15 GB disk cache each, a link of 25 ms and 110 MiB/s:
the metas and bloom filters of all objects take about 136 MB per CN. While a CN runs, its
memory meta cache holds them all. After a restart that keeps the disk cache, 912 and 874
of the 954 and 896 metas the CNs read again came from object storage: the data of the
queries before had evicted them.

## Contracts

- A read of an object's meta or bloom filter carries `ObjectMetaRead`. Only `objectio`
  sets it.
- The disk cache keeps the entries such a read writes in its meta layer, the `meta`
  directory under its path. A whole object, or a range of data that holds meta bytes,
  stays in the data layer: a meta read served from it writes nothing.
- A read finds an entry whichever layer holds it. A delete of an object removes its files
  in both layers. A restart keeps each file in its layer.
- The two layers share the disk cache's capacity. When room is needed, of the least
  recently used file of each layer, the one evicted is the one whose idle time over its
  value is the larger. A file's value is the time a read of it from object storage takes,
  the measured latency plus its size over the measured bandwidth, per byte. Of files of
  one layer the least recently used goes first, as before.
- Neither layer has a share of its own: one with nothing to evict leaves the other the
  whole capacity.
- A read of a meta or bloom filter served by the memory meta cache counts as a use of its
  meta-layer file, at most once every 10 seconds per entry; counting a use does no I/O.
  Without it, a meta in use is idle to the disk cache and ages out.

## Decisions

- No configured share of the capacity for metas (2026-11-19): the share follows the
  metas in use. Metas of objects deleted elsewhere, which the cache is not told of, age
  out as their idle time grows.
- The memory cache is used by meta reads as before (2026-11-19).
