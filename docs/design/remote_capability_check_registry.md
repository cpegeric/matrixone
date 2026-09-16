# Remote capability-check registry

Status: proposal — design for review.
Scope: mixed-version fencing of features that put new data on the wire, at the point a query
fragment is serialized to another node.

## 1. Problem

MatrixOne clusters are upgraded one node at a time, so a running cluster routinely holds nodes of
different binary versions. A newer coordinator can serialize a query fragment that carries a feature
an older node does not understand, and send it there.

There are two ways an older receiver can react, and only one is safe:

- **Clean rejection.** The receiver does not recognize the construct at all and returns an error.
  Loud, and correct.
- **Silent misread.** The receiver decodes the payload, ignores the part it does not know, and runs
  the older behavior — returning a *wrong result with no error*. This is the dangerous class:
  silent data loss during an upgrade window.

Each version-gated feature guards against this today by hand-rolling its own send-boundary check and
adding a call to the send path. That has four problems:

1. **Duplicated.** Every feature repeats the same "detect → confirm the destination supports it →
   refuse" shape.
2. **Opt-in and unenforced.** Nothing requires a feature that changes the wire to add a check; it is
   remembered, or it is not.
3. **Inconsistently applied.** The most dangerous slice — the silent-misread fences — is the least
   covered, while broad but less-dangerous checks are common.
4. **Not uniform across paths.** One serialization path carried the checks while a sibling path did
   not, so a fragment routed through the second path was unguarded.

## 2. Goals and non-goals

**Goals**

- A single point that evaluates *all* remote capability checks, run by *every* path that serializes
  a fragment to another node.
- A feature declares its requirement once — the minimum protocol version it needs and how to detect
  that a fragment uses it — without editing the send path.
- **Fail closed.** When a required capability cannot be confirmed on the destination, refuse the
  send. A refused query is a loud, correct error; the alternative is a silent wrong answer.
- A guard that keeps registered checks well-formed, so a malformed or ineffective fence cannot ship.

**Non-goals**

- Changing how the cluster negotiates its protocol version.
- Graceful degradation for every feature. Some features already degrade at scheduling time (running
  on fewer nodes instead of refusing); that is a separate, preserved mechanism.
- A complete "you changed the wire but forgot a check" detector. As §6 explains, that is not
  deterministically achievable and is not claimed.

## 3. Model

A **capability check** is two things: the minimum protocol version a feature requires, and a
predicate that reports whether the fragment about to be sent actually uses the feature.

At the send boundary, for each registered check whose predicate fires, the destination must meet the
minimum version or the send is refused. The version consulted is the destination's **current**
negotiated version, read at send time — so a version rollback that happens between planning a query
and sending it is still caught (a plan-time decision alone would not survive that rollback).

Detection is feature-specific and lives with the feature, because it inspects feature-specific
payload shape. The registry itself is generic: it knows how to run checks and nothing about any
particular feature. Features contribute their checks from their own layer, so the generic registry
never depends on the layers above it.

Two check shapes are supported: the common **require-or-refuse** gate described above, and an
**escape hatch** for a feature whose contract is not plain fail-closed — for example one that also
degrades — which supplies its own check body instead.

## 4. Key decisions and trade-offs

- **Fail closed, not silent degrade.** Correctness during an upgrade outranks availability. Refusing
  a small set of queries in a transient mixed-version window is acceptable; returning wrong rows is
  not. Features that *can* degrade safely still do so at scheduling time.
- **Enforce at the sender, on the destination's current version.** A plan-time gate is necessary but
  not sufficient: the negotiated version can drop between planning and sending. The authoritative
  check is at the boundary, against the version in force then.
- **Generic registry; features register from their own layer.** Detectors that need higher-layer
  knowledge stay in that layer; the generic registry stays dependency-free and reusable. The cost is
  that a detector may re-derive information the send path already computed — a small, send-time-only
  cost, accepted for clean layering. If it ever became material, the send path could compute shared
  detection once and pass it down.
- **One registration site.** All of a layer's checks are collected in one place, so adding a feature
  is one entry rather than edits scattered through the send path.

## 5. Invariants

- Every path that serializes a fragment to another node runs the complete set of checks; no path is
  exempt.
- A check never permits a send it cannot positively confirm: an unknown or unverifiable destination
  is treated as unsupported and the send is refused.
- Detection is superset-safe: when unsure whether a feature is present, run the check rather than
  skip it.
- The generic registry contains no knowledge of any specific feature.

## 6. Risks and limits

- **Completeness is not guaranteed by construction.** Nothing forces a new wire-changing feature to
  register a check. A consistency guard can verify that the checks that *are* registered are
  well-formed, but "a wire change with no registered check" cannot be detected deterministically —
  there is no enumeration of which changes are dangerous. Closing that gap is a review-time /
  tripwire concern, not something a unit test can guarantee. This is stated plainly rather than
  implied away.
- **Order independence is assumed.** Checks gate distinct features and do not interact, so evaluation
  order is immaterial. A future check that depended on another running first would break this
  assumption and must not be added without revisiting it.

## 7. Alternatives considered

- **Coarse, release-boundary protocol version.** Simpler and low-churn, but cannot gate a single
  feature during a mixed-version window — it can only say "same release or not."
- **Keep per-feature hand-rolled checks.** The status quo: duplicated, opt-in, and inconsistently
  applied — the problem this design exists to remove.
- **Put both the machinery and the feature detectors in the generic layer.** Rejected: it couples
  the generic layer to specific features and inverts the dependency direction.

## 8. Companion: the local (plan-time) gate

There are two kinds of version check, and they must not be conflated:

- **Local, plan-time.** "Is the local cluster's negotiated version at least what this feature
  needs?" It probes no one; it reads the local rollout gate and compares. This is where a planner
  decides whether to *emit* a feature at all.
- **Remote, send-time.** The subject of this document: "does the *destination* support it?", run at
  the boundary as the backstop.

The two are complementary, not redundant. The plan-time gate is necessary (do not emit a feature the
cluster cannot run) but insufficient on its own (the negotiated version can change between planning
and sending), which is exactly why the remote backstop exists.

They share one thing: the definition of "the current negotiated protocol version." Both read that
one primitive, so the whole system has a single source of truth for it. The local gate is a plain
read-and-compare, not a registry — a registry is only warranted where the check must run uniformly
at a boundary many features cross. Keeping both behind the shared primitive is the design intent;
folding the many scattered local read-and-compare sites onto it is a separate, mechanical follow-up,
not part of this design.
