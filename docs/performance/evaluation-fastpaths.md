# Evaluation fast paths

These implementation optimizations preserve native decisions, observations,
physical transitions and research metrics. They do not change evaluation stopping
conditions or authorize a research performance claim. Use the repository's
`evaluation-profile` and `research-code-review` skills for measurement and review.

- Native Base DQN with `training=False`, `selection="argmax"` and one legal
  candidate returns that candidate without preparing tensors or computing Q.
  Network ownership is still resolved. Training and sampling keep their original
  inference, RNG draws and records. Custom learner subclasses, encoders, observations, networks,
  packers, replaced modules and forward hooks retain normal inference. This
  shortcut does not apply to PPO and adds no checkpoint fields.
- Dispatcher construction reuses ready/PRE/processing summaries and PRE work
  already present in its native station rows. Reuse ends before the action is
  yielded; subsequent requests observe updated reservations, jobs and services,
  even at the same physical tick. Quota calls and their diagnostic effects remain.
- The native Mover retrieves its owner distance map once per vehicle's parking
  scoring pass. Candidate cells, score arithmetic and tie breaks are unchanged.
  Public route changes are observed on the next pass. Custom route classes and
  instance `to_owner` overrides retain per-cell public queries.

Check both standard workloads and longer arrival streams with the same checkpoint,
seeds, arrival intensity, metrics and thread count. Compare complete reports
(excluding explicit wall-time fields), rewards, request/action traces and physical
transitions before interpreting timings. Measure without profiler hooks in serial,
alternating repeated runs; phase/cProfile runs are separate diagnostic evidence.
Generated measurements remain outside version control.

The next evaluation optimization group adds three separately measurable paths:

- Synchronous native read scopes reuse inventory tuples and an AGV reservation
  index during initial availability scans, request construction and invariant
  checks. Scopes end before yielding actions, custom observation callbacks or
  mutations. Quota/credit diagnostics still execute. Central protocols and
  instance-overridden fact queries retain uncached behavior.
- Native social DQN singleton argmax evaluation omits the tensor forward while
  retaining social packing, identity/shape validation and enabled-stage checks.
  Training, sampling, custom learners/packers/modules and forward hooks still
  execute normally. Social attention and pooling modules have no forward state.
- The native Mover keeps at most 1024 parking rankings by current cell and target
  owner. Contents of distance maps and the road-cell sequence are checked before
  reuse, preserving public map edits and replacement. Port membership, occupancy
  and selected goals are filtered live. Score arithmetic and coordinate ties
  match the previous filtered minimum; disconnected/noninteger maps and custom
  routes use that original path. Rankings are derived data, not checkpoint state.

Each path and their combination require identical frozen evaluation inputs,
serial alternating uninstrumented measurements and full semantic/trace checks.

Further read-phase reuse caches POST credit components, job vectors and ordered
job summaries in the same synchronous scopes. Credit diagnostics remain outside
memoization. The keys are owner/job identities or the complete ordered job tuple,
including duplicates; every scope discards its results before any policy or
physical mutation.

Native remaining-work estimates use a bounded value cache of each immutable
operation declaration and its ordered eligible machine/rate pairs. Machine
eligibility and rates are read again for each lookup, so declaration replacement
is observed when the existing per-tick job-work cache refreshes. Each operation's
rounded term is still accumulated left to right, and the current processing term
is calculated from live progress. Custom operation implementations bypass the
new cache. This caches neither sampled durations nor realized completion times.

Social packing builds link indices, masks and metadata as Python rows before
one tensor conversion per output. Query/peer tensor conversions retain their
original inference, broadcasting and shape validation. Outputs retain default
floating dtype/device and fresh storage on every call; no tensor crosses requests.
The same packer is used by training, so native learning checks remain required.

Reservation checks build a local drop-off lock counter once per check, keyed by
owner and slot. Every original capacity and credit assertion and every check
boundary remains in place.

Native candidate construction now opens its own short read scopes around
machine choices, dispatcher candidates, acceptance candidates and on-site
service candidates. These scopes end before `make_request` can invoke external
observation code, before yielding and before committing actions. Ready-stock
queries and nominal work are memoized only inside these scopes, with native
query overrides retaining their original calls.

Singleton model eligibility inspects the live native module graph iteratively,
including current hooks, types and replaced children. Custom module enumeration
retains the original traversal. Port lookup validation retains all identity/type
checks using direct loops. Physical invariants retain every assertion and its
original error order while reusing local job rows and active-demand membership.

Native fixed-case panels can opt into independent processes with
`concurrency.evaluation_processes: 4` (or `--evaluation-processes 4`). The default
is one process. This operation setting applies to standalone evaluation and the
same evaluator used by validation/final panels; it does not change scientific
inputs, case identities, seeds or numerical threads. It currently requires
native execution (`tune: off`), disabled recovery protocols and built-in
components. Custom extensions retain the one-process path unless separately
qualified for independent process state.

A bounded spawn pool constructs and loads a fresh policy for each case. The
parent alone holds the stage lock, commits receipts, writes tracking/progress and
reduces summaries in frozen order. Completed receipts are reused on continuation;
uncommitted cases run again. Each case retains its own replay directory. Pause,
stop and errors cancel children cooperatively and join them before releasing the
stage lock. CPU/memory admission reserves the evaluation worker envelope; the
trial-level concurrency limit remains separate from workers inside each trial.

Physical stopping conditions remain unchanged. Wall-clock deadlines still measure
real elapsed time, so a deadline-censored result can depend on available hardware
or contention. Equal-work performance comparisons must establish that no wall
budget was hit and compare complete reports and traces. Pool startup, input/model
loading, publication and cleanup belong in end-to-end panel timings.
