# 0056 — Rule-credit Base / ALL development experiment

Date: 2026-10-31
Status: User authorized implementation and launch after engineering gates, at
18:45 UTC. This decision supersedes ADR0055 for this new contract; historical v1 is retained.

Both arms select discrete named heuristics resolving to currently legal physical
candidates. This is paper-inspired, not a reproduction of Gui's continuous rule
weights or ample-buffer assumptions. The source remains a39664c-based. No old
results or checkpoints are overwritten. Shared actor/critic tensors initialize
equally with seed4101; native ALL candidate-linked attention adds its original
extra parameters. It is not a capacity-matched social ablation.

Each retained rule gets an explicit role/stage/rule/mode identity in three unused
common feature slots45:48. Stable semantic tie breaking consumes no RNG. Aliases
selecting the same physical candidate retain one fixed-priority representative;
no duplicate voting. Full original SocialLink objects follow the resolved physical
candidate by identity, not list index. Rule resolution uses common public fields,
never social-only information. Compression can reduce physical-action coverage.

Actions:
- Empty Dispatcher: nearest / most ready / most urgent ready / fewest commitments.
- Loaded Dispatcher: nearest / least PRE nominal work / fewest PRE jobs. All
  current machines have equal processing rates, so next-operation SPT is removed.
- PRE accept: EDD / nearest requester / least remaining nominal route work.
- POST/Input accept: nearest / longest continuously unassigned-empty idle age.
  No persistent request clock is invented; current request-age is always zero.
  Equal idle ages make that rule unavailable; aliases still collapse.
- PRE service: EDD / arrival priority / least remaining nominal work.
- POST/Input service: EDD / longest residence / least or most remaining nominal
  work / most currently feasible next machines / least downstream PRE work.
  Resolve a legal job/vehicle pair; tie by vehicle port-arrival priority then
  semantic identity. Dynamic downstream rules exclude terminal/no-feasible jobs.
- Machine: EDD / SPT for the actual legal mode / longest residence, crossed with
  existing legal modes. Mover and Output retain their automatic behavior.

Free STOP and EMPTY_WAIT are disabled in both arms. No legal candidate means
automatic wait/no admission; no fallback illegal action. Buffer selection remains
learned whenever distinct rule representatives exist. Rule availability, aliases,
selection and projected candidate sizes are recorded, plus first two full mappings
per stage per episode/case (at most18). This does not imply every request offers
multiple meaningful actions; singleton/alias rates must be reported. Protocol
metrics count each owner/application batch once, waiting/quota totals and
oversubscribed rounds; sequential accept decisions have a separate denominator.
POST/Input idle all-zero, all-equal (including zero), and distinct counts expose
real ranking opportunities. Input automatic admission counts rounds, not decisions.

New POST free pickup credit is max(0,R+P-C): R includes all finished physical POST
jobs, including loading locks; P is the one actual already-started machine job
(processing, paused, blocked or local READY), never unstarted PRE jobs; C counts
accepted unfulfilled pickups until actual loading completes. Completion moves P
into R atomically; loading completion removes one R and one C. No concrete job is
bound before the vehicle arrives. A processing credit allows waiting at source,
never loading unfinished work. Original fault-related cancellation remains intact.
Unexpected negative balance is logged and fails the invariant; no invented scrap
or silent cancellation. Four ready plus one blocked machine job can support five
promises while physical POST capacity remains four. Input has no processing credit.

Idle starts when empty/unassigned/no service, stops on assignment; zero-time
reject/retry does not reset it. It measures unassigned idle, not stationary time.
Both arms see identical common idle/downstream/credit fields; the old aggregate
critic remains unchanged. Current feasible next-machine count requires capability,
up state, positive PRE quota and reachability; it is not a future guarantee or
new destination reservation. These observations, credit protocol and removal of
free STOP/WAIT are changes relative to historical raw-action runs, so that
historical comparison is not a pure action ablation. Within this new pair only
native social pathways differ.

Original social-long factory,320 orders/40 initial, arrivalmean14.4 and
normal/low/medium/high fault cycle remain. One seed4101 and original train-world
stream; names refer to faults, not arrival pressure. Native common reward,
reward/1000 learner scale, gamma1, three PPO epochs, truncation bootstrap, critic,
optimizers and physics timing remain. Fresh initialization, no pilot continuation.

Formal target1000episodes/arm. Deadline2026-11-01 05:24UTC stops training; final
checkpoint is the largest equally committed complete four-condition cycle.
Retain all episode ledgers, ep0, multiples4, latest2, and early160k checkpoint.
Barrier limits arms to one-episode difference. Interrupted uncommitted work is
not counted; last.pt is transaction authority and sidecars reconcile from it.
No performance-based checkpoint choice or replacement world.

The fixed held-out evaluation namespace is rule-credit-20261031. Each arm runs
final sampling on four fault-condition worlds, final argmax on all four, then
initial sampling on the same four:24 total cases. Each has one fixed action seed.
Case wall limits600s sampling/1200s argmax bound worst-case160min per arm;
190min reserved before08:34UTC evaluation stop and08:44 hard audit deadline.
Wall cuts remain separately recorded incomplete cases, never complete production
metrics; physical horizon remains last release+2000, never shortened to fit wall
budget. All four conditions are attempted even if earlier cases time out.

New pilot uses original train world indices0 and3 (normal/high), two full episodes
with updates/save per arm. Distinct pilot evaluation worlds normal/high each run
sample+argmax:8 total cases. Sample cap600s, argmax cap300s for bounded timing;
argmax cuts give lower bounds, not successful scientific evaluations. Gate: two
committed full episodes per arm, four completed sample cases, four argmax attempts
with full/cut receipts, exact source/config identity and no unexplained failures.
Pilot training cap20min and total eval cap60min, also absolute deadlines. Measured
new throughput and right-censored argmax guide honest1000episode feasibility.

Tests, independent review, clean local tag, manifest identity and explicit launch
receipt are required. Latest user authorization covers launch; no repeated request
is necessary. Two single-thread workers only,6GiB RSS ceiling each, no push/upload.
