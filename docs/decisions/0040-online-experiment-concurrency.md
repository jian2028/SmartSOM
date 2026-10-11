# 0040 — Online experiment concurrency with reusable machine/task evidence

Date: 2026-10-10
Status: Development execution contract

The user selected daily execution that adjusts concurrency during real training,
with a complete performance test retained as an optional tool. This supersedes
the mandatory pre-training performance search for new V4 train-evaluate inputs
in ADRs 0027/0032/0033. ADR 0034's explicit off path, existing frozen plans and
the scientific continuation/ownership contracts of ADR 0023 remain intact.

New V4 train-evaluate inputs without an explicit executor or tuning selection
compile to Tune auto and calibration_level online. Explicit native/off inputs
retain manual execution. When online max_concurrent is omitted, the expanded
entry count supplies its ceiling; an explicitly authored ceiling remains binding.
Directory batches retain their existing global maximum and per-file ceilings.
Train-only and rule evaluation keep their native paths. No optional dependency
is added and the simulator does not import Ray.

Online setup freezes the authored thread/environment/sampling layout and does
not run disposable performance probes. Engineering preflight still applies.
Daily online Rich sessions show preflight then execution; the execution panel
contains the concurrency/throughput card. Waiting for online resources stays in
execution. Only explicit isolated quick/full tests retain a calibration stage.
The existing single-node Ray FIFO/resource-changing adapter owns admission,
checkpoint-safe pause/resume and actual actor/child lifetimes. The new controller
changes only concurrent independent experiments. It never prunes a scientific
trial or changes its budget, seed, network, reward, sampler or device.

Each comparable task group starts with one trial. The first verified commit and
observed process-tree RSS establish a conservative resource request. Two stable
wall-clock windows, normally at least 30 seconds each and two committed updates
per live trial per window, establish useful aggregate physical ticks/second.
The windows include validation/checkpoint overhead. Startup, staging, stopping,
membership changes, final evaluation and insufficient tail waves do not supply
comparisons. More than 15% variation holds the setting for further observations.
An accepted operating point permits one additional experiment. Less than 5%
improvement returns to the last accepted count; excess work pauses at a complete
update and its reservation remains charged until owned processes exit. Live
CPU/RAM/GPU admission and pressure handling continue independently.

Online groups include map topology, workload shape, algorithm/network parameters,
sampler layout and validation/save schedule. Unlike standalone mixed probes,
online windows keep incomparable groups separate. This is conservative single-
group scaling, not a claim of the fastest possible mixed workload schedule.

Each Tune run retains online-performance.json with windows and decisions. The
existing local performance directory also retains online.json, separate from
probe index.json. Its key includes host architecture, allocated CPUs/RAM/limits,
resource policy, task shape, implementation and dependency versions. Only a row
still present in its original report supplies a cached hint. A fresh execution
or resume starts at one again; a compatible hint shortens confirmation windows
below its measured count to 15 seconds, without bypassing complete updates or
memory/resource checks. Retained history is evidence, not fresh measured speed.

quick/full calibration and recommend remain available for explicit hardware
experiments, including pre-training thread/sampling searches. A recommend request
with no explicit level selects quick; online cannot produce a recommendation
without running real work. Online/off have zero isolated calibration budget.
These engineering checks do not establish speedups on Mac, rented Linux or USC
nodes, or promote a research milestone. Formal experiments still require an
integrated main commit or explicit tag and recorded source identity.
