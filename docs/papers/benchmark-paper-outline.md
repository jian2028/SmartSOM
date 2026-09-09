# SmartSOM-Bench journal paper outline

Status: working skeleton, no result claim

Target delivery: complete manuscript v1 in Fall Week 12; submission package in
Week 15 after advisor approval.

## Working title

**SmartSOM-Bench: A Compositional and Reproducible Benchmark for Manufacturing
Scheduling Under Heterogeneity and Volatility**

## Venue positioning

Keep the manuscript venue-neutral through the Week 10 claim freeze, then choose
with Professor Jin based on the executed emphasis:

1. **Primary positioning — [Journal of Manufacturing Systems](https://www.sciencedirect.com/journal/journal-of-manufacturing-systems):**
   best fit if the paper's strongest contribution is the compositional
   manufacturing-system benchmark, simulator/evidence architecture, and system
   performance under controlled H/V.
2. **Alternative — [Robotics and Computer-Integrated Manufacturing](https://www.sciencedirect.com/journal/robotics-and-computer-integrated-manufacturing):**
   stronger if the final evidence centers on integrated scheduling/control,
   AGV/buffer logistics, and multi-agent coordination.
3. **Alternative — [Journal of Intelligent Manufacturing](https://link.springer.com/journal/10845):**
   stronger if the headline is intelligent dynamic scheduling and the
   cross-method benchmark/learning evidence.
4. **Alternative — [Computers & Industrial Engineering](https://www.sciencedirect.com/journal/computers-and-industrial-engineering):**
   stronger if the paper becomes a planning/scheduling methodology and fair
   computational-evaluation contribution rather than a simulator-centered paper.

This is a fit shortlist, not a submission decision. Journal, article type,
authorship, and formatting are frozen only after advisor approval.

## Relationship to the Phase-I paper

The existing IDETC/CIE Phase-I paper is the pilot and prior work, not the result
to repeat. It established a bounded 2 x 2 comparison of SPT, Centralized PPO,
and SO-MARL under one heterogeneity/volatility construction. The journal paper
must make a substantial extension through all of the following:

1. a clean, compositional benchmark implementation rather than one fixed study;
2. separate task/environment heterogeneity from team/agent heterogeneity;
3. separate volatility sources and report their parameters rather than collapse
   them into one score;
4. reproduce and adapt representative external Rule/CP/DRL/MARL methods under
   a common semantic and evaluation contract;
5. test capability boundaries and transfer across selected benchmark profiles;
6. retain failures, deviations, provenance, replay, and system cost as evidence.

If these extensions are not achieved, the manuscript must be described as a
Phase-I replication/engineering report rather than a new journal benchmark.

## Intended contribution

SmartSOM-Bench should let researchers construct matched, replayable scheduling
scenarios that vary heterogeneity and volatility without silently changing
workload, information visibility, completion semantics, or algorithm budgets.
It should expose where methods remain admissible, where rankings change, and
where an external method cannot be transferred faithfully.

The paper does **not** need an algorithm-superiority claim. A valid negative or
null result can still support the benchmark contribution.

## Research questions

### RQ1 — Benchmark construction

How can a compositional, trace-reproducible benchmark independently control
task/environment heterogeneity and volatility across static/online JSP and FJSP,
with layered FAJSP, transport, and buffer profiles?

### RQ2 — Capability boundary

How do representative Rule, CP, DRL, and MARL methods rank across matched
heterogeneity-volatility conditions, and where do failures, inadmissibility, or
rank reversals occur?

### RQ3 — Transfer

After an external method is first reproduced in its original setting, which
behaviors transfer to SmartSOM profiles and H/V conditions, with what deviations,
system costs, and failure modes?

## Construct decisions

### Heterogeneity

Do not create a single composite `H`.

- **Task/environment heterogeneity (`H_task`)** is the primary cross-algorithm
  construct. Manipulations and descriptors may include operation/capability
  composition, routing or eligibility dispersion, processing-mode diversity,
  and resource-role composition. The protocol must predeclare the manipulated
  component and match job count, total expected work, effective capacity, and
  eligibility density where the contrast requires them.
- **Team/agent heterogeneity (`H_agent`)** is a separate MARL/SO-MARL sub-study.
  It may describe controller roles, observation/action contracts, or team-type
  composition. Team size, effective action coverage, and capacity must be
  reported separately. `H_agent` is not applicable to every centralized method
  and therefore cannot be forced into the universal algorithm table.

### Volatility

Do not create a single composite `V`.

- **Primary confirmatory source:** arrival/task-stream volatility. The main
  low/high contrast should match total released work and mean arrival rate while
  varying burstiness and/or predeclared structural change magnitude.
- **Load control:** arrival intensity is reported separately. A higher mean
  arrival rate is a load change, not by itself proof of higher volatility.
- **Supporting sources:** machine outage frequency/duration and processing-time
  perturbation magnitude are independent slices. Their effects are not summed
  into the primary arrival-volatility axis.
- Robustness, adaptability, recovery time, and degradation are method responses,
  not volatility definitions.

## Benchmark profiles

The profiles are layered, not a full Cartesian product.

| Profile | Required role | Main evidence |
| --- | --- | --- |
| P0 Static JSP/FJSP | semantic reference and exact-baseline layer | JSP-as-one-mode equivalence, FJSP modes, Rule/CP/DRL checks |
| P1 Online JSP/FJSP | universal dynamic comparison layer | hidden future jobs, paired materialized arrivals, Rule/CP/DRL/MARL where compatible |
| P2 FAJSP | structural/DAG capability layer | assembly precedence and completion semantics; compatible methods only |
| P3 Finite transport | logistics capability layer | loaded/empty movement, availability, transport-aware completion |
| P4 Transport + buffer | selected coupled stress layer | finite capacity, blocking/deadlock diagnostics, selected compatible algorithms |

Only P0/P1 and one preselected coupled profile should enter the main comparison
tables. P2/P3 may be capability-validation profiles. Worker is a stretch profile
and must not delay the Week 12 manuscript.

## Algorithm portfolio

By the Week 8 integration gate, at least one method from each family must run
through a declared benchmark interface on at least one compatible canonical
profile:

- fixed dispatching Rule baseline;
- CP/exact or bounded rolling-horizon solver baseline;
- one reproduced/adapted DRL method;
- one reproduced/adapted public MARL method.

The methods do not need to support every profile. Unsupported settings must be
recorded as structured incompatibility, not imputed results. The Phase-I
SO-MARL is reproduced only after the new benchmark implements the semantics it
requires; it is a secondary internal prior-work baseline, not a substitute for
the public MARL reproduction.

## Reproduction contract

Every external method receives a `PaperReproductionSpec`-style record containing:

- paper and public repository identifiers, pinned commit, license status, and
  environment image;
- original instance/scenario, checkpoint or training budget, seeds, metric
  definition, and expected reported result;
- achieved reproduction level and tolerance;
- SmartSOM adapter mapping for state, action, reward, objective, completion,
  visibility, and failure semantics;
- all known deviations and whether they prevent direct comparison;
- original-setting evidence, SmartSOM compatibility evidence, and H/V extension
  evidence as distinct artifacts.

See [`reproduction-roadmap.md`](reproduction-roadmap.md) for levels and fallbacks.

## Evaluation protocol

Freeze the protocol before formal test runs.

- Materialize scenarios and event streams before any algorithm sees them.
- Use disjoint train/validation/test identities and domain-separated seeds.
- Pair compared algorithms on the same factory, demand, event stream, and
  simulator tie-break seeds.
- Predeclare primary outcomes, supporting outcomes, budgets, stopping rules,
  censoring/drain behavior, and multiplicity handling.
- Retain timeout, deadlock, invalid action, OOM, and incomplete outcomes.
- Report quality and systems evidence: makespan/tardiness/throughput as applicable,
  admissibility/pass rate, wall time, policy/solver time, memory, and trace cost.
- Never select checkpoints or tune hyperparameters on the frozen test set.

### Main study

Run Rule/CP/DRL/MARL on the common compatible P0/P1 subset across the frozen
`H_task x V_arrival` contrast. The primary volatility contrast holds mean arrival
rate and total expected work fixed while changing burstiness/structural change.

### Supporting studies

1. outage and processing-time-noise slices on a bounded P1 subset;
2. cross-profile transfer from P0/P1 into one of P2/P3/P4 where the method is
   semantically compatible;
3. `H_agent x V_arrival` comparison restricted to public MARL and Phase-I
   SO-MARL-compatible settings;
4. failure and systems-cost analysis across every attempted method/profile pair.

## Claim-evidence map

| Planned claim | Minimum admissible evidence | Falsifier |
| --- | --- | --- |
| compositional benchmark | module-off equivalence, semantic trace/replay, profile validation | behavior changes when a disabled module is present |
| controlled H/V | matched controls and independently recomputed construct values | workload/capacity or visibility confounds the contrast |
| fair method comparison | shared protocol, paired scenarios, declared budgets and deviations | incomparable objectives, horizons, or completion semantics |
| transfer insight | original reproduction plus SmartSOM adapter and repeated profile results | original result not reproduced or adapter changes the method materially |
| reproducibility | pinned source, resolved config, seeds, manifests, retained failures, regenerated table | result cannot be reconstructed from persisted artifacts |

## Manuscript skeleton

### Abstract

Write only after the evidence freeze. Include: the comparison problem, benchmark
design, evaluated method families/profiles, one or two executed findings, and
artifact availability. Do not use planned features or preliminary single-seed
results as completed contributions.

### 1. Introduction

1. Dynamic manufacturing scheduling methods are difficult to compare because
   problem semantics, disturbances, information access, objectives, and system
   resources vary across studies.
2. The Phase-I study revealed useful H/V behavior but was bounded to one design
   and limited controllers.
3. Existing environments and paper repositories provide strong components but
   do not by themselves supply the planned controlled, compositional H/V evidence.
4. State RQ1-RQ3 and the contribution list.

### 2. Related work

- JSP/FJSP/FAJSP benchmark environments and exact/reference solvers;
- static and dynamic DRL scheduling;
- MARL and self-organizing manufacturing control;
- transport, buffer, worker, and production-logistics extensions;
- reproducibility and fair evaluation gaps.

Separate sources used to define the benchmark from sources whose algorithms are
actually reproduced.

### 3. SmartSOM-Bench design

- scope and non-goals;
- shared semantic entities, actions, events, schedules, and completion truth;
- P0-P4 profile composition;
- H and V construct definitions and matched controls;
- scenario generation, benchmark registry, and provenance;
- interface profiles for Rule/CP/DRL/MARL;
- trace, replay, metrics, failure, and systems-cost contracts.

### 4. External-method reproduction and adaptation

- reproduction levels and selection criteria;
- per-method original result and environment;
- adapter mappings and deviations;
- admissibility matrix across profiles;
- anchor methods selected for full training.

### 5. Experimental protocol

- research questions and hypotheses/estimands;
- factor design and selected profiles;
- train/validation/test materialization;
- budgets, seeds, pairing, metrics, failure handling, and statistics;
- CARC and local hardware/software identities.

### 6. Results

Fill only from frozen artifacts:

1. benchmark validation and replay;
2. original-setting reproduction success/deviation table;
3. common-profile `H_task x V_arrival` comparison;
4. capability and transfer boundary;
5. MARL `H_agent` sub-study if the go/no-go gate passes;
6. outage/noise sensitivity;
7. failures and system costs.

### 7. Discussion

- where algorithm rankings or admissibility change;
- which paper results transfer and which depend on original semantics;
- what heterogeneity and volatility must be reported separately;
- implications for manufacturing scheduling benchmark design.

### 8. Limitations and threats to validity

- selected profiles are not every manufacturing constraint;
- public repositories may be stale, incomplete, or unlicensed for code reuse;
- checkpoint reproduction is weaker than full retraining;
- adapter equivalence may be partial;
- finite compute and seed coverage;
- synthetic scenarios versus industrial traces;
- worker, route conflict avoidance, quality/rework, and energy are outside V1.

### 9. Conclusion

Answer RQ1-RQ3 at the strength supported by executed evidence. Close with the
benchmark capability and empirical boundary, not a generic future-platform claim.

## Planned figures and tables

- Figure 1: one semantic core, layered profiles, and algorithm adapters.
- Figure 2: reproduction ladder from original paper to SmartSOM H/V extension.
- Figure 3: H/V factor construction and matched controls.
- Figure 4: capability/rank boundary across selected profiles.
- Table 1: comparison with related environments and benchmark papers.
- Table 2: profile semantics and module-off controls.
- Table 3: reproduced methods, commits, levels, deviations, and licenses.
- Table 4: protocol, budgets, seeds, and outcomes.
- Table 5: primary results including failures and uncertainty.

## Writing and submission gates

- **Week 4:** freeze RQ and construct specification v1.
- **Week 8:** freeze V1 feature scope and four-family algorithm gate.
- **Week 9:** freeze formal evaluation protocol before CARC test runs.
- **Week 10:** freeze main-paper claim inventory; no new feature family.
- **Week 12:** complete manuscript v1 with missing evidence explicitly marked.
- **Week 13-14:** only advisor-requested or claim-critical reruns/revisions.
- **Week 15:** journal-ready package; submit only after advisor approves journal,
  authorship, and final manuscript.

## Initial source anchors

Local Zotero remains the first search source. Useful current anchors include:

- Phase-I manuscript: `zotero://select/library/items/W5CIUEYD`.
- Reijnen et al., *Job Shop Scheduling Benchmark: Environments and Instances for
  Learning and Non-learning Methods*: `zotero://select/library/items/AKLBZDU5`.
- Hoss et al., *A Production Scheduling Framework for Reinforcement Learning
  Under Real-World Constraints*: `zotero://select/library/items/VAP7DURR`.
- Destouet et al., dynamic FJSP systematic review:
  `zotero://select/library/items/GJYX9H7H`.
- Dauzere-Peres et al., FJSP review:
  `zotero://select/library/items/FP2L98EF`.
- Liu et al., deep MARL for dynamic JSP:
  `zotero://select/library/items/XMJKTDFW`.
- He et al., dynamic integrated FJSP with transport robot:
  `zotero://select/library/items/XDNVPIBQ`.

Public repositories are tracked in the reproduction roadmap and must be pinned
before execution.
