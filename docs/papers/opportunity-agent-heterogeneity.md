# Conditional paper: agent heterogeneity and SO-MARL

Status: go/no-go outline only

## Candidate claim

Team/agent heterogeneity changes the feasible and effective action structure of
self-organizing manufacturing control under volatility, and this effect cannot
be inferred from task/environment heterogeneity alone.

## Boundary with the benchmark paper

The benchmark paper defines `H_agent` separately and may include one bounded
MARL sub-study. This follow-up requires a mechanism-level claim about team
composition, coordination, or effective action coverage.

## Go gate

Expand this outline only if all are true:

1. one public MARL method and the Phase-I SO-MARL baseline run under a common,
   documented profile;
2. at least two team compositions are matched on team size/capacity or a clear
   adjustment model is predeclared;
3. effective action coverage or another mechanism measure varies non-trivially
   and reproducibly across `H_agent`;
4. the effect is not explained solely by task `H`, workload, observation access,
   or incompatible action interfaces.

## No-go conditions

- only performance changes are observed without a coordination mechanism;
- agent roles are labels with identical observations/actions;
- joint-action infeasibility or simulator defects dominate results;
- the required AGV/buffer semantics are not stable by the Week 10 scope freeze.

## Skeleton after go

1. Introduction: why agent heterogeneity is distinct from shop heterogeneity.
2. Team-role and effective-action constructs.
3. Common MARL/SO-MARL interface and matched compositions.
4. `H_agent x V_arrival` experiment.
5. Coordination mechanism, performance, failure, and latency results.
6. Design implications for self-organizing manufacturing systems.
