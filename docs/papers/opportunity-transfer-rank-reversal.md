# Conditional paper: transfer and rank reversal under H/V

Status: go/no-go outline only

## Candidate claim

Scheduling-method rankings reported in an original paper may not transfer under
matched changes in task heterogeneity, arrival volatility, or production-logistics
profiles, even when the method is reproduced and adapted under a common contract.

## Why this might be separate

The benchmark paper needs some transfer evidence to validate the benchmark. A
separate empirical paper is justified only if the cross-paper pattern is strong
enough to become the central contribution rather than one benchmark result.

## Go gate

Expand this outline only if all are true:

1. at least three external methods reach R2 and at least two reach R4;
2. at least two predeclared, replicated ranking changes or transfer failures
   occur across H/V or profile conditions;
3. the changes remain after objective, horizon, budget, and failure handling are
   aligned;
4. the benchmark paper can remain complete after moving detailed transfer
   mechanisms to a follow-up without duplicating its essential validation.

## No-go conditions

- only one method transfers;
- ranking changes are post-hoc, single-seed, or caused by known adapter mismatch;
- the only result is that an old repository no longer runs;
- removing the analysis would leave the benchmark paper unsupported.

## Skeleton after go

1. Introduction: external-validity problem in scheduling-paper comparisons.
2. Reproduction and semantic-alignment protocol.
3. Transfer taxonomy: environment, objective, interface, scale, H/V, logistics.
4. Multi-method experiments and preregistered rank contrasts.
5. Results: stable ranks, reversals, failures, and systems cost.
6. Implications for reporting and method selection.
