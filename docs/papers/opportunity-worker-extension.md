# Conditional paper: worker-flexible scheduling under H/V

Status: post-V1 stretch outline

## Candidate claim

Worker flexibility introduces a distinct capability and uncertainty layer whose
interaction with task heterogeneity and volatility changes scheduling-method
admissibility and transfer.

## Source starting point

The local Zotero item `X2IJRAPN`, *A Benchmarking Suite for Flexible Job Shop
Scheduling Problems with Worker Flexibility under Uncertainty*, and its public
[FJSSP-W repository](https://github.com/jrc-rodec/FJSSP-W-Competition) provide a
safer starting point than inventing a worker benchmark from scratch.

## Go gate

Expand this outline only if all are true:

1. the Week 8 V1 scope (static/online JSP/FJSP, FAJSP, transport, buffer) passes;
2. the Week 12 benchmark manuscript is not delayed;
3. worker capability, eligibility, calendars, and uncertainty can be added as a
   composition without changing existing semantic truth;
4. the public benchmark result reaches R1 before SmartSOM adaptation begins;
5. the worker layer produces a question not already answered by the transport or
   task-heterogeneity studies.

## No-go conditions

- worker is added only as another resource label;
- it forces a rewrite of the core or invalidates V1 profiles;
- it consumes the CARC or writing budget reserved for the journal paper;
- the result is only a larger full-factorial table without a new mechanism.

## Skeleton after go

1. Worker-flexible scheduling gap and benchmark requirements.
2. Worker/resource/task heterogeneity definitions.
3. Uncertainty and workload-matched scenario construction.
4. Rule/CP/learning compatibility and transfer.
5. Capability, robustness, failure, and system-cost results.
6. Implications for human-resource-aware manufacturing control.
