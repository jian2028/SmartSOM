# 0099 — One composite Rule with critical-ratio speed selection

Status: user-authorized implementation; engineering verification recorded separately.

Supersedes ADR0092's three fixed-speed Rule comparator inventory and the default
Base Rule's normal-mode preference. ADR0098's standalone evaluation deferral stays:
this change does not claim a completed comparator panel adapter or formal result.

Retain one Rule per scenario size in the maintained campaign. Random remains an
internal debugging reference, excluded from paper baselines. Keep Base without SL
and implicit SL as the learning comparison; Central's design/adaptation is deferred.
Historical fixed-speed recipes, source references, frozen runs and calibration
reports stay intact. New rule actions do not rewrite inherited drain provenance.

Choose the earliest-due legal job first, then current-machine normal duration and
semantic job ID. Compute CR = (due_at - now) / remaining normal processing ticks.
The denominator is the current operation's normal duration on the deciding machine,
plus each subsequent operation's minimum normal duration on compatible machines.
Use machine rates and per-operation ceiling, with a minimum of one tick. Base's
normal reference mode has time_scale=1. Do not double count the current operation.
The estimate uses only released-job public routes and static machine capability,
not transport, queueing, future outages, sampled processing time or latent defects.

CR < 1 chooses fast; 1 <= CR < 1.2 chooses normal; CR >= 1.2 chooses slow.
Use exact cross multiplication for thresholds. When the target mode is absent from
legal candidates for the selected job, use normal if available, else shortest legal
duration and candidate identity. This preserves single-mode and restricted-action
execution without inventing an illegal action or switching to another job.

No new tunable weights or policy framework. The 1.2 threshold is an explicit
heuristic, not a fitted optimum. The campaign's slower modes have lower declared
defect rates; the heuristic does not inspect error draws or directly optimize J.
Other Base rule decisions, the central rule implementation and Random stay unchanged.
Extra processing estimates are built only for named Rule observations; generic
rule projection and Base/SL feature construction retain their previous behavior.

Validate threshold boundaries, overdue jobs, EDD/ties, restricted modes, public-only
heterogeneous route estimates and RNG invariance. Compare short fixed-input runs
against the pre-change dirty-source snapshot, recording changed work and outcomes
as intentional policy differences rather than claiming trajectory equivalence.
