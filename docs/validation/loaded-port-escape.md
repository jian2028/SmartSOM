# Loaded AGV transit-port escape

Engineering change, 2026-10-25. This is a controller repair, not a simulator
transition change, formal experiment, or claim of universal deadlock freedom.

## Problem and change

The 700-order, 160-processing-tick no-outage control developed a head-on wait
at tick 12,080. Two loaded vehicles needed the same transit port, and each
occupied the other's forward exit. Other vehicles queued behind them. Seven
vehicles were eventually stationary for more than 1,000 ticks.

`blocks_transit_departure` previously asked only unloaded blockers to vacate an
exit. It now also asks loaded, non-servicing blockers, using the existing joint
priority-inheritance assignment. They can temporarily move away from their
destination to let another vehicle through. Movement masks, service locks,
cell/edge reservations, no-swap checks and port-clearance validation still apply.

The conflict-free shortcut, route-distance calculation, policy observations and
checkpoint fields are unchanged. The extra assignment is attempted only when an
AGV enters a transit port whose forward exit is occupied. There is no new global
search, dependency, topology-specific ID or simulator-private policy input.

## Completed bounded checks

- The new regression reproduces the old loaded head-on failure with previously
  visited detours and passes for either initial priority order after the fix.
- Existing tests retain distinct station-side access, shared-exit safety,
  immovable/service blockers, safe following and rotations without swaps.
- 208 focused planner, dispatcher, three-map integration and related
  physics/public-contract checks pass; Ruff lint/format and the V4 controller
  configuration check pass. The final planner regression is checked separately
  after refining its initial traffic history.
- Twelve-job checks finish on all three templates with zero rejected actions:

| Template | Four one-tick operations | 30/40/50/40-tick route |
| --- | ---: | ---: |
| 7, Small | 289 | 793 |
| 8, Medium | 208 | 752 |
| 9, Large | 183 | 729 |

These are finish ticks for engineering cases, not a claim that every short
workload improves. They do not replace long-horizon or multi-seed validation.

## Exact original-gridlock recovery

Semantic replay of all 15,000 original control ticks matched 123,722 events,
every per-tick shipment count, all 30 saved state hashes and the exact final
state. Mover history was reconstructed at the ordinary public Mover boundary.
The stopping horizon was then explicitly extended by 100 ticks without editing
physical state, hidden quality, loads or destinations. With the corrected
planner, all eight vehicles moved on the first intervention tick, zero actions
were rejected, and two additional orders were delivered within 100 ticks.

This intervention preserves the original evidence; it is not a replacement run.

## Computer execution cost

Alternating old/new 200-tick checks on the same 700-job frozen input measured
median process CPU times of 46.555 and 45.969 seconds, respectively: ratio 0.9874.
There are two observations per version, so this establishes no material slowdown
in that bounded check, not a precise 1.3% speedup. A static recorded-gridlock
probe measured about 2 ms per planner call for both versions.

Fresh 700-job control and machine-outage regressions both reached the fixed
15,000-tick horizon with the same frozen inputs and seeds as the original pair.
Neither corrected case ended with a loaded, targeted vehicle stationary for
1,000 ticks or more, or a loaded job carried for 1,000 ticks or more. Both had
zero rejected actions.

| Case | Original deliveries | Corrected deliveries | Original final rate | Corrected final rate |
| --- | ---: | ---: | ---: | ---: |
| No-outage control | 504 | 562 | 14.5 | 38.5 |
| Machine 3 unavailable, ticks 5,000-9,000 | 484 | 485 | 37.5 | 38.0 |

Final rates are original jobs delivered per 1,000 ticks over ticks
13,000-15,000. The corrected control maintains its pre-gridlock throughput;
the outage case retains its drop and recovery behavior. These are deliveries
within the horizon, not completion of all 700 jobs.

Whole-run wall times were 44.34 versus 46.57 minutes for the control and 46.36
versus 49.41 minutes for the outage case. The corrected runs took about 5-7%
longer, so this is not a claim of shorter fixed-horizon execution. The control
delivered 11.5% more jobs, using about 5.8% less wall time per delivery. Whole-run
timings include host contention and changed traffic; the corrected pair ran
side by side. The bounded alternating benchmark above better isolates execution
cost and found no material additional cost.

The local IMPACT validation report contains the throughput comparison, frozen
inputs, source hashes, commands, telemetry, and reproduction scripts. Original
evidence is preserved separately.

## Throughput phases in the corrected pair

Both cases first reach an observed normal plateau of about 39 deliveries per
1,000 ticks. The machine-outage case then reaches a lower plateau and recovers
approximately the healthy delivery speed after repair:

| Phase | Estimated onset tick | Total deliveries at onset | Ticks since phase event | Deliveries since phase event | Confirmation tick / total deliveries | Mean rolling rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial saturation, both cases | 1,400 | 39 | 1,400 | 39 | 3,400 / 118 | 38.62 |
| Shutdown plateau | 6,700 | 216 | 1,700 after shutdown | 35 | 8,700 / 252 | 18.57 |
| Recovered plateau | 10,000 | 295 | 1,000 after repair | 37 | 12,000 / 370 | 37.52 |

Rates count original deliveries per 1,000 ticks. Onset is retrospective: use a
trailing 1,000-tick rate and search every 100 ticks, starting after a full
post-event window. Accept the first candidate whose following 2,000 ticks,
sampled every 100 ticks, have coefficient of variation at most 10% and absolute
fitted drift at most 10% of their mean. Confirmation is the end of that span.
This measures an observed plateau, not maximum capacity or a confidence interval.

In matched settled windows, the shutdown case delivers 18.5 versus 38.5 jobs per
1,000 ticks over ticks 7,000-9,000, a 51.95% reduction. Over ticks 13,000-15,000,
its recovered rate is 38.0 versus 38.5, within 1.30% of the control. It stays
within 10% of the concurrent control's trailing rate for ticks 10,000-10,999.

The 1,000-tick recovery delay is constrained by the measurement window and
search rule, not an exact physical restart delay. During the first 500 ticks
after repair, the shutdown case already delivers 19 jobs versus the control's
18. Changing the rate window to 500 or 1,500 ticks shifts estimated normal
onset to ticks 1,000-1,800, shutdown onset to ticks 6,500-6,700 and recovery onset
to ticks 9,500-10,500. These are window-sensitivity ranges from one paired seed.

Healthy speed returns without clearing lost production. At repair, the shutdown
case has delivered 258 jobs versus 332 in the control, a deficit of 74. At the
horizon its deficit is 77. It does not sustain enough excess output to catch up.

## Verification limits

Impact planning selects the full supported Linux/macOS native CI union because
the planner path is conservatively unmapped and a test module changed. That
full gate has not been run on this native Windows host, which is outside the
maintained CI platforms. Focused checks and local probes do not establish that
gate, a release, a research result or roadmap acceptance. Generated runs and
reports remain local and are not committed.
