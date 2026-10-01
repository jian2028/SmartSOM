# 0027 — Balanced and Performance calibration with frozen sampling layout

Date: 2026-09-28
Status: user-authorized implementation; engineering verification is separate.

Supersedes ADR 0023's `office`/`throughput` names, 600-second active-only
calibration limit, baseline-only deadline selection, and exclusion of parallel
sampling for new runs. Historical records retain their original configuration
and execution identity.

New inputs use `balanced` or `performance`. Both maximize measured aggregate
physical-tick throughput for the selected batch. Balanced keeps a larger office
CPU/RAM reserve; Performance keeps a smaller system reserve. Their feasible sets
may select the same profile. A higher mode is not a promise of strictly higher
throughput on every workload or at every later machine load.

Calibration's user budget is total wall time, including resource waiting, process
startup, measurement and cleanup. It may stop early after repeat confirmation.
At the deadline, the best valid measured candidate is usable with an explicit
nonconvergence marker. If none is valid, an explicitly requested runtime layout
or the minimal layout may be attempted, marked uncalibrated. Invalid inputs and
failed preflight never become successful starts.

The new sampler advances independent environments concurrently with one fixed
weight version per collection wave and merges completed results in stable
environment order. The layout, environment identity, random streams and complete
collector state are frozen before formal training and restored unchanged. Only
the prior numerical-thread and batch-concurrency whitelist remains dynamically
resizable after training starts. Stateful custom observation/rule components
without safe independent state handling are excluded from parallel candidates.

No MPS performance claim or Apple-GPU path is introduced. Existing CUDA execution
remains explicit and separate from CPU candidate search. Engineering checks do
not establish research superiority or a universal speedup.
