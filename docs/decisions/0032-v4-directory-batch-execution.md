# 0032 — V4 Experiment directory batches

Date: 2026-09-29
Status: Accepted for development experiments

ADR 0025 defines the four V4 author files, ADR 0027 defines measured resource
calibration, and ADR 0031 adds Algorithm matrices. This decision adds a directory
execution layer without changing the scientific contents of Factory, Workload or
Algorithm. `smartsom check DIR` checks every direct Experiment YAML; `smartsom
batch-run DIR` freezes that file list, source identity and every expanded V4
entry before work starts. A single-file `smartsom run FILE` keeps its old meaning.
No separate batch author file is required.

Optional top-level `batch.stage`, `batch.parallel_files` and `batch.gate` are
execution metadata. Without them, direct YAML members run in filename order,
one at a time. When used, every file must declare a stage. `parallel_files`
limits concurrent non-training child files; training entries from all files in
one stage enter Tune together and use measured resource admission instead. A gate checks all
committed evaluation cases and can require a minimum case count, qualified
deliveries per case and evidence of first pickup. Engineering failures fail the
stage. Later stages do not start after a failed gate. A completed gate is
rechecked on resume; resumed work uses frozen input snapshots rather than
rereading edited YAML.

All `train-evaluate` entries in one stage, including entries from different
Experiment files, enter one frozen Tune batch. Compatible learner/reward/network
groups share one calibration session. Its default wall-clock limit is 30 minutes,
including resource waiting and process cleanup. Disposable probes derive one
512-tick validation case and a short training budget from each frozen input;
they cannot be reported as formal training or held-out results. Each group gets
a bounded first probe. A valid measured layout is selected when available; a
timeout may use the recorded starting layout as uncalibrated. A baseline
engineering error stops the batch. The saved calibration is bound to the frozen
Tune plan and group identities and is reused for the later training stage and
resume.

When two CPU groups have valid single-group measurements, compatible layouts,
enough observed resources and remaining time, a mixed two-worker probe can
compare **estimated training-stage completion time** against separate group
waves. The chosen waves are frozen and enforced during Tune dispatch. Without
valid mixed evidence, group waves remain separate and scheduling is marked
uncalibrated. Mixed scheduling is inferred only when short and full inputs have
the same update quantum and validation-case tick limits; longer formal cases
cannot be projected from a 512-tick case. The probe does not measure final
held-out evaluation time and does not establish a globally optimal
configuration.

The parent batch directory owns progress, Rich attach/monitor, cooperative
stop and recovery. Child runs and their artifacts live beneath it. Calibration
has its own limited budget; the whole batch has no implicit six- or ten-hour
cutoff. Parent resume may explicitly retry failed Tune entries while preserving
completed entries and the frozen plan. Formal research claims still require an integrated `main` commit or
explicit tag in the run manifest. This feature creates development evidence
only until that source condition is met.
