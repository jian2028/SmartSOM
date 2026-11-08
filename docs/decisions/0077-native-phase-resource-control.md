# 0077 — Native phase resource control

Status: implementation under engineering validation; no production migration.

Refines ADR0058's operational controls and ADR0069's admission contract for native
directory workers. Native reservation tasks no longer reserve evaluation parallelism
throughout training. Other framework/resource-model contracts remain unchanged.

`batch-evaluation QUEUE --processes N [--cpus N] [--task ID]` submits a durable,
idempotent operational intent. The default CPU cap equals the requested process
count. A task override takes precedence over the queue default. `batch-budget`
continues to bound the aggregate queue and `batch-limit` the number of experiments.
Old drivers without the capability marker reject the new command. Updating code
never changes existing processes, frozen source, inputs, checkpoints or case IDs.

A native task retains one baseline CPU and one experiment slot. Evaluation grows
that existing lease atomically against the shared CPU/memory budget and observed
host headroom; the actual worker count can be below the requested target. The
baseline permits progress when all tasks enter evaluation together. These are
logical admission quotas, not OS affinity or hard CPU isolation. Existing cooperative
resource guards remain active. Standalone evaluation retains its launch concurrency;
live phase controls belong to native batch drivers.

The evaluation parent owns ordered receipt publication. Each case independently
seeds and constructs its policy. Growth adds workers without restarting the panel;
shrinkage retires idle workers after their current case, joins them, then releases
CPU capacity. Completed buffered results remain in frozen case order. Cancellation
joins all children before returning the task lease to training. Memory high-water
charges can remain conservative until the panel ends. Shared admission records
requested, effective and live evaluation capacity separately.

Explicit resume concurrency overrides replace prior phase intents. Reconstructed
native jobs recompute their training requirement instead of trusting cached whole-run
CPU requirements. A globally stopped addition which never started can return to
queued without a checkpoint; a started training addition still requires a committed
continuation. Scheduling records the start intent before spawning and retains
individual pause/stop holds. No automatic migration of historical run sources is
introduced.
