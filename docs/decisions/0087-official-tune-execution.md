# 0087 — Official Tune execution and phase resources

Status: authorized implementation in an isolated worktree; acceptance pending.

Supersedes the execution and operational controls in ADR0055/0056/0057,
ADR0069, ADR0077 (native phase resources), and ADR0078 as described here.
Scientific contracts in ADR0079/0080/0082/0083 remain authoritative.
Existing archived runs and their sources/environments are never migrated.

Fixed experiments and explicit Optuna searches will share one Tune Trainable
and the native scientific training/evaluation implementation. FIFO is the default.
The simulator remains independent of Ray. Resource declarations belong in profiles;
no runtime performance calibration, measured admission broker, custom process pool,
or monkey patch of Ray's private control/checkpoint machinery remains in the target.
Ray requests are logical admission quantities, not OS limits. Failure retries
will default to zero. Source and complete-checkpoint validation remain mandatory.

Stock ResourceChangingScheduler uses a small phase callback. Ray 2.58 rejects
heterogeneous initial trial requests, so the user explicitly approved a common
lightweight bootstrap stage. This stage does not construct models or sample worlds.
It reports the next phase request. Restore also negotiates the actual allocation
before loading scientific state: Tuner.restore can return a trial to its initial
request. Resource changes use complete durable boundaries and reuse_actors=False.
The callback selects declared resources, never measures throughput or estimates RAM.

Online changes to queue membership, limits, budgets, priorities and evaluation
concurrency are removed. Evaluation concurrency remains a startup option.
Whole-run pause/resume and official failed-trial restoration remain; selective
trial retry and cooperative RSS/host-RAM/disk threshold guards are removed.
Seed identities and scientific stage dependencies remain, without promising
parallel start/completion order. Best/last, fixed panels, receipts, provenance,
Rich and W&B semantics must survive the migration.

Public check/run remain the entry points, with --search for an explicit search
configuration. Search uses a single explicit training seed and explicit periodic-panel
objectives, never final-test objectives. The selected research objectives minimize
makespan and mean tardiness and maximize final passing rate, without priorities or
weights. Existing Rule-based completion limits and censored metrics are retained.
A user-approved public Searcher adapter delegates proposals and persistence to
OptunaSearch. Censored completion is sent through its no-result completion path
(PRUNED), preventing Ray 2.58 from leaving an Optuna trial RUNNING after receiving
null objective values. No score is fabricated and Tune does not retry censoring.
Normal finite-budget termination before any panel is labelled not_evaluated and
also carries no score. Searcher state must survive restore. Failed search trials are retained as terminal
failures: the official OptunaSearch does not update a failed ID after a manual
retry. The user chose to reject in-place HPO retry and open a new search instead.
Unfinished search trials still restore; fixed experiments retain retry --all.
Scientific drain calibration is unrelated to the removed performance calibration.

Acceptance requires real learning and recovery tests, independent engineering,
scientific and cross-test review, and three alternating baseline/candidate pairs
per representative workload. Stable work should take about 5–15 minutes with at
least two periodic panels. Initial startup is reported separately; phase restarts
remain in steady execution costs. Each case must stay within 5% of baseline for
both steady and end-to-end median wall time, with matching scientific work/results.
These engineering tests do not authorize a formal research claim or deployment.
