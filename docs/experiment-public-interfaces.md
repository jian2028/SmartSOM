# Public experiment inputs and ordered directory batches

These opt-in interfaces extend the four-file compiler and existing directory
driver. Experiment-specific scripts and large data plans remain local.

Optional fields such as `world_seed`, `training.groups` and
`evaluation.initial_control` should be authored directly in YAML. `--set` replaces
existing paths in the normalized configuration; it cannot introduce an optional
path omitted from that mapping, including fields excluded at their default value.

## Per-role initialization and independent random streams

Algorithm model selectors already load compatible inference packages per role.
`training.groups: [machine, buffer, dispatcher]` now explicitly chooses learned
groups, including entity override groups, from mixed fresh/model inputs. Unknown,
duplicate, empty and rule groups fail before allocation. Omission preserves the
historical fresh-group selection/fallback. Other model groups remain frozen.

Using the same compatible per-role model source gives exactly equal weights and
encoder state across same-role groups. Policy/action RNG use independent
`policy:<group>` domains from the experiment's policy root. Model initialization
does not import optimizer, replay or sampler continuation. Full resume remains
a separate strict operation.

Optional Experiment fields declare roots explicitly:

```yaml
world_seed: 707
environment_seed: 808
policy_seed: 101
training_environment_seeds: policy
```

World falls back to data_seed, environment falls back to world, and policy falls
back to seed. policy_seed cannot coexist with matrix.seeds, which already selects
per-entry policy roots. Fixed world/environment roots keep held-out worlds
independent of changing policy roots. Each explicit contract freezes exact
numeric split/replication roots and derivation domains under
`smartsom.experiment-seed-streams/v1` in check summaries, preparation and run
records. Training uses `environment:<index>:<episode>` from the frozen environment
root, or directly from the selected policy root when the path is policy.
The alternative explicit path is environment. Omission of all new fields keeps
historical V3/V4 derivation and default serialization unchanged.

## Initial control and unchanged task reward

For train-evaluate, `evaluation.initial_control: true` automatically retains the
start-of-training inference model. Native execution records an authoritative
initial_control stage after primary evaluation, using the same frozen held-out
cases. Completed stages are not repeated on resume; missing/changed evidence
rejects completion. Results are labelled INITIAL-CONTROL and excluded from primary
ranking. Tune reuses its existing initial-control evaluator and dependencies.
Explicit training.record_initial remains available separately.

Experiment `task_reward: {schema: smartsom.shipment-task-reward/v1}` exposes the
existing objective to learning, rule evaluation and controls. Its default weights
remain shipment 0.5, passing 0.3 and tardiness 0.2, with reference_jobs 128 and
reference_ticks 4096; the formula is unchanged. For learning the compiler places
the same contract in training.reward.task. Gamma must be one; enabled validation
must use custom return/max with ineligible failures. Team/role transforms cannot
stack. Rules use task=evaluate without a fake training section. Oracle quality
remains in trainer/evaluation reports; public traces retain their raw legacy
reward ledger.

## Subset, queue preference and strict stage barriers

File stems are stable IDs. Repeated --id selects an ordered whitelist. Repeated
--tag requires all existing output.tags labels:

```sh
smartsom check LOCAL_DIR --id medium_dqn_b --id medium_dqn_a --id medium_ppo --tag medium
smartsom batch-run LOCAL_DIR --id medium_dqn_b --id medium_dqn_a --id medium_ppo --tag medium --calibration-level online --no-background
```

Unknown/duplicate IDs, duplicate stems, empty selection and ID/tag conflicts fail
before allocation. Only selected dependencies are compiled. Tag-only selection
keeps stable filename order. --id/--tag apply to directory check and batch-run.

Use authored batch.stage for completion order, for example DQN files in stage 1
and PPO files in stage 2. All selected files declare stages together; learning
stages cannot mix rule/native tasks. Each learning stage reuses the original Tune
continuous queue, online feedback and broker. Ordered IDs within a stage are
admission preference, permit resource-blocked-head skipping and parallel cohorts,
and retain existing CPU-before-CUDA ordering. Completion order is not implied.
Failed/stopped stages hold later stages; failure retry is explicit.

Optional `batch.requires: [earlier_file_id]` requires a selected file in a strictly
earlier stage. Missing, duplicate and same/later dependencies fail statically.
This is not a generic DAG or expression language. Selection, resolved order,
stages and dependencies enter the parent execution digest; child scientific
identities remain unchanged. Resume consumes the frozen plan without rescanning
or changing selection. Scoped Tune children retain the whole parent's digest and
exact stage entry IDs. No truncated parent plan substitutes for the original.

Multiple learning stages run only the current stage's smoke/calibration after its
barrier. Online/off adds no long isolated calibration. Static checks inspect all
selected inputs before launch. Existing observed process-tree RSS high-water
marks remain charged across continuation restarts, including growing DQN replay;
a short probe does not certify a future full-replay memory ceiling. Live resource
guards and reserves remain active.

One standard parent Rich view and lifecycle is preserved:

```sh
smartsom monitor ACTUAL_RUN_DIRECTORY --once --log-format json
smartsom stop ACTUAL_RUN_DIRECTORY
smartsom resume ACTUAL_RUN_DIRECTORY
smartsom resume ACTUAL_RUN_DIRECTORY --retry-failed
```

Use the actual directory returned by batch-run. Monitor is read-only. Local
engineering evidence, full CI and independent review remain distinct from formal
research execution. No new Mover algorithm or physical transition is introduced.
