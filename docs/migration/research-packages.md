# Research packages and network ownership

The current entry is `experiments/TOPIC/STUDY/VARIANT/experiment.yaml`. Scenario
(factory, jobs and failures) and Algorithm are explicit relative references.
Reuse shared files under `configs/{scenarios,algorithms}`; put study-specific inputs in
`common/`, Python extensions in `extensions/`, and analysis source in `analysis/`.
No YAML inheritance or name registry is required.

```sh
smartsom check network-sharing/01-owner-sharing --profile local
smartsom check peer-pressure/01-peer-on-off --profile server
# Formal runs still require integrated main or an explicit source tag.
smartsom run network-sharing/01-owner-sharing --profile server --background
smartsom runs list
```

A study directory expands variants and training seeds into one admission queue.
`--max-concurrent` is the total independent-trial ceiling. A topic directory keeps
study results separate. A canonical experiment file and its variant directory both
expand train_seeds. Ad hoc files outside the research layout retain the single-task
API (run.seed); old flat directories
must be reorganized into canonical entries. Short-name collisions list candidates.
No check command connects W&B or trains. W&B remains online by default; tests must
select offline or disabled explicitly.

Results mirror `runs/TOPIC/STUDY`. Each `trials/VARIANT/seed-N/` contains `last.pt`,
optional validation-selected `best.pt`, `validation/`, `evaluation/`, `inputs/`,
`source/`, `continuation/`, `tracking/` and logs. A checkpoint contains all entity
networks, not one user-managed file per machine. `last.pt` is the latest committed
update, not proof that a run finished. Copy the whole stopped result bundle for
continuation, with matching source, Python and package versions. Weights alone
cannot continue training. Use `smartsom resume PATH`; never rerun into an existing
output or edit its frozen inputs. Different platforms may produce numerical drift.
Standalone Tune retains its existing saved-plan/seed lifecycle and admission rules.

`report/report.md`, `metrics.csv` and `summary.json` are rebuilt from local results.
Missing metrics are explicitly missing, sample/argmax are separate, and seed coverage
is reported. No automatic algorithm-superiority claim is generated.

## Algorithm configuration

Inside a learning entry under `production.policies`:

```yaml
parameter_sharing:
  default: shared
  roles:
    machine: independent
    dispatcher: shared
role_params:
  machine:
    learning_rate: 0.0001
learner:
  target: smartsom.learning.ppo.create_learner
  params:
    learning_rate: 0.0003
    epochs: 3
```

Roles are machine/dispatcher/pre/post/input. Shared means within one role.
Independent means complete model and optimizer ownership; DQN also separates
replay and targets. Same-role entities receive identical initial parameters and
configuration. The Dispatcher owner is an AGV, not a transient candidate index.
PPO/DQN do not have to use equal compute budgets. Rule/Mover bindings remain
independent of network sharing. Custom learner factories declare
`PARAMETER_SHARING = ("shared", "independent")` and `ROLE_PARAMS = True` only when
their implementation supports these semantics.

## Seeds, panels and milestones

`train_seeds: 3` expands to 101/102/103; lists retain their exact identities.
`validation.unit` is `episodes` (default) or `physical_ticks`; milestones may
include zero. Exact ticks require a tick budget and `budget_boundary: cut`.
PPO bootstraps a truncated rollout; DQN never invents a terminal transition.
Explicit `final_evaluation: true` evaluates the chosen validation checkpoint on
the test panel after training. Migrated research packages enable this; bounded
legacy diagnostic recipes retain their existing opt-in behavior.

`case_sets.validation` and `case_sets.test` reference JSON inventories. A
`smartsom.fixed-cases/v1` inventory records split, factory/workload digests and
cases with id, condition, seed, explicit action_seeds and full scenario snapshots.
A `smartsom.case-recipe/v1` inventory instead carries drain_ticks and each world's
expected scenario_sha256. It is expanded, checked and frozen during compilation;
it cannot silently regenerate a different historic world. The two splits cannot
contain identical physical worlds. Default generated panels retain their fixed
namespaces and freeze executed worlds in the case store.

## Historical migrations and limits

- `peer-pressure/01-peer-on-off`: original 20-machine map, intervals 28.8/7.2,
  original seeds, 1M ticks, 0/500k/1M nodes, drain 4000 and reward divisor 1000.
  OFF retains the social architecture and empties peer inputs through the frozen
  encoder adapter. Original 602901–602908 worlds were compared to frozen snapshots.
- `network-sharing/01-owner-sharing`: low/high pressure × shared/Machine-independent/
  Dispatcher-independent, original single seed 2026102901. Original owner pilot
  had a dirty source patch; it is not represented as a clean historical commit.
- `base-sl/01-role-ablation`: nine source-defined arms, seeds 4101–4103, 1,000
  episodes and original world/action namespaces. The local source recipe is
  available; the remote run manifest/environment has not been fetched. Early-160k
  episode-boundary selection is not relabeled as an exact-cut result.

The peer and owner pilots had one repeatedly inspected panel, not an independent
final test. Their old panel is labeled validation here; a new distinct final test
panel is an explicit protocol addition, not historical evidence. These packages
create new runs; no old model or training state is automatically migrated. Maps
remain editable by Studio, but changing a map changes the scientific identity.
Historical outputs and frozen sources remain at their original locations.
