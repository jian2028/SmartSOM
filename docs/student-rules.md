# Student rule development

Custom rules use a named, versioned registration and detached public decisions.
They require no Torch, Ray or learner. The Algorithm YAML selects a registration;
an explicit CLI `--extension-module` authorizes importing the Python module.
YAML does not contain a Python import path.

## A Machine rule

The following is an inline development example; this guide does not create a
student module or experimental input:

```python
from smartsom.algorithms.rule_registry import register_rule


class RushThenSPT:
    def __init__(self, *, parameters, seed):
        if parameters:
            raise ValueError("this version accepts no parameters")

    def choose(self, request):
        selected = min(
            request.candidates,
            key=lambda candidate: (
                -candidate.features.get("rush", 0.0),
                candidate.feature("processing_time_scaled"),
                candidate.identity,
            ),
        )
        return selected.action


def make_rule(*, parameters, seed):
    return RushThenSPT(parameters=parameters, seed=seed)


register_rule(
    "student.rush_then_spt",
    "1",
    make_rule,
    roles=("machine",),
)
```

Select it in an Algorithm's Machine setting:

```yaml
machine:
  default:
    kind: rule
    name: student.rush_then_spt
    version: "1"
    parameters: {}
```

Load the module explicitly for check/run:

```sh
smartsom check experiment.yaml --extension-module student_rules
smartsom run experiment.yaml --extension-module student_rules
```

The module must be importable in the current Python environment, for example from
an installed project or an explicit `PYTHONPATH`. Registration code should define
rules at import time, not start an experiment. Keep helper sources available to
spawned workers and declare them with `source_files` when registering. A changed
helper must invalidate the same frozen identity as a changed main module.

A rule factory accepts keyword arguments `parameters` and `seed`. Validate your
own accepted parameter keys and types. Use the supplied seed for rule randomness;
do not generate or change the external Workload. Registered names cannot replace
built-in rules or silently replace an existing name/version with different code.

## Public request and actions

`PublicRuleRequest` includes:

- Current `tick`, `stage`, role and stable owner ID.
- A tuple of **legal** `PublicRuleCandidate` values with stable identity, semantic
  `action` and a read-only map of named features.
- Detached public `observation`, selected prefix and required count.
- `job(job_id)` for a detached copy of an already-public Job; unknown/future IDs
  raise an error.

The contract is `smartsom.public-rule/v1`. A rule has no simulator handle, hidden
quality, future arrival list, unrealized outage schedule or methods to advance
physical time. Mutating the detached observation cannot mutate simulator truth.
An action must equal a current legal candidate's semantic action. Returning a
candidate index, incompatible action or stale choice fails explicitly.

| Role | Semantic choice |
| --- | --- |
| Machine | One legal Job/mode choice or legal wait choice |
| Buffer | One legal Job selection; the runner handles the required ordered prefix |
| Dispatcher | A legal semantic target or wait choice |
| Mover | A legal movement/wait action |

Choose `candidate.action` rather than rebuilding its envelope from array slots.
Job, machine, buffer, vehicle and operation identities remain stable semantic
identities. `request.prefix` describes previous selections in a conditional
Buffer decision; do not select an already-removed candidate. A deterministic
request can have one legal candidate and still uses the same interface.

Named features depend on role:

| Role | Features |
| --- | --- |
| Machine and Buffer | `remaining_operations_scaled`, `due_slack_scaled`, `priority_scaled`, `waiting_time_scaled`, `replacement_attempt_scaled`, `quality_unknown`, `quality_pass`, `quality_fail`, `processing_time_scaled`; public rush when present |
| Machine additionally | `mode_time_scale`, `mode_error_rate`, `normal_mode` |
| Dispatcher | `has_target`, `travel_time_scaled`, `target_x_scaled`, `target_y_scaled`, `carrying_job`, `source_supply_scaled`, `source_reserved_scaled`, `inventory_scaled`, `capacity_scaled`, `finite_capacity`, `full_capacity`, `current_target` |
| Mover | `dx`, `dy`, `distance_to_target_scaled`, `has_target`, `carrying_job`, `in_service`, `next_cell_is_port`, `next_cell_occupied` |

`candidate.feature(name)` raises for unavailable names. Scaled values use the
projection's declared time/count scales; they are not raw seconds or counts.
`request.time_scale` exposes the public time scale. For optional rush data,
`candidate.features.get("rush", 0.0)` treats an absent flag as normal. Rush itself
does not force a scheduling choice. Agent decisions receive public realized
state, not offline V classes or initial statistical-history samples.

## Stateful rules and recovery

A stateful rule declares `stateful=True` and implements `reset()`, `state_dict()`
and `load_state_dict(state)` in addition to `choose()`:

```python
class CountingRule:
    def reset(self):
        self.decisions = 0

    def choose(self, request):
        self.decisions += 1
        return min(request.candidates, key=lambda c: c.identity).action

    def state_dict(self):
        return {"decisions": self.decisions}

    def load_state_dict(self, state):
        value = state["decisions"]
        if type(value) is not int or value < 0:
            raise ValueError("invalid rule state")
        self.decisions = value
```

The factory still needs the `parameters`/`seed` signature. Rule state and parameters
must be finite JSON objects. Include RNG state or other persistent learning-free
state if it affects future decisions; restore it instead of seeding again.
Reset initializes a new episode, while load restores a saved state where the
native lifecycle supports continuation. Evaluation interruption restarts its
stage and does not promise exact within-case continuation.

Frozen run identity records registration name/version, declared roles, code
hashes, parameters and explicitly loaded module identities. These capabilities
are passed to background and batch workers. Missing code or code drift rejects
execution/recovery. Saved YAML is not permission to import a new arbitrary module.
Changing a rule implementation means a new version and a new scientific run;
editing code beneath an active frozen run does not update that run.

## Models and research extensions

Rules and learned Agents select from the same semantic legal choices. A central
PPO controller replaces the four independent policy declarations; resource mode
can combine learned role groups with rule partners under one explicit learner.
Observation, reward and encoder extensions have their existing name/version/hash
contracts and backend support checks. They are distinct from the rule registry.
This work supplies extensibility, not a concrete Social Learning implementation.
See [four-file workflow](four-file-workflow.md) for Algorithm ownership and the
[command workflow](command-workflow.md) for stop/recovery limits.
