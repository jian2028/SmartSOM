# Public learning extension examples

These are short engineering demonstrations using the existing micro scenario.
Their default budget is 128 environment steps, with updates every 32 steps,
minibatches of 16, two PPO epochs per update, and
one independent evaluation replication. They do not modify the frozen Item 13
recipe, establish a performance target, or imply that the learned policy will
finish every episode. A non-completed evaluation retains its failure evidence
and has no makespan.

Install the optional dependencies for the route you intend to execute, using the
repository's locked environment. From the repository root:

```bash
uv run --locked --extra learning-sb3 python examples/extensions/sb3.py
uv run --locked --extra learning-rllib python examples/extensions/rllib.py
uv run --locked --extra learning-marl python examples/extensions/resource.py
```

Each script uses the public `load_preset`, `train_evaluate`, `evaluate` and
`audit_run` interfaces. `common.py` explicitly registers a real custom
feed-forward Torch encoder with its name, version, parameters, supported
backends and implementation source files. Actor and critic have separate encoder
parameters and different widths. The resource example additionally gives AGVs a
different architecture from machines; resources of the same role share their
role policy. The mask and semantic action mapping retain their original meaning.

The input is a fixed-shape Dict derived only from public observations. A team
reward transform and a resource AGV transform demonstrate the separate raw,
research and learner reward streams. The encoder is stateless apart from normal
Torch parameters, which the framework checkpoint saves. For a stateful public
observation/reward extension, see the state protocol in
[`docs/learning-extensions.md`](../../docs/learning-extensions.md).

Use `--output-root PATH` to retain all artifacts elsewhere, `--steps 256` for a
different explicit demonstration budget, or `--replications 5` for more paired
independent evaluation worlds. Every invocation creates a new attempt. No script
deletes prior runs or writes into a supplied checkpoint.

To evaluate a saved demonstration model, rerun its matching script with
`--checkpoint-source RUN_DIRECTORY_OR_CHECKPOINT`. This explicitly registers the
same custom implementation before checkpoint loading. Ordinary CLI evaluation
does not execute arbitrary Python from a model or configuration; a custom
registration must already be available in the evaluation process. Retain the
same example source when reproducing a checkpoint, because source changes alter
its extension identity.

The scripts print the true evaluation completion/failure counts and aggregate
audit status. A verified incomplete prefix is distinct from a complete schedule.
Exit status is nonzero for engineering or audit failures. Full end-to-end support
must be verified with the integrated backend drivers; network unit tests alone
are insufficient.
