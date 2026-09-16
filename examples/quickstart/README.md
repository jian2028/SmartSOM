# Python training quickstart

Run from the repository root. Install the default MARL dependencies once:

```sh
uv sync --locked --extra learning-marl --extra cpu --extra tensorboard --inexact
source .venv/bin/activate
```

Train and independently evaluate the small `marl_micro` case:

```sh
python examples/quickstart/train_evaluate.py
```

Or run the two stages separately. Replace `RUN_DIRECTORY` with the path printed
by training:

```sh
python examples/quickstart/train.py
python examples/quickstart/evaluate.py RUN_DIRECTORY --checkpoint last --baseline spt
```

Training prints the experiment and checkpoint paths; evaluation prints its own
output directory and status. Outputs are local under the configured output root.
These examples exercise the training API, not a performance benchmark or a
promise that the learned policy completes every order.

`train.py` accepts `--steps`, `--seed`, `--num-envs`, `--device` and `--preset`.
Use `--help` for each script. For `rllib_micro` or `sb3_micro`, install
`learning-rllib` or `learning-sb3` instead of `learning-marl`.
To watch evaluation, install `studio` and use:

```sh
smartsom evaluate RUN_DIRECTORY --checkpoint last --render-mode human
```

See the [experiment guide](../../docs/experiments.md) for checkpoint selection,
configuration and resume, or [Template 1](../../docs/template1-replay.md) for a
rule-based production demo requiring no training.
