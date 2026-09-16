# Examples

Start from the repository root with `source .venv/bin/activate`.
For installation and a live four-machine demo, follow the [README](../README.md).

| What you want | Example | Requirements |
| --- | --- | --- |
| Watch a rule-controlled factory and replay it | [Template 1](../docs/template1-replay.md) | `studio` extra; no model needed |
| Train or evaluate from Python | [Quickstart](quickstart/README.md) | Learning extras for your chosen backend |
| Customize networks and rewards | [Extensions](extensions/README.md) | Matching learning backend |
| Inspect the historical 167/121-order comparison | [Historical replay](../docs/historical-replay.md) | Separate local evidence bundle; not runnable from a clone alone |

The Template 1 rule demo and the micro training example are different cases.
The historical comparison uses a frozen older runtime and checkpoint; it is not
an evaluation of the current simulator.
