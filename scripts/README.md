# Scripts

These are validation and maintenance tools, not the main entry point for new
users. Start with the [factory demo](../README.md#try-a-factory-and-replay-it) or
[Python examples](../examples/README.md).

Run from the repository root after `source .venv/bin/activate`. Most scripts
accept `--help`; the Template 1 validator runs its fixed cases immediately.
Generated evidence belongs in local `artifacts/` or `runs/` directories.

| Purpose | Entry points | Notes |
| --- | --- | --- |
| Verify Template 1 and generate recordings | `python scripts/validation/template1_validate.py` | Static, isolated disturbances and combined case; independent ledger plus replay audit |
| Train the Template 1 curriculum | `python scripts/validation/template1_train.py --help` | Requires learning extras; starts training unless a frozen selection is supplied; no accepted model yet |
| Verify transport, buffers, quality and events | `validate_transport.py`, `validate_buffers.py`, `validate_quality.py`, `validate_machine_events.py` | Focused physical checks; see [validation records](../docs/validation/) |
| Audit trained models and experiment lifecycle | `validate_learning.py`, `validate_resource_learning.py`, `validate_usability.py` | Require separately generated training inputs and the corresponding optional dependencies |
| Prepare and audit IDETC reference inputs | `prepare_idetc.py`, `validate_idetc.py` | Historical provenance and configured validation; see [guide](../docs/validation/idetc-integration.md) |
| Run a configured batch | `smartsom batch STUDY.yaml` | `run_study.sh` is a convenience wrapper |
| Restore the historical Rule/MARL comparison | `validation/historical_replay.py` | Requires a separate frozen evidence bundle; [instructions](../docs/historical-replay.md) |

Other files in `validation/` are shared auditors, source-evidence helpers or
external-reference checks used by these entry points and tests. They are retained
for reproducibility. Historical acceptance records describe their recorded source
versions, not automatic acceptance of current `main`.

The `configs/` directory includes both current grid examples and historical
reference recipes. Use the linked examples first; CP-SAT recipes are not runnable
through the current grid simulator.
