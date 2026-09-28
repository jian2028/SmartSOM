# Historical configuration path migration — 2026-09-26

The 195 repository demonstration/verification configurations have moved under
`configs/test/`. Their schema versions and scientific inputs are retained.
References to repository data and output roots are rebased to the new location.

| Historical path prefix | Current path prefix |
| --- | --- |
| `configs/factories/` | `configs/test/factories/` |
| `configs/workloads/` | `configs/test/workloads/` |
| `configs/scenarios/` | `configs/test/scenarios/` |
| `configs/algorithms/` | `configs/test/algorithms/` |
| `configs/runs/` | `configs/test/runs/` |
| `configs/studies/` | `configs/test/studies/` |

This mapping applies to the historical files, not the user's newly saved
`configs/factories/large.yaml`. Bundled presets and Studio templates keep their
existing package paths. Current tests, scripts and usage documentation use the
new paths; frozen validation reports and ADRs retain their original commands,
commit identities, measurements and acceptance boundaries.

Historical local records are archived under `backup/2026-09-26/records/`, with
their original paths beneath it. Original manifests and reports are unchanged.
The archive's inventory, action ledger and `retention.json` describe removed
model/recovery files. Archived records are not resumable model packages.
