# Current CI ownership inventory

The executable source of truth is `.github/ci/test-ownership.json`. Ubuntu Linux
and macOS are maintained platforms; local macOS evidence does not substitute for
hosted Linux runs. Native Windows remains outside this maintained union.

Historical test retirements are recorded in
the external `retired-tests.json` and `removed-files.json` listed in the [archive index](archive/index.json); shared control, replay, metrics, checkpoint and
Studio tests were migrated. Counts below describe selector ownership, not test
quality or measured runtime savings. `selection.py` rejects unowned test files.

| Suite | Selectors | Primary | Platforms |
| --- | ---: | --- | --- |
| fast | 61 | True | ubuntu-latest |
| gym | 1 | True | ubuntu-latest |
| learning | 11 | True | ubuntu-latest |
| tuning | 9 | True | ubuntu-latest |
| resource-learning | 4 | True | ubuntu-latest |
| usability | 2 | True | ubuntu-latest |
| studio | 7 | True | ubuntu-latest |
| control | 9 | False | ubuntu-latest, macos-latest |
| diagnostics | 1 | False | ubuntu-latest, macos-latest |
| macos-learning | 3 | False | macos-latest |
| learning-public | 1 | True | ubuntu-latest |

Real SDK tests explicitly choose W&B offline; other automated runs default to
disabled. The fast environment installs no optional frameworks. Native suites
require the modules actually installed by their declared extras. Unknown impact,
retired tests, dependency/workflow/selector changes expand to full verification.
No older RLlib/SB3 or online lifecycle result is inherited as current acceptance.
