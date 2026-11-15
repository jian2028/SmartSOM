# Historical authoring archives

These directories are reference/test fixtures, not maintained launch campaigns.
`experiments/current` is the maintained campaign.

* `legacy_runtime`: byte-preserved V2, implicit-SL and diagnostic authoring from
  the development working tree, including its uncommitted edits. A configuration
  snapshot preserves relative references. Its manifest records original hashes.
* `v3_1e91c81`: exact V3 authoring/scenario/algorithm files and generator from
  commit `1e91c81`, with a source manifest.

Archiving does not migrate outputs, resume checkpoints, or modify any running
server checkout. Native-runtime recipes are not advertised as executable by the
new framework. Archived Python generators are historical test fixtures only.
