# Historical rule-credit experiments

These documents preserve original protocols, not current runtime options or new
launch authorization. Cloud PPO source is `4ee989c36690fd5f481cd6ea10df461642781f09`;
server Double DQN source is `b269f557300ad0084ea569a0e53361d194e26ae7`.

`source-index.json` records local archive locations, hashes and verification scope.
The cloud source archive was reconstructed from exact base `a39664c` plus the
handoff patch; its complete Git tree matches
`e1d038ed67a939f02540bf798bcb1cf9fbe03d77`. Server source was archived directly from
its recorded commit. Both include their own `uv.lock`, Python configuration,
source and original configs. Backup archives remain local and are not PR payloads.

To inspect or resume old work, extract the matching archive into a separate
location, use its Python version and locked dependencies, then follow its original
driver and complete-checkpoint contract. Preserve the original complete run
bundle and environment evidence. Do not import its checkpoint into current main.
The archive hash verifies preserved source; it does not establish that an arbitrary
historical checkpoint can evaluate or fully continue. Neither historical run was
stopped, modified or resumed during this migration.

The original cloud absolute dates and server operating guards describe those
launches. Do not copy expired deadlines into a fresh experiment. Original files
are retained verbatim; current migration differences are documented separately.
