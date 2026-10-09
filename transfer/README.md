# SmartSOM transfer

Temporary transfer folder between the owner's Mac and Windows computers.

- Only README.md is eligible for Git tracking; all other contents are ignored.
- Transfer static scripts, frozen inputs, and checksum manifests only.
- Verify SHA-256 checksums after synchronization, then copy files to the intended experiment directory before running.
- Do not run experiments directly in this folder.
- Do not synchronize active checkpoints, live results, virtual environments, Git metadata, credentials, or private keys.
