# peer-pressure

Original 20-machine layout and workloads. Run `smartsom check peer-pressure/01-peer-on-off` before `smartsom run peer-pressure/01-peer-on-off`. Profiles only change operations.

Historical source: peer-pressure-pilot-20261028-v1 (274397ddaf949cabb747699552292ecb66d71073); owner-independent pilot additionally used its frozen dirty patch. New checkpoints are not compatible with those archives.

The original 602901–602908 panel was inspected at all checkpoints; it is retained as **validation**, never described as an unseen test. The separate 604901–604908 final test panel is a new protocol addition, not an old result. Case recipes verify exact physical-world digests and freeze full worlds before execution.

Training conditions, arrival intervals, 1M tick budget, 0/500k/1M nodes, reward divisor 1000 and original training seeds are retained. Initialization is role-template-identical; each independent entity owns its complete network and optimizer.

Results: `runs/peer-pressure/01-peer-on-off/trials/METHOD/seed-SEED/`; models are `last.pt`, optional `best.pt`, and validation-node `model.pt`; batch report is `report/report.md`. No formal run was launched by this migration.
