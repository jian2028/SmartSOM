# Base / SL role ablation

Nine arms, seeds 4101–4103, 1,000 episodes, balanced normal/low/medium/high worlds, original 20-machine map, drain 2000, reward divisor 1000. Validation nodes: 0/100/200/300/500/750/1000.

Mapping source: frozen `social_long.py` at a39664c, available in the local mac-dispatcher-six-arm checkout. Remote live run manifests and environment were not fetched: this is a source-recipe migration, not proof that every remote execution used these inputs. Original early-160k snapshot was selected at an episode boundary, not an exact cut; it is not relabeled as an exact tick result here.

Validation/test world namespaces and action seeds are kept distinct. Case recipes freeze full worlds during compilation. New checkpoints cannot resume old training. Compare historical receipts before making a reproduction claim.

`smartsom check base-sl/01-role-ablation`; run only after the source requirement for formal research is met. Results and reports mirror this study under `runs/`.
