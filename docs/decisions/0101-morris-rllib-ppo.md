# 0101 — PR42 Morris/E1a-current on official RLlib PPO

Status: user-authorized implementation; engineering acceptance recorded separately.

Supersedes the centralized-learning deferral in ADR0098 for PR42's E1a-current
controller only. PR42 was merged as `cd7074b` and is already the development
checkout's base. Preserve its CONWIP limits (Small 48, Medium 96), semantic
process/transport/ADVANCE order, public global observation, owner associations,
64×64 candidate scorer and separate 64×64 critic. ADVANCE is masked while a work
task is legal. This is not the older global network over Base reservation roles.

The maintained `morris-ppo` recipes use official single-agent PPO, EnvRunner,
Learner, GAE and Tune through the existing `smartsom` entry. Decision-step
iteration/update semantics and official initialization supersede the native
learner. Explicit PR42 hyperparameters map to their official names; other values
use locked Ray 2.60 defaults. There is no native execution fallback.

PR42's chosen recipes use the default overdue/failed-output reward, not the
separate historical E1a shipment/starvation reward. Preserve that objective and
its `0.001` learner scale in `create_current_reward`, before official GAE. Base/SL's
normalized J conversion does not apply. The default factory reward implementation
and simulator outcomes remain unchanged. All methods report J; full fixed sampled
PPO panels select best by minimum mean J, not by Morris training return.

One central decision maps an integer slot to the actual semantic candidate.
Selecting ADVANCE invokes the existing physical advance; it does not insert a
second learned WAIT. Truncation exposes the real post-advance public state for
bootstrap. No hidden fault/quality information is added. Actor/critic use the same
public global scales. Layout and inventory changes fail explicitly.

Gym Sequence observations retain real candidate rows in episode storage.
The public Gym vector-entry-point wrapper provides a one-factory batch dimension;
public ConnectorV2 hooks pad the current batch and its old categorical logits.
`episodes_to_numpy=False` retains varying lengths until that connector. The
original factorized scorer gathers real rows before neural computation. Padding
never removes a candidate; no custom PPO loss, optimizer or scheduler is added.
Peak memory still grows with the longest candidate row in a training batch and
must be measured on representative dense candidates. Each EnvRunner has one factory.

Official RLlib/Tune checkpoints and existing best/last references own restoration.
No native checkpoint migration or second model store is supported. Fixed worlds,
action seeds, final panels and committed event publication use the common runtime.
Current campaign scenarios, fixed panels and inherited Rule-calibrated drain
provenance are shared with Base/SL; these are not newly calibrated bounds.

Acceptance requires dense candidate score/gradient and memory probes, real PPO
updates, public training and fixed panels, best/last, official checkpoint recovery
followed by updates, affected performance comparison, Linux CI and independent
engineering/science/testing review. Short probes do not establish formal campaign
performance. No formal experiment, publication, merge or existing-process change
is authorized. Other ADR0098 deferrals remain.
