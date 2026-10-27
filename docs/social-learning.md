# Social observation on the reservation Base

This is the implementation specification for the user-approved observation-only
SL increment. The common starting implementation is `87b4e7e`; existing Mac and
server experiments retain their own frozen identities. This document does not
claim that the reconstructed Mac Base and the older server Base are identical.
See [ADR0054](decisions/0054-base-social-observation.md).

## Selection and invariants

The Base algorithm accepts `social_arm`: `base` (default), `machine`,
`dispatcher-pickup`, `dispatcher-delivery`, `dispatcher`, `pre`, `post`, `input`,
or `all`. `social_pool` is `attention` (default) or `mean-max`. These selectors
only affect actor observations/parameters. Base already contains operational
coordination information. The increment is not a replacement for that information.

The same scene, public Base inputs, width, seeds, masks, reward scaling, PPO
settings, critic, physical clock and fixed Mover must be used across a comparison.
STOP/WAIT exist only when the common Base enables them. Dispatcher still has one
parameter group with two stage heads. A single-stage intervention can indirectly
affect its other stage through subsequent shared-parameter training.

No scientific training schedule is selected here. The existing bounded Base
diagnostic entrypoint consumes these fields from its algorithm YAML and freezes
that file. Calibration identities still must match; do not reuse a mismatched
receipt. Historical overnight scripts are not SL launchers. Development tests
are not nine-arm performance experiments. Formal runs need integrated main or an
explicit tag, paired seeds/worlds, fixed budgets and validation-only selection.

## Detached observation schema

`BaseDecision.social` is absent/None for disabled roles/stages. Otherwise it holds
one candidate-keyed entry per semantic action, including STOP/WAIT. Each entry
contains zero or more `SocialLink`s: a semantic relation ID (audit only), kind,
48-field public query, terminal bit, peer count and a tuple of public peer rows.
Entity and relation IDs are never numerical network features. Candidate keys
align the data after permutations; they never encode priority.

All rows use the existing fixed scales: time/200, static distance/50, count/20
by default, taken from the common algorithm. No per-arm fitted normalization,
clipping, NaN sentinel or hidden state. Feature vectors are zero-padded to 48.
An empty set produces a fixed zero pooled vector. Terminal=1 means the released
job has no next processing step; terminal=0 with zero peers is a different state.
Missing active SL observations are an error, not silently treated as no peers.

| Peer kind | Ordered fields before zero padding | Scope |
| --- | --- | --- |
| resource | down, processing, blocked; pre resident, post resident, post ready; pre capacity, post capacity; inbound count, outbound count; inbound quota, outbound quota; pre nominal work, current remaining standard work | Machine downstream; Pre alternatives |
| station | Unmodified Base `station_vectors(pre_owner)`: pre/process/post tagged rows, including already-allowed public summaries | Pre downstream; Post/Input direct next stations |
| vehicle | not-arrived, arrived-awaiting-service, static distance, current continuous arrival wait | Dispatcher effective reservations at this candidate owner |

Resource rows exclude remote urgency/job details. Station rows retain information
already allowed to the Buffer Base. Only current standard work is used; hidden
actual remaining work, latent quality, future releases/faults and repair calendars
are excluded. Count metadata is the number of compatible stations for resource/
station links. Vehicle count metadata is zero: the Base candidate already retains
the corresponding reservation count.

## Role association and event timing

* Machine: one resource link for each job×mode. Skip the consecutive operations
  compatible with the current machine; include all stations capable of the first
  external operation, including down/full stations. Different modes share peer
  rows but have distinct candidate queries. Terminal jobs have no Output peer.
* Empty/loaded Dispatcher: one vehicle link per legal source/destination, only
  other cars with effective reservations of the same pickup/delivery kind.
  Unaccepted intentions and carried-job details are excluded. No downstream
  extension for loaded Dispatcher. WAIT has no synthetic destination/link.
* Pre: each applicant/unload vehicle's job has two separately tagged links:
  downstream after mandatory local continuation (Base station rows), and other
  stations capable of the job's current operation (resource rows). Deduplicate
  stations within each relation, not across distinct relations. STOP reads the
  same applicant-job links; it cannot redirect a vehicle.
* Post/Input service: one station link per concrete job–AGV candidate, using the
  job's next/first operation. Post stock already points to its next step: do not
  skip it a second time.
* Post/Input accept: candidates have no assigned job. Each applicant sees the
  same set of local job→downstream links, conditioned by its own candidate query.
  Post includes resident pre, processing and post jobs as the Base supply does;
  skip mandatory local continuation only for jobs not yet in post. Input uses
  resident jobs. Post STOP reads this same unbound local set. No new Input STOP.

Snapshots are assembled when the existing protocol yields each request. Within
a negotiation round all Dispatcher applications precede Buffer acceptance; only
the next round sees newly accepted peers. No clock advance or order change.
Vehicle arrival wait is zero off the owner's legal ports and after service starts.
At a port, it is now minus the later of reservation acceptance and last actual
entry. Existing `accept` initializes `arrived_at` and `advance` resets it on entry;
checking stage/location prevents treating travel as arrival waiting. Leaving and
re-entering starts a new segment. The current one-tick service is finished before
the next Dispatcher epoch; servicing peers are excluded, not given a third stage.

## Network and rollout interface

First construct all five unchanged Base networks, then attach enabled actor
branches. Thus common Base weights and critics have identical initial values for
the same seed. No SL parameters are allocated for `base`. Critic forward/input
and loss are unchanged; each run trains its own critic.

With common hidden width H, the social branch has a shared per-peer MLP
`48+kind(3) → H → H`. A single-head dot-product attention uses a projected query
from Base candidate H, Base context H, and link query 48; key/value are learned
H projections, scaled by sqrt(H). Only nonempty sets enter softmax. The pooling
control uses the same encoded rows, masked mean and componentwise max followed
by `2H → H`, with the identical zero-empty rule.

For Machine/Buffer, concatenate each link's query, pooled peers and metadata
(kind one-hot, terminal, scaled count), encode to H, then pool links conditioned
on candidate+context. This preserves Post/Input's unbound per-job associations
and Pre's two relations. For Dispatcher the single peer pool is used directly:
no peers is exactly zero, with no learned empty embedding or duplicated count.

Enabled candidate heads score `[candidate H, context H, social H]`; STOP scores
`[context H, own allowed social H]`. WAIT has a second pool over legal sources,
encoding `[source Base candidate H, source social H]`, queried by own context H;
zero-peer sources remain included, and zero legal sources returns zero. WAIT
scores `[context H, source-summary H]`. Disabled stages use their original heads.
All pools use the selected attention/fixed mode and preserve permutation symmetry.

Rollouts retain detached social payloads, never recompute them from later world
state. Batched padding masks cover candidates, links and peers. Batches
pack peer rows only for actual candidate-links, then scatter small H-dimensional
summaries into candidate/link slots; STOP's larger link inventory must not expand
every other candidate's raw peer storage. SL checkpoints
use `smartsom.social-ppo/v1`, store the complete selectors/options and reject
Base/SL or arm/pool mismatches. Old Base checkpoints lacking the two optional
selector fields remain readable only as default Base, with all other options
checked. No cross-arm warm start is implied.

Attention versus same-input pooling isolates an aggregation choice, not exact
parameter capacity. Parameter-matched and relation-shuffle controls require a
separate precise experimental recipe before causal attribution. Reward shaping,
temporal history, trust and second-hop propagation are outside this increment.

## Engineering acceptance

Verify scopes and feature order with constructed worlds, terminal/local-chain
cases, down/full peers, reservation timing and arrival wait. Perturb hidden truth
and unrelated peers without changing observations. Verify semantic permutations,
padding/empty sets, STOP/WAIT direct dependence, mixed Dispatcher stages, disabled
Base equivalence and unchanged critic. Run real finite PPO updates, checkpoint
round trips and bounded CLI/config entrypoints. Review code independently before
publishing. These checks establish executable behavior, not SL learning gains.

The small engineering recipe is `configs/test/social/experiment.yaml` (eight
orders, width eight, one PPO epoch). Example bounded commands, with fresh local
output directories:

```sh
uv run --no-sync python -m smartsom.experiments.base_diagnostic check configs/test/social/experiment.yaml
uv run --no-sync python -m smartsom.experiments.base_diagnostic calibrate configs/test/social/experiment.yaml --output runs/sl-calibration --seconds 120 --cases 1 --max-drain 4000
uv run --no-sync python -m smartsom.experiments.base_diagnostic train configs/test/social/experiment.yaml --output runs/sl-smoke --calibration runs/sl-calibration/calibration.json --ticks 300 --seconds 120
uv run --no-sync python -m smartsom.experiments.base_diagnostic evaluate configs/test/social/experiment.yaml --output runs/sl-evaluation --calibration runs/sl-calibration/calibration.json --method ppo --checkpoint runs/sl-smoke/last.pt --cases 4 --split validation --selection sample
```

This runner retains the Base diagnostic's reward processing; the historical
overnight reward/1000 wrapper is a separate frozen driver. Do not compare runs
from different drivers/scales as if only SL changed. Formal schedules and any
common long-training driver integration remain separate from this smoke recipe.
