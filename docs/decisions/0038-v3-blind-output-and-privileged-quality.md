# 0038 — V3 blind OUTPUT and privileged shipment quality

Date: 2026-10-07
Status: User-authorized implementation; engineering verification separate.

Supersedes ADR0017 and ADR0020's V3 qualified-demand completion and Output-driven
replacement, and ADR0037's action/observation identities. Current V3 decisions
are v3.2 and physical metadata adds `output_semantics: blind-shipment/v1`.
The prior dispatch and learning liveness repairs remain. No SimulationV4 is added.
Historical recordings/results require their original source and contracts.

OUTPUT only receives. A finished UNKNOWN or PASS attempt ships its original
order once; OUTPUT performs no inspection, quality reveal, rejection or
replacement. Public observable quality stays as it was. Known early-inspection
FAIL still must go to scrap, with replacement retaining original identity,
release and due date. Inspection/disposal physics otherwise remain unchanged.

The core tracks a shipped-original set and exact shipment times; its completion
set now aliases that shipment identity. Qualified shipments form a distinct
private set, derived from latent accumulated defects. No private quality counter,
latent defect flag or quality component enters V3 snapshots/traces, events,
public summary, actor/critic observations or rule inputs. Explicit privileged
methods provide trainer/evaluation quality statistics; archived frozen scenario
inputs and continuation state remain privileged reproducibility data.

Flow, completion, finite termination, original-demand tardiness and throughput
use shipment identity/time, including bad shipments. Dynamic runs retain their
configured horizon. Censored makespan stays null if not every order ships.
Public replay quality metrics are unavailable. Privileged evaluation reports
separate good/bad shipments and passing rate (unavailable with no shipments);
it does not call bad shipments rejections. Old snapshots remain viewable under
their recorded semantics. Models, continuations and semantic replay with old
physical/decision identities reject rather than silently reinterpret results.
