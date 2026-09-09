# Paper and project reproduction roadmap

Status: fall execution plan, not proof of reproduction

## Weekly rule

From Week 2 onward, close at least one reproducibility record every week. A week
may close either a new original-setting reproduction or the SmartSOM adaptation
of a previously reproduced project; it does not require training a new model
from scratch every week.

The target is a staggered two-week pipeline:

```text
Week N: environment + original result (R1)
Week N+1: SmartSOM compatibility + small H/V extension (R2/R3)
```

Only three or four anchor methods receive full retraining on CARC. Other weekly
records may use an official checkpoint, a documented table/figure regeneration,
or a short bounded training run.

## Reproduction levels

| Level | Meaning | Required evidence |
| --- | --- | --- |
| R0 | source qualified | paper, repo, commit, license, environment and target result recorded |
| R1 | original-setting result reproduced | official checkpoint or bounded training reproduces a declared metric/figure within a stated tolerance |
| R2 | SmartSOM-compatible | method runs through a declared adapter on a semantically compatible profile; deviations are explicit |
| R3 | H/V extension | same adapted method runs on a paired, materialized SmartSOM H/V contrast |
| R4 | anchor-quality | full selected training/evaluation protocol, multi-seed results, failures and system costs retained |

A repository that only imports or launches is not R1. A result produced after
changing objective, horizon, completion semantics, or information visibility is
not an exact reproduction; record it as compatible/partial or failed.

## Selection policy

1. Search the local Zotero collection `Jin's Group / RL for Scheduling` first.
2. Prefer papers with an author/public repository, checkpoints, instances, and
   explicit reproduction commands.
3. Pin repository commit and inspect the license before copying code. Publicly
   visible code is not automatically reusable code.
4. Prefer checkpoint/table reproduction for weekly throughput; choose R4 anchors
   for scientific relevance and interface coverage.
5. Keep a primary and backup for every high-risk week. Abandon the primary after
   its predeclared time box rather than consuming the development milestone.

## Candidate portfolio

| Candidate | Role | Local evidence | Public code | Initial risk and use |
| --- | --- | --- | --- | --- |
| JobShopLib | static JSP reference, Rule/OR-Tools | existing local practice checkout | [repo](https://github.com/Pabloo22/job_shop_lib) | Low; modern MIT project and Week 2 backup/reference |
| Reijnen benchmark | JSP/FJSP/FAJSP environments, Rule/CP/DRL | Zotero `AKLBZDU5`; local research context | [repo](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances) | Low-medium; primary framework/CP anchor |
| FJSP-DRL | static FJSP DRL checkpoint | Zotero `YQ5TUGSC`; local mirror under SmartSOM3 external | [repo](https://github.com/wrqccc/FJSP-DRL) | Medium; old Python/Torch stack and no license located in initial audit; checkpoint first |
| DRL_ADD_JSSP | online arrivals, PPO-AC + GNN | add to Zotero if retained | [repo](https://github.com/Nour0602/DRL_ADD_JSSP) | Medium-high; explicit result scripts and checkpoints, but solver dependencies may block full comparison |
| Deep-MARL Dynamic JSP | public MARL anchor | Zotero `XMJKTDFW` | [paper repo](https://github.com/RK0731/Deep-MARL-for-Dynamic-JSP) | High; authors disclose a job-overstay simulator bug. Reproduce with disclosure, never treat its simulator as benchmark truth |
| DRLforDynamicScheduling | maintained project backup | web fallback | [repo](https://github.com/RK0731/DRLforDynamicScheduling) | Medium; project reproduction, not an exact reproduction of the older COR paper |
| FJSPT-Scheduler/HGS | FJSP with transportation | transport literature in Zotero | [repo](https://github.com/msh0576/FJSPT-Scheduler) | High; checkpoints and benchmark scripts exist, but old DGL/PyG stack and no license located in initial audit |
| JobShopLab | transport, buffer, breakdown and PPO | Zotero `VAP7DURR`; local copies and legacy artifacts | [repo](https://github.com/proto-lab-ro/jobshoplab) | Medium; strong feature-validation project; inspect code-license implications before reuse |
| Phase-I SO-MARL | prior-work baseline under H/V | Zotero `W5CIUEYD`; legacy project evidence | local legacy only | Medium-high; reproduce after matching its AGV/buffer/controller semantics; not the public MARL requirement |
| Hutter FJSSP-W suite | worker stretch profile | Zotero `X2IJRAPN` | [repo](https://github.com/jrc-rodec/FJSSP-W-Competition) | Medium; open benchmark, but worker cannot delay V1 paper |

## Anchor portfolio

Provisional R4 anchors, subject to R1 success and adapter admissibility:

1. Reijnen framework Rule/CP result for static reference and profile coverage.
2. FJSP-DRL for a static FJSP DRL anchor.
3. DRL_ADD_JSSP for an online-arrival DRL anchor.
4. Deep-MARL Dynamic JSP for the public MARL family, with the disclosed simulator
   defect treated as part of the reproduction finding.

FJSPT-Scheduler and JobShopLab are primary feature/profile validations. Promote
one to R4 only if transport/buffer becomes part of a headline empirical claim.
Phase-I SO-MARL is a secondary R3/R4 prior-work baseline once semantically ready.

## Week-by-week pipeline

| Week | R1 close target | R2/R3 follow-up | Backup/time-box |
| --- | --- | --- | --- |
| 1 | reconstruct one Phase-I scenario/table mapping from source artifacts | record deviations required by the journal extension | JobShopLib hand-checkable schedule; retrospective only |
| 2 | JobShopLib static JSP Rule/OR-Tools result | map entities/actions/metrics to the new shared core | Reijnen toy/static result; 6-hour setup cap |
| 3 | Reijnen static JSP/FJSP Rule/CP result | run the same fixture through SmartSOM and verify JSP-as-one-mode equivalence | JobShopLab static dispatch result; 8-hour cap |
| 4 | FJSP-DRL official checkpoint result | adapt to compatible P0 FJSP observation/action/metric contract | L2D checkpoint or Reijnen DANIEL result; 10-hour environment cap |
| 5 | DRL_ADD_JSSP published/checkpoint result under arrivals | run a small paired SmartSOM arrival-burstiness contrast if P1 is ready | FJSP-DRL R3 or local DDQN paper data audit; skip solver dependency |
| 6 | Deep-MARL Dynamic JSP checkpoint/figure with bug disclosure | adapt only the common online-JSP subset | maintained DRLforDynamicScheduling project smoke; 10-hour cap |
| 7 | Reijnen FAJSP profile result | SmartSOM P2 assembly/DAG compatibility record | hand-checkable FAJSP instance or defer P2 from main table |
| 8 | FJSPT-Scheduler/HGS checkpoint and benchmark script | SmartSOM finite-transport compatibility; start JobShopLab buffer path | JobShopLab transport demo; old-stack cap 10 hours |
| 9 | JobShopLab transport+buffer policy/result | SmartSOM P4 compatibility and first small H/V slice | FJSPT R3; no worker implementation |
| 10 | Phase-I SO-MARL baseline if P4 semantics match | compare with public MARL on only the common admissible profile | JobShopLab R3 or Deep-MARL R3 |
| 11 | first CARC R4 anchor rerun | transfer/H/V result for a second anchor | checkpoint-only evidence if training exceeds frozen budget |
| 12 | second/third selected R4 anchor rerun | regenerate one manuscript table from frozen artifacts | reduce seeds only by protocol revision, never silently |
| 13 | clean-environment relocation rerun of one anchor | reproduce manuscript result from manifests only | critical advisor-requested reproduction |
| 14 | independent regeneration of one primary table/figure | verify corrected result after any approved rerun | Thanksgiving: no new high-risk environment |
| 15 | one-command reproducibility-package smoke | audit source-to-claim chain for the submission result | withdraw or narrow any claim that cannot be regenerated |

## Weekly reproduction record

Create one record per attempted project/result with:

```yaml
id:
paper_zotero_key:
paper_doi_or_url:
repository_url:
repository_commit:
license_status:
environment_image_or_lock:
target_result:
reproduction_level:
status: complete | partial | failed | blocked
expected_value_or_artifact:
observed_value_or_artifact:
tolerance_or_acceptance_rule:
runtime_and_hardware:
deviations:
failure_class:
smartsom_profile:
adapter_mapping:
hv_extension_artifact:
next_decision:
```

Failed and partial records count as research evidence but do not satisfy the
weekly completion target unless the failure itself is independently reproduced
and documented as the declared target (for example, the disclosed MARL simulator
defect). Use the backup when the primary exceeds its time box.
