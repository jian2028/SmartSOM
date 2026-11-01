"""Human-readable views of an already compiled plan; never execute a run."""

from pathlib import Path

import yaml

from smartsom.config.launch import flatten, scientific_documents


def _path(value):
    path = Path(value)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _list(values, limit=8):
    values = list(dict.fromkeys(str(v) for v in values))
    shown = ", ".join(values[:limit])
    return shown + (f" … (+{len(values) - limit})" if len(values) > limit else "")


def _common(values):
    values = list(dict.fromkeys(values))
    return values[0] if len(values) == 1 else "各任务不同（见 --verbose）"


def _method(member):
    algorithm = yaml.safe_load(dict(member.documents)["algorithm"])
    policies = algorithm["production"]["policies"]
    selected = set(algorithm["production"]["bindings"].values())
    selected.add(algorithm["production"]["default"])
    entries = []
    for name in sorted(selected):
        policy = policies[name]
        component = (
            policy["learner"]
            if policy["kind"] == "learning"
            else policy["implementation"]
        )
        target = component.get("target") or component.get("path")
        entries.append(f"{name}: {target}")
    return algorithm, entries


def _policy_label(algorithm):
    selected = {
        algorithm["production"]["default"],
        *algorithm["production"]["bindings"].values(),
    }
    names = []
    for name in sorted(selected):
        policy = algorithm["production"]["policies"][name]
        component = (
            policy["learner"]
            if policy["kind"] == "learning"
            else policy["implementation"]
        )
        target = component.get("target") or component.get("path")
        known = {
            "smartsom.learning.ppo.create_learner": "PPO",
            "smartsom.learning.dqn.create_learner": "DQN",
            "smartsom.policies.rule.create_policy": "Rule",
            "smartsom.policies.random.create_policy": "Random",
        }
        label = known.get(target, f"{name} (custom)")
        if policy["kind"] == "learning":
            observation = policy["observation"].get("target") or policy[
                "observation"
            ].get("path")
            label += " / " + {
                "smartsom.representations.base.observation.create_observation": "Base",
                "smartsom.representations.social.observation.create_observation": "Social",
            }.get(observation, "custom observation")
        names.append(label)
    return _list(names)


def _differences(members):
    rows = [
        flatten(scientific_documents({k: yaml.safe_load(v) for k, v in m.documents}))
        for m in members
    ]
    keys = sorted(set().union(*(r.keys() for r in rows)))
    missing = object()
    return [
        k
        for k in keys
        if any(r.get(k, missing) != rows[0].get(k, missing) for r in rows[1:])
    ]


def _budget(ex):
    if ex.run.tune != "off":
        return f"Tune {ex.run.tune}（候选预算见 --verbose）"
    if ex.run.task == "train":
        if ex.train_episodes is not None:
            return f"{ex.train_episodes:,} episodes / 任务"
        return f"{ex.train_ticks:,} physical ticks / 任务"
    if ex.run.task == "smoke":
        return f"smoke {ex.run.smoke_ticks:,} ticks / 方法"
    return f"{ex.run.task}（不配置训练预算）"


def render_check(plan, *, verbose=False):
    """Keep common settings once, bound task rows, and disclose heterogeneous plans."""
    if hasattr(plan, "files"):
        members = [r["compiled"] for r in plan.files]
        labels = [r["id"] for r in plan.files]
        source, output, operations = plan.directory, plan.output_root, plan.operations
    else:
        members = [plan]
        source = dict((name, path) for name, path, _ in plan.source_files)["experiment"]
        labels = [Path(source).parent.name]
        output, operations = plan.experiment.run.output, plan.operations
    experiments = [m.experiment for m in members]
    algorithms = [_method(m)[0] for m in members]
    # Compilation rejects empty selections; no placeholder success view is needed.
    methods = []
    for ex, algorithm in zip(experiments, algorithms):
        configured = _policy_label(algorithm)
        methods.append(
            _list(
                configured if method == "configured" else method
                for method in ex.run.methods
            )
            if ex.run.task in {"smoke", "evaluate"}
            else configured
        )
    differences = _differences(members)
    world = []
    for member in members:
        docs = dict(member.documents)
        workload = yaml.safe_load(docs["workload"])
        factory = next(
            path
            for name, path, _ in member.source_files
            if name in {"scenario", "factory"}
        )
        world.append(
            f"{Path(factory).name} / {member.machines} machines / {member.agvs} AGVs / "
            f"{member.orders} orders / arrival {workload['mean_interarrival']:g} / "
            f"conditions {_list(member.experiment.training_conditions)}"
        )
    ownership = []
    for member in members:
        roles = member.component_report.get("network_ownership", {})
        ownership.append(
            ", ".join(
                f"{role}={v['mode']}({v['networks']})" for role, v in roles.items()
            )
            or "无学习网络"
        )
    validation = [
        f"{e.validation.unit}: {_list(e.validation.milestones) or '无节点'}; "
        f"{_list(e.validation.modes)}"
        for e in experiments
    ]
    panels = [
        f"validation {e.validation_cases} worlds / test {e.test_cases} worlds; "
        f"sample repeats {e.validation.sample_action_seeds}; final test={e.final_evaluation}"
        for e in experiments
    ]
    budgets = [_budget(e) for e in experiments]
    total = ""
    if all(
        e.run.task == "train" and e.run.tune == "off" and e.train_episodes is None
        for e in experiments
    ):
        total = f"；总计 {sum(e.train_ticks for e in experiments):,} ticks"
    lines = [
        f"✓ 配置检查通过 · {len(members)} 个任务",
        f"入口：{_path(source)}",
        f"任务：{_common([e.run.task for e in experiments])} / control={_common([a['control'] for a in algorithms])}",
        f"策略：{_common(methods)}",
        f"场景：{_common(world)}",
        f"配置网络：{_common(ownership)}",
        f"Seed：{_list(seed for e in experiments for seed in (e.train_seeds if e.run.tune != 'off' else (e.run.seed,)))}",
        f"预算：{_common(budgets)}{total}",
        f"验证配置：{_common(validation)}",
        f"案例配置：{_common(panels)}",
        "组间科学差异："
        + (
            _list(
                (
                    k.removeprefix("algorithm.").removeprefix("decision_protocol.")
                    for k in differences
                ),
                limit=3,
            )
            or "无"
        ),
        "",
        "方法 / seed                         Pre STOP   Post STOP",
    ]
    for label, member, algorithm in list(zip(labels, members, algorithms))[:8]:
        variant = member.experiment.tracking.variant or label
        # Canonical paths already carry the seed; keep the method label readable.
        if "--" in variant:
            variant = variant.rsplit("--seed-", 1)[0].split("--")[-1]
        protocol = algorithm["decision_protocol"]
        switches = [
            "—"
            if algorithm["control"] == "centralized"
            else ("on" if protocol[s]["accept"]["allow_stop"] else "off")
            for s in ("pre", "post")
        ]
        seeds = (
            _list(member.experiment.train_seeds)
            if member.experiment.run.tune != "off"
            else str(member.experiment.run.seed)
        )
        lines.append(f"{variant} / {seeds}    {switches[0]}         {switches[1]}")
    if len(members) > 8:
        lines.append(f"… 另有 {len(members) - 8} 个任务，使用 --verbose 查看全部")
    tracking = _common(
        [
            f"{m.operations['tracking']['mode']} / {m.operations['tracking']['project']}"
            f"; TensorBoard={m.operations['tracking']['tensorboard']}"
            for m in members
        ]
    )
    origins = _common([m.origins["concurrency.max_concurrent"] for m in members])
    overrides = sum(len(m.overrides) for m in members)
    lines.extend(
        [
            "",
            f"运行：{operations['launch']['mode']} / {operations['concurrency']['mode']} / "
            f"最大并发 {operations['concurrency']['max_concurrent']}（来源：{_path(origins)}）",
            f"监控：W&B {tracking}（未连接或验证登录）",
            f"科学覆盖：{overrides} 项" + ("（见 --verbose）" if overrides else ""),
            f"输出：{_path(output)}",
        ]
    )
    if Path(output).exists():
        lines.append(
            "提示：输出目录已存在；启动会执行已有结果保护，本次未验证可恢复性。"
        )
    lines.append("未启动实验；未检查运行资源余量。详情 --verbose；完整结构 --json。")
    if verbose:
        for label, member in zip(labels, members):
            algorithm, entries = _method(member)
            detail = {
                "files": {
                    name: {"path": str(path), "sha256": sha}
                    for name, path, sha in member.source_files
                },
                "scientific_sha256": member.scientific_sha256,
                "differing_fields": differences,
                "overrides": list(member.overrides),
                "schedule": {
                    "task": member.experiment.run.task,
                    "train_ticks": member.experiment.train_ticks,
                    "train_episodes": member.experiment.train_episodes,
                    "seed": member.experiment.run.seed,
                    "training_conditions": list(member.experiment.training_conditions),
                    "validation": member.experiment.validation.model_dump(mode="json"),
                    "case_sets": member.experiment.case_sets,
                    "final_evaluation": member.experiment.final_evaluation,
                    "evaluation": member.experiment.evaluation.model_dump(mode="json"),
                    "drain_ticks": member.experiment.drain_ticks,
                    "wall_seconds": member.experiment.run.seconds,
                },
                "policy_entries": entries,
                "algorithm": algorithm,
                "components": {
                    k: v
                    for k, v in member.component_report.items()
                    if k != "network_ownership"
                },
                "network_ownership": {
                    role: {
                        "mode": v["mode"],
                        "networks": v["networks"],
                        "entity_count": len(v["entities"]),
                    }
                    for role, v in member.component_report.get(
                        "network_ownership", {}
                    ).items()
                },
                "operations": member.operations,
                "origins": member.origins,
            }
            lines.extend(
                [
                    f"\n--- {label} ---",
                    yaml.safe_dump(
                        detail, sort_keys=False, allow_unicode=True
                    ).rstrip(),
                ]
            )
    return "\n".join(lines)
