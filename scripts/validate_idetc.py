"""Frozen 60-run engineering acceptance, never a learning or benchmark framework."""

import argparse
import csv
import json
import multiprocessing
import subprocess
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from uuid import uuid4

from validation.idetc_audit import audit_run, require
from validation.idetc_inputs import CASES, ROOT, verify_source

from smartsom.config import resolve_study
from smartsom.config.codec import digest
from smartsom.experiments import run_batch
from smartsom.experiments.batch import _latest, _load_plan, execution_identity
from smartsom.experiments.evidence import source_identity, write_json

STUDY = ROOT / "configs/studies/idetc_spt.yaml"


def require_integrated_source():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    require(
        git("branch", "--show-current") == "main", "formal acceptance requires main"
    )
    # Research/governance edits are explicitly outside this implementation.
    dirty = (
        subprocess.check_output(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    unexpected = [
        row
        for row in dirty
        if row and row[3:] != "AGENTS.md" and not row[3:].startswith("docs/papers/")
    ]
    require(not unexpected, f"commit acceptance inputs/code first: {unexpected}")
    return git("rev-parse", "HEAD")


def audit_task(row):
    try:
        result = audit_run(
            Path(row["run_dir"]),
            expected_jobs=100,
            expected_operations=283 if row["case_id"] in ("S00", "S10") else 291,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    return {"entry_id": row["entry_id"], **result}


def summarize(rows, audits, paper):
    by_id = {a["entry_id"]: a for a in audits}
    groups, paired = defaultdict(list), defaultdict(dict)
    output = []
    for row in rows:
        audit = by_id.get(row["entry_id"], {})
        successful = row["status"] == "completed"
        verified = successful and audit.get("status") == "passed"
        record = {
            **row,
            "audit_status": audit.get("status", "not_run"),
            "audit_error": audit.get("error"),
            "accepted": verified,
        }
        output.append(record)
        groups[(row["case_id"], row["algorithm_id"])].append(record)
        if verified:
            paired[(row["case_id"], row["replication"])][row["algorithm_id"]] = set(
                audit["passed_job_ids"]
            )
    paired_checks = []
    for case in CASES:
        for replication in range(5):
            values = paired[(case, replication)]
            available = all(f"SPT-M{i}" in values for i in range(3))
            passed = (
                available and values["SPT-M0"] >= values["SPT-M1"] >= values["SPT-M2"]
            )
            paired_checks.append(
                {
                    "case_id": case,
                    "replication": replication,
                    "status": "passed"
                    if passed
                    else "failed"
                    if available
                    else "incomplete",
                }
            )
    references = {(p["case_id"], f"SPT-{p['quality_mode']}"): p for p in paper}
    aggregate = []
    for key, group in sorted(groups.items()):
        complete = [r for r in group if r["status"] == "completed"]
        metrics = {}
        for metric in ("makespan", "passing_rate"):
            values = [r[metric] for r in complete]
            metrics[metric + "_mean"] = mean(values) if values else None
            metrics[metric + "_sample_sd"] = stdev(values) if len(values) > 1 else None
        aggregate.append(
            {
                "case_id": key[0],
                "algorithm_id": key[1],
                "completed": len(complete),
                "failed_or_incomplete": len(group) - len(complete),
                "audited": sum(r["accepted"] for r in group),
                **metrics,
                "paper_reference": references[key],
            }
        )
    identities = {
        (r["case_id"], r["algorithm_id"], r["replication"], r["variant_id"])
        for r in rows
    }
    expected = {
        (c, f"SPT-M{m}", n, "control")
        for c in CASES
        for m in range(3)
        for n in range(5)
    }
    accepted = (
        len(rows) == 60
        and identities == expected
        and all(r["accepted"] for r in output)
        and all(c["status"] == "passed" for c in paired_checks)
    )
    return {
        "accepted": accepted,
        "completed": sum(r["status"] == "completed" for r in rows),
        "audited": sum(r["accepted"] for r in output),
        "rows": output,
        "groups": aggregate,
        "paired_quality": paired_checks,
    }


def report_markdown(result):
    lines = [
        "# IDETC SPT 集成验收",
        "",
        f"验收通过：{result['accepted']}；执行完成 {result['completed']}/60；replay 与 observation 审计通过 {result['audited']}/60。",
        f"证据类别：{result['evidence_kind']}；源码：`{result['source']['git']['commit']}`。Linux 实际执行：待验证。",
        "",
        "均值与样本标准差仅使用成功执行的样本；审计数独立列出。失败指标为空，不计作零。五次重复仅用于工程集成观察。",
        "",
        "| Case | 档位 | 完成/审计 | Makespan 均值 ± SD | 合格率均值 ± SD | 论文 makespan | 论文合格率 ± SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def value(v):
        return "—" if v is None else f"{v:.4f}".rstrip("0").rstrip(".")

    for row in result["groups"]:
        paper = row["paper_reference"]
        lines.append(
            f"| {row['case_id']} | {row['algorithm_id']} | {row['completed']}/{row['audited']} | {value(row['makespan_mean'])} ± {value(row['makespan_sample_sd'])} | {value(row['passing_rate_mean'])} ± {value(row['passing_rate_sample_sd'])} | {value(paper['makespan'])} | {value(paper['passing_rate'])} ± {value(paper['passing_rate_sample_sd'])} |"
        )
    lines.extend(
        [
            "",
            "逐次结果见 runs.csv / report.json；每个运行的审计见 audits/。",
            "",
            "输入路线已包含旧 task-map edits。当前使用加工优先 SPT、最短空载加载货运输、专属预订与 blocking、逐工序配对质量随机数，以及最后实际出库时间。旧实现的候选接受/拒绝、全局质量 RNG 和终态时钟处理不同。三档在本批输入上没有实际取整差异。论文数字仅作参考，不作为通过阈值。",
            "",
            "质量包含关系检查要求每个配对世界中 M0 合格集合包含 M1，M1 包含 M2；不要求整厂 makespan 单调。",
            "",
            "失败与缺口：",
        ]
    )
    failures = [r for r in result["rows"] if not r["accepted"]]
    lines.extend(
        f"- {r['case_id']} {r['algorithm_id']} rep={r['replication']}: {r.get('audit_error') or r.get('failure_reason') or r['status']} ({r.get('run_dir')})"
        for r in failures
    )
    if not failures:
        lines.append("- 无运行或 replay 失败。")
    return "\n".join(lines) + "\n"


def audit_study(directory: Path, *, workers=2, development=False):
    source = verify_source()
    if not development:
        require_integrated_source()
    entries = _load_plan(
        directory
    )  # Verify immutable plan, snapshots and execution identity.
    expected = resolve_study(STUDY)
    require(
        digest([e.entry_id for e in entries]) == expected.plan_sha256,
        "study differs from frozen 60-run acceptance",
    )
    # Re-read verified attempt evidence rather than trusting aggregate metrics.
    identity = execution_identity()
    rows = []
    for entry in entries:
        try:
            status = _latest(directory, entry)
            if status["status"] == "completed":
                actual = json.loads(
                    (Path(status["run_dir"]) / "manifest.json").read_text()
                )["source"]
                require(
                    actual["git"]["commit"] == identity["commit"]
                    and actual["packages"] == identity["packages"]
                    and actual["python"] == identity["python"],
                    "child source or dependencies differ from study",
                )
        except Exception as exc:
            status = {
                "status": "evidence_invalid",
                "run_dir": None,
                "failure_reason": str(exc),
            }
        rows.append(
            dict(
                entry_id=entry.entry_id,
                case_id=entry.case_id,
                algorithm_id=entry.algorithm_id,
                replication=entry.replication,
                variant_id=entry.variant_id,
                **status,
            )
        )
    output = (
        directory
        / "acceptance"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex)
    )
    (output / "audits").mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "schema": "smartsom.idetc-acceptance-protocol/v1",
            "source": source_identity(),
            "execution_identity": execution_identity(),
            "reference_sha256": digest(source),
            "plan_sha256": expected.plan_sha256,
            "required_completed": 60,
            "required_audited": 60,
            "quality_pairing": "M0 superset M1 superset M2 per case and replication",
            "paper_values_are_thresholds": False,
        },
    )
    audits = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        tasks = {
            executor.submit(audit_task, row): row
            for row in rows
            if row["status"] == "completed"
        }
        with (output / "progress.log").open("w", buffering=1) as log:
            for future in as_completed(tasks):
                row = tasks[future]
                try:
                    audit = future.result()
                except Exception as exc:
                    audit = {
                        "entry_id": row["entry_id"],
                        "status": "failed",
                        "error": f"audit worker exited: {exc}",
                    }
                audits.append(audit)
                write_json(output / "audits" / f"{row['entry_id']}.json", audit)
                message = f"audit {len(audits)}/{len(tasks)} {row['case_id']} {row['algorithm_id']} rep={row['replication']} {audit['status']}"
                print(message, flush=True)
                log.write(message + "\n")
    for row in rows:
        if row["status"] != "completed":
            run = (
                Path(row["run_dir"])
                if row.get("run_dir")
                else Path(row.get("attempt_dir", directory))
            )
            for name in ("failure.json", "worker_failure.json"):
                if (run / name).exists():
                    row["failure_reason"] = json.loads((run / name).read_text())
                    break
    result = summarize(rows, audits, source["paper_reference"]["results"])
    result.update(
        schema="smartsom.idetc-acceptance/v1",
        study_dir=str(directory),
        source=source_identity(),
        evidence_kind="development" if development else "integrated_main",
        linux_execution="unverified",
        reference_sha256=digest(source),
    )
    if not development:
        require_integrated_source()
        require(
            execution_identity()
            == json.loads((directory / "manifest.json").read_text())[
                "execution_identity"
            ],
            "source changed during audit",
        )
    write_json(output / "report.json", result)
    (output / "report.md").write_text(report_markdown(result), encoding="utf-8")
    fields = (
        "case_id",
        "algorithm_id",
        "replication",
        "status",
        "audit_status",
        "makespan",
        "passing_rate",
        "run_dir",
        "audit_error",
        "failure_reason",
    )
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    print(f"accepted={result['accepted']} report={output / 'report.md'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        help="audit a finished frozen study instead of launching another",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--development",
        action="store_true",
        help="explicitly mark pre-commit diagnostic evidence",
    )
    args = parser.parse_args()
    require(args.workers > 0, "workers must be positive")
    if not args.development:
        require_integrated_source()
    directory = args.study_dir
    if directory is None:
        last = 0.0

        def progress(event):
            nonlocal last
            now = time.monotonic()
            if "counts" in event and now - last >= 10:
                print(
                    f"batch {event['counts']} elapsed={event['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                last = now

        batch = run_batch(
            resolve_study(STUDY), workers=args.workers, on_progress=progress
        )
        directory = batch.study_dir
        print(
            f"study={directory} completed={batch.completed} failed={batch.failed}",
            flush=True,
        )
        if batch.interrupted:
            return 130
    return (
        0
        if audit_study(directory, workers=args.workers, development=args.development)[
            "accepted"
        ]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
