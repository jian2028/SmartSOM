"""User-facing execution subsets and complete-directory recovery contracts."""

import pytest

from smartsom.experiments import author_batch, batch_append, runtime_control
from smartsom.experiments.batch_selection import parse_selector
from smartsom.experiments.cli import _parser
from smartsom.experiments.evidence import write_json
from smartsom.experiments.launch import compile_launch, run_launch
from tests.unit.test_batch_growth_order import add_large
from tests.unit.test_incremental_batch import batch, hashes
from tests.unit.test_incremental_batch import lightweight as _lightweight


@pytest.fixture
def lightweight(monkeypatch):
    return _lightweight.__wrapped__(monkeypatch)


def from_cli(folder, *args):
    parsed = _parser().parse_args(["batch-run", str(folder), *args])
    return compile_launch(
        folder,
        ids=parsed.batch_ids,
        selectors=parsed.batch_selectors,
        first_train_seeds=parsed.first_train_seeds,
        resume_missing=parsed.resume_missing,
        operations={"output_root": str(folder.parent / "results")},
    )


def test_user_chain_subsets_full_resume_and_new_members(tmp_path, lightweight):
    folder = batch(tmp_path)
    first = from_cli(
        folder,
        "--id",
        "medium-sl",
        "--select",
        "train_seeds: [4101]",
        "--id",
        "medium-base",
        "--select",
        "train_seeds: [4102, 4103]",
    )
    run_launch(first)
    assert lightweight == [
        "medium-sl--seed-4101",
        "medium-base--seed-4102",
        "medium-base--seed-4103",
    ]
    root = first.output_root
    old = hashes(root / "trials/medium-base/seed-4102")
    assert {row["compiled"].experiment.train_seeds for row in first.files} == {
        (4101, 4102, 4103)
    }
    result = run_launch(from_cli(folder, "--resume-missing"))
    assert result["completed_tasks"] == 12 and result["submitted_tasks"] == 9
    assert len(lightweight) == len(set(lightweight)) == 12
    assert run_launch(from_cli(folder, "--resume-missing"))["submitted_tasks"] == 0
    add_large(folder)
    result = run_launch(from_cli(folder, "--resume-missing"))
    assert result["completed_tasks"] == 18 and result["submitted_tasks"] == 6
    assert len(lightweight) == len(set(lightweight)) == 18
    assert hashes(root / "trials/medium-base/seed-4102") == old


def test_repeated_id_disjoint_seeds_keep_request_order(tmp_path):
    folder = batch(tmp_path)
    plan = from_cli(
        folder,
        "--id",
        "medium-sl",
        "--select",
        "train_seeds: [4103]",
        "--id",
        "medium-base",
        "--select",
        "train_seeds: [4102]",
        "--id",
        "medium-sl",
        "--select",
        "train_seeds: [4101, 4102]",
    )
    assert plan.selection["tasks"] == [
        "medium-sl--seed-4103",
        "medium-base--seed-4102",
        "medium-sl--seed-4101",
        "medium-sl--seed-4102",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "train_seeds: [4101, 4101]",
        "train_seeds: true",
        "train_seeds: []",
        "train_seeds: -1",
        "train_seeds: '3'",
        "train_seeds: [1.5]",
        "train_seeds: [4101]\ntrain_seeds: [4102]",
        "learning_rate: 0.1",
        "{}",
        "train_seeds: [4101]\nunknown: 2",
        "!!python/object/apply:os.system ['false']",
    ],
)
def test_invalid_select_parser(value):
    with pytest.raises((ValueError, __import__("yaml").YAMLError)):
        parse_selector(value)


def test_count_semantics_are_existing_yaml_not_first_n(tmp_path):
    assert parse_selector("train_seeds: 3") == {"train_seeds": [101, 102, 103]}
    folder = batch(tmp_path)
    with pytest.raises(ValueError, match="not in authored"):
        from_cli(folder, "--id", "medium-sl", "--select", "train_seeds: 1")
    plan = from_cli(folder, "--id", "medium-sl", "--first-train-seeds", "1")
    assert plan.selection["tasks"] == ["medium-sl--seed-4101"]


@pytest.mark.parametrize(
    "args",
    [
        ["--select", "train_seeds: [4101]"],
        [
            "--id",
            "medium-sl",
            "--select",
            "train_seeds: [4101]",
            "--select",
            "train_seeds: [4102]",
        ],
        ["--id", "medium-sl", "--order", "medium-sl"],
    ],
)
def test_ambiguous_parser_scope_rejected(args):
    with pytest.raises(SystemExit):
        _parser().parse_args(["batch-run", "folder", *args])


def test_invalid_submission_is_atomic_and_overrides_conflict(tmp_path, lightweight):
    folder = batch(tmp_path)
    run_launch(from_cli(folder, "--id", "medium-base", "--first-train-seeds", "1"))
    before = hashes(folder.parent / "results")
    for ids, selectors in [
        (["unknown"], [None]),
        (["medium-sl"] * 2, [None, {"train_seeds": [4101]}]),
        (["medium-sl"], [{"train_seeds": [999]}]),
    ]:
        with pytest.raises(ValueError):
            run_launch(
                compile_launch(
                    folder,
                    ids=ids,
                    selectors=selectors,
                    resume_missing=True,
                    operations={"output_root": str(folder.parent / "results")},
                )
            )
    for extra in [
        {"first_train_seeds": 1},
        {"seed": 4101},
        {"sets": ["experiment.train_seeds=[4101]"]},
    ]:
        with pytest.raises(ValueError, match="conflicts"):
            compile_launch(
                folder, ids=["medium-sl"], selectors=[{"train_seeds": [4101]}], **extra
            )
    assert hashes(folder.parent / "results") == before


def test_failure_recovery_validates_before_any_activation(tmp_path, lightweight):
    folder = batch(tmp_path)
    first = from_cli(folder, "--id", "medium-base", "--first-train-seeds", "1")
    run_launch(first)
    root = first.output_root
    _, _, state = author_batch.load(root)
    state["files"]["medium-base--seed-4101"]["status"] = "failed"
    state["status"] = "failed"
    write_json(root / "batch.json", state)
    model = root / "trials/medium-base/seed-4101/continuation/generation-0/model.pt"
    original = model.read_bytes()
    model.write_bytes(b"corrupt")
    before = hashes(root)
    with pytest.raises(ValueError, match="digest mismatch"):
        run_launch(from_cli(folder, "--resume-missing"))
    assert hashes(root) == before
    model.write_bytes(original)
    result = run_launch(from_cli(folder, "--resume-missing"))
    assert result["completed_tasks"] == 12
    assert lightweight.count("medium-base--seed-4101") == 2  # explicit compatible retry


def test_manual_hold_blocks_without_clearing_or_allocating(tmp_path, lightweight):
    folder = batch(tmp_path)
    run_launch(from_cli(folder, "--id", "medium-base", "--first-train-seeds", "1"))
    root = folder.parent / "results"
    with runtime_control.transaction(root) as control:
        control["paused_tasks"] = ["small-sl--seed-4101"]
    before = hashes(root)
    with pytest.raises(ValueError, match="manual holds"):
        run_launch(from_cli(folder, "--resume-missing"))
    assert hashes(root) == before


def test_active_append_honors_disjoint_repeated_id_selection(tmp_path, lightweight):
    from tests.unit.test_batch_append_ownership import open_queue

    folder = batch(tmp_path)
    root = open_queue(folder)
    add_large(folder)
    receipt = batch_append.append(
        root,
        folder,
        ids=["large-sl", "large-base", "large-sl"],
        selectors=[
            {"train_seeds": [4103]},
            {"train_seeds": [4101]},
            {"train_seeds": [4101]},
        ],
    )
    assert receipt["tasks"] == [
        "large-sl--seed-4103",
        "large-base--seed-4101",
        "large-sl--seed-4101",
    ]
    _, _, saved = author_batch.load(root)
    assert saved["files"]["large-sl--seed-4102"]["status"] == "deferred"


@pytest.mark.parametrize("local", [True, False])
def test_parent_run_override_cannot_bypass_seed_conflict(tmp_path, local):
    folder = batch(tmp_path)
    selection = (
        {"selectors": [{"train_seeds": [4101]}]} if local else {"first_train_seeds": 1}
    )
    with pytest.raises(ValueError, match="conflicts"):
        compile_launch(
            folder,
            ids=["medium-sl"],
            sets=["experiment.run={task: train, seed: 4102}"],
            **selection,
        )


def test_api_id_only_append_uses_owned_selection_path(tmp_path, monkeypatch):
    folder = batch(tmp_path)
    root = tmp_path / "legacy-parent"
    root.mkdir()
    write_json(root / "plan.json", {"incremental": False})
    calls = []
    monkeypatch.setattr(
        batch_append,
        "append",
        lambda *args, **kwargs: calls.append(kwargs) or {"routed": True},
    )
    assert runtime_control.append(root, folder, ids=["medium-sl"]) == {"routed": True}
    assert calls[0]["ids"] == ["medium-sl"]
