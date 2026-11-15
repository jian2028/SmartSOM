"""User-facing execution subsets and complete-directory recovery contracts."""

import pytest

from smartsom.experiments import author_batch
from smartsom.experiments.batch_selection import parse_selector
from smartsom.experiments.cli import _parser
from smartsom.experiments.evidence import write_json
from smartsom.experiments.launch import compile_launch, run_launch
from tests.unit.test_incremental_batch import batch, hashes
from tests.unit.test_incremental_batch import lightweight as _lightweight


@pytest.fixture
def lightweight(monkeypatch):
    return _lightweight.__wrapped__(monkeypatch)


def from_cli(folder, *args):
    parsed = _parser().parse_args(["run", str(folder), *args])
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
    assert set(lightweight) == {
        "medium-sl--seed-4101",
        "medium-base--seed-4102",
        "medium-base--seed-4103",
    }
    root = first.output_root
    old = hashes(root / "trials/medium-base/seed-4102")
    assert {row["compiled"].experiment.train_seeds for row in first.files} == {
        (4101, 4102, 4103)
    }
    result = run_launch(from_cli(folder, "--resume-missing"))
    assert result["completed_tasks"] == 12 and result["submitted_tasks"] == 9
    assert len(lightweight) == len(set(lightweight)) == 12
    assert run_launch(from_cli(folder, "--resume-missing"))["submitted_tasks"] == 0
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
        _parser().parse_args(["run", "folder", *args])


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
    with pytest.raises(ValueError, match="require resume or retry"):
        run_launch(from_cli(folder, "--resume-missing"))
    assert hashes(root) == before
    model.write_bytes(original)
    with pytest.raises(ValueError, match="require resume or retry"):
        run_launch(from_cli(folder, "--resume-missing"))
    assert lightweight.count("medium-base--seed-4101") == 1


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


def test_run_directory_rejects_file_only_selector(tmp_path):
    from smartsom.experiments.cli import main

    assert main(["run", str(tmp_path), "--select", "train_seeds: [4101]"]) == 2


@pytest.mark.parametrize("command", ["check", "run"])
def test_check_and_run_share_file_selection_syntax(command):
    args = _parser().parse_args(
        [command, "experiment.yaml", "--select", "train_seeds: [4101]"]
    )
    assert args.file_selector == {"train_seeds": [4101]}


def test_standalone_file_expands_every_training_seed(tmp_path):
    import yaml

    from tests.helpers.algorithm import rllib_recipe as recipe
    from tests.unit.test_algorithm_components import author

    path = author(tmp_path, recipe("ppo"))
    document = yaml.safe_load(path.read_text())
    document.setdefault("run", {})["task"] = "train"
    document["train_seeds"] = [71, 73]
    document["drain_frozen"] = True
    path.write_text(yaml.safe_dump(document))
    plan = compile_launch(path)
    assert [row["compiled"].experiment.run.seed for row in plan.files] == [71, 73]
    assert len({row["child"] for row in plan.files}) == 2
    assert len({row["scientific_sha256"] for row in plan.files}) == 2
    document["train_seeds"] = [71]
    path.write_text(yaml.safe_dump(document))
    singleton = compile_launch(path)
    assert singleton.experiment.run.seed == 71
    one = compile_launch(path, seed=73)
    assert one.experiment.run.seed == 73
    assert one.experiment.train_seeds == (73,)
