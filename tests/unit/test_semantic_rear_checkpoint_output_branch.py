"""Exact419 receipt and artifact routing checks; no simulator or PPO credit."""
from copy import deepcopy
import json

import pytest

from wlr50_clean.ppo import semantic_cli as cli, semantic_training as training
from wlr50_clean.ppo import semantic_rear_policy_timing_migration as initial
from wlr50_clean.ppo.semantic_migration import digest
from wlr50_clean.ppo.semantic_policy_distribution import policy_contract
from wlr50_clean.ppo.semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT,
)


BRANCH = "ancestor220544_recapture_v2"
COUNTS = dict(global_policy_decisions=220544, ppo_updates=1688, optimizer_steps=33760)
SOURCE_HEAD = "fa4b98ed506eb2230e61e13bd93ceecdcb6cfdad"
SCHEMA = "wlr50_clean.rear_recapture_same419.v1"
FACTOR = "rear_recapture_same419_factor"
RECEIPT = "rear_recapture_migration"
SELECTION = {
    "source_role": "explicit_initial419_ancestor_recapture_branch",
    "checkpoint_sha256": "e83b7340febb760fdd00aa5cf44c07bbf45809a2f9a175b1eafd0d670c3cef85",
    "manifest_sha256": "3fba5f7d8b32f98420924fc5103b4ea4f58205cef004651658152ea011dc05f6",
    "counters": COUNTS,
    "output_branch": BRANCH,
}


def metadata():
    policy = policy_contract(REAR_POLICY_TIMING_POLICY,
                             observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    old = {"experiment_id": initial.EXPERIMENT, "source_git_commit": SOURCE_HEAD,
           "runtime_content_sha256": "e" * 64}
    current = {**old, "source_git_commit": "f" * 40, "runtime_content_sha256": "c" * 64}
    append = {
        "schema": initial.SCHEMA,
        "source_checkpoint_sha256": initial.SOURCE_SHA,
        "target_contract_sha256": digest(old),
        "target_git_commit": SOURCE_HEAD,
        "target_runtime_content_sha256": old["runtime_content_sha256"],
        initial.FACTOR_KEY: {
            "source_role": "explicit_front_and_RR_validated_ancestor_continuation",
            "target_policy_contract": deepcopy(policy), "counter_origin": deepcopy(COUNTS),
        },
    }
    revision = {
        "schema": SCHEMA,
        "source_checkpoint_sha256": SELECTION["checkpoint_sha256"],
        "source_manifest_sha256": SELECTION["manifest_sha256"],
        "source_git_commit": SOURCE_HEAD,
        "source_contract_sha256": digest(old),
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_contract_sha256": digest(current),
        "target_runtime_content_sha256": current["runtime_content_sha256"],
        "target_git_commit": current["source_git_commit"],
        "source_selection": deepcopy(SELECTION),
        FACTOR: {
            "schema": SCHEMA,
            "source_policy_contract": deepcopy(policy), "target_policy_contract": deepcopy(policy),
            "source_mode": "rr_capture_before_rl_transfer_v1",
            "target_mode": "rr_recapture_current_support_v2",
            "source_selection": deepcopy(SELECTION),
            "original_branch_origin": deepcopy(COUNTS), "revision_counter_origin": deepcopy(COUNTS),
            "preserved_metadata_sha256": {initial.MIGRATION: digest(append)},
            "added_policy_decisions": 0, "added_ppo_updates": 0,
            "added_optimizer_steps": 0, "added_auxiliary_updates": 0,
        },
    }
    return {
        **COUNTS, "seed": 1001, "semantic_version": "v3", "policy_contract": policy,
        "runtime_contract": current, initial.MIGRATION: append, RECEIPT: revision,
        "rear_policy_timing_branch": {
            "schema": initial.SCHEMA, "counter_origin": deepcopy(COUNTS),
            "source_checkpoint_sha256": initial.SOURCE_SHA,
            "source_role": "explicit_front_and_RR_validated_ancestor_continuation",
            "migration_added_updates": 0,
        },
        "rear_policy_timing_branch_counts": dict.fromkeys(COUNTS, 0),
        "stage_requested_decisions": dict.fromkeys(training.STAGE_BUDGETS, 0),
    }


def write_checkpoint(path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic419 route only")
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture
def paths(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    base = tmp_path / "outputs" / ("ppo_" + initial.EXPERIMENT)
    monkeypatch.setattr(cli, "version_paths", lambda *a, **kw: (runs, base, tmp_path / "configs"))
    source = write_checkpoint(base / "checkpoints/history/ancestor_recapture.pt", metadata())
    return runs, base, source


def args(paths, *, source=None, name=BRANCH, command="train", phase="P01", migration=None):
    argv = [command, "--run-dir", str(paths[0] / "test"), "--expected-head", "f" * 40,
            "--semantic-version", "v3", "--experiment-id", initial.EXPERIMENT,
            "--checkpoint", str(source or paths[2]),
            "--stage", "full_episode" if phase == "P01" else "phase_suffix", "--from-phase", phase]
    argv += ["--mode", "semantic_residual_eval", "--seed", "4001"] if command == "eval" else ["--decisions", "128"]
    if phase != "P01":
        argv += ["--prefix-source", "checkpoint_policy"]
    if name is not None:
        argv += ["--checkpoint-output-branch", name]
    if migration is not None:
        argv += ["--resume-migration", str(migration)]
    return cli.parser().parse_args(argv)


def route(base):
    return {"schema": "wlr50_clean.checkpoint_output_routing.v1", "branch": BRANCH,
            "output_root": str((base / "branches" / BRANCH).resolve()),
            "main_latest_pointer_promotion": False, "source_selection": deepcopy(SELECTION)}


def test_recapture_parent_eval_allowed_but_train_requires_explicit_branch(paths):
    request = args(paths, name=None, command="eval")
    cli.validate_request(request)
    assert request.checkpoint == paths[2].resolve()
    assert cli._checkpoint_output_root(request) == paths[1].resolve()
    assert not hasattr(request, "_checkpoint_output_routing")
    with pytest.raises(ValueError, match="explicit output branch|explicit output routing"):
        cli.validate_request(args(paths, name=None))


@pytest.mark.parametrize("phase", ["P01", "P04"])
def test_recapture_parent_routes_natural_or_checkpoint_policy_prefix(paths, phase):
    request = args(paths, phase=phase)
    cli.validate_request(request)
    assert request.checkpoint == paths[2].resolve()
    assert request._checkpoint_output_routing == route(paths[1])
    assert cli._checkpoint_output_root(request) == paths[1] / "branches" / BRANCH
    assert request.teacher_offset_decisions == 0
    if phase != "P01":
        assert request.prefix_source == "checkpoint_policy"
    assert not (paths[1] / "branches").exists()


@pytest.mark.parametrize("name", ["..", "../escape", "a/b", r"a\b", "C:/escape", "CON", "con", "lpt9", "trailing.", ""])
def test_recapture_rejects_unsafe_branch_names_without_output(paths, name):
    with pytest.raises(ValueError):
        cli.validate_request(args(paths, name=name))
    assert not (paths[1] / "branches").exists()


@pytest.mark.parametrize("bad", ["foreign_source", "mutable_parent", "foreign_branch", "occupied", "wrong_branch", "missing_routing"])
def test_recapture_rejects_foreign_source_or_occupied_output(paths, bad):
    runs, base, source = paths
    meta = metadata()
    destination = base / "branches" / BRANCH
    name = BRANCH
    if bad == "foreign_source":
        source = write_checkpoint(runs / "foreign.pt", meta)
    elif bad == "mutable_parent":
        source = write_checkpoint(base / "checkpoints/checkpoint_last.pt", meta)
    elif bad == "foreign_branch":
        source = write_checkpoint(base / "branches/another/checkpoints/history/source.pt", meta)
    elif bad == "occupied":
        write_checkpoint(destination / "checkpoints/history/existing.pt", meta)
    elif bad == "wrong_branch":
        name = "another"
    else:
        source = write_checkpoint(destination / "checkpoints/history/source.pt", meta)
    with pytest.raises(ValueError):
        cli.validate_request(args(paths, source=source, name=name))


@pytest.mark.parametrize("command,phase", [("train", "P01"), ("train", "P04"), ("eval", "P01")])
def test_recapture_same_branch_immutable_and_pointer_resolution(paths, command, phase):
    base = paths[1]
    destination = base / "branches" / BRANCH
    meta = metadata()
    increments = dict(global_policy_decisions=128, ppo_updates=1, optimizer_steps=20)
    meta.update({k: COUNTS[k] + increments[k] for k in COUNTS})
    meta["rear_policy_timing_branch_counts"] = increments
    meta["checkpoint_output_routing"] = route(base)
    source = write_checkpoint(destination / "checkpoints/history/checkpoint_step_000220672.pt", meta)
    training._publish_last(source, source.with_name(source.stem + "_manifest.json"), destination)
    for selected in (source, destination / "checkpoints/checkpoint_last.pt"):
        request = args(paths, source=selected, command=command, phase=phase)
        cli.validate_request(request)
        assert request.checkpoint == source.resolve()
        assert request._checkpoint_output_routing == route(base)
    pointer = destination / "checkpoints/checkpoint_last_pointer.json"
    binding = json.loads(pointer.read_text())
    binding["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        cli.validate_request(args(paths, source=destination / "checkpoints/checkpoint_last.pt", command=command, phase=phase))


def test_recapture_namespace_keeps_append_and_revision_receipts_intact(paths):
    meta = metadata()
    before = deepcopy(meta)
    record = initial.build_rear_policy_output_routing(meta, meta["runtime_contract"], paths[1] / "branches" / BRANCH)
    assert record == route(paths[1])
    initial.validate_rear_policy_namespace(meta, meta["runtime_contract"], paths[1] / "branches" / BRANCH,
                                         checkpoint_output_routing=record)
    assert meta == before
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(meta, meta["runtime_contract"], paths[1],
                                             checkpoint_output_routing=record)


def test_original_append_receipt_keeps_its_original_namespace_validation(paths):
    meta = metadata()
    meta.pop(RECEIPT)
    meta["runtime_contract"].update(source_git_commit=SOURCE_HEAD, runtime_content_sha256="e" * 64)
    before = deepcopy(meta)
    initial.validate_rear_policy_namespace(meta, meta["runtime_contract"], paths[1])
    assert meta == before
    with pytest.raises(ValueError):
        initial.build_rear_policy_output_routing(meta, meta["runtime_contract"], paths[1] / "branches" / BRANCH)


@pytest.mark.parametrize("bad", ["missing_append", "missing_revision", "empty_revision", "wrong419", "foreign_role", "wrong_source_sha", "wrong_source_manifest", "wrong_factor_selection", "broken_chain", "wrong_target_contract", "wrong_target_head", "wrong_origin", "changed_append", "borrowed_counts", "borrowed_branch_counts"])
def test_recapture_namespace_rejects_invalid_lineage_before_output(paths, bad):
    meta = metadata()
    receipt = meta[RECEIPT]
    if bad == "missing_append":
        meta.pop(initial.MIGRATION)
    elif bad == "missing_revision":
        meta.pop(RECEIPT)
    elif bad == "empty_revision":
        meta[RECEIPT] = {}
    elif bad == "wrong419":
        meta["policy_contract"]["observation_layout"] = "role410"
        receipt[FACTOR]["source_policy_contract"] = deepcopy(meta["policy_contract"])
        receipt[FACTOR]["target_policy_contract"] = deepcopy(meta["policy_contract"])
        meta[initial.MIGRATION][initial.FACTOR_KEY]["target_policy_contract"] = deepcopy(meta["policy_contract"])
        receipt[FACTOR]["preserved_metadata_sha256"][initial.MIGRATION] = digest(meta[initial.MIGRATION])
    elif bad == "foreign_role":
        receipt["source_selection"]["source_role"] = "latest_learned_continuation"
    elif bad == "wrong_source_sha":
        receipt["source_checkpoint_sha256"] = "0" * 64
    elif bad == "wrong_source_manifest":
        receipt["source_manifest_sha256"] = "0" * 64
    elif bad == "wrong_factor_selection":
        receipt[FACTOR]["source_selection"]["counters"]["ppo_updates"] += 1
    elif bad == "broken_chain":
        receipt["source_contract_sha256"] = "0" * 64
    elif bad == "wrong_target_contract":
        receipt["target_contract_sha256"] = "0" * 64
    elif bad == "wrong_target_head":
        receipt["target_git_commit"] = "0" * 40
    elif bad == "wrong_origin":
        receipt[FACTOR]["revision_counter_origin"]["global_policy_decisions"] += 128
    elif bad == "changed_append":
        meta[initial.MIGRATION]["source_checkpoint_sha256"] = "0" * 64
    elif bad == "borrowed_counts":
        meta["global_policy_decisions"] += 128
    else:
        meta["rear_policy_timing_branch_counts"]["global_policy_decisions"] = 128
    write_checkpoint(paths[2], meta)
    with pytest.raises(ValueError):
        cli.validate_request(args(paths))
    assert not (paths[1] / "branches").exists()
