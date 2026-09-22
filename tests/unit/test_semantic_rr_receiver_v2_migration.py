"""Same389 v1-to-v2 state transfer; all data and counts below are synthetic."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from test_semantic_rr_workspace_migration import boundary
from wlr50_clean.ppo import semantic_rr_workspace_migration as m


def v2_boundary(events=4):
    meta, old, new, reviewed, source, target = boundary()
    old["files"][m.MODULE] = "b"*64
    origin = dict(global_policy_decisions=207872, ppo_updates=1593, optimizer_steps=31860)
    meta["rr_postcross_workspace_branch"] = {
        "schema": m.SCHEMA, "semantics": m.MODE, "counter_origin": origin,
        "front_rehearsal_auxiliary": {
            "schema": "wlr50_clean.front_rehearsal_auxiliary.v1",
            "events": [{"id": f"event_{i+1}", "accepted": 32, "attempted": 32,
                        "opaque_future_field": {"keep": i}} for i in range(events)],
            "accepted": 32*events, "attempted": 32*events},
    }
    meta["rr_postcross_workspace_migration"] = {
        "schema": m.SCHEMA, m.FACTOR_KEY: {"target_semantics": m.MODE},
        "immutable_v1_evidence": {"preserve": "entire object"}}
    meta = m.rr_workspace_branch_counts(meta)
    source[m.MODE_KEY] = m.MODE
    target = {**deepcopy(source), m.MODE_KEY: m.MODE_V2}
    reviewed = {p:sha for p,sha in new["files"].items() if old["files"].get(p) != sha}
    return meta, old, new, reviewed, source, target


def factor(values):
    meta, old, new, review, source, target = values
    return m.rr_workspace_factor(meta, old, new, reason="synthetic receiver v2 boundary",
        reviewed_code_sha256=review, source_task_spec=source, target_task_spec=target)


@pytest.mark.parametrize("events", [3, 4])
def test_v2_factor_hashes_entire_existing_aux_and_origins(events):
    from wlr50_clean.ppo.semantic_migration import digest
    values = v2_boundary(events)
    meta = values[0]
    before = deepcopy(meta)
    f = factor(values)
    assert meta == before
    assert f["schema"] == m.SCHEMA_V2
    assert f["source_semantics"] == m.MODE and f["target_semantics"] == m.MODE_V2
    assert f["counter_origin"] == {k:meta[k] for k in m.COUNTERS}
    for key in m.preserved_keys(meta):
        assert f["preserved_metadata_sha256"][key] == digest(meta[key])
    assert f["observation_semantics_changed"][0]["index"] == 17
    assert f["observation_contract"]["observation_dimension"] == 389
    for key in ("caps_changed", "sigma_changed", "policy_kernel_changed", "nominal_changed",
                "controller_predicates_changed", "capture_assist_changed",
                "reward_coefficients_and_return_profile_changed", "same_mdp_claimed"):
        assert f[key] is False
    assert f["reward_changed"] is True and f["discard_old_rollout_storage"] is True


@pytest.mark.parametrize("bad", ["source_none", "source_v2", "source_unknown", "target_unknown",
    "repeat_branch", "repeat_receipt", "missing_v1", "missing_front_aux", "missing_old_aux",
    "origin", "count", "reward_config", "capture_config", "sigma", "extra_file", "review", "task"])
def test_v2_rejects_unrelated_or_repeated_boundary(bad):
    values = v2_boundary()
    meta, old, new, review, source, target = values
    if bad == "source_none": source.pop(m.MODE_KEY)
    elif bad == "source_v2": source[m.MODE_KEY] = m.MODE_V2
    elif bad == "source_unknown": source[m.MODE_KEY] = "unknown"
    elif bad == "target_unknown": target[m.MODE_KEY] = "unknown"
    elif bad == "repeat_branch": meta[m.V2_BRANCH] = {}
    elif bad == "repeat_receipt": meta[m.V2_MIGRATION] = {}
    elif bad == "missing_v1": meta.pop("rr_postcross_workspace_migration")
    elif bad == "missing_front_aux": meta["rr_postcross_workspace_branch"].pop("front_rehearsal_auxiliary")
    elif bad == "missing_old_aux": meta["task_conditioned_hip_wheel_branch"].pop("auxiliary_mean_learning")
    elif bad == "origin": meta["rr_postcross_workspace_branch"]["counter_origin"]["ppo_updates"] = -1
    elif bad == "count": meta["rr_postcross_workspace_branch_counts"]["ppo_updates"] += 1
    elif bad in ("reward_config", "capture_config"):
        name = "reward_config.yaml" if bad == "reward_config" else "execution_profile.yaml"
        path = new["selected_configuration"][name]["path"]
        new["files"][path] = "8"*64
        new["selected_configuration"][name]["sha256"] = "8"*64
        review[path] = "8"*64
    elif bad == "sigma": meta["policy_contract"]["rho"] = .123
    elif bad == "extra_file":
        new["files"]["src/wlr50_clean/ppo/semantic_capture_feedback_migration.py"] = "8"*64
        review["src/wlr50_clean/ppo/semantic_capture_feedback_migration.py"] = "8"*64
    elif bad == "review": review.pop(m.SUPERVISOR)
    else: target["episode_maximum_duration_s"] += 1
    with pytest.raises(ValueError):
        factor(values)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
        capture_output=True, text=True).stdout.strip()


def _real_plan_fixture(tmp_path, metadata):
    # Real immutable git/byte bindings in a temp repo. This is not a claim that
    # synthetic marker modules are production semantics or a real training run.
    from wlr50_clean.ppo import semantic_migration as shared
    values = v2_boundary()
    _, old, new, _, source_spec, target_spec = values
    root = tmp_path / "source_and_target"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "unit-test@example.invalid")
    _git(root, "config", "user.name", "Synthetic unit test")
    _git(root, "config", "core.autocrlf", "false")
    paths = set(old["files"]) | {
        "src/wlr50_clean/ppo/semantic_migration.py",
        "src/wlr50_clean/ppo/semantic_training.py"}
    for path in paths:
        text = (yaml.safe_dump(source_spec) if path == m.TASK_SPEC else
                "unchanged: true\n" if path.endswith(".yaml") else "# synthetic v1 fixture\n")
        _write(root/path, text)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "synthetic source")

    def contract(template):
        result = deepcopy(template)
        result["files"] = {path:shared.file_sha(root/path) for path in paths}
        result["runtime_content_sha256"] = shared.digest(result["files"])
        result["source_git_commit"] = _git(root, "rev-parse", "HEAD")
        for binding in result["selected_configuration"].values():
            binding["sha256"] = result["files"][binding["path"]]
        return result

    old = contract(old)
    for path in (m.SUPERVISOR, m.MODULE, "src/wlr50_clean/ppo/semantic_migration.py",
                 "src/wlr50_clean/ppo/semantic_training.py"):
        _write(root/path, "# synthetic v2 fixture\n")
    _write(root/m.TASK_SPEC, yaml.safe_dump(target_spec))
    _git(root, "add", ".")
    _git(root, "commit", "-m", "synthetic target")
    new = contract(new)
    metadata["runtime_contract"] = old
    return root, old, new


def test_official_plan_publish_reload_and_actual_save_whitelist_carry(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from wlr50_clean.ppo import semantic_training as training, semantic_migration as shared
    from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv
    assert not torch.cuda.is_available(), "Run this CPU contract test with CUDA_VISIBLE_DEVICES=-1"

    def make():
        return training.construct_semantic_runner(_ObservationOnlyEnv(389), seed=1001, device="cpu",
            policy_version=m.P05_CAPTURE_POLICY, observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]

    runner = make()
    # Populate exact, nonempty synthetic Adam state without a forward, loss or fit.
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 2.25e-5
        for parameter in group["params"]:
            runner.alg.optimizer.state[parameter] = {
                "step": torch.tensor(9.), "exp_avg": torch.full_like(parameter, .001),
                "exp_avg_sq": torch.full_like(parameter, .002)}
    runner.alg.learning_rate = 2.25e-5
    meta = v2_boundary(events=4)[0]
    root, old, new = _real_plan_fixture(tmp_path, meta)
    source, _ = training.save_semantic_checkpoint(runner, tmp_path/"synthetic_source.pt", meta)
    source_meta = shared.checkpoint_metadata(source)
    reviewed = {p:sha for p,sha in new["files"].items() if old["files"].get(p) != sha}
    plan = m.build_rr_workspace_migration(source, new, reason="synthetic exact state v2",
        reviewed_code_sha256=reviewed, project_root=root)
    plan_path = tmp_path/"migration_plan.json"
    _write(plan_path, json.dumps(plan))
    # Only root routing is wrapped. Both real validators execute, no result stub.
    original_validate = shared.validate_migration_plan
    monkeypatch.setattr(shared, "PROJECT_ROOT", root)
    monkeypatch.setattr(shared, "validate_migration_plan",
        lambda c, runtime, p, **kw: original_validate(c, runtime, p, project_root=root))
    verified = shared.validate_migration_plan(source, new, plan_path)
    assert verified["schema"] == m.SCHEMA_V2
    tampered = deepcopy(plan)
    tampered[m.FACTOR_KEY]["counter_origin"]["ppo_updates"] += 1
    bad_path = tmp_path/"tampered_plan.json"
    _write(bad_path, json.dumps(tampered))
    with pytest.raises(ValueError):
        shared.validate_migration_plan(source, new, bad_path)

    receipt = m.publish_rr_workspace_checkpoint(source, new, plan_path, tmp_path/"synthetic_v2.pt")
    migrated = shared.checkpoint_metadata(Path(receipt["checkpoint"]))
    for key in m.preserved_keys(source_meta):
        assert migrated[key] == source_meta[key]
    assert migrated[m.V2_MIGRATION] == verified
    assert migrated[m.V2_BRANCH]["counter_origin"] == {k:source_meta[k] for k in m.COUNTERS}
    assert migrated["rr_receiver_retirement_v2_branch_counts"] == dict.fromkeys(m.COUNTERS, 0)
    fresh = make()
    loaded = training.load_semantic_checkpoint(fresh, Path(receipt["checkpoint"]), contract=new, seed=1001)
    assert fresh.alg.storage.step == 0 and fresh.alg.transition.actions is None
    for key in ("actor_parameter_sha256", "critic_parameter_sha256", "optimizer_state_sha256",
                "normalizer_state_sha256", "training_rng_state", "optimizer_learning_rate"):
        assert loaded[key] == source_meta[key]

    # Read the actual normal train save-carry whitelist; do not invent a second
    # list which could pass while train_semantic silently drops the new ledger.
    tree = ast.parse(inspect.getsource(training.train_semantic))
    carry = [ast.literal_eval(node.iter) for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
        and any(isinstance(x, ast.Constant) and x.value == "new_mdp_warm_start" for x in node.iter.elts)]
    assert len(carry) == 1
    assert {m.V2_BRANCH, m.V2_MIGRATION, "rr_postcross_workspace_branch",
            "rr_postcross_workspace_migration", "task_conditioned_hip_wheel_branch"} <= set(carry[0])
    later_infos = {k:v for k,v in loaded.items() if not k.endswith(("_branch", "_migration"))}
    for key in carry[0]:
        if key in loaded:
            later_infos[key] = loaded[key]
    # Counter-only propagation fixture, not an optimizer update or PPO credit.
    later_infos.update(global_policy_decisions=300128, ppo_updates=2001, optimizer_steps=40020)
    later, _ = training.save_semantic_checkpoint(fresh, tmp_path/"synthetic_later.pt", later_infos)
    later_meta = shared.checkpoint_metadata(later)
    assert later_meta["rr_receiver_retirement_v2_branch_counts"] == {
        "global_policy_decisions":128, "ppo_updates":1, "optimizer_steps":20}
    for key in ("rr_postcross_workspace_branch", "rr_postcross_workspace_migration",
                "task_conditioned_hip_wheel_branch", m.V2_BRANCH, m.V2_MIGRATION):
        assert later_meta[key] == loaded[key]
