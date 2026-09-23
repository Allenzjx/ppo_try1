"""Synthetic CPU RR410 identity migration; no simulator or learning credit."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from test_semantic_p05_preedge_migration import boundary as prior_boundary
from wlr50_clean.ppo import semantic_rr_capture_knee_migration as m
from wlr50_clean.ppo.semantic_migration import continuation_topology, digest
from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES, policy_contract
from wlr50_clean.ppo.semantic_rr_capture_assist import RR_CAPTURE_ASSIST_FEATURE_NAMES, _SCALES
from wlr50_clean.ppo.semantic_training import semantic_runner_config


def boundary():
    meta = prior_boundary()[0]
    canonical = policy_contract(m.RR_CAPTURE_POLICY, observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT)
    paths = {f"configs/ppo_{m.EXPERIMENT}/{name}":"a" * 64 for name in CONFIG_NAMES}
    paths.update({p:"b" * 64 for p in m.ALLOWED_FILES - {m.PROFILE, m.MODULE}})
    paths.update({"src/wlr50_clean/ppo/semantic_reward.py":"c" * 64,
                  "src/wlr50_clean/ppo/semantic_rr_capture_actor.py":"c" * 64})
    old = {"experiment_id":m.EXPERIMENT, "semantic_version":"v3", "physics_hz":120.,
        "files":paths, "source_git_commit":m.SOURCE_HEAD, "runtime_content_sha256":digest(paths),
        "selected_configuration":{name:{"path":f"configs/ppo_{m.EXPERIMENT}/{name}", "sha256":"a" * 64}
                                  for name in CONFIG_NAMES}}
    new = deepcopy(old); new["source_git_commit"] = "f" * 40
    new["files"].update({p:"d" * 64 for p in m.ALLOWED_FILES})
    new["selected_configuration"]["execution_profile.yaml"]["sha256"] = "d" * 64
    new["runtime_content_sha256"] = digest(new["files"])
    source = dict(revision=m.SOURCE_REVISION, rr_capture_assist_mode="rr_hip_only_capture_v1",
                  rr_capture_wheel_mode="off", rr_capture_feedback_revision=m.SOURCE_FEEDBACK_REVISION, physics_hz=120., residual={"unchanged_all12":True})
    target = {**deepcopy(source), "revision":m.TARGET_REVISION, "rr_capture_feedback_revision":m.FEEDBACK_REVISION}
    meta.update(runtime_contract=old, policy_contract=canonical, seed=1001,
        runner_config=semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
            policy_version=m.RR_CAPTURE_POLICY, observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT),
        sampling="P01_full_task_only_initial_version")
    meta["execution_topology"] = continuation_topology(meta["sampling"], None,
        observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT)
    for name in ("p05_preedge_approach_recovery", "rr_capture_transfer"):
        meta[name + "_branch"] = {"counter_origin":{k:meta[k] for k in m.COUNTERS},
            "schema":"wlr50_clean.rr_capture_transfer_append.v1" if name == "rr_capture_transfer" else "old-preserved"}
        meta[name + "_migration"] = {"opaque_source_receipt":name}
    for name in m.PRIOR_BRANCHES:
        meta[name + "_branch_counts"] = {k:meta[k]-meta[name + "_branch"]["counter_origin"][k] for k in m.COUNTERS}
    bind_prior_receipt(meta, old)
    review = {p:sha for p,sha in new["files"].items() if old["files"].get(p) != sha}
    return meta, old, new, review, source, target


def bind_prior_receipt(meta, old):
    meta[m.PRIOR_FEEDBACK_MIGRATION] = {
        "schema":"wlr50_clean.rr_capture_feedback_same410.v2",
        "target_git_commit":old["source_git_commit"], "target_contract_sha256":digest(old),
        "target_runtime_content_sha256":old["runtime_content_sha256"],
        m.PRIOR_FEEDBACK_FACTOR:{"schema":"wlr50_clean.rr_capture_feedback_same410.v2",
            "target_feedback_revision":m.SOURCE_FEEDBACK_REVISION,
            "observation_contract":{"target_policy_contract":deepcopy(meta["policy_contract"])},
            "counter_origin":{k:meta[k] for k in m.COUNTERS}},
        "opaque_prior_receipt":{"must_survive":[7,8,96,96,103,104]}}


def factor(values):
    meta, old, new, review, source, target = values
    return m.rr_capture_knee_factor(meta, old, new,
        reason="reviewed finite hip-then-knee search; unchanged wheel and peak feedback",
        reviewed_code_sha256=review, source_profile=source, target_profile=target)


def test_same410_v3_keeps_all_weights_contracts_origins_and_opaque_aux():
    values = boundary(); before = deepcopy(values[0]); value = factor(values)
    assert values[0] == before and not value["creates_new_branch"]
    assert value["observation_contract"]["observation_dimension"] == 410
    assert value["observation_contract"]["source_policy_contract"] == value["observation_contract"]["target_policy_contract"]
    assert not value["capture_owner_acquisition_changed"] and not value["pre_cross_descent_allowed"]
    assert value["discard_old_rollout_storage"] and not value["same_mdp_claimed"]
    assert not value["observation_shape_changed"] and not value["observation_codec_changed"]
    assert value["source_feedback_revision"] == "window_peak_progress_v2"
    assert value["current_effective_execution_semantics"]["total_exposure_limit_s"] == 32
    assert value["current_effective_execution_semantics"]["combined_travel_limit_deg"] == 40
    assert not value["legacy_action_transform_is_current_execution_semantics"]
    assert value["legacy_action_transform_description"] == before["policy_contract"]["action_transform"]
    assert value["source_feedback_v2_receipt_sha256"] == digest(before[m.PRIOR_FEEDBACK_MIGRATION])
    for flag in ("policy_kernel_changed", "reward_changed", "nominal_changed", "caps_changed", "sigma_changed",
                 "physical_dynamics_changed", "physical_task_acceptance_rules_changed"):
        assert value[flag] is False
    for key in m.preserved_keys(before):
        assert value["preserved_metadata_sha256"][key] == digest(before[key])
    # Future opaque auxiliary events must survive whole, not a fixed seven/eight summary.
    before["rr_postcross_workspace_branch"]["front_rehearsal_auxiliary"]["events"].append({"opaque_later_event":True})
    value = factor((before, *values[1:]))
    assert value["preserved_metadata_sha256"]["rr_postcross_workspace_branch"] == digest(before["rr_postcross_workspace_branch"])


@pytest.mark.parametrize("bad", ["source_head", "repeat", "missing_origin", "wrong_counts", "negative_origin",
    "old389", "missing_aux", "LR", "counter", "runtime_rate", "deleted_file", "unknown_file",
    "protected_file", "existing_module", "missing_review", "extra_review", "config_binding",
    "task_config", "profile_extra", "wrong_revision", "source_already_v3", "missing_v2", "v2_bad_type",
    "v2_wrong_head", "v2_wrong_contract", "v2_wrong_revision", "v2_not_zero_update", "missing_hook_delta", "backend_delta"])
def test_reject_unrelated_unbound_or_repeated_changes(bad):
    values = boundary(); meta, old, new, review, source, target = values
    if bad == "source_head": old["source_git_commit"] = "1" * 40
    elif bad == "repeat": meta[m.MIGRATION] = {}
    elif bad == "missing_origin": meta.pop("rr_capture_transfer_branch")
    elif bad == "wrong_counts": meta["rr_capture_transfer_branch_counts"]["ppo_updates"] += 1
    elif bad == "negative_origin": meta["rr_capture_transfer_branch"]["counter_origin"]["ppo_updates"] = -1
    elif bad == "old389": meta["policy_contract"]["observation_dimension"] = 389
    elif bad == "missing_aux": meta["task_conditioned_hip_wheel_branch"].pop("auxiliary_mean_learning")
    elif bad == "LR": meta["optimizer_learning_rate"] = 0.
    elif bad == "counter": meta["global_policy_decisions"] = True
    elif bad == "runtime_rate": new["physics_hz"] = 240.
    elif bad == "deleted_file": new["files"].pop(m.ASSIST)
    elif bad in ("unknown_file", "protected_file"):
        path = "src/wlr50_clean/ppo/unknown.py" if bad == "unknown_file" else "src/wlr50_clean/ppo/semantic_reward.py"
        new["files"][path] = review[path] = "8" * 64
    elif bad == "existing_module": old["files"][m.MODULE] = "8" * 64
    elif bad == "missing_review": review.pop(m.ASSIST)
    elif bad == "extra_review": review["unknown"] = "8" * 64
    elif bad == "config_binding": new["selected_configuration"]["reward_config.yaml"]["sha256"] = "8" * 64
    elif bad == "task_config":
        path = new["selected_configuration"]["stage_task_spec.yaml"]["path"]
        new["files"][path] = review[path] = "8" * 64
        new["selected_configuration"]["stage_task_spec.yaml"]["sha256"] = "8" * 64
    elif bad == "profile_extra": target["physics_hz"] = 60.
    elif bad == "wrong_revision": target["rr_capture_feedback_revision"] = "unknown"
    elif bad == "missing_v2": meta.pop(m.PRIOR_FEEDBACK_MIGRATION)
    elif bad == "v2_bad_type": meta[m.PRIOR_FEEDBACK_MIGRATION] = None
    elif bad == "v2_wrong_head": meta[m.PRIOR_FEEDBACK_MIGRATION]["target_git_commit"] = "8" * 40
    elif bad == "v2_wrong_contract": meta[m.PRIOR_FEEDBACK_MIGRATION]["target_contract_sha256"] = "8" * 64
    elif bad == "v2_wrong_revision": meta[m.PRIOR_FEEDBACK_MIGRATION][m.PRIOR_FEEDBACK_FACTOR]["target_feedback_revision"] = "wrong"
    elif bad == "v2_not_zero_update": meta[m.PRIOR_FEEDBACK_MIGRATION][m.PRIOR_FEEDBACK_FACTOR]["counter_origin"]["ppo_updates"] -= 1
    elif bad == "missing_hook_delta":
        path = "src/wlr50_clean/ppo/semantic_training.py"
        new["files"][path] = old["files"][path]; review.pop(path)
    elif bad == "backend_delta": new["files"]["src/wlr50_clean/ppo/semantic_backend.py"] = review["src/wlr50_clean/ppo/semantic_backend.py"] = "8" * 64
    else: source["rr_capture_feedback_revision"] = m.FEEDBACK_REVISION
    with pytest.raises(ValueError): factor(values)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.mark.parametrize("bad", [None, "scale", "name_order", "window", "search", "revision"])
def test_explicit_semantics_keep_all14_numeric_positions_and_scales(bad):
    base = {"RR_CAPTURE_ASSIST_FEATURE_NAMES":RR_CAPTURE_ASSIST_FEATURE_NAMES, "_SCALES":_SCALES,
            "RR_CAPTURE_FEEDBACK_REVISION":m.SOURCE_FEEDBACK_REVISION,
            "RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS":m.WINDOW_REFERENCE_SEMANTICS}
    target = {**base, "RR_CAPTURE_FEEDBACK_REVISION":m.FEEDBACK_REVISION,
              "RR_CAPTURE_SEARCH_SEMANTICS":m.CAPTURE_SEARCH_SEMANTICS}
    if bad == "scale": target["_SCALES"] = (*_SCALES[:5], 40., *_SCALES[6:])
    elif bad == "name_order": target["RR_CAPTURE_ASSIST_FEATURE_NAMES"] = tuple(reversed(RR_CAPTURE_ASSIST_FEATURE_NAMES))
    elif bad == "window": target["RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS"] = "different"
    elif bad == "search": target["RR_CAPTURE_SEARCH_SEMANTICS"] = "hidden_knee_clock"
    elif bad == "revision": target["RR_CAPTURE_FEEDBACK_REVISION"] = m.SOURCE_FEEDBACK_REVISION
    def code(values): return "\n".join(f"{k} = {v!r}" for k,v in values.items())
    if bad:
        with pytest.raises(ValueError): m._validate_assist_semantics(code(base), code(target))
    else:
        m._validate_assist_semantics(code(base), code(target))


def test_builder_rejects_non_published_source_bytes_before_any_load(tmp_path):
    path = tmp_path / "not_source.pt"; path.write_bytes(b"synthetic unrelated bytes")
    with pytest.raises(ValueError, match="exact published"):
        m.build_rr_capture_knee_migration(path, {}, reason="not allowed", reviewed_code_sha256={})


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def fixture_repo(tmp_path, meta, monkeypatch):
    from wlr50_clean.ppo import semantic_migration as shared
    _, old, new, _, source, target = boundary()
    root = tmp_path / "synthetic_git"; root.mkdir()
    git(root, "init"); git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Synthetic CPU test"); git(root, "config", "core.autocrlf", "false")
    code = f"RR_CAPTURE_ASSIST_FEATURE_NAMES = {RR_CAPTURE_ASSIST_FEATURE_NAMES!r}\n_SCALES = {_SCALES!r}\n"
    paths = set(old["files"])
    for path in paths:
        write(root / path, yaml.safe_dump(source) if path == m.PROFILE else (
            code + f"RR_CAPTURE_FEEDBACK_REVISION = {m.SOURCE_FEEDBACK_REVISION!r}\n"
            f"RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS = {m.WINDOW_REFERENCE_SEMANTICS!r}\n"
            if path == m.ASSIST else "# synthetic unchanged\n"))
    git(root, "add", "."); git(root, "commit", "-m", "synthetic source")
    def contract(template):
        result = deepcopy(template)
        result["files"] = {p:shared.file_sha(root / p) for p in paths}
        result["runtime_content_sha256"] = digest(result["files"])
        result["source_git_commit"] = git(root, "rev-parse", "HEAD")
        for binding in result["selected_configuration"].values(): binding["sha256"] = result["files"][binding["path"]]
        return result
    old = contract(old)
    # Only synthetic git's identity is substituted, never a validator result.
    monkeypatch.setattr(m, "SOURCE_HEAD", old["source_git_commit"])
    paths.add(m.MODULE)
    for path in m.ALLOWED_FILES:
        text = yaml.safe_dump(target) if path == m.PROFILE else (
            code + f"RR_CAPTURE_FEEDBACK_REVISION = {m.FEEDBACK_REVISION!r}\n"
            f"RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS = {m.WINDOW_REFERENCE_SEMANTICS!r}\n"
            f"RR_CAPTURE_SEARCH_SEMANTICS = {m.CAPTURE_SEARCH_SEMANTICS!r}\n"
            if path == m.ASSIST else "# synthetic reviewed target\n")
        write(root / path, text)
    git(root, "add", "."); git(root, "commit", "-m", "synthetic target")
    new = contract(new); meta["runtime_contract"] = old
    bind_prior_receipt(meta, old)
    return root, old, new


def test_official_v3_publish_reload_and_carry_preserve_full_state(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch"); pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training, semantic_migration as shared
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available(), "CPU-only synthetic publication"
    def make():
        return training.construct_semantic_runner(_shape_env(410, "cpu"), seed=1001, device="cpu",
            policy_version=m.RR_CAPTURE_POLICY, observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT, initialize_actor=False)[0]
    runner = make()
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 2.25e-5
        for parameter in group["params"]:
            runner.alg.optimizer.state[parameter] = {"step":torch.tensor(9.), "exp_avg":torch.full_like(parameter, .001),
                                                    "exp_avg_sq":torch.full_like(parameter, .002)}
    runner.alg.learning_rate = 2.25e-5
    meta = boundary()[0]; root, old, new = fixture_repo(tmp_path, meta, monkeypatch)
    source, _ = training.save_semantic_checkpoint(runner, tmp_path / "source.pt", meta)
    source_meta = shared.checkpoint_metadata(source)
    monkeypatch.setattr(m, "SOURCE_CHECKPOINT_SHA256", shared.file_sha(source))
    review = {p:sha for p,sha in new["files"].items() if old["files"].get(p) != sha}
    plan = m.build_rr_capture_knee_migration(source, new, reason="synthetic peak and pre-cross hold identity boundary",
        reviewed_code_sha256=review, project_root=root)
    path = tmp_path / "plan.json"; write(path, json.dumps(plan))
    monkeypatch.setattr(shared, "PROJECT_ROOT", root)
    original_validate = shared.validate_migration_plan
    monkeypatch.setattr(shared, "validate_migration_plan", lambda c, contract, p, **kw:
        original_validate(c, contract, p, project_root=root))
    verified = shared.validate_migration_plan(source, new, path)
    assert verified["schema"] == m.SCHEMA
    bad = deepcopy(plan); bad[m.FACTOR_KEY]["reward_changed"] = True
    bad_path = tmp_path / "tampered.json"; write(bad_path, json.dumps(bad))
    with pytest.raises(ValueError): shared.validate_migration_plan(source, new, bad_path)
    published = m.publish_rr_capture_knee_checkpoint(source, new, path, tmp_path / "target.pt")
    target = shared.checkpoint_metadata(Path(published["checkpoint"]))
    assert all(target[k] == source_meta[k] for k in m.preserved_keys(source_meta))
    assert target[m.PRIOR_FEEDBACK_MIGRATION] == source_meta[m.PRIOR_FEEDBACK_MIGRATION]
    assert target[m.MIGRATION] == verified and published["observation_dimension"] == 410
    assert published["original_rr_branch_preserved"] and published["migration_added_optimizer_steps"] == 0
    with pytest.raises(FileExistsError): m.publish_rr_capture_knee_checkpoint(source, new, path, source)
    fresh = make()
    loaded = training.load_semantic_checkpoint(fresh, Path(published["checkpoint"]), contract=new, seed=1001)
    assert fresh.alg.storage.step == 0 and fresh.alg.transition.actions is None
    # Same numeric input retains Gaussian mean/sigma and critic exactly, even for
    # the newly declared travel/exposure range. These are synthetic probes only.
    probes = _shape_env(410, "cpu").get_observations()
    probes["policy"][..., 0] = 0.; probes["policy"][..., 8] = 1.
    probes["policy"][..., 391] = -.1; probes["policy"][..., 394] = 1.5
    probes["policy"][..., 395] = 2.; probes["critic"] = probes["policy"].clone()
    with torch.no_grad():
        runner.alg.actor(probes, stochastic_output=True)
        old_distribution = tuple(x.clone() for x in runner.alg.actor.output_distribution_params)
        fresh.alg.actor(probes, stochastic_output=True)
        assert all(torch.equal(a,b) for a,b in zip(old_distribution, fresh.alg.actor.output_distribution_params))
        assert torch.equal(runner.alg.critic(probes), fresh.alg.critic(probes))
    # Assert the real production train carry list, then ordinary official save.
    tree = ast.parse(inspect.getsource(training.train_semantic))
    carry = [ast.literal_eval(n.iter) for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)
             and any(isinstance(x, ast.Constant) and x.value == "new_mdp_warm_start" for x in n.iter.elts)]
    assert len(carry) == 1 and m.MIGRATION in carry[0]
    later = {k:v for k,v in loaded.items() if not k.endswith(("_branch", "_migration"))}
    for key in carry[0]:
        if key in loaded: later[key] = loaded[key]
    later.update(global_policy_decisions=loaded["global_policy_decisions"] + 128,
                 ppo_updates=loaded["ppo_updates"] + 1, optimizer_steps=loaded["optimizer_steps"] + 20)
    cp, _ = training.save_semantic_checkpoint(fresh, tmp_path / "later_synthetic_ledger.pt", later)
    last = shared.checkpoint_metadata(cp)
    assert last[m.MIGRATION] == verified and last["rr_capture_transfer_branch"] == source_meta["rr_capture_transfer_branch"]
    assert last[m.PRIOR_FEEDBACK_MIGRATION] == source_meta[m.PRIOR_FEEDBACK_MIGRATION]
    assert last["rr_capture_transfer_branch_counts"] == {"global_policy_decisions":128, "ppo_updates":1, "optimizer_steps":20}
    for key in ("rr_postcross_workspace_branch", "task_conditioned_hip_wheel_branch"):
        assert last[key] == source_meta[key]
    # These counters are a synthetic save-carry fixture, not optimizer credit.
    restored = make(); restored_info = training.load_semantic_checkpoint(restored, source, contract=old, seed=1001)
    restored.alg.storage.step = 1
    with pytest.raises(RuntimeError, match="rollout"):
        m.record_loaded_rr_capture_knee(restored, restored_info, verified)
