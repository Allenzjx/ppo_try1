"""Reviewed height boundary: synthetic Git, real CPU learner, no Isaac or saved user CP."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as m
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, git, source_metadata, write_json

CONFIG = "configs/ppo_fsm_reference_p09_stable_v2"
GEOMETRY = "src/wlr50_clean/ppo/semantic_nominal_geometry.py"
BACKEND = "src/wlr50_clean/ppo/semantic_backend.py"
VIDEO = "src/wlr50_clean/ppo/semantic_video.py"


def candidate(name="FL_minus4", fl=4., rl=0.):
    return {"mode": m.HEIGHT_RECOVERY_MODE, "candidate_id": name,
        "preparation_reduction_deg": {"front_left_hip": fl, "rear_left_hip": rl},
        "post_lift_recovery_deg": {"front_left_hip": 0., "rear_left_hip": 0.},
        "recovery_rate_deg_s": 15.}


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {name: {"path": f"{CONFIG}/{name}",
        "sha256": contract["files"][f"{CONFIG}/{name}"]} for name in NAMES}


def yaml_edit(root, path, change):
    value = yaml.safe_load((root / path).read_bytes())
    change(value)
    (root / path).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("height_recovery_git")
    paths = (m.HEIGHT_FILES - m.HEIGHT_NEW_FILES) | {f"{CONFIG}/{n}" for n in NAMES} | {PROTECTED}
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((m.PROJECT_ROOT / path).read_bytes())
    yaml_edit(root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].pop("height_recovery", None))
    yaml_edit(root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].pop("final_stop_owner", None))
    yaml_edit(root, m.HEIGHT_EXECUTION, lambda v: v.update(nominal_geometry_advisory="functional_rr_preplace_nominal_advisory_v2"))
    modules = {
        m.SUPERVISOR: "LOCKED = 1\nclass NominalMotionProvider:\n    amount = 1\nclass TaskEvaluator:\n    pass\n",
        GEOMETRY: "LOCKED = 1\ndef capture_nominal_geometry_context():\n    return 1\ndef correct_nominal_geometry():\n    return 1\n",
        BACKEND: "LOCKED = 1\ndef load_execution_profile():\n    return 1\nclass SemanticIsaacBackend:\n    def __init__(self):\n        self.mode = 1\n    def _atomic_apply(self):\n        return 1\n    def reset(self):\n        return 0\n",
        VIDEO: "LOCKED = 1\ndef capture_semantic_video():\n    return 1\nclass EndpointObserver:\n    def __init__(self):\n        self.mode = 1\n    def __call__(self):\n        return 1\n"}
    for p, value in modules.items(): (root / p).write_text(value, encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "Height fixture")
    git(root, "config", "user.email", "height@example.invalid")
    old = {"source_git_commit": commit(root, "before height"), "files": dict.fromkeys(sorted(paths)),
        "semantic_version": "v3", "experiment_id": "fsm_reference_p09_stable_v2",
        "frozen_A_files": {"frozen": "a"*64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "CPU only"}}
    bind(root, old)
    yaml_edit(root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].update(height_recovery=candidate()))
    yaml_edit(root, m.HEIGHT_EXECUTION, lambda v: v.update(nominal_geometry_advisory=m.HEIGHT_GEOMETRY_MODE))
    for p, value in modules.items():
        (root / p).write_text(value.replace("return 1", "return 2").replace("amount = 1", "amount = 2"), encoding="utf-8")
    for p in m.HEIGHT_NEW_FILES: (root / p).write_bytes((m.PROJECT_ROOT / p).read_bytes())
    new = copy.deepcopy(old)
    new["files"].update(dict.fromkeys(m.HEIGHT_NEW_FILES))
    new["source_git_commit"] = commit(root, "height candidate")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque fixture, not torch loaded")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **options):
    delta = sorted(p for p in set(f.old["files"]) | set(f.new["files"])
        if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason="Height/P02 reviewed candidate, not physical equivalence",
        height_recovery_review={"reason": "Reviewed exact candidate code and bounded configuration",
            "reviewed_code_sha256": {p: f.new["files"].get(p) for p in m.HEIGHT_CODE_FILES}},
        project_root=f.root, **options)


def test_bound_plan_round_trip_and_immutable_source(f):
    immutable = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    plan = record(f)
    factor = plan["height_recovery_factor"]
    assert factor["schema"] == m.HEIGHT_RECOVERY_SCHEMA
    assert factor["source_candidate"] is None and factor["target_candidate"] == candidate()
    assert plan["observation_dimension"] == 372 and plan["action_dimension"] == 12
    assert plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"]
    assert plan["discard_old_rollout_storage"] and plan["physics_resume"] == "fresh_legal_P01_reset"
    assert factor["physical_mdp_changed"] and not factor["reward_changed"]
    path = f.root / "plan.json"
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)["height_recovery_factor"] == factor
    plan["height_recovery_factor"]["target_candidate"]["recovery_rate_deg_s"] = 60.
    write_json(path, plan)
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)
    assert immutable == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())


def test_config_only_candidate_revision_from_height_source(f):
    f.old = copy.deepcopy(f.new)
    f.metadata["runtime_contract"] = copy.deepcopy(f.old)
    write_json(f.sidecar, f.metadata)
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].update(height_recovery=candidate("RL_minus3", 0., 3.)))
    bind(f.root, f.new)
    plan = record(f)
    assert plan["allowed_changed_files"] == [m.TIMING_ONLY_SPEC]
    assert plan["height_recovery_factor"]["source_candidate"] == candidate()
    assert plan["height_recovery_factor"]["target_candidate"] == candidate("RL_minus3", 0., 3.)


def test_reviewed_final_stop_owner_preserves_physical_acceptance(f):
    f.old = copy.deepcopy(f.new)
    f.metadata["runtime_contract"] = copy.deepcopy(f.old)
    write_json(f.sidecar, f.metadata)
    yaml_edit(f.root, m.TIMING_ONLY_SPEC,
              lambda v: v["nominal"].update(final_stop_owner=m.HEIGHT_FINAL_STOP_MODE))
    bind(f.root, f.new)
    plan = record(f)
    factor = plan["height_recovery_factor"]
    assert plan["allowed_changed_files"] == [m.TIMING_ONLY_SPEC]
    assert factor["final_stop_nominal_owner"] == {
        "source_mode": None, "target_mode": m.HEIGHT_FINAL_STOP_MODE,
        "task_acceptance_and_post_completion_window_changed": False,
        "policy_channels_or_action_distribution_changed": False,
    }
    assert not factor["task_acceptance_changed"] and not factor["action_ranges_changed"]
    assert plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"]
    assert plan["discard_old_rollout_storage"]
    path = f.root / "final_stop_plan.json"
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)["height_recovery_factor"] == factor


@pytest.mark.parametrize("mode", [False, True, "skip_observation", 1, {}])
def test_final_stop_owner_rejects_unreviewed_modes(f, mode):
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].update(final_stop_owner=mode))
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="final-stop owner"):
        record(f)


def test_final_stop_owner_does_not_open_observer_or_task_thresholds(f):
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].update(final_stop_owner=m.HEIGHT_FINAL_STOP_MODE))
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["final"].update(post_completion_observation_s=0.1))
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="task spec"):
        record(f)


def test_final_stop_owner_cannot_silently_remove_existing_owner(f):
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].update(final_stop_owner=m.HEIGHT_FINAL_STOP_MODE))
    bind(f.root, f.new)
    f.new["source_git_commit"] = commit(f.root, "reviewed final stop source")
    f.old = copy.deepcopy(f.new)
    f.metadata["runtime_contract"] = copy.deepcopy(f.old)
    write_json(f.sidecar, f.metadata)
    yaml_edit(f.root, m.TIMING_ONLY_SPEC, lambda v: v["nominal"].pop("final_stop_owner"))
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="final-stop owner"):
        record(f)


@pytest.mark.parametrize("case", ["negative", "too_large", "nan", "bool", "other_leg", "rate_zero", "rate_large",
    "mode", "extra", "no_id", "missing", "task", "range", "reward", "obs", "physics", "history",
    "supervisor_outside", "geometry_outside", "backend_reset", "video_outside", "helper_setter", "helper_extra"])
def test_rejects_outside_the_exact_boundary(f, case):
    def bad_spec(v):
        c = v["nominal"]["height_recovery"]
        if case in ("negative", "too_large", "nan", "bool"):
            c["preparation_reduction_deg"]["front_left_hip"] = {"negative": -1., "too_large": 10.1, "nan": float("nan"), "bool": True}[case]
        elif case == "other_leg": c["preparation_reduction_deg"]["rear_right_hip"] = 0.
        elif case in ("rate_zero", "rate_large"): c["recovery_rate_deg_s"] = 0 if case == "rate_zero" else 60.1
        elif case == "mode": c["mode"] = "generic372"
        elif case == "extra": c["unreviewed"] = 1
        elif case == "no_id": c["candidate_id"] = " "
        elif case == "missing": del v["nominal"]["height_recovery"]
        elif case == "task": v["rear_leg_order"] = "RL_first"
    if case in ("negative", "too_large", "nan", "bool", "other_leg", "rate_zero", "rate_large", "mode", "extra", "no_id", "missing", "task"):
        yaml_edit(f.root, m.TIMING_ONLY_SPEC, bad_spec)
    elif case == "range": yaml_edit(f.root, m.HEIGHT_EXECUTION, lambda v: v["residual"].update(servo_rate_deg_s=100.))
    elif case in ("reward", "obs"):
        p = f.root / (PROTECTED if case == "reward" else f"{CONFIG}/observation_schema.json")
        p.write_bytes(p.read_bytes() + b"\n")
    elif case == "physics": f.new["physics_hz"] = 60.
    elif case == "history":
        f.metadata["policy_contract"]["rho"] = .8
        write_json(f.sidecar, f.metadata)
    else:
        name = {"supervisor_outside": m.SUPERVISOR, "geometry_outside": GEOMETRY,
            "backend_reset": BACKEND, "video_outside": VIDEO,
            "helper_setter": "src/wlr50_clean/ppo/semantic_height_diagnostics.py",
            "helper_extra": "src/wlr50_clean/ppo/semantic_height_recovery.py"}[case]
        p = f.root / name
        text = p.read_text()
        if case.endswith("outside"): text = text.replace("LOCKED = 1", "LOCKED = 2")
        elif case == "backend_reset": text = text.replace("return 0", "return 9")
        elif case == "helper_setter": text = text.replace("before = self._clock()", "before = self._clock()\n        self.robot.set_joint_position_target(None)")
        else: text += "\ndef unreviewed():\n    return 1\n"
        p.write_text(text, encoding="utf-8")
    bind(f.root, f.new)
    with pytest.raises(ValueError): record(f)


def test_review_required_exclusive_and_hash_bound(f):
    with pytest.raises(ValueError, match="cannot mix"): record(f, timing_review={"reason": "no"})
    delta = sorted(p for p in f.new["files"] if f.old["files"].get(p) != f.new["files"][p])
    with pytest.raises(ValueError):
        m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta, reason="no review", project_root=f.root)
    with pytest.raises(ValueError, match="every exact"):
        m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta, reason="bad review", project_root=f.root,
            height_recovery_review={"reason": "bad", "reviewed_code_sha256": {}})


def test_existing_cli_accepts_height_factor(f, monkeypatch):
    path = f.root / "plan.json"
    write_json(path, record(f))
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    original = m.validate_migration_plan
    monkeypatch.setattr(m, "validate_migration_plan", lambda cp, c, p: original(cp, c, p, project_root=f.root))
    args = cli.parser().parse_args(["train", "--semantic-version", "v3", "--experiment-id",
        "fsm_reference_p09_stable_v2", "--run-dir", str(f.root / "runs/ppo_fsm_reference_p09_stable_v2/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint), "--resume-migration", str(path),
        "--stage", "full_episode", "--from-phase", "P01", "--decisions", "128", "--seed", "1001", "--device", "cpu"])
    cli.validate_request(args)
    cli._preflight_checkpoint(args, f.new)
    assert args._migration_record["height_recovery_factor"]["schema"] == m.HEIGHT_RECOVERY_SCHEMA


def test_real_cpu372_keeps_adam_rng_counters_and_continues(prototype, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as t
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        root = tmp_path / "real_cpu"
        shutil.copytree(prototype.root, root)
        t.seed_training_rngs(1001)
        source, source_env = make(True)
        result = t.train_semantic(source, source_env, run_dir=root / "old_run", output_root=root / "old_output",
            stage="full_episode", decisions=128, contract=prototype.old, seed=1001)
        metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
        for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"): metadata.pop(key)
        source.alg.learning_rate = 1e-5
        for group in source.alg.optimizer.param_groups: group.update(lr=1e-5, betas=(.87, .996), eps=2e-8)
        cp = root / "learned.pt"
        _, sidecar = t.save_semantic_checkpoint(source, cp, metadata)
        saved = m.checkpoint_metadata(cp)
        assert source.alg.optimizer.state_dict()["state"]
        f = SimpleNamespace(root=root, old=prototype.old, new=prototype.new, checkpoint=cp)
        plan = root / "plan.json"
        write_json(plan, record(f))
        original = m.validate_migration_plan
        monkeypatch.setattr(m, "validate_migration_plan", lambda cp, c, p: original(cp, c, p, project_root=root))
        verified = m.validate_migration_plan(cp, prototype.new, plan)
        target, env = make(True)
        loaded = t.load_semantic_checkpoint(target, cp, contract=prototype.new, seed=1001, migration=verified)
        for role in ("actor", "critic"):
            assert t.parameter_hash(getattr(source.alg, role)) == t.parameter_hash(getattr(target.alg, role))
        assert t.state_hash(target.alg.optimizer.state_dict()) == saved["optimizer_state_sha256"]
        assert t.state_hash(t._normalizers(target)) == saved["normalizer_state_sha256"]
        assert t.capture_training_rng_state(seed=1001) == saved["training_rng_state"]
        assert target.alg.learning_rate == t.optimizer_learning_rate(target) == 1e-5
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None and env.core.calls == 0
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions"):
            assert loaded[key] == saved[key]
        target.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="partial old rollout"):
            t.load_semantic_checkpoint(target, cp, contract=prototype.new, seed=1001, migration=verified)
        target.alg.storage.step = 0
        continued = t.train_semantic(target, env, run_dir=root / "new_run", output_root=root / "new_output",
            stage="full_episode", decisions=128, contract=prototype.new, seed=1001, resume_infos=loaded)
        final, _ = make(True)
        infos = t.load_semantic_checkpoint(final, Path(continued["checkpoints"][-1]["checkpoint"]),
            contract=prototype.new, seed=1001)
        assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (256, 2, 40)
        assert infos["resume_ancestry"]["resume_migration"]["height_recovery_factor"]["schema"] == m.HEIGHT_RECOVERY_SCHEMA
    finally:
        torch.set_num_threads(original_threads)
