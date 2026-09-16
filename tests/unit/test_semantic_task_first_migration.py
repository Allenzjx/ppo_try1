"""Directed reward-boundary and selective mean-head tests; no Isaac or real CP writes."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES, HISTORY_TEMPERED_POLICY, policy_contract
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


@pytest.fixture
def migration_case(tmp_path):
    from wlr50_clean.ppo.semantic_training import semantic_runner_config
    source_root = "configs/ppo_fsm_reference_p09_stable_v2"
    target_root = "configs/ppo_task_first_recovery_v1"
    paths = []
    for name in CONFIG_NAMES:
        for namespace in (source_root, target_root):
            relative = f"{namespace}/{name}"
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((m.PROJECT_ROOT / source_root / name).read_bytes())
            paths.append(relative)
    reward = tmp_path / target_root / "reward_config.yaml"
    data = yaml.safe_load(reward.read_bytes())
    data.update(revision="task_first_recovery_epsilon_zero_v1", objective_profile="task_first_recovery_v1", quality_epsilon=0.)
    for key in ("body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"):
        data["family_weights"][key] = 0.
    reward.write_text(yaml.safe_dump(data), encoding="utf-8")
    for relative in (m.BODY_REWARD_CODE, m.SUPERVISOR):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic unchanged production bytes\n", encoding="utf-8")
    shared = [m.BODY_REWARD_CODE, m.SUPERVISOR]
    def contract(namespace, experiment, head, relative_paths):
        files = {p: m.file_sha(tmp_path / p) for p in relative_paths}
        return {"files": files, "runtime_content_sha256": m.digest(files), "source_git_commit": head * 40,
            "semantic_version": "v3", "experiment_id": experiment, "physics_hz": 120., "decision_hz": 15.,
            "task_timeout_s": 200., "timeout_bootstrap": False,
            "selected_configuration": {n: {"path": f"{namespace}/{n}", "sha256": files[f"{namespace}/{n}"]} for n in CONFIG_NAMES}}
    old = contract(source_root, "fsm_reference_p09_stable_v2", "a", shared + [p for p in paths if source_root in p])
    new = contract(target_root, "task_first_recovery_v1", "b", shared + paths)
    cp = tmp_path / "source.pt"
    cp.write_bytes(b"synthetic metadata-only checkpoint")
    metadata = {"schema": "wlr50_clean.semantic_checkpoint.v1", "checkpoint_path": str(cp.resolve()),
        "checkpoint_sha256": m.file_sha(cp), "save_load_round_trip": True,
        "semantic_version": "v3", "runtime_contract": old, "seed": 1001,
        "runner_config": semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
            policy_version=HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT),
        "sampling": "P01_full_task_only_initial_version",
        "execution_topology": m.continuation_topology("P01_full_task_only_initial_version", None,
            observation_layout=ROLE_OBSERVATION_LAYOUT),
        "policy_contract": policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)}
    sidecar = cp.with_name("source_manifest.json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, cp=cp, sidecar=sidecar, old=old, new=new, metadata=metadata,
                           source_root=source_root, target_root=target_root)


def build(f, **extra):
    delta = sorted(p for p in f.old["files"].keys() | f.new["files"].keys() if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.cp, f.new, allowed_changed_files=delta, reason="task-first boundary",
        task_first_reward_review={"reason": "epsilon-zero isolated reward profile",
            "reviewed_code_sha256": {p: f.new["files"][p] for p in delta if not p.startswith(f.target_root+"/")}},
        project_root=f.root, **extra)


def test_task_first_plan_roundtrip_preserves_N_and_old_checkpoint(migration_case):
    f = migration_case
    old_bytes = f.cp.read_bytes(), f.sidecar.read_bytes()
    plan = build(f)
    factor = plan["task_first_reward_factor"]
    assert factor["quality_epsilon"] == 0.
    assert factor["reward_changed"] and not factor["nominal_control_changed"]
    assert not factor["action_ranges_changed"] and not factor["kernel_changed"]
    assert factor["observation_contract"]["observation_dimension"] == 372
    assert plan["discard_old_rollout_storage"] and factor["migration_added_updates"] == 0
    path = f.root / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert m.validate_migration_plan(f.cp, f.new, path, project_root=f.root)["task_first_reward_factor"] == factor
    assert old_bytes == (f.cp.read_bytes(), f.sidecar.read_bytes())


def rebind(f, path):
    f.new["files"][path] = m.file_sha(f.root / path)
    f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    name = path.rsplit("/", 1)[-1]
    if name in f.new["selected_configuration"]:
        f.new["selected_configuration"][name]["sha256"] = f.new["files"][path]


@pytest.mark.parametrize("kind", ["nominal", "nonreward_config", "epsilon", "failure_cost", "physics", "sigma_kernel"])
def test_task_first_cannot_relax_task_or_change_controller(migration_case, kind):
    f = migration_case
    if kind == "physics":
        f.new["physics_hz"] = 60.
    elif kind in ("nominal", "sigma_kernel"):
        p = m.SUPERVISOR if kind == "nominal" else "src/wlr50_clean/ppo/semantic_history_actor.py"
        (f.root / p).parent.mkdir(parents=True, exist_ok=True)
        (f.root / p).write_text("# changed\n", encoding="utf-8")
        rebind(f, p)
    else:
        name = "execution_profile.yaml" if kind == "nonreward_config" else "reward_config.yaml"
        p = f"{f.target_root}/{name}"
        data = yaml.safe_load((f.root / p).read_bytes())
        if kind == "epsilon": data["quality_epsilon"] = .1
        elif kind == "failure_cost": data["failure_cost"] = 0.
        else: data["nominal_geometry_advisory"] = "changed"
        (f.root / p).write_text(yaml.safe_dump(data), encoding="utf-8")
        rebind(f, p)
    with pytest.raises(ValueError): build(f)


def test_task_first_rejects_mixed_factors(migration_case):
    with pytest.raises(ValueError, match="cannot mix"):
        build(migration_case, exploration_temperature_review={})


def test_task_first_requires_review_hashes_and_immutable_receipt(migration_case):
    f = migration_case
    plan = build(f)
    plan["task_first_reward_factor"]["optimizer"] = "reset all"
    path = f.root / "tampered.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.cp, f.new, path, project_root=f.root)


@pytest.fixture
def mean_case():
    torch = pytest.importorskip("torch")
    from tensordict import TensorDict
    from wlr50_clean.ppo.semantic_history_actor import SemanticTemperedHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import semantic_runner_config, parameter_hash, state_hash
    prior_threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(90123)
    cfg = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
        policy_version=HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)["actor"]
    cfg = copy.deepcopy(cfg)
    cfg.pop("class_name")
    obs = TensorDict({"policy": torch.zeros(2, 372)}, batch_size=[2])
    actor = SemanticTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **cfg)
    critic = torch.nn.Linear(372, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-5, amsgrad=True)
    for parameter in [*actor.parameters(), *critic.parameters()]: parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer,
        storage=SimpleNamespace(step=0), transition=SimpleNamespace(actions=None)),
        _semantic_policy_version=HISTORY_TEMPERED_POLICY, _semantic_observation_layout=ROLE_OBSERVATION_LAYOUT)
    infos = {"policy_contract": policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT),
        "runtime_contract": {"experiment_id": "task_first_recovery_v1"},
        "global_policy_decisions": 176640, "ppo_updates": 1345, "optimizer_steps": 26900,
        "resume_source_checkpoint": {"checkpoint_sha256": "a"*64, "manifest_sha256": "b"*64},
        "actor_parameter_sha256": parameter_hash(actor), "critic_parameter_sha256": parameter_hash(critic),
        "optimizer_state_sha256": state_hash(optimizer.state_dict())}
    yield SimpleNamespace(runner=runner, infos=infos, obs=obs)
    torch.set_rng_state(rng)
    torch.set_num_threads(prior_threads)


def test_mean_reset_only_mean_rows_and_corresponding_adam_moments(mean_case):
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash, parameter_hash
    f = mean_case
    actor = f.runner.alg.actor
    old_actor = copy.deepcopy(actor.state_dict())
    old_optimizer = copy.deepcopy(f.runner.alg.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    result = m.apply_task_recovery_mean_head(f.runner, f.infos, branch_id="mean_head_recovery_v1", reason="same mean failed after new objective block")
    for key, value in actor.state_dict().items():
        if key in ("mlp.4.weight", "mlp.4.bias"):
            assert not bool(value[:12].any())
            assert torch.equal(value[12:], old_actor[key][12:])
        else: assert torch.equal(value, old_actor[key])
    new_optimizer = f.runner.alg.optimizer.state_dict()
    assert old_optimizer["param_groups"] == new_optimizer["param_groups"]
    head_ids = {next(i for i, p in enumerate(f.runner.alg.optimizer.param_groups[0]["params"]) if p is head)
                for head in (actor.mlp[4].weight, actor.mlp[4].bias)}
    for index, state in new_optimizer["state"].items():
        if index not in head_ids: assert state_hash(state) == state_hash(old_optimizer["state"][index])
        else:
            assert torch.equal(state["step"], old_optimizer["state"][index]["step"])
            for moment in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                assert not bool(state[moment][:12].any())
                assert torch.equal(state[moment][12:], old_optimizer["state"][index][moment][12:])
    assert parameter_hash(f.runner.alg.critic) == f.infos["critic_parameter_sha256"]
    assert torch.equal(rng, torch.get_rng_state())
    assert not bool(actor(f.obs).any())
    f.obs["policy"][:,195:207] = 1.
    assert torch.allclose(actor(f.obs), torch.full((2,12), .9))  # HISTORY is not silently bypassed.
    assert result["global_policy_decisions"] == 176640
    branch = result["task_recovery_branch"]
    assert branch["branch_policy_decisions"] == branch["branch_ppo_updates"] == 0
    assert branch["initialization_is_not_PPO_learning_success"]
    assert "task_recovery_branch" not in f.infos


@pytest.mark.parametrize("bad", ["partial_rollout", "transition", "wrong_hash", "wrong_profile", "already_branch", "missing_moment"])
def test_invalid_mean_branch_leaves_live_state_unchanged(mean_case, bad):
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    f = mean_case
    if bad == "partial_rollout": f.runner.alg.storage.step = 1
    elif bad == "transition": f.runner.alg.transition.actions = torch.zeros(1,12)
    elif bad == "wrong_hash": f.infos["actor_parameter_sha256"] = "0"*64
    elif bad == "wrong_profile": f.infos["runtime_contract"]["experiment_id"] = "fsm_reference_p09_stable_v2"
    elif bad == "already_branch": f.infos["task_recovery_branch"] = {}
    else:
        del f.runner.alg.optimizer.state[f.runner.alg.actor.mlp[4].bias]["exp_avg"]
        f.infos["optimizer_state_sha256"] = state_hash(f.runner.alg.optimizer.state_dict())
    before = state_hash(f.runner.alg.actor.state_dict()), state_hash(f.runner.alg.optimizer.state_dict())
    with pytest.raises(ValueError):
        m.apply_task_recovery_mean_head(f.runner, f.infos, branch_id="mean_head_recovery_v1", reason="explicit")
    assert before == (state_hash(f.runner.alg.actor.state_dict()), state_hash(f.runner.alg.optimizer.state_dict()))


def test_offline_initializer_actual_official_save_load_without_physics(tmp_path, monkeypatch):
    import importlib.util
    import torch
    from wlr50_clean.ppo import semantic_cli
    from wlr50_clean.ppo.semantic_training import (SemanticRslAdapter, construct_semantic_runner,
        save_semantic_checkpoint, seed_training_rngs)
    script = m.PROJECT_ROOT / "outputs/ppo_task_first_recovery_v1/initialize_mean_head_recovery.py"
    spec = importlib.util.spec_from_file_location("offline_mean_initializer_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    contract = {"experiment_id": "task_first_recovery_v1", "synthetic_fixture": True}
    monkeypatch.setattr(semantic_cli, "runtime_contract", lambda **kwargs: contract)
    class NoPhysics:
        def reset(self, *, seed): return (0.,) * 372
        def step(self, raw): raise AssertionError("offline migration must never collect physics")
    prior_threads, rng = torch.get_num_threads(), torch.get_rng_state()
    try:
        torch.set_num_threads(1)
        seed_training_rngs(1001)
        env = SemanticRslAdapter(NoPhysics(), seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, _ = construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=HISTORY_TEMPERED_POLICY, initialize_actor=False,
            observation_layout=ROLE_OBSERVATION_LAYOUT)
        for group in runner.alg.optimizer.param_groups:
            for p in group["params"]: p.grad = torch.ones_like(p)
        runner.alg.optimizer.step()
        runner.alg.optimizer.zero_grad(set_to_none=True)
        cp = tmp_path / "outputs/ppo_task_first_recovery_v1/checkpoints/history/synthetic_source.pt"
        infos = {"seed": 1001, "runtime_contract": contract, "semantic_version": "v3",
            "global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 1,
            "sampling": "P01_full_task_only_initial_version",
            "execution_topology": m.continuation_topology("P01_full_task_only_initial_version", None,
                observation_layout=ROLE_OBSERVATION_LAYOUT),
            "stage_requested_decisions": {"smoke": 128, "phase_suffix": 0, "full_episode": 0}}
        save_semantic_checkpoint(runner, cp, infos)
        before = m.file_sha(cp)
        result = module.initialize(cp, expected_head="a"*40, branch_id="cpu_fixture_branch",
                                   reason="synthetic fixture only; not physical success")
        assert result["actual_save_load_round_trip"] and result["source_checkpoint_unchanged"]
        assert result["offline_initialization"]["physics_steps"] == 0
        assert m.file_sha(cp) == before
        new_meta = m.checkpoint_metadata(__import__("pathlib").Path(result["candidate_checkpoint"]))
        assert new_meta["global_policy_decisions"] == 128
        assert new_meta["task_recovery_branch_counts"] == {"global_policy_decisions": 0, "ppo_updates": 0, "optimizer_steps": 0}
        assert not (cp.parents[1] / "checkpoint_last_pointer.json").exists()
        with pytest.raises(FileExistsError):
            module.initialize(cp, expected_head="a"*40, branch_id="cpu_fixture_branch", reason="cannot overwrite")
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(prior_threads)
