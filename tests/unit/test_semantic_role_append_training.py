"""Official CPU load/pad/update/resume; pure plan validation is separate."""
import copy
import json
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from test_semantic_v3_continuation import Core324
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, policy_contract
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


class Core372(Core324):
    def reset(self, **kwargs):
        return super().reset(**kwargs) + (.4,) * 48

    def step(self, raw):
        result = super().step(raw)
        result.observation += (.4,) * 48
        return result


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(count)


def make(append=False):
    env = training.SemanticRslAdapter(Core372() if append else Core324(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
        policy_version=HISTORY_POLICY, initialize_actor=False,
        observation_layout=ROLE_OBSERVATION_LAYOUT if append else None)
    return runner, env


@pytest.fixture
def source(tmp_path):
    training.seed_training_rngs(1001)
    runner, env = make()
    old_contract = {"revision": "synthetic_old324"}
    result = training.train_semantic(runner, env, run_dir=tmp_path / "source_run",
        output_root=tmp_path / "source_output", stage="full_episode", decisions=128,
        contract=old_contract, seed=1001)
    metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        metadata.pop(key)
    metadata["new_mdp_origin_global_policy_decisions"] = 0
    checkpoint = tmp_path / "source.pt"
    training.save_semantic_checkpoint(runner, checkpoint, metadata)
    record = {"source_runtime_contract": old_contract,
        "target_runtime_contract": {"revision": "synthetic_new372"},
        "target_stage_requested_decisions": metadata["stage_requested_decisions"],
        "new_mdp_origin_global_policy_decisions": 0,
        "observation_append_transition": {
            "schema": "wlr50_clean.transfer_roles_observation_append_transition.v1",
            "source_observation_dimension": 324, "target_observation_dimension": 372,
            "source_observation_layout": None, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
            "first_layer_parameter": "mlp.0.weight", "appended_columns": [324, 372],
            "source_policy_contract": policy_contract(HISTORY_POLICY),
            "target_policy_contract": policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)}}
    return runner, checkpoint, record


def load(source, monkeypatch):
    old, checkpoint, record = source
    monkeypatch.setattr(migration, "build_v3_warm_start_record", lambda *args, **kwargs: copy.deepcopy(record))
    target, env = make(True)
    infos = training.load_semantic_checkpoint(target, checkpoint,
        contract=record["target_runtime_contract"], seed=1001, warm_start=record)
    return target, env, infos


def test_real_append_preserves_weights_function_rng_counters_and_fresh_adam(source, monkeypatch):
    old, checkpoint, record = source
    before_bytes = checkpoint.read_bytes()
    target, env, infos = load(source, monkeypatch)
    for role in ("actor", "critic"):
        a, b = getattr(old.alg, role).state_dict(), getattr(target.alg, role).state_dict()
        for key in a:
            if key == "mlp.0.weight":
                assert torch.equal(a[key], b[key][:, :324])
                assert torch.count_nonzero(b[key][:, 324:]) == 0
            else:
                assert torch.equal(a[key], b[key])
    assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (128, 1, 20)
    assert training.capture_training_rng_state(seed=1001) == infos["training_rng_state"]
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert target.alg.actor.obs_normalizer.state_dict() == target.alg.critic.obs_normalizer.state_dict() == {}
    assert env.core.calls == 0 and env.core.resets == 1
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert checkpoint.read_bytes() == before_bytes
    tensor = torch.randn(13, 372, generator=torch.Generator().manual_seed(777))
    new_obs = TensorDict({"policy": tensor, "critic": tensor}, batch_size=[13])
    old_obs = TensorDict({"policy": tensor[:, :324], "critic": tensor[:, :324]}, batch_size=[13])
    with torch.inference_mode():
        for role in ("actor", "critic"):
            torch.testing.assert_close(getattr(old.alg, role)(old_obs), getattr(target.alg, role)(new_obs), rtol=2e-5, atol=3e-6)
    assert infos["observation_append_evidence"]["physical_steps_added"] == 0


def test_appended_inputs_train_and_roundtrip_with_exact_resume(source, monkeypatch, tmp_path):
    target, env, infos = load(source, monkeypatch)
    contract = source[2]["target_runtime_contract"]
    result = training.train_semantic(target, env, run_dir=tmp_path / "new_run",
        output_root=tmp_path / "new_output", stage="full_episode", decisions=128,
        contract=contract, seed=1001, resume_infos=infos)
    assert result["global_policy_decisions"] == 256 and result["ppo_updates_this_run"] == 1
    assert torch.count_nonzero(target.alg.actor.mlp[0].weight[:, 324:]) > 0
    assert torch.count_nonzero(target.alg.critic.mlp[0].weight[:, 324:]) > 0
    metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
    assert metadata["execution_topology"]["observation_dimension"] == 372
    assert metadata["observation_append_evidence"] == infos["observation_append_evidence"]
    fresh, _ = make(True)
    restored = training.load_semantic_checkpoint(fresh, Path(metadata["checkpoint_path"]), contract=contract, seed=1001)
    assert restored["ppo_updates"] == 2 and restored["optimizer_steps"] == 40
    assert training.parameter_hash(fresh.alg.actor) == training.parameter_hash(target.alg.actor)
    assert training.state_hash(fresh.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
    rollout = torch.load(tmp_path / "new_run/rollouts/rollout_000002.pt", weights_only=False)
    assert rollout["observations"]["policy"].shape[-1] == 372
    assert rollout["policy_contract"] == policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)


def test_no_silent_append_or_partial_rollout_reuse(source, monkeypatch):
    old, checkpoint, record = source
    target, env = make(True)
    with pytest.raises(RuntimeError, match="observation layout"):
        training.load_semantic_checkpoint(target, checkpoint, contract=record["source_runtime_contract"], seed=1001)
    monkeypatch.setattr(migration, "build_v3_warm_start_record", lambda *args, **kwargs: copy.deepcopy(record))
    target.alg.storage.step = 1
    before = training.parameter_hash(target.alg.actor)
    with pytest.raises(RuntimeError, match="fresh N1"):
        training.load_semantic_checkpoint(target, checkpoint, contract=record["target_runtime_contract"], seed=1001, warm_start=record)
    assert training.parameter_hash(target.alg.actor) == before
