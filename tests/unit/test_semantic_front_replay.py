"""Small deferred CPU regressions for declared front replay; zero robot credit.

Normal production imports after the staged implementation is applied.
Do not execute while the sole Isaac process is active. No checkpoints are read.
"""
import copy
import hashlib
import json

import pytest

from wlr50_clean.ppo.semantic_front_replay import (
    DATA_SCHEMA, SCHEMA, VERSION, FrontReplayRegularizer,
    add_actor_gradient, current_front_gaussian, validate_replay_spec,
)


def row(index, phase="P01"):
    observation = [0.0] * 439
    observation[int(phase[1:]) - 1] = 1.0
    return dict(decision_index=index, phase=phase, observation=observation,
                reference_mean=[0.0] * 12, reference_sigma=[0.1] * 12)


def dataset_spec(tmp_path, data=None):
    if data is None:
        data = dict(schema=DATA_SCHEMA, training_rows=[row(1), row(2, "P05")],
                    heldout_rows=[row(3, "P06")])
    path = tmp_path / "front_replay.json"
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    return dict(schema=SCHEMA, version=VERSION, coefficient=1.0, minibatch_size=32,
                dataset_path=str(path.resolve()),
                dataset_sha256=hashlib.sha256(payload).hexdigest())


def test_validate_explicit_partition_and_input_unchanged(tmp_path):
    spec = dataset_spec(tmp_path)
    before = copy.deepcopy(spec)
    data = validate_replay_spec(spec)
    assert spec == before
    assert [r["decision_index"] for r in data["training_rows"]] == [1, 2]
    assert [r["decision_index"] for r in data["heldout_rows"]] == [3]


@pytest.mark.parametrize("case", ["overlap", "rear_phase", "width", "nan",
                                   "zero_sigma", "phase_mismatch", "too_many"])
def test_validate_rejects_wrong_reference_rows(tmp_path, case):
    data = dict(schema=DATA_SCHEMA, training_rows=[row(1)], heldout_rows=[row(2)])
    if case == "overlap":
        data["heldout_rows"][0]["decision_index"] = 1
    elif case == "rear_phase":
        data["training_rows"][0] = row(1, "P07")
    elif case == "width":
        data["training_rows"][0]["observation"].pop()
    elif case == "nan":
        data["training_rows"][0]["reference_mean"][0] = float("nan")
    elif case == "zero_sigma":
        data["training_rows"][0]["reference_sigma"][0] = 0.0
    elif case == "phase_mismatch":
        data["training_rows"][0]["phase"] = "P05"
    else:
        data["training_rows"] = [row(i) for i in range(256)]
        data["heldout_rows"] = [row(256)]
    with pytest.raises(ValueError):
        validate_replay_spec(dataset_spec(tmp_path, data))


def test_validate_rejects_byte_change_and_nonexplicit_objective(tmp_path):
    spec = dataset_spec(tmp_path)
    changed = dict(spec, coefficient=0.0)
    with pytest.raises(ValueError):
        validate_replay_spec(changed)
    from pathlib import Path
    Path(spec["dataset_path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="bytes changed"):
        validate_replay_spec(spec)


@pytest.fixture
def cpu_torch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    torch = pytest.importorskip("torch")
    old_state, old_threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_rng_state(old_state)
        torch.set_num_threads(old_threads)


def test_gradient_sum_equals_joint_loss_before_clip_and_same_adam(cpu_torch):
    torch = cpu_torch
    # Deliberately large terms activate the existing combined norm clip.
    combined = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(),
                                   torch.nn.Linear(4, 2)).double()
    hooked = copy.deepcopy(combined)
    x = torch.tensor([[0.2, -0.4, 0.7], [0.5, 0.8, -0.3]], dtype=torch.float64)
    ref_x = x.flip(0) * 1.7
    def ppo_loss(actor):
        return (actor(x) - 4.0).square().mean()
    def replay_loss(actor):
        return 1.3 * (actor(ref_x) + 2.0).square().mean()
    opt_joint = torch.optim.Adam(combined.parameters(), lr=1e-5)
    opt_hook = torch.optim.Adam(hooked.parameters(), lr=1e-5)
    (ppo_loss(combined) + replay_loss(combined)).backward()
    parameters = tuple(hooked.parameters())
    addition = torch.autograd.grad(replay_loss(hooked), parameters)
    rng = torch.get_rng_state().clone()
    with add_actor_gradient(parameters, addition) as calls:
        ppo_loss(hooked).backward()
        for p, q in zip(combined.parameters(), parameters):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
        norm_joint = torch.nn.utils.clip_grad_norm_(combined.parameters(), 0.3)
        norm_hook = torch.nn.utils.clip_grad_norm_(parameters, 0.3)
        torch.testing.assert_close(norm_joint, norm_hook, rtol=1e-12, atol=1e-12)
        assert float(norm_joint) > 0.3
        opt_joint.step()
        opt_hook.step()
    assert calls == [1] * len(parameters)
    assert torch.equal(rng, torch.get_rng_state())
    for p, q in zip(combined.parameters(), parameters):
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
        assert opt_joint.state[p]["step"] == opt_hook.state[q]["step"] == 1
    # Hooks have been removed: a later ordinary backward does not add replay.
    for p in parameters:
        p.grad = None
    plain = copy.deepcopy(hooked)
    ppo_loss(hooked).backward()
    ppo_loss(plain).backward()
    for p, q in zip(parameters, plain.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)


def test_gradient_hook_cleanup_on_exception_and_missing_backward(cpu_torch):
    torch = cpu_torch
    p = torch.nn.Parameter(torch.tensor(2.0))
    with pytest.raises(RuntimeError, match="not consumed once"):
        with add_actor_gradient((p,), (torch.tensor(3.0),)):
            pass
    with pytest.raises(ValueError, match="intentional"):
        with add_actor_gradient((p,), (torch.tensor(3.0),)):
            raise ValueError("intentional")
    (p.square()).backward()
    assert p.grad.item() == 4.0


def test_exact439_kernel_preserves_cache_rng_and_official_head_hook(cpu_torch):
    torch = cpu_torch
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict
    from wlr50_clean.ppo.semantic_training import construct_semantic_runner
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    from wlr50_clean.ppo.semantic_rear_owner_profile import (
        REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT,
    )
    runner, _ = construct_semantic_runner(
        _shape_env(439, "cpu"), seed=1001, device="cpu", initialize_actor=False,
        policy_version=REAR_OWNER_POLICY,
        observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)
    actor = runner.alg.actor
    x = torch.zeros(2, 439)
    x[:, 0] = 1.0
    x[:, 195:207] = torch.linspace(-0.2, 0.2, 12)
    observation = TensorDict({"policy": x, "critic": x.clone()}, batch_size=[2])
    raw = actor(observation, stochastic_output=True).detach()
    logp = actor.get_output_log_prob(raw).detach().clone()
    cache = tuple(value.detach().clone() for value in actor.output_distribution_params)
    entropy = actor.output_entropy.detach().clone()
    x_before, rng = x.clone(), torch.get_rng_state().clone()
    official_heads = []
    handle = actor.mlp.register_forward_hook(
        lambda _m, _args, value: official_heads.append(value))
    try:
        mean, log_sigma = current_front_gaussian(actor, x)
        assert official_heads == []  # Replay bypasses only this official head hook.
        torch.testing.assert_close(mean, cache[0], rtol=0, atol=0)
        torch.testing.assert_close(log_sigma.exp(), cache[1], rtol=0, atol=0)
        torch.autograd.grad(mean.square().mean() + log_sigma.square().mean(),
                            tuple(actor.parameters()), allow_unused=True)
        assert official_heads == []
        assert torch.equal(x, x_before)
        assert torch.equal(rng, torch.get_rng_state())
        for actual, saved in zip(actor.output_distribution_params, cache):
            assert torch.equal(actual, saved)
        assert torch.equal(actor.output_entropy, entropy)
        assert torch.equal(actor.get_output_log_prob(raw), logp)
        actor(observation, stochastic_output=False)
        assert len(official_heads) == 1
    finally:
        handle.remove()


def test_heldout_only_evaluated_not_fit_and_exposures_not_ppo(tmp_path, cpu_torch, monkeypatch):
    torch = cpu_torch
    import wlr50_clean.ppo.semantic_front_replay as module
    data = dict(schema=DATA_SCHEMA, training_rows=[row(1), row(2)],
                heldout_rows=[row(99)])
    for r in data["training_rows"]:
        r["observation"][100] = 1.0
    data["heldout_rows"][0]["observation"][100] = 9.0
    regularizer = FrontReplayRegularizer(dataset_spec(tmp_path, data), device="cpu")
    actor = torch.nn.Linear(439, 24)
    seen_grad_batches = []
    def kernel(model, observations):
        if torch.is_grad_enabled():
            seen_grad_batches.append(observations[:, 100].detach().tolist())
        output = model(observations).reshape(-1, 2, 12)
        return output[:, 0], output[:, 1]
    monkeypatch.setattr(module, "current_front_gaussian", kernel)
    before = regularizer.begin_update(actor)
    with regularizer.minibatch(actor, global_minibatch_index=0):
        actor(torch.zeros(1, 439)).square().mean().backward()
    report = regularizer.finish_update(actor, before, expected_minibatches=1)
    assert len(seen_grad_batches) == 1 and seen_grad_batches[0] == [1.0] * 32
    assert report["actual_replay_row_exposures"] == 32
    assert report["on_policy_samples_added"] == 0
    assert report["separate_auxiliary_optimizer_steps"] == 0
    assert report["before"]["heldout_rows"]["rows"] == 1
    assert report["after"]["heldout_rows"]["rows"] == 1
    assert 99 not in report["minibatches"][0]["source_decision_indices"]
