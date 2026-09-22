"""Synthetic CPU current-389 tests. No real checkpoint update/physical credit."""
import copy
from dataclasses import replace

import pytest
import torch
import front_rehearsal as a
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(617)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_rng_state(before)
    torch.set_num_threads(threads)


def runner():
    run = a.training.construct_semantic_runner(_ObservationOnlyEnv(389), seed=1001, device="cpu",
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)[0]
    # Establish nonzero learned-looking std/trunk weights and Adam state. This
    # is synthetic fixture setup, not an AUX step or real PPO learning credit.
    for p in [*run.alg.actor.parameters(), *run.alg.critic.parameters()]:
        p.grad = torch.linspace(-.03, .05, p.numel()).reshape_as(p)
    run.alg.optimizer.step()
    run.alg.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        run.alg.actor.mlp[4].weight[12:] += torch.linspace(-.03, .03, 12 * 256).reshape(12, 256)
    return run


def states(run):
    x = torch.zeros(6, 389)
    x[torch.arange(6), torch.tensor([0, 1, 0, 1, 0, 1])] = 1
    x[:, 20] = .01
    x[:, 30] = torch.linspace(-.1, .1, 6)
    x[:, 195:207] = torch.linspace(-.2, .2, 72).reshape(6, 12)
    v = x.clone()
    v[:, 30] += .023
    v[:, 195:207] *= .7
    h = torch.zeros(11, 389)
    h[torch.arange(11), torch.arange(2, 13)] = 1
    h[:, 20] = .01
    h[:, 157] = 1
    h[:, 195:207] = torch.linspace(-.1, .1, 132).reshape(11, 12)
    tx, vx, ix = a.tensors(x), a.tensors(v), a.tensors(h)
    with torch.no_grad():
        target = a.distribution(run.alg.actor, tx)["mean"] + torch.linspace(.02, .05, 12)
        vt = a.distribution(run.alg.actor, vx)["mean"] + torch.linspace(.02, .05, 12)
    return tx, target, vx, vt, ix


def budget(**changes):
    return replace(a.Budget(4, .05, (3.,) * 8 + (.15,) * 4,
                           (3.,) * 8 + (.15,) * 4, .05, .05), **changes)


@pytest.mark.parametrize("changes", [{"max_attempts": 33}, {"max_attempts": True},
    {"max_attempts": 0}, {"learning_rate": float("nan")}, {"learning_rate": 0},
    {"maximum_train_request_shift_full12": (1.,) * 11},
    {"maximum_validation_request_shift_full12": (1.,) * 11 + (False,)},
    {"maximum_per_state_full_gaussian_kl": float("inf")}, {"maximum_abs_log_sigma_change": 0}])
def test_finite_explicit_budget(changes):
    with pytest.raises(ValueError):
        budget(**changes).validate()


def test_authorization_and_fresh_rollout_required():
    run = runner(); data = states(run)
    with pytest.raises(ValueError, match="authorization"):
        a.fit(run, *data, budget=budget())
    for step, action in ((1, None), (0, torch.zeros(1, 12))):
        run.alg.storage.step = step; run.alg.transition.actions = action
        with pytest.raises(ValueError, match="boundary"):
            a.fit(run, *data, budget=budget(), authorized=True)


@pytest.mark.parametrize("bad", ["soft", "multibit", "train_P03", "invariance_P02", "shared", "old372"])
def test_illegal_phase_or_partition_rejected(bad):
    run = runner(); tx, target, vx, vt, ix = states(run)
    if bad == "soft": tx["policy"][0, 0] = .9
    elif bad == "multibit": tx["policy"][0, 1] = 1
    elif bad == "train_P03": tx["policy"][0, 0] = 0; tx["policy"][0, 2] = 1
    elif bad == "invariance_P02": ix["policy"][0, 2] = 0; ix["policy"][0, 1] = 1
    elif bad == "shared": vx = tx
    elif bad == "old372":
        with pytest.raises(ValueError, match="389"): a.tensors(torch.zeros(2, 372))
        return
    with pytest.raises(ValueError):
        a.inspect(run.alg.actor, tx, target, vx, vt, ix)


def test_actual_kernel_matches_official_mu_and_receiving_sigma():
    run = runner(); tx, _, _, _, ix = states(run)
    for obs in (tx, ix):
        d = a.distribution(run.alg.actor, obs)
        assert torch.equal(d["mean"], run.alg.actor(obs))
        rng = torch.get_rng_state().clone()
        run.alg.actor(obs, stochastic_output=True)
        assert torch.equal(d["sigma"], run.alg.actor.output_std)
        assert torch.equal(d["mean"], run.alg.actor.output_mean)
        torch.set_rng_state(rng)


def test_genuine_point_one_history_derivative_and_no_live_grads():
    run = runner(); tx, _, _, _, _ = states(run)
    leaf = torch.nn.Parameter(a.first_layer(run.alg.actor).weight[:, :2].detach().clone())
    d = a.distribution(run.alg.actor, tx, leaf)
    network, = torch.autograd.grad(d["network_mean"].sum(), leaf, retain_graph=True)
    actual, = torch.autograd.grad(d["mean"].sum(), leaf)
    # Separate float32 backward summation orders incur small cancellation noise.
    torch.testing.assert_close(actual, .1 * network, rtol=1e-4, atol=1e-7)
    assert float(actual.norm() / network.norm()) == pytest.approx(.1, rel=1e-6)
    head = torch.ones(1, 2, 12, dtype=torch.float64, requires_grad=True)
    conditioned = a.history_conditioned_head(head, torch.zeros(1, 12, dtype=torch.float64), .9)
    chain, = torch.autograd.grad(conditioned[:, 0, :].sum(), head)
    assert torch.equal(chain[:, 0, :], torch.full((1, 12), 1. - .9, dtype=torch.float64))
    assert torch.count_nonzero(chain[:, 1, :]) == 0
    assert all(p.grad is None for p in run.alg.actor.parameters())


def test_inspection_is_read_only_with_JVP_not_hidden_SGD(monkeypatch):
    run = runner(); data = states(run)
    hashes = (a.training.parameter_hash(run.alg.actor), a.training.parameter_hash(run.alg.critic),
              a.training.state_hash(run.alg.optimizer.state_dict()))
    rng = a.training.capture_training_rng_state(seed=1001)
    def forbid(*args, **kwargs): raise AssertionError("inspection stepped an optimizer")
    monkeypatch.setattr(torch.optim.SGD, "step", forbid)
    monkeypatch.setattr(torch.optim.Adam, "step", forbid)
    result = a.inspect(run.alg.actor, *data)
    assert result["selected_gradient_l2"] > 0
    assert result["frozen_first_gradient_response"]["optimizer_steps_performed"] == 0
    assert result["frozen_first_gradient_response"]["raw_half_MSE_per_unit_lr"] < 0
    assert hashes == (a.training.parameter_hash(run.alg.actor), a.training.parameter_hash(run.alg.critic),
                      a.training.state_hash(run.alg.optimizer.state_dict()))
    assert a.training.capture_training_rng_state(seed=1001) == rng
    assert all(p.grad is None for p in run.alg.actor.parameters())


def test_full_gaussian_KL_detects_sigma_only_change():
    b = {"mean": torch.zeros(2, 12), "sigma": torch.ones(2, 12),
         "log_sigma": torch.zeros(2, 12), "request": torch.zeros(2, 12)}
    c = {**b, "sigma": torch.full((2, 12), 1.1), "log_sigma": torch.full((2, 12), .0) + torch.tensor(1.1).log()}
    delta = a.gaussian_change(b, c)
    assert bool((delta["kl_original_to_candidate"] > 0).all())
    assert bool((delta["kl_candidate_to_original"] > 0).all())
    assert torch.count_nonzero(delta["raw_mean_delta"]) == 0
    assert float(delta["log_sigma_delta"].abs().min()) > 0
    p = torch.distributions.Normal(b["mean"].double(), b["sigma"].double())
    q = torch.distributions.Normal(c["mean"].double(), c["sigma"].double())
    torch.testing.assert_close(delta["kl_original_to_candidate"],
                               torch.distributions.kl_divergence(p, q).sum(-1), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(delta["kl_candidate_to_original"],
                               torch.distributions.kl_divergence(q, p).sum(-1), rtol=1e-12, atol=1e-14)


def test_accept_changes_only512_and_preserves_Adam_RNG_Identity_critic(monkeypatch):
    run = runner(); data = states(run)
    actor = copy.deepcopy(run.alg.actor.state_dict())
    hashes = {"critic": a.training.parameter_hash(run.alg.critic),
              "adam": a.training.state_hash(run.alg.optimizer.state_dict()),
              "normalizers": a.training.state_hash(a.training._normalizers(run))}
    rng = a.training.capture_training_rng_state(seed=1001)
    mode = run.alg.actor.training
    def forbid(*args, **kwargs): raise AssertionError("PPO Adam was stepped")
    monkeypatch.setattr(run.alg.optimizer, "step", forbid)
    report = a.fit(run, *data, budget=budget(), authorized=True)
    assert report["accepted_auxiliary_updates"] == 4
    assert report["after"]["train"]["conditional_raw_mse"] < report["before"]["train"]["conditional_raw_mse"]
    assert torch.count_nonzero(run.alg.actor.mlp[0].weight[:, :2] - actor["mlp.0.weight"][:, :2]) > 0
    for key, value in run.alg.actor.state_dict().items():
        assert torch.equal(value[:, 2:], actor[key][:, 2:]) if key == "mlp.0.weight" else torch.equal(value, actor[key])
    assert hashes == {"critic": a.training.parameter_hash(run.alg.critic),
                      "adam": a.training.state_hash(run.alg.optimizer.state_dict()),
                      "normalizers": a.training.state_hash(a.training._normalizers(run))}
    assert a.training.capture_training_rng_state(seed=1001) == rng
    assert run.alg.actor.training == mode
    assert report["PPO_decisions_added"] == report["PPO_updates_added"] == report["PPO_optimizer_steps_added"] == 0
    assert report["teacher_deployed"] is False and report["physical_success_claimed"] is False
    assert any(torch.count_nonzero(torch.tensor(step["train_change_from_original"]["log_sigma_delta"]))
               for step in report["steps"])
    assert all(step["real_holdout_entire_Gaussian_bitwise_equal"]
               and step["synthetic_P03_P13_math_probes_bitwise_equal"] for step in report["steps"])


@pytest.mark.parametrize("limit", ["request", "sigma", "KL"])
def test_rejection_restores_entire_leaf_and_never_searches_LR(limit):
    run = runner(); data = states(run)
    before = a.training.parameter_hash(run.alg.actor)
    options = {"maximum_train_request_shift_full12": (1e-12,) * 12} if limit == "request" else (
        {"maximum_abs_log_sigma_change": 1e-12} if limit == "sigma" else {"maximum_per_state_full_gaussian_kl": 1e-12})
    report = a.fit(run, *data, budget=budget(**options), authorized=True)
    assert report["attempted_auxiliary_optimizer_steps"] == 1
    assert report["accepted_auxiliary_updates"] == 0
    assert report["steps"][0]["rejected_full512_leaf_restored"]
    assert a.training.parameter_hash(run.alg.actor) == before


def test_second_rejection_retains_only_first_accepted_proposal(monkeypatch):
    run = runner(); data = states(run)
    baseline = copy.deepcopy(run.alg.actor.state_dict())
    one = a.fit(run, *data, budget=budget(max_attempts=1), authorized=True)
    expected = copy.deepcopy(run.alg.actor.state_dict())
    assert one["accepted_auxiliary_updates"] == 1
    run.alg.actor.load_state_dict(baseline)
    original = torch.optim.SGD.step
    count = 0
    def sabotaged(optimizer, *args, **kwargs):
        nonlocal count
        count += 1
        result = original(optimizer, *args, **kwargs)
        if count == 2:
            with torch.no_grad(): optimizer.param_groups[0]["params"][0].add_(1000.)
        return result
    monkeypatch.setattr(torch.optim.SGD, "step", sabotaged)
    report = a.fit(run, *data, budget=budget(), authorized=True)
    assert report["accepted_auxiliary_updates"] == 1 and report["attempted_auxiliary_optimizer_steps"] == 2
    assert all(torch.equal(v, expected[k]) for k, v in run.alg.actor.state_dict().items())


def test_zero_gradient_does_not_step_or_claim_aux(monkeypatch):
    run = runner()
    with torch.no_grad(): run.alg.actor.mlp[4].weight[:12].zero_()
    data = states(run)
    def forbid(*args, **kwargs): raise AssertionError("zero gradient should not invoke SGD")
    monkeypatch.setattr(torch.optim.SGD, "step", forbid)
    result = a.fit(run, *data, budget=budget(), authorized=True)
    assert result["stop_reason"] == "zero_selected_gradient_no_optimizer_step"
    assert result["accepted_auxiliary_updates"] == result["attempted_auxiliary_optimizer_steps"] == 0
    assert result["actor_parameter_sha256_before"] == result["actor_parameter_sha256_after"]


def test_float_rounding_noop_attempt_rejected_without_credit(monkeypatch):
    run = runner(); data = states(run)
    monkeypatch.setattr(torch.optim.SGD, "step", lambda *args, **kwargs: None)
    result = a.fit(run, *data, budget=budget(), authorized=True)
    assert result["accepted_auxiliary_updates"] == 0 and result["attempted_auxiliary_optimizer_steps"] == 1
    assert "numerically_unchanged_full512_proposal" in result["steps"][0]["rejection_reasons"]
    assert result["actor_parameter_sha256_before"] == result["actor_parameter_sha256_after"]


def test_exception_after_copy_restores_complete_actor(monkeypatch):
    run = runner(); data = states(run)
    baseline = copy.deepcopy(run.alg.actor.state_dict())
    original = a.inspect
    calls = 0
    def fail_final(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("synthetic final inspection failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(a, "inspect", fail_final)
    with pytest.raises(RuntimeError, match="synthetic final"):
        a.fit(run, *data, budget=budget(), authorized=True)
    assert all(torch.equal(v, baseline[k]) for k, v in run.alg.actor.state_dict().items())


def test_JVP_uses_negative_initial_gradient_with_small_finite_difference():
    run = runner(); tx, target, vx, vt, ix = states(run)
    report = a.inspect(run.alg.actor, tx, target, vx, vt, ix)
    leaf = a.first_layer(run.alg.actor).weight[:, :2].detach().clone()
    gradient = torch.tensor(report["selected_gradient_full256x2"])
    step = .2  # Read-only functional evaluation, no optimizer/parameter write.
    with torch.no_grad():
        old = a.distribution(run.alg.actor, tx, leaf)
        perturbed = a.distribution(run.alg.actor, tx, leaf - step * gradient)
    predicted = torch.tensor(report["frozen_first_gradient_response"]["train"]["requested_residual_per_unit_lr_full12"])
    measured = (perturbed["request"] - old["request"]) / step
    torch.testing.assert_close(measured, predicted, rtol=.025, atol=3e-5)


def test_device_paths_are_explicit_without_GPU_execution():
    # Static review guard only: GPU execution remains root's authorized task.
    import inspect
    source = inspect.getsource(a)
    assert 'device=x.device' in source
    assert 'len(tx), tx.device' in source and 'len(vx), vx.device' in source
    assert 'for row in tx.detach().cpu()' in source
    assert '.new_tensor(limit)' in source
    assert 'candidate is CPU-only' not in source
