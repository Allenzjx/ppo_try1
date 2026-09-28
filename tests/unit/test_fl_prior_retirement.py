"""Bounded CPU synthetic wiring tests, not simulation or new PPO credit."""
import copy
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests/unit'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import torch
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_fl_prior_retirement_actor import PriorRetirementActor, OBSERVATION_LAYOUT, audited_retirement_request
from wlr50_clean.ppo.semantic_fl_prior_retirement_task import RetirementClock, PriorRetirementCore
from wlr50_clean.ppo.semantic_fl_forward_actor import WindowFLActor, audited_fl_request
from wlr50_clean.ppo.semantic_history_actor import HISTORY_RHO
from test_fl_window_task import CFG, FakeCore, Step, frame

CHECKPOINT = ROOT/'outputs/ppo_finish_advance_fl_forward_v1/checkpoints/history/checkpoint_CP233547_FL000587_data002048_gd226fafd8bc0.pt'
CHECKPOINT_SHA = '39c35998298fe25541f7c6595076486ea629cbf8f5b4d833f2bce3cce1810458'
MANIFEST_SHA = 'e28866bd440c629c94d1bf1b6f727bba9ee83983c77a26f3b978a7b6a3f83785'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('GPU initialization forbidden')
    monkeypatch.setattr(torch.cuda, 'init', denied)
    monkeypatch.setattr(torch.cuda, '_lazy_init', denied)
    old_threads = torch.get_num_threads()
    state = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        yield
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_rng_state(state)
        torch.set_num_threads(old_threads)


@lru_cache(maxsize=1)
def package():
    # Read only the completed B1 update package, never a live DET source.
    side = CHECKPOINT.with_name(CHECKPOINT.stem+'_manifest.json')
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA
    assert hashlib.sha256(side.read_bytes()).hexdigest() == MANIFEST_SHA
    meta = json.loads(side.read_text())
    return torch.load(CHECKPOINT, map_location='cpu', weights_only=False), meta


def observation(flags=(1, 0, 1), lambdas=(0., 0., 1.)):
    x = torch.zeros(len(flags), 536)
    x[:, 5] = 1.  # P06; synthetic values, no real robot state claims.
    x[:, 20] = .1
    x[:, 203] = -1.1747  # Previous raw FL history, not wheel rad/s.
    x[:, 531] = torch.tensor(flags)
    x[:, 532] = .5
    x[:, 533] = 1.
    x[:, 535] = torch.tensor(lambdas)
    return TensorDict({'policy':x}, batch_size=[len(flags)])


def make_actor(new=True, obs=None):
    state, meta = package()
    obs = observation() if obs is None else obs
    cfg = copy.deepcopy(meta['runner_config']['actor'])
    cfg.pop('class_name')
    if new:
        cfg['observation_layout'] = OBSERVATION_LAYOUT
        actor = PriorRetirementActor(obs, {'actor':['policy']}, 'actor', 12, **cfg)
    else:
        old_obs = {'policy':obs['policy'][:,:535]}
        actor = WindowFLActor(old_obs, {'actor':['policy']}, 'actor', 12, **cfg)
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    return actor


def test_clock_real_elapsed_monotonic_stop_and_reset():
    clock = RetirementClock()
    assert clock.observe(400, False)['retirement_lambda'] == 0.
    assert clock.observe(408, True)['retirement_lambda'] == 0.
    assert clock.observe(468, True)['retirement_lambda'] == .5
    assert clock.observe(468, True)['retirement_lambda'] == .5
    assert clock.observe(528, True)['retirement_lambda'] == 1.
    assert clock.observe(700, True)['retirement_lambda'] == 1.
    assert clock.observe(701, False)['retirement_lambda'] == 0.
    assert clock.observe(702, True)['retirement_lambda'] == 0.
    with pytest.raises(ValueError): clock.observe(701, True)
    with pytest.raises(ValueError): clock.observe(702, False)
    clock.reset()
    assert clock.observe(0, False)['retirement_lambda'] == 0.
    assert clock.snapshot()['owner_indices'] == []


def test_lambda_zero_identity_and_off_exact_no_new_rng():
    obs = observation((1, 1, 0), (0., 0., 0.))
    candidate, old = make_actor(obs=obs), make_actor(False, obs)
    with torch.no_grad():
        expected = old({'policy':obs['policy'][:,:535]}, stochastic_output=False)
        actual = candidate(obs, stochastic_output=False)
    assert torch.equal(actual, expected)
    assert torch.equal(candidate._last_forward_evidence['conditional_std_FL'], old._last_forward_evidence['conditional_std_FL'])
    off = observation((0, 0), (0., 0.))
    seen = []
    hook = candidate.mlp.register_forward_hook(lambda *args: seen.append(True))
    rng = torch.get_rng_state().clone()
    actual = candidate(off, stochastic_output=True)
    hook.remove()
    assert seen == [] and torch.equal(rng, torch.get_rng_state())
    expected = old({'policy':off['policy'][:,:535]}, stochastic_output=False)
    assert torch.equal(actual, expected)
    assert torch.equal(candidate.get_output_log_prob(actual), torch.zeros(2))


def test_endpoints_midpoint_same_executed_history_other11_and_std():
    obs = observation((1, 1, 1), (0., .5, 1.))
    actor = make_actor(obs=obs)
    calls = []
    handle = actor.frozen_A.register_forward_hook(lambda *args: calls.append(True))
    mean = actor(obs, stochastic_output=False)
    handle.remove()
    assert calls == [True]
    e = actor._last_forward_evidence
    assert torch.equal(e['history_center_FL'], actor.frozen_A._last_forward_evidence['history_center'][:,8:9])
    assert mean[0,8] == e['original_B_conditional_mean'][0,8]
    assert mean[1,8] == e['original_B_conditional_mean'][1,8]-.5*e['old_A_innovation_FL'][1,0]
    assert mean[2,8] == HISTORY_RHO*e['history_center_FL'][2,0]+actor.mlp.mean_delta
    other = [i for i in range(12) if i != 8]
    assert torch.equal(mean[:,other], e['accepted_raw'][:,other])
    assert torch.equal(e['conditional_std_FL'], actor.mlp.log_std.exp().expand(3,1))
    assert e['public_prior_retirement_is_not_learned_delta']


def test_history_is_actual_cap_handoff_not_naive_raw_slot():
    obs = observation((1,), (1.,))
    obs['policy'][:,20] = 0.  # Genuine observed cap-handoff conditions in synthetic input.
    obs['policy'][:,162] = 1.
    obs['policy'][:,215] = -.1
    actor = make_actor(obs=obs)
    actor(obs, stochastic_output=False)
    h = actor._last_forward_evidence['history_center_FL']
    assert torch.equal(h, actor.frozen_A._last_forward_evidence['history_center'][:,8:9])
    assert not torch.equal(h[:,0], obs['policy'][:,203])


def test_true_one_dimensional_likelihood_KL_gradient_and_negative_support():
    obs = observation((1, 0, 1), (.25, 0., 1.))
    actor = make_actor(obs=obs)
    raw = actor(obs, stochastic_output=True).detach()
    old = tuple(v.detach().clone() for v in actor.output_distribution_params)
    take = old[2][:,0].bool()
    expected = torch.zeros(3)
    expected[take] = torch.distributions.Normal(old[0][take,0], old[1][take,0]).log_prob(raw[take,8])
    assert torch.equal(actor.get_output_log_prob(raw), expected)
    with torch.no_grad(): actor.mlp.mean_delta.add_(.005); actor.mlp.log_std.add_(.01)
    actor(obs, stochastic_output=True)
    new = actor.output_distribution_params
    expected[take] = torch.distributions.Normal(new[0][take,0], new[1][take,0]).log_prob(raw[take,8])
    assert torch.equal(actor.get_output_log_prob(raw), expected)
    kl = torch.zeros(3)
    kl[take] = torch.distributions.kl_divergence(
        torch.distributions.Normal(old[0][take,0],old[1][take,0]),
        torch.distributions.Normal(new[0][take,0],new[1][take,0]))
    assert torch.equal(actor.get_kl_divergence(old,new), kl)
    loss = -actor.get_output_log_prob(raw).sum()-.01*actor.output_entropy.sum()
    loss.backward()
    assert actor.mlp.mean_delta.grad is not None and actor.mlp.mean_delta.grad.abs() > 0.
    assert actor.mlp.log_std.grad is not None and actor.mlp.log_std.grad.abs() > 0.
    actor.assert_frozen_state()
    broken = raw.clone(); broken[0,3] += .01
    with pytest.raises(ValueError, match='deterministic support'): actor.get_output_log_prob(broken)


def test_explicit_auditor_and_schema_rejections():
    obs = observation((1,), (.5,))
    actor = make_actor(obs=obs)
    raw, audit = audited_retirement_request(actor, obs, lambda:actor(obs, stochastic_output=True), stochastic=True)
    assert audit['selected_raw_log_probability'] == float(actor.get_output_log_prob(raw)[0].detach())
    assert audit['retirement_lambda'] == .5 and audit['sampling_draws'] == 1
    assert audit['public_prior_retirement_is_not_learned_delta'] and not audit['extra_actuator_projection']
    with pytest.raises(ValueError, match='explicit scalar'): audited_fl_request(actor, obs, lambda:None, stochastic=False)
    for bad in (float('nan'), -.01, 1.01):
        altered = obs.clone(); altered['policy'][:,535] = bad
        with pytest.raises(ValueError): actor(altered)
    altered = obs.clone(); altered['policy'][:,531] = 0.
    with pytest.raises(ValueError, match='OFF'): actor(altered)
    with pytest.raises(ValueError, match='536'): actor({'policy':obs['policy'][:,:535]})


def test_exact_current_actor_state_and_Adam_moments_LR_preserved(tmp_path):
    state, meta = package()
    actor = make_actor()
    current = actor.state_dict()
    assert current.keys() == state['actor_state_dict'].keys()
    assert all(torch.equal(v, state['actor_state_dict'][k]) for k,v in current.items())
    assert sum(p.numel() for p in actor.trainable_parameters()) == 2
    oldopt = state['optimizer_state_dict']
    ids = oldopt['param_groups'][0]['params'][:2]
    group = copy.deepcopy(oldopt['param_groups'][0]); group['params'] = [0,1]
    subset = dict(state={i:copy.deepcopy(oldopt['state'][p]) for i,p in enumerate(ids)}, param_groups=[group])
    optimizer = torch.optim.Adam(actor.trainable_parameters(), lr=meta['learning_rate'])
    optimizer.load_state_dict(subset)
    assert optimizer.param_groups[0]['lr'] == meta['learning_rate'] == .01
    for i,p in enumerate(actor.trainable_parameters()):
        for key,value in subset['state'][i].items():
            actual = optimizer.state[p][key]
            assert torch.equal(value,actual) if torch.is_tensor(value) else value == actual
    actor.assert_frozen_state(optimizer)
    path = tmp_path/'candidate_synthetic_roundtrip.pt'
    torch.save(actor.state_dict(),path)
    other = make_actor(); other.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
    assert torch.equal(actor(observation()),other(observation()))


def test_core_tail_recording_stop_reset_and_continuous_GAE():
    inner = FakeCore(); core = PriorRetirementCore(inner, CFG)
    seen = []; core.tick_observer = lambda a,b,p: seen.append(b.physics_tick)
    assert core.task is inner.task and core.backend is inner.backend
    x = core.reset()
    assert len(x) == 536 and x[:531] == (0.,)*531 and x[-1] == 0.
    step = core.step([0.]*12)
    assert seen == [6001] and step.observation[-1] == 1./120.
    assert step.info['fl_prior_retirement']['request']['retirement_lambda'] == 0.
    assert step.info['fl_prior_retirement']['native_observation_contexts'][0]['retirement_lambda'] == 1./120.
    assert not step.terminated and not step.truncated
    stopped = frame(6002, cursor=3056)
    inner.next_frame = stopped
    step = core.step([0.]*12)
    assert step.observation[-1] == 0. and step.observation[531] == 0.
    assert not step.terminated  # Source permission change is not an episode boundary.
    inner.next_frame = frame(6003, phase='P07')
    assert core.step([0.]*12).observation[-1] == 0.
    assert len(seen) == 3 and core.reset()[-1] == 0.


def test_native_observer_monotone_configured_ramp_and_external_callback_once():
    inner = FakeCore()
    seen = []
    inner.tick_observer = lambda a,b,p: seen.append(b.physics_tick)
    core = PriorRetirementCore(inner, CFG, ramp_duration_s=2.)
    core.reset()
    values = []
    for _ in range(240):
        values.append(core.step([.1]*12).observation[-1])
    assert len(seen) == 240 and len(set(seen)) == 240
    assert values == sorted(values) and values[119] == .5 and values[-1] == 1.
    assert core.retirement.snapshot()['ramp_duration_s'] == 2.
    assert core.retirement.snapshot()['ramp_is_declared_control_change']
    # Replacing the public recorder setter changes only the external callback.
    after = []
    core.tick_observer = lambda a,b,p: after.append(b.physics_tick)
    core.step([.1]*12)
    assert len(seen) == 240 and after == [6241]
    assert inner.tick_observer.__self__ is core
    core.reset()
    assert core.retirement.value == 0. and core.retirement.entry_tick == 6000


class EightTickCore(FakeCore):
    """Synthetic original raw-hold boundary, not a physics simulator."""
    def __init__(self, stop_after=None):
        super().__init__()
        self.stop_after = stop_after
        self.raw_seen = []

    def step(self, raw):
        for i in range(1,9):
            before = self.frame
            after = frame(before.physics_tick+1, x=.6+i*.001)
            if self.stop_after is not None and i >= self.stop_after:
                after.nominal_action_full12[8:] = [0.]*4
            self.frame = after
            self.raw_seen.append((after.physics_tick, tuple(raw)))
            self.tick_observer(before,after,None)
        return Step((0.,)*531, 0., False, False, dict(termination_reason=None))


def test_eight_tick_request_native_clock_separation_without_action_rewrite():
    inner = EightTickCore()
    core = PriorRetirementCore(inner, CFG)
    core.reset()
    raw = tuple(.01*i for i in range(12))
    result = core.step(raw)
    evidence = result.info['fl_prior_retirement']
    assert evidence['request']['retirement_lambda'] == 0.
    assert evidence['current']['retirement_lambda'] == 8./120.
    assert [x['retirement_lambda'] for x in evidence['native_observation_contexts']] == [i/120. for i in range(1,9)]
    assert all(v == raw for _,v in inner.raw_seen)
    assert evidence['original_eight_native_tick_raw_hold']
    assert evidence['no_native_instantaneous_raw_cancellation_claim']
    assert all(x['owner_indices'] == [] for x in evidence['native_observation_contexts'])
    assert not result.terminated and not result.truncated


def test_native_stop_resets_observed_lambda_without_hidden_instant_sample():
    inner = EightTickCore(stop_after=3)
    core = PriorRetirementCore(inner, CFG)
    core.reset()
    raw = tuple(-.1*i for i in range(12))
    result = core.step(raw)
    rows = result.info['fl_prior_retirement']['native_observation_contexts']
    assert [x['retirement_lambda'] for x in rows] == [1./120.,2./120.]+[0.]*6
    assert result.observation[531] == result.observation[535] == 0.
    assert all(v == raw for _,v in inner.raw_seen)
    assert not result.terminated  # Native source stop is not a done/GAE cut.
    assert all(not x['source_clock_or_stop_modified'] for x in rows)
