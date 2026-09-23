"""CPU/synthetic419 tests; no Isaac or genuine policy-update/task credit."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
TensorDict = pytest.importorskip('tensordict').TensorDict
from wlr50_clean.ppo.semantic_rr_capture_actor import SemanticRRCaptureHistoryMLPModel
from wlr50_clean.ppo.semantic_rr_capture_profile import (
    RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT, RR_TASK_FIELDS, RR_TASK_START,
)
from wlr50_clean.ppo.semantic_rear_policy_timing_actor import (
    SemanticRearPolicyTimingHistoryMLPModel, rear_policy_timing_effective_log_std,
)
from wlr50_clean.ppo.semantic_rear_policy_timing_profile import *
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_observation import (
    load_semantic_observation_schema, SemanticObservationBuilder, SemanticObservationError, HISTORY_GROUPS,
)
from wlr50_clean.ppo.semantic_policy_distribution import (
    policy_contract, supported_heteroscedastic_contract_version, configure_policy_distribution,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_scope():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20260923)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def latent(phase=8, count=1):
    x = torch.zeros(count, 419)
    x[:, phase] = 1
    x[:, 20] = .025
    x[:, 158:158+phase] = 1
    x[:, 195:207] = torch.linspace(-.7, .9, 12)
    return x


def obs(x):
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=list(x.shape[:-1]))


def model(x, new=True, **overrides):
    options = dict(observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT if new else RR_CAPTURE_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25, obs_normalization=False,
        distribution_cfg={'class_name': 'HeteroscedasticGaussianDistribution', 'std_type': 'log', 'init_std': .15})
    options.update(overrides)
    cls = SemanticRearPolicyTimingHistoryMLPModel if new else SemanticRRCaptureHistoryMLPModel
    return cls(obs(x), {'actor': ['policy']}, 'actor', 12, **options)


def zero_append(state):
    result = deepcopy(state)
    result['mlp.0.weight'] = torch.cat((state['mlp.0.weight'], torch.zeros(256, 9)), dim=-1)
    return result


@pytest.mark.parametrize('phase', range(13))
def test_old410_zero_columns_preserve_mean_history_and_parent_sigma_when_gates_off(phase):
    x = latent(phase)
    x[:, 157] = float(phase >= 9)
    old, new = model(x[:, :410], False), model(x)
    new.load_state_dict(zero_append(old.state_dict()))
    with torch.no_grad():
        expected = old(obs(x[:, :410]))
        # New timing fields are visible but zero-weighted, not actor hidden state.
        x[:, 411] = 1
        x[:, 414:417] = torch.tensor([.2, .3, .15])
        x[:, 417:] = 1
        actual = new(obs(x))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        old(obs(x[:, :410]), stochastic_output=True)
        new(obs(x), stochastic_output=True)
        torch.testing.assert_close(new.output_mean, old.output_mean, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(new.output_std, old.output_std, atol=2e-7, rtol=2e-6)
        assert new.exploration_std_temperature == .25
        for key, value in old.state_dict().items():
            if key != 'mlp.0.weight': assert torch.equal(value, new.state_dict()[key])


@pytest.mark.parametrize('carry,reachable,prep', [(False,False,False), (True,False,False),
    (False,True,False), (True,True,False), (False,False,True), (True,True,True)])
def test_only_declared_raw_sigma_channels_change_and_no_rng_or_target_bias(carry,reachable,prep):
    x = latent(count=3)
    x[:,410], x[:,RR_TASK_START+1], x[:,412] = float(carry), float(reachable), float(prep)
    learned = torch.linspace(-3., -.2, 36).reshape(3,12)
    original = x.clone()
    before, parent_evidence = receiving_wheel_effective_log_std(learned, x[:,:372], .25)
    rng = torch.get_rng_state()
    after, evidence = rear_policy_timing_effective_log_std(learned,x,.25)
    expected = torch.ones(3,12)
    if carry and reachable: expected[:,6] = 4.
    if prep: expected[:,[1,3]] = 2.
    assert torch.equal(evidence['rear_local_sigma_multiplier_full12'], expected)
    assert torch.equal(after, before + expected.log())
    assert torch.equal(x, original) and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(evidence['effective_innovation_sigma_multiplier_full12'],
        parent_evidence['effective_innovation_sigma_multiplier_full12'] * expected)
    assert bool((after.exp() > 0).all())


def test_one_gaussian_sample_old_current_logp_and_entropy_use_the_same_kernel():
    x = latent(count=2)
    x[0,410] = x[0,404] = 1
    x[1,412] = 1
    actor = model(x)
    calls = []
    handle = actor.mlp.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        raw = actor(obs(x), stochastic_output=True)
        old_logp = actor.get_output_log_prob(raw).clone()
        assert len(calls) == 1
        mean, sigma = actor.output_mean.clone(), actor.output_std.clone()
        normal = torch.distributions.Normal(mean,sigma)
        assert torch.equal(old_logp,normal.log_prob(raw).sum(-1))
        torch.testing.assert_close(actor.output_entropy,normal.entropy().sum(-1))
        actor(obs(x),stochastic_output=True)
        assert torch.equal(actor.get_output_log_prob(raw),old_logp)
        # Different samples, same current conditional density. Raw action is not a target.
        assert raw.shape == (2,12) and actor.output_mean.shape == (2,12)
    handle.remove()


def test_local_sigma_activation_cannot_change_deterministic_mean_after_zero_append():
    x = latent(count=2)
    old, new = model(x[:,:410],False), model(x)
    new.load_state_dict(zero_append(old.state_dict()))
    x[0,410] = x[0,404] = 1
    x[1,412] = 1
    with torch.no_grad():
        torch.testing.assert_close(new(obs(x)),old(obs(x[:,:410])),atol=2e-7,rtol=2e-6)
    with pytest.raises(NotImplementedError): new.as_jit()
    with pytest.raises(NotImplementedError): new.as_onnx()


@pytest.mark.parametrize('stochastic', [False,True])
def test_production_request_audit_uses_same_419_kernel_without_extra_forward(stochastic):
    from wlr50_clean.ppo.semantic_training import audited_history_policy_request
    x=latent();x[:,410]=x[:,404]=1
    actor=model(x);observation=obs(x)
    calls=[]
    handle=actor.mlp.register_forward_hook(lambda *_:calls.append(1))
    with torch.no_grad():
        raw,receipt=audited_history_policy_request(actor,observation,
            lambda:actor(observation,stochastic_output=stochastic),stochastic=stochastic)
        assert len(calls)==1
        assert receipt['policy_version']==REAR_POLICY_TIMING_POLICY
        assert receipt['rear_task_assists_enabled'] is False
        assert receipt['rear_policy_timing_observed_features']==x[0,410:].tolist()
        expected=[1.]*12;expected[6]=4.
        assert receipt['rear_local_sigma_multiplier_full12']==expected
        assert receipt['sampling_draws']==int(stochastic)
        if stochastic:
            normal=torch.distributions.Normal(actor.output_mean,actor.output_std)
            assert torch.equal(actor.get_output_log_prob(raw),normal.log_prob(raw).sum(-1))
    handle.remove()


@pytest.mark.parametrize('index,value', [(410,.5),(417,-1),(404,.1),(414,-.1),(415,1.001),(418,float('nan'))])
def test_malformed_observed_sigma_state_fails_closed(index,value):
    x = latent(); x[:,index] = value
    with pytest.raises(ValueError): rear_policy_timing_effective_log_std(torch.zeros(1,12),x)


@pytest.mark.parametrize('options', [dict(observation_layout=None),dict(exploration_std_temperature=.5),
    dict(obs_normalization=True),dict(distribution_cfg={'class_name':'GaussianDistribution','std_type':'scalar','init_std':.15})])
def test_actor_rejects_implicit_layout_temperature_or_normalizer(options):
    with pytest.raises(ValueError): model(latent(),**options)


def schema_data():
    data = json.loads((ROOT/'configs/ppo_rr_capture_then_rl_transfer_v1/observation_schema.json').read_text())
    data['rear_policy_timing_features_version'] = REAR_POLICY_TIMING_OBSERVATION_LAYOUT
    data['feature_groups'].append({'name':REAR_POLICY_TIMING_GROUP,'size':9,'scale':1.})
    return data


def write_schema(tmp_path,data):
    path = tmp_path/'schema.json'
    path.write_text(json.dumps(data))
    return load_semantic_observation_schema(path)


def live_frame():
    from test_semantic_observation_reward_env import _frame,ZERO12
    from wlr50_clean.ppo.semantic_capture_assist import HipOnlyCaptureAssist
    from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist
    frame = _frame(stage='P09')
    frame.info['semantic_task'].update(
        capture_continuation={'mode':'p05_hip_only_continuation_v1','scheduler_advanced_pending':False},
        fl_capture_pending=False,allow_capture_continuation=False,p05_local_deadline_warning=False,capture_pending_elapsed_s=0.)
    frame.info['capture_assist'] = HipOnlyCaptureAssist().snapshot()
    frame.info['rr_capture_assist'] = RRHipOnlyCaptureAssist().snapshot()
    frame.info['rr_capture_transfer_context'] = dict.fromkeys(RR_TASK_FIELDS,False)
    frame.info['rear_policy_timing'] = {key:0. if key in REAR_POLICY_TIMING_TIME_FIELDS else False for key in REAR_POLICY_TIMING_FIELDS}
    return frame,dict.fromkeys(HISTORY_GROUPS,ZERO12)


def test_append_exact_nine_strict_fields_and_preserve410_numerical_codec(tmp_path):
    old = load_semantic_observation_schema(ROOT/'configs/ppo_rr_capture_then_rl_transfer_v1/observation_schema.json')
    new = write_schema(tmp_path,schema_data())
    assert new.dimension == 419 and new.groups[:-1] == old.groups
    assert new.observation_layout == REAR_POLICY_TIMING_OBSERVATION_LAYOUT
    frame,history = live_frame()
    frame.info['rear_policy_timing'].update(rr_carry_capture=True,p09_source_time_s=23.,
        p12_source_time_s=5.,p12_rl_source_time_s=.5,p12_dependency_wait=True)
    before = old.encode(SemanticObservationBuilder(old).build(frame,history).groups)
    after = new.encode(SemanticObservationBuilder(new).build(frame,history).groups)
    assert after[:410] == before
    assert after[410:] == (1.,0.,0.,0.,23./200.,5./200.,.5/200.,0.,1.)
    assert after[389:403] == (0.,)*14


@pytest.mark.parametrize('kind', ['missing','extra','numeric_flag','bool_clock','negative_clock','infinite_clock','over_horizon'])
def test_no_hidden_fallback_or_invalid_rear_state(tmp_path,kind):
    schema = write_schema(tmp_path,schema_data())
    frame,history = live_frame()
    timing = frame.info['rear_policy_timing']
    if kind == 'missing': del timing['p12_rl_source_time_s']
    if kind == 'extra': timing['hidden_cursor'] = 0
    if kind == 'numeric_flag': timing['rr_carry_capture'] = 1
    if kind == 'bool_clock': timing['p09_source_time_s'] = True
    if kind == 'negative_clock': timing['p09_source_time_s'] = -.1
    if kind == 'infinite_clock': timing['p12_source_time_s'] = float('inf')
    if kind == 'over_horizon': timing['p12_rl_source_time_s'] = 200.1
    with pytest.raises(SemanticObservationError): SemanticObservationBuilder(schema).build(frame,history)


@pytest.mark.parametrize('kind', ['marker','size','scale','parent','reordered'])
def test_schema_rejects_malformed_append(tmp_path,kind):
    data = schema_data()
    if kind == 'marker': del data['rear_policy_timing_features_version']
    if kind == 'size': data['feature_groups'][-1]['size'] = 8
    if kind == 'scale': data['feature_groups'][-1]['scale'] = 200
    if kind == 'parent': del data['rr_capture_features_version']
    if kind == 'reordered': data['feature_groups'][-1],data['feature_groups'][-2] = data['feature_groups'][-2],data['feature_groups'][-1]
    with pytest.raises(SemanticObservationError): write_schema(tmp_path,data)


def test_contract_exact_registration_rear_authority_and_legacy_unchanged():
    old = deepcopy(policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT))
    contract = policy_contract(REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    assert supported_heteroscedastic_contract_version(contract) == REAR_POLICY_TIMING_POLICY
    assert contract['preserved_observation_prefix_dimension'] == 410 and contract['observation_dimension'] == 419
    assert contract['rho'] == .9 and contract['history_slice'] == [195,207]
    assert 'rear_task_assists_and_rear_task_wheel_projection_OFF' in contract['action_transform']
    assert contract['rear_sigma_status'].endswith('not_claimed_converged')
    assert old == policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    config = {'actor':{'distribution_cfg':{'init_std':.15}}}
    configure_policy_distribution(config,REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    assert config['actor']['class_name'] == REAR_POLICY_TIMING_ACTOR_CLASS
    assert config['actor']['exploration_std_temperature'] == .25
    bad = deepcopy(contract); bad['rear_local_sigma_gates']['RR_hip']['multiplier'] = 8.
    with pytest.raises(ValueError): supported_heteroscedastic_contract_version(bad)
    with pytest.raises(ValueError): policy_contract(REAR_POLICY_TIMING_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
