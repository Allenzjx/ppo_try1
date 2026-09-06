"""CPU fake physics: credit/state continuity, not live suffix availability."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
from wlr50_clean.ppo.semantic_env import WRITE_COUNTERS
from wlr50_clean.ppo.semantic_prefix import PrefixSemanticIsaacBackend
from wlr50_clean.ppo.residual_direct_env import ResidualStep
from wlr50_clean.ppo.semantic_policy_distribution import STATE_DEPENDENT_POLICY, policy_contract
from wlr50_clean.ppo.semantic_checkpoint_prefix import (
    CheckpointPolicyPrefixRequest as Request,
    CheckpointPolicyPrefixRslAdapter as Adapter,
    RESULT_SCOPE, sampling_label,
)

torch.set_num_threads(1)
ZERO = (0.,)*12


def provenance():
    return {"checkpoint_path": "C:/verified/checkpoint.pt", "checkpoint_sha256": "a"*64,
            "actor_parameter_sha256": "b"*64, "source_global_policy_decisions": 84992,
            "source_ppo_updates": 629, "policy_contract": policy_contract(STATE_DEPENDENT_POLICY)}


class FrozenPolicy:
    def __init__(self):
        self.calls = []
        self._provenance = provenance()

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def __call__(self, observation):
        self.calls.append(tuple(observation))
        return (.125 + observation[0]/100.,)*12


class Core:
    observation_dimension = 324

    def __init__(self, plans=None, fault=None):
        self.backend = object.__new__(SemanticIsaacBackend)
        self.tick_observer = None
        self.plans = plans or [[("P06", 8, None)]]
        self.fault = fault
        self.resets, self.actions, self.pre_action_history = 0, [], []
        self._total_decisions = 0
        # Stand-ins whose identities must survive credit opening.
        self.bridge, self.reward_calculator = object(), object()
        self.mapper, self.nominal, self.contacts = object(), object(), object()

    def _frame(self, phase, tick):
        return SimpleNamespace(state_id=phase, physics_tick=tick, sim_time_s=tick/120.,
            info={"semantic_task": {"remaining_task_time_s": 200.-tick/120.,
                                    "placed_history": {"FR": True, "FL": True, "RR": False, "RL": False}}})

    def reset(self, seed=1001, options=None):
        self.resets += 1
        self.frame = self._frame("P01", 0)
        self.done, self.decision_count, self._episode_return = False, 0, 0.
        self.observation = (0.,)*324
        self._history = {"residual": ZERO, "physics_ticks": 0}
        self.plan = self.plans[min(self.resets-1, len(self.plans)-1)]
        return self.observation

    def step(self, raw):
        assert not self.done
        self.actions.append(tuple(raw))
        self.pre_action_history.append(deepcopy(self._history))
        before = self.frame
        phase, count, reason = self.plan[min(self.decision_count, len(self.plan)-1)]
        self.frame = self._frame(phase, before.physics_tick+count)
        for tick in range(before.physics_tick+1, self.frame.physics_tick+1):
            if self.tick_observer:
                self.tick_observer(self._frame(before.state_id, tick-1), self._frame(phase, tick), None)
        self.done = reason is not None
        self.decision_count += 1
        self._total_decisions += 1
        self._episode_return += 1.
        self.observation = (float(self.decision_count),)*324
        self._history.update(residual=tuple(raw), physics_ticks=self.frame.physics_tick)
        native = {"schema": "wlr50_clean.actuator_target_effect_audit.v1", "verified": True,
            "actual_mapping_matches_dispatch": True, "setter_dispatch_targets_equal": True,
            "same_tick_counterfactual": True, "raw_policy_action_full12": tuple(raw),
            "target_dtype": "torch.float32", "changed_target_channel_count": 1,
            "physics_tick": self.frame.physics_tick+600,
            "actual_native_targets": {"servo_position_rad": [0.]*8, "wheel_velocity_rad_s": [0.]*4}}
        ticks = [{"episode_physics_tick": tick, "command_physics_tick": tick+600,
                  "verified": True, "source_phase_id": before.state_id,
                  "actual_native_effect": True, "own_phase_request_effect": True}
                 for tick in range(before.physics_tick+1, self.frame.physics_tick+1)]
        info = {"phase_id": before.state_id, "end_phase_id": phase, "physics_tick": self.frame.physics_tick,
            "sim_time_s": self.frame.sim_time_s, "physics_ticks": count, "decision_count": self.decision_count,
            "raw_policy_action_full12": tuple(raw), "applied_action_full12": tuple(raw),
            "projected_residual_full12": tuple(raw), "actual_drive_target_full12": tuple(raw),
            "actuator_target_effect_audit": native, "actuator_target_effect_audit_ticks": ticks,
            "actuator_target_effect_audit_summary": {"all_ticks_verified": True,
                "physics_ticks": count, "verified_tick_count": count,
                "actual_native_effect_tick_count": count, "own_phase_request_effect_tick_count": count},
            "no_in_episode_state_writes_verified": True, **dict.fromkeys(WRITE_COUNTERS, 0),
            "termination_reason": reason, "task_success": reason == "SUCCESS",
            "episode_return": self._episode_return, "task_outcome_label": "FULL_TASK_SUCCESS" if reason == "SUCCESS" else reason,
            "full_task_success": reason == "SUCCESS"}
        if self.fault:
            self.fault(info)
        return ResidualStep(self.observation, 1., self.done, False, info)

    def telemetry_summary(self):
        return {"decisions": self._total_decisions, "episodes": self.resets}


def adapter(core=None, request=None):
    core, records = core or Core(), []
    env = Adapter(core, seed=1001, device="cpu", evidence_sink=records.append,
                  request=request or Request())
    return env, core, records


@pytest.mark.parametrize("phase", [f"P{i:02}" for i in range(6, 14)])
def test_bootstrap_and_exact_target_reached_without_second_reset(phase):
    env, core, records = adapter(Core([[(phase, 8, None)]]), Request(phase))
    objects = (core.bridge, core.reward_calculator, core.mapper, core.nominal, core.contacts)
    assert core.resets == 1 and not core.actions and env.total_decisions == 0
    assert env.cfg["prefix_policy_provenance"] is None
    assert env.get_observations()["policy"].shape == (1, 324)
    policy = FrozenPolicy()
    result = env.install_prefix_policy(policy, policy.provenance)
    assert result["mode"] == RESULT_SCOPE and result["actual_phase"] == phase
    assert result["physics_tick"] == 8 and result["remaining_task_time_s"] == pytest.approx(200.-8/120.)
    assert core.resets == 1 and core.actions == [(.125,)*12]
    assert (core.bridge, core.reward_calculator, core.mapper, core.nominal, core.contacts) == objects
    assert env._observation == core.observation and env._observation != (0.,)*324
    assert env.core.prefix_decisions == 1 and env.core.prefix_ticks == 8
    assert env.total_decisions == env.core.credited_decisions == 0
    assert env.episode_length_buf.item() == 0 and not env.completed_episodes
    assert all(row["policy_credit"] is False for row in records)
    assert env.cfg["reset_sampling"] == sampling_label(Request(phase))


def test_first_credit_inherits_residual_and_return_without_handoff_zero():
    env, core, records = adapter()
    policy = FrozenPolicy()
    env.install_prefix_policy(policy, policy.provenance)
    actions = torch.full((1, 12), -.25)
    _, reward, done, extras = env.step(actions)
    info = extras["semantic_decisions"][0]
    assert core.resets == 1 and core.actions == [(.125,)*12, (-.25,)*12]
    assert core.pre_action_history[1]["residual"] == (.125,)*12
    assert info["physical_core_decision_count_including_prefix"] == 2
    assert info["decision_count"] == info["episode_return"] == 1
    assert info["physical_episode_return_including_prefix"] == 2
    assert info["task_result_scope"] == RESULT_SCOPE
    assert info["prefix_checkpoint_policy_data_in_ppo_storage"] is False
    assert not done.item() and reward.item() == 1
    assert len(policy.calls) == 1 and env.total_decisions == env.core.credited_decisions == 1


def test_offset_counts_decisions_after_observed_entry_without_resets():
    env, core, _ = adapter(Core([[('P05',8,None),('P06',8,None),('P06',8,None),('P06',8,None)]]),
                           Request('P06', teacher_offset_decisions=2))
    policy = FrozenPolicy()
    env.install_prefix_policy(policy, policy.provenance)
    assert len(policy.calls) == 4 and core.resets == 1
    assert env.core.start_record["target_first_observed_decision"] == 2
    assert env.core.start_record["physics_tick"] == 32


@pytest.mark.parametrize("plan,prefix_request,reason", [
    ([("P06",8,None),("P07",8,None)], Request("P06",1), "requested_phase_ended_before_offset"),
    ([("P08",8,None)], Request("P07"), "target_not_observed_at_decision_end"),
    ([("P05",8,None)], Request("P06",maximum_prefix_decisions=2), "bounded_prefix_exhausted"),
    ([("P09",2,"BODY_COLLISION")], Request("P09"), "physical_prefix_terminal"),
    ([("P13",3,"SUCCESS")], Request("P13"), "physical_prefix_terminal"),
    ([("P06",8,"TASK_TIMEOUT")], Request("P06"), "physical_prefix_terminal"),
])
def test_one_fresh_p01_fallback_excludes_every_prefix_terminal(plan,prefix_request,reason):
    env, core, records = adapter(Core([plan, [("P01",8,None)]]), prefix_request)
    policy = FrozenPolicy()
    env.install_prefix_policy(policy, policy.provenance)
    assert core.resets == 2 and core.frame.physics_tick == 0 and core.frame.state_id == "P01"
    assert env.core.start_record["mode"] == "fresh_P01_fallback"
    assert env.core.attempts[-1]["miss"]["reason"] == reason
    assert env.total_decisions == 0 and not env.completed_episodes
    assert env.telemetry_summary()["success_count"] == 0
    assert env.core.prefix_ticks == sum(row[1] for row in plan) if len(plan)>1 else env.core.prefix_ticks > 0
    assert all(row.get("full_task_success") is not True for row in records)
    _, _, _, extras = env.step(torch.zeros(1,12))
    assert extras["semantic_decisions"][0]["curriculum_start"]["from_P01_current_policy"] is True
    assert len(policy.calls) == env.core.prefix_decisions


def test_suffix_success_and_following_reset_reuse_identical_frozen_callback():
    env, core, records = adapter(Core([[('P06',8,None),('P13',1,'SUCCESS')], [('P06',8,None)]]))
    policy = FrozenPolicy()
    env.install_prefix_policy(policy, policy.provenance)
    _, _, dones, extras = env.step(torch.zeros(1,12))
    summary = extras["episode_summaries"][0]
    assert dones.item() and summary["task_outcome_label"] == "SUFFIX_SUCCESS"
    assert summary["full_task_success"] is False
    assert summary["policy_decisions"] == 1 and summary["duration_s"] == pytest.approx(9/120.)
    assert extras["time_outs"].item() is False
    assert extras["terminal_observation"]["policy"][0,0].item() == 2
    assert env._observation[0] == 1 and core.resets == 2
    assert env.total_decisions == 1 and env.core.prefix_decisions == 2
    assert env.core.prefix_ticks == 16 and env.core.credited_ticks == 1
    assert len(policy.calls) == 2 and policy.calls[0] == policy.calls[1]
    assert env.core._policy is policy
    assert env.telemetry_summary()["success_count"] == 0
    assert env.telemetry_summary()["checkpoint_policy_initialized_task_success_count"] == 1


def test_only_fresh_fallback_full_policy_episode_can_be_full_success():
    env, core, _ = adapter(Core([[('P08',8,None)], [('P13',8,'SUCCESS')], [('P06',8,None)]]), Request('P06'))
    policy = FrozenPolicy()
    env.install_prefix_policy(policy, policy.provenance)
    _, _, _, extras = env.step(torch.zeros(1,12))
    assert extras['episode_summaries'][0]['task_outcome_label'] == 'FULL_TASK_SUCCESS'
    assert env.telemetry_summary()['success_count'] == 1


def test_prebind_step_and_second_reset_rejected_without_action():
    env, core, _ = adapter()
    with pytest.raises(RuntimeError, match="install"):
        env.step(torch.zeros(1,12))
    with pytest.raises(RuntimeError, match="install"):
        env.core.reset()
    assert not core.actions and core.resets == 1


@pytest.mark.parametrize("mutation", [
    lambda c: c.step(ZERO),
    lambda c: c._history.update(residual=(1.,)*12),
    lambda c: setattr(c, 'observation', (1.,)*324),
    lambda c: c.frame.info['semantic_task'].update(remaining_task_time_s=199.),
])
def test_install_rejects_changed_bootstrap(mutation):
    env, core, _ = adapter()
    mutation(core)
    policy = FrozenPolicy()
    with pytest.raises(RuntimeError, match='untouched'):
        env.install_prefix_policy(policy, policy.provenance)


def test_provenance_deep_copy_and_rebinding_or_mutation_rejected():
    env, core, _ = adapter()
    policy, supplied = FrozenPolicy(), provenance()
    env.install_prefix_policy(policy, supplied)
    supplied['policy_contract']['actor_hidden_dims'][0] = 1
    assert env.cfg['prefix_policy_provenance']['policy_contract']['actor_hidden_dims'] == [256,256]
    with pytest.raises(RuntimeError, match='immutable'):
        env.install_prefix_policy(policy, policy.provenance)
    env.cfg['prefix_policy_provenance']['source_ppo_updates'] += 1
    with pytest.raises(RuntimeError, match='binding changed'):
        env.step(torch.zeros(1,12))


@pytest.mark.parametrize('key,value', [('checkpoint_sha256','bad'),('actor_parameter_sha256','F'*64),
    ('source_global_policy_decisions',True),('source_ppo_updates',-1),('checkpoint_path',''),
    ('policy_contract',{}),('extra',float('nan')),('extra',object())])
def test_provenance_rejects_invalid_before_any_rollin(key,value):
    env, core, _ = adapter()
    record=provenance();record[key]=value
    with pytest.raises(ValueError):
        env.install_prefix_policy(FrozenPolicy(), record)
    assert not core.actions and core.resets == 1


@pytest.mark.parametrize('key', WRITE_COUNTERS)
def test_each_of_four_write_counters_fails_closed(key):
    env, core, _ = adapter(Core(fault=lambda info: info.update({key:1})))
    policy=FrozenPolicy()
    with pytest.raises(RuntimeError, match='writes'):
        env.install_prefix_policy(policy, policy.provenance)
    assert core.resets == 1 and not env.core._credit_open and env.total_decisions == 0


@pytest.mark.parametrize('fault', [
    lambda i: i['actuator_target_effect_audit_summary'].update(all_ticks_verified=False),
    lambda i: i['actuator_target_effect_audit_summary'].update(actual_native_effect_tick_count=0),
    lambda i: i['actuator_target_effect_audit_summary'].update(own_phase_request_effect_tick_count=True),
    lambda i: i['actuator_target_effect_audit_ticks'][0].update(verified=False),
    lambda i: i['actuator_target_effect_audit_ticks'][0].update(episode_physics_tick=999),
    lambda i: i['actuator_target_effect_audit'].update(raw_policy_action_full12=ZERO),
    lambda i: i.update(no_in_episode_state_writes_verified=False),
])
def test_audit_fault_is_not_a_retriable_physical_prefix_miss(fault):
    env, core, _=adapter(Core(fault=fault));policy=FrozenPolicy()
    with pytest.raises(RuntimeError):env.install_prefix_policy(policy,policy.provenance)
    assert core.resets==1 and not env.core._credit_open


def test_existing_tick_observer_receives_prefix_and_credited_ticks():
    core=Core();seen=[]
    core.tick_observer=lambda before,after,projection:seen.append(after.physics_tick)
    env,_,_=adapter(core);policy=FrozenPolicy()
    env.install_prefix_policy(policy,policy.provenance)
    env.step(torch.zeros(1,12))
    assert seen==list(range(1,17))
    assert env.core.prefix_ticks==env.core.credited_ticks==8


@pytest.mark.parametrize('kwargs', [dict(target_phase='P05'),dict(target_phase='P14'),
    dict(teacher_offset_decisions=True),dict(teacher_offset_decisions=-1),
    dict(maximum_prefix_decisions=0),dict(maximum_prefix_decisions=True),
    dict(maximum_prefix_decisions=3001),dict(maximum_prefix_decisions=2,teacher_offset_decisions=2)])
def test_request_validation(kwargs):
    with pytest.raises(ValueError):Request(**kwargs)


def test_request_schema_and_teacher_backend_rejection():
    request=Request()
    assert request.as_dict()['schema']=='wlr50_clean.checkpoint_policy_prefix_request.v1'
    assert request.as_dict()['source']=='frozen_checkpoint_policy'
    assert 'checkpoint_path' not in request.as_dict()
    core=Core();core.backend=object.__new__(PrefixSemanticIsaacBackend)
    with pytest.raises(ValueError,match='ordinary'):adapter(core)


def test_policy_contract_boolean_cannot_be_replaced_with_number():
    env,core,_=adapter();record=provenance()
    record['policy_contract']['state_dependent_std']=1
    with pytest.raises(ValueError,match='exact heteroscedastic'):
        env.install_prefix_policy(FrozenPolicy(),record)
    assert not core.actions


def test_zero_residual_is_valid_prefix_without_required_effect_gate():
    def zero_effect(info):
        info['actuator_target_effect_audit']['changed_target_channel_count']=0
        for row in info['actuator_target_effect_audit_ticks']:
            row.update(actual_native_effect=False,own_phase_request_effect=False)
        info['actuator_target_effect_audit_summary'].update(
            actual_native_effect_tick_count=0,own_phase_request_effect_tick_count=0)
    env,core,_=adapter(Core(fault=zero_effect))
    env.install_prefix_policy(lambda obs:ZERO,provenance())
    assert env.core._credit_open and core.actions==[ZERO]
