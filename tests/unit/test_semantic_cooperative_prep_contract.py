"""Local contract/potential checks, not physical-success evidence."""
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from wlr50_clean.ppo.semantic_backend import load_execution_profile
from wlr50_clean.ppo.semantic_reward import load_semantic_reward_config
from wlr50_clean.ppo.semantic_supervisor import load_task_spec
from wlr50_clean.ppo.semantic_cooperative_preparation import MODE, REWARD_CONFIG, workspace_progress

CFG=Path(__file__).resolve().parents[2]/'configs/ppo_rr_rl_timing_policy_learning_v1'

def test_explicit_modes_keep_rear_assists_off_and_full12_open():
    profile=load_execution_profile(CFG/'execution_profile.yaml')
    spec=load_task_spec(CFG/'stage_task_spec.yaml')
    reward=load_semantic_reward_config(CFG/'reward_config.yaml')
    assert profile['cooperative_preparation_mode']==spec['cooperative_preparation_mode']==MODE
    assert profile['rr_capture_assist_mode'] is None
    assert profile['nominal_geometry_advisory'] is None
    assert profile['rr_capture_wheel_mode']=='off'
    assert profile['capture_assist_mode']=='p05_hip_only_continuation_v1'
    assert profile['residual']['allowed_channels']=='all_12_in_every_phase'
    assert profile['residual']['closed_channels']=={}
    assert profile['episode_timeout_s']==spec['episode_maximum_duration_s']==200
    assert reward.values['cooperative_preparation']==REWARD_CONFIG
    assert reward.failure_avoidance_bound < reward.values['failure_cost']

def test_cost_is_included_in_failure_avoidance_upper_bound():
    from wlr50_clean.ppo.semantic_reward import SemanticRewardConfig
    new=load_semantic_reward_config(CFG/'reward_config.yaml')
    old=deepcopy(dict(new.values));old.pop('cooperative_preparation')
    prior=SemanticRewardConfig(old,new.path)
    assert new.failure_avoidance_bound-prior.failure_avoidance_bound == pytest.approx(.01/15/(1-.9985))

def test_stationary_preparation_is_not_an_infinite_bonus():
    diag=dict(valid=True,relevant=True,fl_range_credit=1.,rl_space_credit=1.)
    value=workspace_progress(1.,1.,diag)
    assert value==1.
    # The same fixed physical potential is differenced, not rewarded each tick.
    assert 5*(.9985*value-value)-.02/15 < 0.
    assert workspace_progress(.3,.7,None)==.5

def test_undeclared_cost_or_preparation_weight_is_rejected(tmp_path):
    values=yaml.safe_load((CFG/'reward_config.yaml').read_text())
    values['cooperative_preparation']['counterroll_cost_per_s']=1.
    selected=tmp_path/'reward.yaml';selected.write_text(yaml.safe_dump(values,sort_keys=False))
    with pytest.raises(ValueError,match='bounded explicit'):
        load_semantic_reward_config(selected)
