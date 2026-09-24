"""Actual prefix adapter gates with fake ordinary backend; zero physical credit."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_semantic_checkpoint_prefix import Core, ZERO
from wlr50_clean.ppo.semantic_checkpoint_prefix import (
    CheckpointPolicyPrefixRequest, CheckpointPolicyPrefixRslAdapter,
    _CheckpointPolicyCreditCore, _provenance)
from wlr50_clean.ppo.semantic_policy_distribution import policy_contract
from wlr50_clean.ppo.semantic_rear_owner_profile import (
    REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_DIM, REAR_OWNER_OBSERVATION_LAYOUT)


class OwnerCore(Core):
    observation_dimension = REAR_OWNER_OBSERVATION_DIM

    def __init__(self, dimension=REAR_OWNER_OBSERVATION_DIM, layout=REAR_OWNER_OBSERVATION_LAYOUT):
        super().__init__([[('P10',8,None)]])
        self.observation_dimension=dimension
        self.observation_schema=SimpleNamespace(observation_layout=layout)

    def reset(self, seed=1001, options=None):
        super().reset(seed=seed,options=options)
        self.observation=(0.,)*self.observation_dimension
        return self.observation

    def step(self, raw):
        step=super().step(raw)
        self.observation=tuple(step.observation)+(0.,)*(self.observation_dimension-324)
        return replace(step,observation=self.observation)


def provenance(source):
    if source=='successful_nominal':
        return dict(schema='wlr50_clean.successful_nominal_prefix.v1',source=source,
            execution_profile_sha256='a'*64,stage_task_spec_sha256='b'*64,runtime_content_sha256='c'*64,
            interface_contract=dict(observation_dimension=REAR_OWNER_OBSERVATION_DIM,
                observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,action_dimension=12),
            raw_action_full12=[0.]*12,policy_credit=False)
    return dict(checkpoint_path='C:/synthetic/owner439.pt',checkpoint_sha256='a'*64,
        actor_parameter_sha256='b'*64,source_global_policy_decisions=225280,source_ppo_updates=1725,
        policy_contract=policy_contract(REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT))


@pytest.mark.parametrize('source',['successful_nominal','frozen_checkpoint_policy'])
def test_real_prefix_adapter_439_init_install_and_credit(source):
    core=OwnerCore(); records=[]
    env=CheckpointPolicyPrefixRslAdapter(core,seed=1001,device='cpu',evidence_sink=records.append,
        request=CheckpointPolicyPrefixRequest('P10',source=source))
    assert type(env.core) is _CheckpointPolicyCreditCore
    assert env.get_observations()['policy'].shape==(1,439)
    assert core.resets==1 and env.total_decisions==0
    binding=provenance(source); assert _provenance(binding)==binding
    start=env.install_prefix_policy(lambda observation: ZERO,binding)
    assert start['actual_phase']=='P10' and core.resets==1
    assert env.core.prefix_decisions==1 and env.core.prefix_ticks==8 and env.total_decisions==0
    assert all(r['policy_credit'] is False for r in records)
    _,_,_,extras=env.step(torch.zeros((1,12)))
    assert env.total_decisions==1 and env.core.credited_decisions==1
    assert extras['semantic_decisions'][0]['prefix_checkpoint_policy_data_in_ppo_storage'] is False
    assert env.get_observations()['policy'].shape==(1,439)


@pytest.mark.parametrize('dimension,layout',[(438,REAR_OWNER_OBSERVATION_LAYOUT),
    (422,REAR_OWNER_OBSERVATION_LAYOUT),(439,'role419_p02_progress_v1'),(439,None)])
def test_wrong_live_pair_rejected_before_reset(dimension,layout):
    core=OwnerCore(dimension,layout)
    with pytest.raises(ValueError,match='supported observation layout'):
        CheckpointPolicyPrefixRslAdapter(core,seed=1001,device='cpu',evidence_sink=lambda row:None,
            request=CheckpointPolicyPrefixRequest('P10'))
    assert core.resets==0 and core.actions==[]


@pytest.mark.parametrize('field,value',[('observation_dimension',438),
    ('observation_layout','role419_p02_progress_v1'),('action_dimension',11)])
def test_wrong_nominal_provenance_pair_rejected(field,value):
    binding=provenance('successful_nominal'); binding['interface_contract'][field]=value
    with pytest.raises(ValueError,match='supported explicit layout'):
        _provenance(binding)
