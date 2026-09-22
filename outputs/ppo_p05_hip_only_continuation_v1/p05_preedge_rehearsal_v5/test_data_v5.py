"""CPU data-selection negatives; no optimization or real checkpoint writes."""
from copy import deepcopy
import numpy as np
import pytest
import data_v5 as d


def evidence():
    state={'mode':0.,'initialized':0.,'mode_name':'WAIT'}
    return state,{'state_before':deepcopy(state),'state_after':deepcopy(state),'owner_indices':[],
        'candidate_before_assist_full12':[0.]*12,'candidate_after_assist_full12':[0.]*12}


def test_uninitialized_WAIT_is_eligible():
    state,ev=evidence()
    assert d.unassisted_entire_interval(state,ev)


@pytest.mark.parametrize('kind',['input_initialized','last_tick_initialized','endpoint_initialized','owner','hidden_transform','released'])
def test_endpoint_empty_owner_alone_is_insufficient(kind):
    state,ev=evidence()
    if kind=='input_initialized':state['initialized']=1
    if kind=='last_tick_initialized':ev['state_before']['initialized']=1
    if kind=='endpoint_initialized':ev['state_after']['initialized']=1
    if kind=='owner':ev['owner_indices']=[0,1]
    if kind=='hidden_transform':ev['candidate_after_assist_full12'][0]=1
    if kind=='released':ev['state_after'].update(mode=5.,mode_name='RELEASED',initialized=1.)
    assert not d.unassisted_entire_interval(state,ev)


@pytest.fixture
def arrays():
    with np.load(d.DATA,allow_pickle=False) as loaded:
        return {k:loaded[k].copy() for k in loaded.files}


def test_actual_data_fixed_split_has_no_dropped_eligible_rows(arrays):
    d.validate_arrays(arrays)
    assert len(arrays['observations389'])==438
    assert len(arrays['train_indices'])==291 and len(arrays['validation_indices'])==147
    for ep in (0,1,2):
        assert sum(arrays['episode_indices']==ep)==146
        assert sum(arrays['episode_indices'][arrays['train_indices']]==ep)==97
        assert sum(arrays['episode_indices'][arrays['validation_indices']]==ep)==49
    assert set(arrays['train_indices'])|set(arrays['validation_indices'])==set(range(438))


@pytest.mark.parametrize('kind',['wrong_phase','initialized_input','mean_label','nonfinite','split','P05_protection'])
def test_invalid_actual_copy_rejected(arrays,kind):
    if kind=='wrong_phase':arrays['observations389'][0,4]=0;arrays['observations389'][0,5]=1
    if kind=='initialized_input':arrays['observations389'][0,373]=1
    if kind=='mean_label':arrays['actual_raw12'][0]=arrays['source_mean12'][0]
    if kind=='nonfinite':arrays['actual_raw12'][0,0]=np.nan
    if kind=='split':arrays['validation_indices'][0]=0
    if kind=='P05_protection':arrays['protection_observations'][0,:13]=0;arrays['protection_observations'][0,4]=1
    with pytest.raises(ValueError):d.validate_arrays(arrays)


def test_actual_read_loader_adds_no_labels_to_protection_or_learning_credit():
    metadata=d.read(d.SOURCE.with_name(d.SOURCE.stem+'_manifest.json'))
    metadata['checkpoint_path']=str(d.SOURCE)
    result=d.load_reviewed_data(metadata,metadata['runtime_contract'])
    assert result['train_observations'].shape==(291,389)
    assert result['validation_observations'].shape==(147,389)
    assert result['protection_observations'].shape==(8,389)
    assert 'protection_raw_targets' not in result
    assert not result['receipt']['AUX_authorized'] and result['receipt']['PPO_or_AUX_credit_added']==0
    assert result['receipt']['X17_remapping_performed'] is False
    metadata['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='currentCP216448'):
        d.load_reviewed_data(metadata,metadata['runtime_contract'])


def test_placement_qualification_does_not_relabel_assist_actions_as_raw_positive():
    m=d.read(d.MANIFEST)
    assert m['exclusions']['assisted_or_initialized_interval']==152
    assert all(e['first_FL_placed_endpoint']['assist_owner_indices']==[0,1] for e in m['episode_local_qualification'])
    assert all(r['owners']==[] and r['whole_N1_unassisted_by_monotonic_uninitialized_WAIT'] for r in m['rows_provenance'])
    assert m['coverage']['source_near_last_live_FL_front_and_gap_within10mm']['rows']==0
