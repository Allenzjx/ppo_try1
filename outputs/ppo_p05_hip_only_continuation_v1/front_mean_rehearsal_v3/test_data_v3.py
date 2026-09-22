"""Pure metadata/composition safeguards; never fits or modifies a checkpoint."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('data_v3_tested',HERE/'data_v3.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


def fixture():
    manifest=data.read(data.MANIFEST)
    reference=data.read(data.OUT/'checkpoints/history/checkpoint_aux_detP02_step_000211968_v2_manifest.json')
    metadata=data.read(data.OUT/'checkpoints/history/checkpoint_step_000212352_manifest.json')
    return metadata,deepcopy(metadata['runtime_contract']),manifest,reference


def test_later_normal_PPO_metadata_keeps_reference_separate():
    args=fixture();before=deepcopy(args)
    receipt=data.validate_current_metadata(*args)
    assert receipt['actual_caller_PPO_counters']['global_policy_decisions']==212352
    assert receipt['old_admission_was_rebound_to_current_weights'] is False
    assert receipt['current_weights_or_trajectory_equivalent_to_demonstration_claimed'] is False
    assert args==before


@pytest.mark.parametrize('kind',('runtime','origin','event1','old_aux','normalizer','older_counter'))
def test_changed_semantics_or_lineage_rejected(kind):
    metadata,contract,manifest,reference=fixture()
    if kind=='runtime':contract['source_git_commit']='0'*40
    elif kind=='origin':metadata['p05_capture_assist_branch']['counter_origin']['global_policy_decisions']+=1
    elif kind=='event1':metadata['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']['events'][0]['event_index']=99
    elif kind=='old_aux':metadata['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']['accepted_auxiliary_updates_total']+=1
    elif kind=='normalizer':metadata['normalizer_state_sha256']='0'*64
    elif kind=='older_counter':metadata['global_policy_decisions']=211967
    with pytest.raises(ValueError):data.validate_current_metadata(metadata,contract,manifest,reference)


def test_manifest_does_not_label_probe_or_claim_invariance():
    manifest=data.read(data.MANIFEST)
    assert manifest['positive_labels']['P01']['independent_validation_rows']==0
    assert manifest['positive_labels']['P02']['validation_range']==[1,254,3]
    assert manifest['protection_only']['stochastic_P01_raw_used_as_deterministic_label'] is False
    assert manifest['protection_only']['mean_head_P03plus_mathematical_invariance_claimed'] is False
    assert manifest['positive_labels']['loss_weights_selected_by_data_loader'] is False


@pytest.mark.parametrize('qualified,crossed,placed,eligible',((False,False,False,True),(True,False,False,True),(True,True,False,False),(True,True,True,True)))
def test_rear_probe_eligibility_does_not_confuse_placed_bypass(qualified,crossed,placed,eligible):
    ev={'history':{'active_lift':{'RR':qualified},'front_edge_crossed':{'RR':crossed},'placed':{'RR':placed}}}
    assert (data.rr_rear_probe_reason(ev) is not None)==eligible
