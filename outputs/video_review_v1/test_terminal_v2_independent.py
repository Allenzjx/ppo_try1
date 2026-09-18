"""Independent request continuity checks, not physical success evidence."""
import copy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_semantic_p13_final_stop_owner import contract,provider,issue
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider


def candidate(contract,version):
    spec=copy.deepcopy(provider(contract).spec)
    spec['nominal']['final_stop_owner']='source_home_after_physical_stop_'+version
    return NominalMotionProvider.from_handoff(contract,spec=spec,stage_id='P13',
        nominal_full12=contract.phase('P13').start_full12,tracking_servo_names=())


def test_v2_identical_to_v1_before_physical_stop_acquisition(contract):
    old,new=candidate(contract,'v1'),candidate(contract,'v2')
    for tick in range(100,230):
        assert issue(old,contract,tick,started=False)==issue(new,contract,tick,started=False)
        assert old.tracking_servo_names==new.tracking_servo_names
        assert old.normal_drive_bias_full12==new.normal_drive_bias_full12
    assert new._final_home_recovery is None and new._final_stop_owner is None


def test_delayed_home_starts_from_retained_owner_nominal_not_future_source(contract):
    p=candidate(contract,'v2'); prior=p.nominal_full12
    def uncontrolled(task,_):
        task['physical_evaluator'].update(final_controlled=False,task_completed_controlled=False)
    for tick in range(100,120):
        assert issue(p,contract,tick,change=uncontrolled)==prior[:8]+(0.,)*4
    entry_owner=p._final_stop_owner['held_nominal_servo_deg']
    assert issue(p,contract,120)==entry_owner+(0.,)*4
    assert p._final_home_recovery['start_servo_deg']==entry_owner
    assert p._final_home_recovery['entry_observation_tick']==120


def test_no_residual_channel_mask_in_v2_home_ramp(contract):
    from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    p=candidate(contract,'v2')
    nominal=issue(p,contract,100)
    bridge=PhaseTransitionBridge(build_semantic_projector(ROOT/'configs/ppo_task_first_recovery_v1/execution_profile.yaml'))
    projection=bridge.project_tick((.2,)*12,state_id='P13',nominal_action_full12=nominal,
        reference_action_full12=nominal,reference_delta_full12=(0.,)*12,dt_s=1/120).projection
    assert projection.effective_action_mask_full12==(1,)*12
    assert all(v!=0 for v in projection.safe_projected_residual_full12)
    assert all(v!=0 for v in projection.applied_action_full12[8:])
