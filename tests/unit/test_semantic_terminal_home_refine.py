"""Source-home nominal refinement; synthetic checks are not physical success."""
from copy import deepcopy

import pytest

from test_semantic_p13_final_stop_owner import (
    contract, provider, issue, at_first_source_stop, layer,
)
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider

MODE = "source_home_after_physical_stop_v1"


def refined(contract):
    spec = deepcopy(provider(contract).spec)
    spec['nominal']['final_stop_owner'] = MODE
    return NominalMotionProvider.from_handoff(contract, spec=spec, stage_id='P13',
        nominal_full12=contract.phase('P13').start_full12, tracking_servo_names=())


def test_no_change_before_real_controlled_post_window(contract):
    new, old = refined(contract), provider(contract)
    for tick in range(100, 230):
        assert issue(new, contract, tick, started=False) == issue(old, contract, tick, started=False)
        assert new.tracking_servo_names == old.tracking_servo_names
        assert new.normal_drive_bias_full12 == old.normal_drive_bias_full12
    assert new._final_home_recovery is None and new._final_stop_owner is None


def test_source_home_without_old_wheel_pulse_and_no_evaluator_change(contract):
    p = refined(contract)
    expected = tuple(contract.phase('P13').end_full12[:8])+(0.,)*4
    for tick in range(100, 150):
        assert issue(p, contract, tick) == expected
        assert p.tracking_servo_names == ()
        assert p.normal_drive_bias_full12 == (0.,)*12
    d = p.nominal_suggestion_diagnostics['final_stop_owner']
    assert d['home_recovery']['entry']['entry_observation_tick'] == 100
    assert not d['home_recovery']['entry']['source_wheel_pulse_replayed']
    assert not d['home_recovery']['home_pose_is_success_gate']
    assert not d['home_recovery']['physical_observation_window_changed']
    assert not d['task_success_awarded']
    assert p.spec['final']['post_completion_observation_s'] == 1.


def test_transient_speed_loss_can_stop_but_cannot_start_home_until_controlled(contract):
    p = refined(contract)
    previous = p.nominal_full12[:8]
    def uncontrolled(t, _):
        t['physical_evaluator'].update(final_controlled=False, task_completed_controlled=False)
    assert issue(p, contract, 100, change=uncontrolled) == previous+(0.,)*4
    assert p._final_stop_owner is not None and p._final_home_recovery is None
    assert issue(p, contract, 101) == tuple(contract.phase('P13').end_full12[:8])+(0.,)*4
    assert p._final_home_recovery['entry_observation_tick'] == 101
    # Do not fall back to an obsolete source pulse during the new motion.
    assert issue(p, contract, 102, change=uncontrolled)[8:] == (0.,)*4
    assert p._final_home_recovery['entry_observation_tick'] == 101


@pytest.mark.parametrize('bad', ['region', 'support', 'unverified', 'not_placed', 'terminal'])
def test_invalid_physical_state_cannot_start_refinement(contract, bad):
    p = refined(contract)
    def changed(t, _):
        ev = t['physical_evaluator']
        if bad == 'region': ev['final_region_valid'] = False
        elif bad == 'support': ev['final_support_available'] = False
        elif bad == 'unverified': ev['physical_evidence_status'] = 'CONTACT_BEARING_UNVERIFIED'
        elif bad == 'not_placed': ev['history']['placed']['RR'] = False
        else: ev['termination_reason'] = 'TASK_FAILURE_BODY_COLLISION'
    issue(p, contract, 100, change=changed)
    assert p._final_home_recovery is None and p._final_stop_owner is None


def test_existing_mode_still_holds_prior_pose(contract):
    p = at_first_source_stop(contract)
    old = p.nominal_full12[:8]
    assert issue(p, contract, 102) == old+(0.,)*4
    assert p._final_home_recovery is None
    assert 'home_recovery' not in p.nominal_suggestion_diagnostics['final_stop_owner']


def test_home_refinement_does_not_mask_nonzero_residual(contract):
    from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    from test_semantic_p13_final_stop_owner import ROOT
    p = refined(contract)
    nominal = issue(p, contract, 100)
    bridge = PhaseTransitionBridge(build_semantic_projector(
        ROOT/'configs/ppo_task_first_recovery_v1/execution_profile.yaml'))
    r = bridge.project_tick((.2,)*12, state_id='P13', nominal_action_full12=nominal,
        reference_action_full12=nominal, reference_delta_full12=(0.,)*12,
        dt_s=1/120).projection
    assert r.effective_action_mask_full12 == (1,)*12
    assert all(x != 0. for x in r.safe_projected_residual_full12)
    assert all(x != 0. for x in r.applied_action_full12[8:])
