"""Synthetic source ownership/permission checks, not physical height claims."""
from copy import deepcopy

import pytest
import yaml

from test_semantic_source_partial_order_v1 import ROOT, contract, current_input
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.ppo.semantic_height_recovery import (
    validate_height_candidate, source_owner_height_target, current_rr_recovery_permission,
)


def spec():
    return yaml.safe_load((ROOT / 'configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml').read_text())


@pytest.mark.parametrize('value,entry,expected', [(49.2,22.8,45.2),(24.,22.8,22.8),(18.,22.8,18.)])
def test_fixed_entry_reduction_does_not_integrate(value, entry, expected):
    for _ in range(100):
        assert source_owner_height_target(source_value=value, source_entry=entry,
            reduction_deg=4., recovery_deg=0.) == pytest.approx(expected)


def test_validation_rejects_unbounded_or_shared_reduction():
    item = spec()['nominal']['height_recovery']
    assert validate_height_candidate(item) == item
    for key, value in [('preparation_reduction_deg', {'front_left_hip':11.,'rear_left_hip':0.}),
                       ('post_lift_recovery_deg', {'front_left_hip':3.}), ('recovery_rate_deg_s',61.)]:
        bad = deepcopy(item); bad[key] = value
        with pytest.raises(ValueError): validate_height_candidate(bad)


def test_fl_only_preserves_source_clock_atomic_group_and_later_owner(contract):
    selected = spec()
    selected['nominal']['height_recovery']['preparation_reduction_deg'] = {'front_left_hip':4.,'rear_left_hip':0.}
    selected['nominal']['height_recovery']['post_lift_recovery_deg'] = {'front_left_hip':0.,'rear_left_hip':0.}
    baseline = deepcopy(selected); baseline['nominal'].pop('height_recovery')
    options = dict(stage_id='P07', nominal_full12=contract.phase('P07').start_full12, tracking_servo_names=())
    a = NominalMotionProvider.from_handoff(contract, spec=selected, **options)
    b = NominalMotionProvider.from_handoff(contract, spec=baseline, **options)
    for tick in range(1150):
        phase = 'P07' if tick < 220 else 'P08' if tick < 280 else 'P09'
        item, raw = current_input(contract, phase, tick, space=True, fr_motion=3.,rl_motion=3.,rr_load=.1,
                                  continuing=phase != 'P07')
        ca, cb = a.evaluate(item,raw), b.evaluate(item,raw)
        assert ca[1:] == cb[1:]
        for la,lb in zip(a._continuous_layers,b._continuous_layers):
            assert la['ticks'] == lb['ticks']
            if la['sample'] is None:
                assert lb['sample'] is None
            else:
                assert la['sample'].atomic_groups == lb['sample'].atomic_groups
        if tick >= 1000:
            assert ca == cb  # P09's later FL owner replaces the P07 adjustment.
    assert a.nominal_suggestion_diagnostics['height_recovery']['task_or_support_credit_awarded'] is False


def test_live_recovery_requires_current_lift_and_other_measured_bearing(contract):
    s = spec(); task, _ = current_input(contract, 'P09', 300, continuing=True)
    ev = task['physical_evaluator']; ev['history']['active_lift']['RR'] = True
    for leg in ('FR','RL'):
        ev['current_legs'][leg].update(air=False, ground_contact=True,
            bearing_force_n=1., support=True, bearing_verified=True)
    assert current_rr_recovery_permission(task, support_spec=s['support'])
    ev['current_legs']['RR']['current_lift_valid'] = False
    assert not current_rr_recovery_permission(task, support_spec=s['support'])
    ev['current_legs']['RR']['current_lift_valid'] = True
    ev['current_legs']['RL']['bearing_force_n'] = None
    assert not current_rr_recovery_permission(task, support_spec=s['support'])
