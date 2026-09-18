"""Directional report wiring; no physical success fixtures or optimizers."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('p02_chain_test',Path(__file__).with_name('sealed_p02_wheel_chain.py'))
chain=importlib.util.module_from_spec(spec); spec.loader.exec_module(chain)


def row():
    return {'tick':400,'simulation_time_s':400/120,'phase':'P02',
        'N_canonical_rad_s':[.3]*4,'mapped_N_canonical_rad_s':[.3]*4,
        'actual_residual_permission_mask':[1]*4,'zero_current_policy_target_canonical_rad_s':[.3]*4,
        'same_tick_effective_target_delta_canonical_rad_s':[-.28,0,0,.2],
        'final_target_canonical_rad_s':[.02,.3,.3,.5],'measured_qd_canonical_rad_s':[.03,.01,.2,.4]}


def test_policy_cancellation_not_mask_and_tracking_is_separate():
    r=row(); fl=' '.join(chain.explain(r,0)); fr=' '.join(chain.explain(r,1))
    assert 'policy effect reduces' in fl and 'permission off' not in fl
    assert 'target present but weak measured tracking' in fr and 'policy effect reduces' not in fr


def test_residual_permission_zero_does_not_zero_N():
    r=row(); r['actual_residual_permission_mask'][1]=0
    assert 'residual permission off' in ' '.join(chain.explain(r,1))
    assert r['N_canonical_rad_s'][1]==r['final_target_canonical_rad_s'][1]==.3


def test_first_layer_timestamps_not_claimed_for_unobserved_layers():
    r=row(); first=chain.first_observations({400:r})
    assert first['FL']['policy_reduces_positive_same_state_baseline_below_half']['tick']==400
    assert first['FL']['residual_permission_off'] is None
    assert first['FR']['policy_reduces_positive_same_state_baseline_below_half'] is None
    assert first['FR']['measured_rotation_near_zero_while_target_present']['tick']==400


def test_empty_positive_nominal_stats_remain_unknown_not_zero():
    assert chain.list_stats([]) is None


def test_bounded_prefix_stops_without_reading_rest(tmp_path):
    p=tmp_path/'fixture.jsonl'; p.write_text('{"tick":1}\n{"tick":2}\n{"tick":3}\n')
    rows,binding=chain.bounded_rows(p,lambda r:r['tick']<2)
    assert [r['tick'] for _,r in rows]==[1]
    assert binding['read_prefix_lines']==2 and binding['whole_file_hashed'] is False
