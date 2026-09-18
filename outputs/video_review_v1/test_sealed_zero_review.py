"""Small output-helper wiring tests; fixtures do not represent physical runs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

spec=importlib.util.spec_from_file_location('zero_review_test_target',Path(__file__).with_name('sealed_zero_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


def dump(path,value):path.write_text(json.dumps(value),encoding='utf8')


def dump_rows(path,values):path.write_text(''.join(json.dumps(x)+'\n' for x in values),encoding='utf8')


def test_active_run_rejected_before_any_large_log(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    dump(tmp_path/'run_manifest.json',{'lifecycle':'RUNNING'})
    with pytest.raises(RuntimeError,match='not finalized'):review.sealed_source(source)


def test_unsealed_run_rejected(tmp_path):
    with pytest.raises(RuntimeError,match='sealed manifest'):review.sealed_source(tmp_path)


def test_keyframes_bind_next_real_frame_and_do_not_invent_missing_leg_events():
    m={'physical_episode':{'physical_task_evaluation':{'history':{'event_ticks':{
        'active_lift':{'FR':23},'front_edge_crossed':{'FR':29},'placed':{'FR':30}}}}}}
    result=review.keyframes(m,{'phase_samples':{},'source_home_entry':None},[NS(sim_step=t) for t in (8,16,24,32,35)])
    by_label={x['label']:x for x in result}
    assert by_label['FR_active_lift']['actual_frame_tick']==24
    assert by_label['FR_front_edge_crossed']['actual_frame_tick']==32
    assert by_label['terminal']['actual_frame_tick']==35
    assert not any(x['label'].startswith('RR') for x in result)


def fixture(source,raw_value=0):
    home={'entry_observation_tick':1,'target_servo_deg':[2.]*8,'source':'fixture source home'}
    owner={'active':True,'home_recovery':{'entry':home}}
    dump_rows(source/'video_policy_decisions.jsonl',[{'decision':1,'end_tick':3,
        'raw_policy_action_full12':[raw_value]*12,'step_info':{'semantic_task':{'nominal':{'final_stop_owner':owner}}}}])
    dump_rows(source/'native_tick_audit.jsonl',[{'episode_physics_tick':t,'source_phase_id':'P13',
        'nominal_full12':[2.]*8+[0.]*4,'projected_residual_full12':[0.]*12,
        'native_audit':{'raw_policy_action_full12':[0.]*12,'native_drive_target_full12':[2.]*8+[0.]*4,
            'actual_native_targets':{'fixture':True}}} for t in (1,2,3)])
    dump_rows(source/'physical_observations.jsonl',[{'physics_tick':t,'simulation_time_s':t/120,
        'actual_full12':[1.5]*8+[0.]*4,'commanded_full12':[2.]*8+[0.]*4,'contacts':{},
        'base':{'position_w_m':[0,0,0],'linear_velocity_w_m_s':[.03,.04,0],
            'angular_velocity_w_rad_s':[0,0,.2]}} for t in (0,1,2,3)])
    return {'episode_physics_ticks':3,'physical_episode':{'task_success':False},
        'success_candidate':False,'camera':{}}


def test_home_request_actual_error_and_body_velocity_are_separate(tmp_path):
    result=review.aggregate(tmp_path,fixture(tmp_path))
    assert result['source_home_entry']['entry_observation_tick']==1
    assert result['maximum_abs_terminal_home_error_deg']==.5
    assert result['terminal_physical']['body_speed_m_s']==.05
    assert result['source_success_candidate'] is False
    assert result['all_issued_and_dispatched_policy_and_projected_residuals_zero'] is True
    assert result['manual_camera_QA_pending'] is True


def test_nonzero_policy_cannot_be_labelled_Nplus0(tmp_path):
    with pytest.raises(RuntimeError,match='all twelve'):
        review.aggregate(tmp_path,fixture(tmp_path,raw_value=.001))
