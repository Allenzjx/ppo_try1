"""Measured synthetic counterexamples; no Isaac, policy or historical verdict edits."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import math

import pytest

from test_semantic_supervisor import observation, advance
from test_sensing_stack import _fake_adapter, _FakeContactBackend
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, load_task_spec, LEG_ORDER
from wlr50_clean.ppo.semantic_physical_sensing import (
    physical_contact_surface, SemanticColliderGeometry, SemanticSensorReader,
    SemanticPhysicalObservation,
)
from wlr50_clean.sensing.geometry import ColliderGeometryCache, Aabb, _world_bounds_from_body_local_points
from wlr50_clean.sensing.observation import Observation
from wlr50_clean.sensing.contact_classifier import BASE_BODY, WHEEL_BODIES

ROOT = Path(__file__).resolve().parents[2]


def new_spec():
    spec = load_task_spec(ROOT / 'configs/ppo_semantic_v3/stage_task_spec.yaml')
    spec['physical_acceptance_version'] = 'all_stage_v1'
    spec['final']['post_completion_observation_s'] = 1.0
    spec.pop('transfer_roles', None)  # This file isolates physical acceptance, not role estimation.
    return spec


def new_observation():
    obs = observation()
    obs['geometry_pose_aware'] = True
    obs['body_bounds_w_m'] = {'base_link': {'minimum_m': [.2,-.05,.05], 'maximum_m': [.4,.05,.15]}}
    obs['contacts']['base_link'] = {'obstacle': dict(pair_verified=True, active=False,
        sensor_body='base_link', other_body='/World/Obstacle', force_w_n=[0.,0.,0.], normal_force_n=0.)}
    for joint in obs['joints'].values():
        joint['command_deg'] = joint['position_deg']
    for name in WHEEL_ORDER:
        wheel = obs['wheels'][name]
        pair = obs['contacts'][wheel['body_name']]
        pair['ground']['force_w_n'] = [0.,0.,7.]
        pair['obstacle'].update(force_w_n=[0.,0.,0.], contact_point_w_m=None)
    return obs


def set_leg(obs, leg, *, x=None, bottom=None, surface='AIR', hip=None):
    index = LEG_ORDER.index(leg)
    wheel = obs['wheels'][WHEEL_ORDER[index]]
    if x is not None:
        wheel['center_w_m'][0] = wheel['bottom_w_m'][0] = x
    if bottom is not None:
        wheel['bottom_w_m'][2], wheel['center_w_m'][2] = bottom, bottom+.05
    ground, obstacle = (obs['contacts'][wheel['body_name']][key] for key in ('ground','obstacle'))
    ground.update(active=surface=='GROUND', normal_force_n=7. if surface=='GROUND' else 0.,
                  force_w_n=[0.,0.,7.] if surface=='GROUND' else [0.,0.,0.])
    obstacle.update(active=surface in ('TOP','WALL','AMBIGUOUS'),
        normal_force_n=7. if surface in ('TOP','WALL','AMBIGUOUS') else 0.,
        force_w_n=[7.,0.,0.] if surface=='WALL' else [0.,0.,7.] if surface in ('TOP','AMBIGUOUS') else [0.,0.,0.],
        contact_point_w_m=([max(.501,wheel['center_w_m'][0]),wheel['center_w_m'][1],.05]
                          if surface=='TOP' else [.5,wheel['center_w_m'][1],.02] if surface=='WALL' else None))
    if hip is not None:
        obs['joints'][SERVO_ORDER[2*index]].update(position_deg=hip, command_deg=hip+.6)


def early_top_place(ev, obs, leg):
    obs = advance(obs); set_leg(obs,leg,bottom=.012,hip=1.5); ev.observe(obs)
    obs = advance(obs); set_leg(obs,leg,x=.498,bottom=.049,surface='TOP',hip=3.); snap=ev.observe(obs)
    assert snap['current_legs'][leg]['top_surface_contact']
    assert snap['history']['active_lift'][leg]
    assert not snap['history']['front_edge_crossed'][leg]
    obs = advance(obs); set_leg(obs,leg,x=.501,bottom=.049,surface='TOP',hip=3.); snap=ev.observe(obs)
    assert snap['history']['front_edge_crossed'][leg] and snap['history']['placed'][leg]
    return obs


@pytest.mark.parametrize('leg', LEG_ORDER)
def test_whole_body_early_top_without_air_above_top_is_accepted(leg):
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    if leg=='RL': obs=early_top_place(ev,obs,'RR')
    obs=early_top_place(ev,obs,leg)
    assert ev.snapshot['termination_reason'] is None
    assert ev.snapshot['history']['placed'][leg]
    assert ev.snapshot['current_legs'][leg]['clearance_m'] < 0.


@pytest.mark.parametrize('leg', LEG_ORDER)
def test_missing_q_is_unverified_not_wheel_only_or_success(leg):
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    for _ in range(3):
        obs=advance(obs); set_leg(obs,leg,x=.501,bottom=.049,surface='TOP'); snap=ev.observe(obs)
    assert snap['termination_reason'] is None and not snap['success']
    assert not snap['history']['placed'][leg]
    assert snap['current_legs'][leg]['crossing_evidence_status']=='UNVERIFIED'


@pytest.mark.parametrize('leg', LEG_ORDER)
def test_positive_powered_wall_ascent_without_joint_demand_is_rejected(leg):
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    for x,bottom in ((.47,.01),(.48,.025),(.501,.049)):
        obs=advance(obs); set_leg(obs,leg,x=x,bottom=bottom,surface='WALL')
        obs['wheels'][WHEEL_ORDER[LEG_ORDER.index(leg)]]['command_rad_s']=.3
        snap=ev.observe(obs)
    assert snap['termination_reason']=='TASK_FAILURE_WHEEL_ONLY_CLIMB'
    assert snap['termination_source']=='WHEEL_ONLY_PROCESS'


def test_ground_revokes_attempt_not_other_leg_history_and_allows_retry():
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    obs=early_top_place(ev,obs,'FR')
    for bottom,hip in ((.012,1.5),(.04,3.)):
        obs=advance(obs); set_leg(obs,'FL',bottom=bottom,hip=hip); ev.observe(obs)
    assert ev.snapshot['history']['active_lift']['FL']
    obs=advance(obs); set_leg(obs,'FL',bottom=0.,surface='GROUND'); snap=ev.observe(obs)
    assert not snap['history']['active_lift']['FL'] and snap['history']['placed']['FR']
    obs=early_top_place(ev,obs,'FL')
    assert ev.snapshot['history']['placed']['FL']


def test_rr_first_still_uses_physical_history():
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    obs=advance(obs); set_leg(obs,'RL',bottom=.012,hip=1.5); ev.observe(obs)
    obs=advance(obs); set_leg(obs,'RL',x=.498,bottom=.049,surface='TOP',hip=3.); ev.observe(obs)
    obs=advance(obs); set_leg(obs,'RL',x=.501,bottom=.049,surface='TOP'); snap=ev.observe(obs)
    assert snap['termination_source']=='RR_FIRST_ORDER' and not snap['history']['placed']['RL']


@pytest.mark.parametrize('missing', ['base_pair','force','body_bounds','pose'])
def test_critical_sensor_missing_is_unverified_not_invented_task_failure(missing):
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation()
    if missing=='base_pair': obs['contacts']['base_link']['obstacle']['pair_verified']=False
    elif missing=='force': obs['contacts']['front_left_wheel']['ground'].pop('force_w_n')
    elif missing=='body_bounds': obs['body_bounds_w_m']={}
    else: obs['geometry_pose_aware']=False
    snap=ev.observe(obs)
    assert snap['valid'] is False and snap['run_validity']=='UNVERIFIED'
    assert snap['termination_reason']=='INFRASTRUCTURE_ERROR'
    assert snap['termination_source']=='UNVERIFIED_SENSOR' and not snap['success']


@pytest.mark.parametrize('surface,expected,bearing', [('TOP','TOP',7.),('WALL','FRONT_WALL',0.),
                                                   ('AMBIGUOUS','OBSTACLE_AMBIGUOUS',0.)])
def test_surface_reaction_and_bearing_are_distinct(surface,expected,bearing):
    obs=new_observation(); set_leg(obs,'RR',x=.498,bottom=.049,surface=surface)
    pair=obs['contacts']['rear_right_wheel']['obstacle']
    result=physical_contact_surface(pair,kind='obstacle',obstacle=obs['obstacle'],tolerance_m=.005,force_noise_floor_n=.2)
    assert result['surface']==expected and result['reaction'] and result['reaction_force_n']==7.
    assert result['bearing_force_n']==bearing


def test_ambiguous_force_does_not_encode_verified_lowload():
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation()
    set_leg(obs,'RR',x=.498,bottom=.049,surface='AMBIGUOUS'); snap=ev.observe(obs)
    assert all(v['load_fraction_valid'] is False for v in snap['current_legs'].values())
    assert snap['physical_evidence_status']=='CONTACT_BEARING_UNVERIFIED'


def finished_setup():
    ev=TaskEvaluator(spec=new_spec()); obs=new_observation(); ev.observe(obs)
    for leg in ('FR','FL','RR','RL'): obs=early_top_place(ev,obs,leg)
    obs=advance(obs)
    for leg in LEG_ORDER: set_leg(obs,leg,x=.7+.03*LEG_ORDER.index(leg),bottom=.05,surface='TOP')
    obs['base']['position_w_m']=[.8,0.,.12]
    obs['body_bounds_w_m']['base_link']={'minimum_m':[.7,-.05,.07],'maximum_m':[.9,.05,.17]}
    return ev,obs


def test_task_complete_and_strict_command_quality_are_separate_after_exact_120_ticks():
    ev,obs=finished_setup()
    for wheel in obs['wheels'].values(): wheel['command_rad_s']=.3
    start=ev.observe(obs)
    assert start['traversal_event_observed'] and start['task_completed_controlled'] and not start['success']
    assert not start['strict_recovery_quality']['controlled_stop']
    for i in range(1,121):
        obs=advance(obs); snap=ev.observe(obs)
        assert snap['success'] is (i==120)
        assert snap['post_completion_elapsed_s']==pytest.approx(i/120)
    assert ev.observe(deepcopy(obs))['post_completion_elapsed_s']==1.
    assert snap['strict_recovery_quality']['commanded_wheels_within_tolerance'] is False


def test_brief_speed_quality_excursion_is_not_permanent_post_completion_failure():
    ev,obs=finished_setup(); ev.observe(obs)
    for i in range(1,121):
        obs=advance(obs); obs['base']['linear_velocity_w_m_s']=[.06 if i==10 else 0.,0.,0.]
        snap=ev.observe(obs)
    assert snap['success'] and not snap['post_completion_loss_observed']


@pytest.mark.parametrize('later', ['retreat','body_collision','hard_limit'])
def test_completion_event_cannot_hide_later_real_loss(later):
    ev,obs=finished_setup(); ev.observe(obs)
    for i in range(1,121):
        obs=advance(obs)
        if i==10:
            if later=='retreat': obs['body_bounds_w_m']['base_link']['minimum_m'][0]=.45
            elif later=='body_collision': obs['body_collision']['detected']=True
            else: obs['joints']['front_left_knee']['position_deg']=-60.01
        if i==11 and later=='retreat': obs['body_bounds_w_m']['base_link']['minimum_m'][0]=.7
        snap=ev.observe(obs)
    assert snap['traversal_event_observed'] and not snap['success']
    assert snap['termination_source']=={'retreat':'POST_COMPLETION_LOSS','body_collision':'BODY_CONTACT','hard_limit':'HARD_JOINT_LIMIT'}[later]


def test_base_origin_alone_cannot_prove_body_passed_and_air_above_thin_band_is_allowed():
    ev,obs=finished_setup(); obs['body_bounds_w_m']['base_link']['minimum_m'][0]=.49
    assert not ev.observe(obs)['final_region_valid']
    obs=advance(obs); obs['body_bounds_w_m']['base_link']['minimum_m'][0]=.7
    set_leg(obs,'FR',x=.75,bottom=.09,surface='AIR')
    snap=ev.observe(obs)
    assert snap['final_region_valid'] and snap['task_completed_controlled']


class LocalPointProvider:
    def __init__(self):
        self._body_local_points={body:tuple((x,y,z) for x in (-.20,.20) for y in (-.01,.01) for z in (-.02,.02))
                                 for body in (BASE_BODY,*WHEEL_BODIES)}
        self._collider_paths={body:(body+'/collider',) for body in self._body_local_points}
    def collision_bounds(self,*args,**kwargs):
        raise AssertionError('cached body-local points must avoid repeated USD traversal')


def test_pose_aware_bounds_rotate_without_changing_frozen_cache_or_assets():
    provider=LocalPointProvider(); legacy=ColliderGeometryCache(provider); geometry=SemanticColliderGeometry(legacy)
    poses={body:(0.,0.,.3) for body in provider._body_local_points}
    q={body:(1.,0.,0.,0.) for body in poses}
    first=geometry.sample(poses,q)
    q={body:(math.sqrt(.5),0.,math.sqrt(.5),0.) for body in poses}
    second=geometry.sample(poses,q)
    assert first.body_bounds_w_m[BASE_BODY].minimum_m[2]==pytest.approx(.28)
    assert second.body_bounds_w_m[BASE_BODY].minimum_m[2]==pytest.approx(.10)
    assert not legacy._shapes


def test_semantic_reader_adds_measured_dto_without_second_read_or_frozen_dataclass_change():
    reader=SemanticSensorReader(_fake_adapter(), contact_backend=_FakeContactBackend(),
        geometry_backend=ColliderGeometryCache(LocalPointProvider()))
    raw=reader.read(physics_tick=10,simulation_time_s=1.,commanded_full12=(2.,)*8+(.4,)*4)
    assert isinstance(raw,SemanticPhysicalObservation) and raw.geometry_pose_aware
    assert 'base_link' in raw.body_bounds_w_m
    assert 'body_bounds_w_m' not in {f.name for f in fields(Observation)}


def test_legacy_spec_default_keeps_old_air_gate_and_dto():
    ev=TaskEvaluator(); obs=observation(); snap=ev.observe(obs)
    assert 'evaluator_version' not in snap and 'load_fraction_valid' not in snap['current_legs']['FL']
