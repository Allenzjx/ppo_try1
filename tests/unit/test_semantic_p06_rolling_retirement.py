"""Measured P06-only prior retirement; no Isaac, fabricated lift, or task gates."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import observation, leg_state
from test_semantic_observation_reward_env import _frame, _raw
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, NominalMotionProvider, SemanticControllerAdapter,
    SemanticObservationError, TaskEvaluator, TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/ppo_semantic_v3'
SPEC=CFG/'stage_task_spec.yaml'
TASK_SPEC=load_task_spec(SPEC)
ZERO=(0.,)*12


@pytest.fixture(scope='module')
def contract():
    return load_motion_contract(ROOT/'configs/recording_motion_contract.json')


@pytest.fixture
def old_finite_tail_spec():
    """Explicit pre-tail-revision fixture; retain the measured retirement rule."""
    spec=load_task_spec(SPEC)
    spec['nominal'].pop('p06_wheel_tail_semantics',None)
    assert spec['nominal']['p06_rolling_retirement']=='measured_workspace_interior_peak'
    return spec


def measured(stage='P06', *, tick=0, rl=-.2175, rr=None, lateral=True):
    obs=observation(tick)
    for joint in obs['joints'].values(): joint['command_deg']=0.
    for leg,x in (('RL',rl),('RR',rl if rr is None else rr)):
        leg_state(obs,leg,x=obs['obstacle']['front_x_m']+x)
    if not lateral:
        obs['wheels']['rear_left_ankle']['center_w_m'][1]=2.
        obs['wheels']['rear_left_ankle']['bottom_w_m'][1]=2.
    ev=TaskEvaluator(spec=TASK_SPEC).observe(obs)
    assert ev['valid'] and ev['termination_reason'] is None
    return {'stage_id':stage,'termination_reason':None,'physical_evaluator':ev},obs


def provider(contract, *, enabled=True, spec=None):
    spec=load_task_spec(SPEC) if spec is None else deepcopy(spec)
    if not enabled:
        # The new tail depends on retirement; disabling retirement selects the
        # old finite-source contract, not an invalid half-enabled new mode.
        spec['nominal'].pop('p06_wheel_tail_semantics',None)
        spec['nominal'].pop('p06_rolling_retirement')
    return NominalMotionProvider(contract,spec=spec)


def diag(p):
    return p.nominal_suggestion_diagnostics['p06_rolling_retirement']


@pytest.mark.parametrize('distance,gain',[
    (-.30,1.),(-.221,1.),(-.220,1.),(-.219,.8),(-.2175,.5),(-.215,0.),(-.20,0.),
])
def test_true_measured_workspace_blend_only_inside_existing_boundary(contract,distance,gain):
    p=provider(contract)
    task,obs=measured(rl=distance)
    sup=TaskStageSupervisor(SPEC)
    goal_before=sup.predicate('rear_approach',task['physical_evaluator'])
    for tick in range(20):
        task,obs=measured(tick=tick,rl=distance)
        command=p.evaluate(task,obs)
    assert diag(p)['wheel_gain']==pytest.approx(gain)
    assert command[8:]==pytest.approx((.3*gain,)*4)
    assert sup.predicate('rear_approach',task['physical_evaluator'])==goal_before
    assert not any(task['physical_evaluator']['history']['active_lift'].values())
    assert not task['physical_evaluator']['success']


@pytest.mark.parametrize('boundary',[-.22,-.215])
def test_gain_continuous_at_both_existing_geometry_boundaries(contract,boundary):
    gains=[]
    for x in (boundary-1e-9,boundary,boundary+1e-9):
        p=provider(contract);task,obs=measured(rl=x);p.evaluate(task,obs)
        gains.append(diag(p)['wheel_gain'])
    assert max(gains)-min(gains)<5e-7


def test_lagging_rear_lateral_validity_peak_retention_and_fresh_reset(contract):
    p=provider(contract)
    task,obs=measured(rl=-.3,rr=-.2);p.evaluate(task,obs)
    assert diag(p)['wheel_gain']==1.
    task,obs=measured(tick=1,rl=-.2175,lateral=False);p.evaluate(task,obs)
    assert diag(p)['wheel_gain']==1. and not diag(p)['lateral_valid']
    task,obs=measured(tick=2,rl=-.2175);p.evaluate(task,obs)
    assert diag(p)['peak_fraction']==pytest.approx(.5)
    task,obs=measured('P07',tick=3,rl=-.215);p.evaluate(task,obs)
    assert diag(p)['wheel_gain']==pytest.approx(0.)
    task,obs=measured('P09',tick=4,rl=-.35);p.evaluate(task,obs)
    assert diag(p)['measured_fraction']==0. and diag(p)['peak_fraction']==pytest.approx(1.)
    assert diag(p)['origin']=='current_live_P06_layer'
    fresh=provider(contract);fresh.evaluate(*measured(rl=-.35))
    assert diag(fresh)['wheel_gain']==1.


@pytest.mark.parametrize('invalid',[
    'missing_evaluator','invalid_evaluator','missing_leg','missing_distance','nan','inf','bool_distance',
    'missing_lateral','integer_lateral','missing_tick','wrong_time','wrong_observation_tick',
])
def test_bad_inputs_fail_before_any_source_or_nominal_mutation(contract,invalid):
    p=provider(contract);p.evaluate(*measured(rl=-.3))
    task,obs=measured(tick=1)
    ev=task['physical_evaluator']
    if invalid=='missing_evaluator': task.pop('physical_evaluator')
    elif invalid=='invalid_evaluator': ev['valid']=False
    elif invalid=='missing_leg': ev['current_legs'].pop('RR')
    elif invalid=='missing_distance': ev['current_legs']['RR'].pop('front_distance_m')
    elif invalid in ('nan','inf','bool_distance'):
        ev['current_legs']['RR']['front_distance_m']={'nan':float('nan'),'inf':float('inf'),'bool_distance':True}[invalid]
    elif invalid=='missing_lateral': ev['current_legs']['RL'].pop('within_lateral_span')
    elif invalid=='integer_lateral': ev['current_legs']['RL']['within_lateral_span']=1
    elif invalid=='missing_tick': ev.pop('physics_tick')
    elif invalid=='wrong_time': ev['simulation_time_s']+=.1
    elif invalid=='wrong_observation_tick': obs['physics_tick']+=1
    state=(p.nominal_full12,p.state_id,p._source_motion._tick_index,deepcopy(diag(p)),
           [(layer['ticks'],layer.get('rolling_retirement_peak')) for layer in p._continuous_layers])
    with pytest.raises(SemanticObservationError): p.evaluate(task,obs)
    assert state==(p.nominal_full12,p.state_id,p._source_motion._tick_index,diag(p),
                   [(layer['ticks'],layer.get('rolling_retirement_peak')) for layer in p._continuous_layers])


def test_missing_initial_geometry_does_not_even_create_p06_layer(contract):
    p=provider(contract)
    with pytest.raises(SemanticObservationError): p.evaluate({'stage_id':'P06'})
    assert not p._continuous_layers and p.state_id is None
    assert p._source_motion._tick_index==0


def test_terminal_sample_does_not_earn_retirement_or_replace_terminal_path(contract):
    p=provider(contract);p.evaluate(*measured(rl=-.3))
    terminal={'stage_id':'P09','termination_reason':'FALL','physical_evaluator':{'valid':False}}
    p.evaluate(terminal)
    assert diag(p)['status']=='terminal_no_new_retirement'
    assert diag(p)['peak_fraction']==0. and diag(p)['measured_fraction'] is None
    assert terminal['termination_reason']=='FALL'


def test_real_early_layers_keep_servo_tracking_new_wheel_owners_and_slew(contract):
    p,old=provider(contract),provider(contract,enabled=False)
    tick=0; saw_reverse=False; saw_new_rolling=False
    for phase,count in (('P06',20),('P07',8),('P08',8),('P09',280)):
        held=p.nominal_full12
        for local in range(count):
            task,obs=measured(phase,tick=tick,rl=-.3 if phase=='P06' else -.21)
            previous=p.nominal_full12
            command=p.evaluate(task,obs); reference=old.evaluate(task,obs)
            if phase!='P06' and local==0: assert command==held
            assert command[:8]==reference[:8]
            assert p.tracking_servo_names==old.tracking_servo_names
            assert p.endpoint_issued==old.endpoint_issued
            assert all(abs(a-b)<=.025+1e-12 for a,b in zip(command[8:],previous[8:]))
            if phase=='P09' and 90<=local<=110:
                assert command[9]==pytest.approx(-.63)
                assert [command[i] for i in (8,10,11)]==pytest.approx([0.]*3)
                saw_reverse=True
            if phase=='P09' and local==279:
                assert command[8:]==pytest.approx((.3,)*4)
                saw_new_rolling=True
            tick+=1
    assert saw_reverse and saw_new_rolling
    for layer,source in zip(p._continuous_layers,old._continuous_layers):
        assert layer['ticks']==source['ticks'] and layer['last']==source['last']
        assert layer['touched']==source['touched'] and layer['sample']==source['sample']


def test_existing_qualified_lift_carry_override_survives_fully_retired_p06(contract):
    from test_semantic_continuous_v3 import live,whole_body_lift
    p=provider(contract);p.evaluate(*measured(rl=-.21))
    ev=TaskEvaluator(SPEC);obs=live()
    for leg in ('RR','RL'): leg_state(obs,leg,x=.3)
    ev.observe(obs);obs=whole_body_lift(ev,obs,top_clearance=.08)
    assert ev.snapshot['history']['active_lift']['RR']
    for tick in range(1,18):
        obs=deepcopy(obs);obs['physics_tick']+=1;obs['simulation_time_s']=obs['physics_tick']/120.
        task={'stage_id':'P09','termination_reason':None,'physical_evaluator':ev.observe(obs)}
        command=p.evaluate(task,obs)
    assert diag(p)['wheel_gain']==pytest.approx(0.)
    assert command[8:]==pytest.approx((.3,)*4)


def test_no_p06_layer_handoff_keeps_actual_teacher_wheels_without_fabricated_peak(contract):
    incoming=ZERO[:8]+(.13,)*4
    p=NominalMotionProvider.from_handoff(contract,spec=load_task_spec(SPEC),stage_id='P09',
        nominal_full12=incoming,tracking_servo_names=())
    command=p.evaluate(*measured('P09',rl=-.21))
    assert command[8:]==incoming[8:]
    assert not diag(p)['layer_present'] and diag(p)['peak_fraction'] is None
    assert [layer['stage'] for layer in p._continuous_layers]==['P09']


def test_p06_handoff_creates_only_current_measured_layer_and_preserves_slew(contract):
    incoming=ZERO[:8]+(.3,)*4
    p=NominalMotionProvider.from_handoff(contract,spec=load_task_spec(SPEC),stage_id='P06',
        nominal_full12=incoming,tracking_servo_names=())
    assert not diag(p)['layer_present'] and diag(p)['peak_fraction'] is None
    task,obs=measured(tick=80,rl=-.215)
    command=p.evaluate(task,obs)
    assert [layer['stage'] for layer in p._continuous_layers]==['P06']
    assert diag(p)['peak_fraction']==pytest.approx(1.)
    assert diag(p)['source_observation_tick']==80
    assert command[8:]==pytest.approx((.275,)*4)  # Existing physical slew, no hard reset.


def test_phase_labels_alone_cannot_retire_an_existing_p06_layer(contract):
    p=provider(contract)
    for tick,phase in enumerate(('P06','P07','P08','P09','P10','P11','P12','P13')):
        p.evaluate(*measured(phase,tick=tick,rl=-.3))
        assert diag(p)['measured_fraction']==0. and diag(p)['peak_fraction']==0.


def test_old_config_behavior_and_finite_p06_zero_tail_preserved(contract,old_finite_tail_spec):
    old=provider(contract,enabled=False,spec=old_finite_tail_spec)
    absent=deepcopy(old_finite_tail_spec);absent['nominal']['p06_rolling_retirement']=None
    compatibility=NominalMotionProvider(contract,spec=absent)
    legacy=NominalMotionProvider(contract,spec=load_task_spec(DEFAULT_TASK_SPEC_PATH))
    assert not legacy.nominal_suggestion_diagnostics
    assert not old.nominal_suggestion_diagnostics
    p=provider(contract,spec=old_finite_tail_spec)
    for tick in range(3090):
        task,obs=measured(tick=tick,rl=-.3)
        assert old.evaluate(task,obs)==compatibility.evaluate(task,obs)
        command=p.evaluate(task,obs)
    assert command[8:]==pytest.approx((0.,)*4)
    assert p.endpoint_issued and diag(p)['wheel_gain']==1.


@pytest.mark.parametrize('mode,width',[
    ('unknown',.005),('measured_workspace_interior_peak',0.),
    ('measured_workspace_interior_peak',.28),('measured_workspace_interior_peak',float('nan')),
])
def test_explicit_revision_requires_existing_valid_geometric_band(tmp_path,mode,width):
    spec=load_task_spec(SPEC);spec['nominal']['p06_rolling_retirement']=mode
    spec['geometry']['xy_measurement_tolerance_m']=width
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(spec),encoding='utf-8')
    with pytest.raises(ValueError): load_task_spec(path)


def test_retired_nominal_does_not_disable_real_residual_native_effect(contract):
    from test_actuator_target_effect import _adapter
    from test_semantic_residual_adapter import plan,dispatch
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    from wlr50_clean.ppo.actuator_target_effect import actuator_target_audit_request,build_actuator_target_effect_audit
    p=provider(contract);nominal=p.evaluate(*measured(rl=-.21))
    assert nominal[8:]==pytest.approx((0.,)*4)
    raw=ZERO[:8]+(.2,)*4
    projection=build_semantic_projector(CFG/'execution_profile.yaml').project(raw,state_id='P06',
        nominal_action_full12=nominal,reference_action_full12=nominal,reference_delta_full12=ZERO)
    adapter=_adapter();previous=tuple(adapter._final_drive_servo_deg.values())
    actuation=plan(projection.safe_projected_residual_full12,nominal=nominal)
    ack=dispatch(adapter,actuation,1)
    audit=build_actuator_target_effect_audit(adapter=adapter,actuation=actuation,raw_ack=ack,
        previous_final_drive_servo_deg=previous,source_phase_id='P06',
        policy_request=actuator_target_audit_request('P06',raw,(1,)*12))
    assert audit['verified'] and all(audit['changed_channels_full12'][8:])
    assert max(abs(x) for x in audit['actual_native_targets']['wheel_velocity_rad_s'])>0.


def test_controller_diagnostics_are_copied_current_suggestions_not_history_or_actor_groups(contract):
    from wlr50_clean.ppo.semantic_observation import HISTORY_GROUPS,SemanticObservationBuilder,load_semantic_observation_schema
    supervisor=TaskStageSupervisor(SPEC,initial_stage_id='P06')
    controller=SemanticControllerAdapter(load_fsm_spec(ROOT/'configs/fsm_states.yaml'),contract,supervisor=supervisor)
    raw=_raw(joint=0.)
    frame=controller.step(raw)
    task=controller.task_snapshot
    diagnostics=task['nominal_provider_diagnostics']
    assert frame.drive_feedback_details['semantic_task']['nominal_provider_diagnostics']==diagnostics
    assert diagnostics['p06_rolling_retirement']['source_observation_tick']==raw.physics_tick
    assert 'not_applied_target' in diagnostics['p06_rolling_retirement']['timing']
    assert 'nominal_provider_diagnostics' not in supervisor.snapshot
    diagnostics['p06_rolling_retirement']['wheel_gain']=123.
    assert controller.task_snapshot['nominal_provider_diagnostics']['p06_rolling_retirement']['wheel_gain']!=123.
    # A real completed-prefix-free task for encoding, without spoofing P06 history.
    sup=TaskStageSupervisor(SPEC);real=sup.observe_and_update(raw)
    shell=_frame(raw=raw);shell.state_id=real['stage_id'];shell.info['semantic_task']=real
    schema=load_semantic_observation_schema(CFG/'observation_schema.json')
    history=dict.fromkeys(HISTORY_GROUPS,ZERO)
    before=SemanticObservationBuilder(schema).build(shell,history)
    shell.info['semantic_task']={**real,'nominal_provider_diagnostics':controller.nominal_provider.nominal_suggestion_diagnostics}
    after=SemanticObservationBuilder(schema).build(shell,history)
    assert schema.dimension==324
    assert schema.encode(before.groups)==schema.encode(after.groups)
