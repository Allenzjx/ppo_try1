"""Source-request/mapper and reset-prefix regressions, not live task success."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math

import pytest
import yaml

from test_all_stage_nominal_transitions import task
from test_semantic_prefix import receipt
from wlr50_clean.fsm.controller import ControllerFrame
from wlr50_clean.fsm.motion_executor import FeedbackCorrection, MotionExecutor
from wlr50_clean.fsm.state_spec import Lifecycle, load_fsm_spec
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, SERVO_COMMAND_SIGN
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.semantic_prefix import ResetOnlyPrefixController, PrefixRequest
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
ZERO = (0.,)*12


@pytest.fixture(scope="module")
def resources():
    return (load_fsm_spec(ROOT/'configs/fsm_states.yaml'),
            load_motion_contract(ROOT/'configs/recording_motion_contract.json'))


def spec(enabled=True):
    value = yaml.safe_load((ROOT/'configs/ppo_all_stage_acceptance_v1/stage_task_spec.yaml').read_text())
    if enabled:
        value['reference_nominal_semantics'] = 'successful_fsm_derived_v2'
    return value


def provider(resources, phase, enabled=True):
    fsm, contract = resources
    return NominalMotionProvider.from_handoff(contract,spec=spec(enabled),stage_id=phase,
        nominal_full12=contract.phase(phase).start_full12,tracking_servo_names=(),fsm_spec=fsm)


def evaluate(p, phase, tick, **kwargs):
    return p.evaluate(task(phase,tick,**kwargs),{'physics_tick':tick,'simulation_time_s':tick/120.})


def test_p06_same_source_prestate_emits_wheels_and_retires_old_tracking_on_causal_tick(resources):
    fsm, contract = resources
    # Trial043 tick3576: source state is phase-aligned for this controlled seam
    # test. These are measured canonical joint angles, not a target/posture gate.
    actual = (25.756556141804033,-12.317211898051788,.3422801242860385,45.81663451757912,
              7.073662535458029,.2981095026994509,-1.5380440086552887,1.090629212061577)
    previous = tuple(contract.phase('P06').start_full12)
    initial_native = (21.55,-12.15,-.7996284021635888,45.347052073602775,5.65,
                      .6421599783343184,-1.3499030061922759,.9543498893273243)
    native = ServoTargetMapper({name:0. for name in SERVO_ORDER})
    native._requested = dict(zip(SERVO_ORDER,previous[:8]))
    native._applied = dict(zip(SERVO_ORDER,initial_native))
    native._compensation = {name:a-b for name,a,b in zip(SERVO_ORDER,initial_native,previous[:8])}
    native._tracking_active[SERVO_ORDER[0]] = True
    native._feedback_tick = 3756
    actual_q = tuple(math.radians(SERVO_COMMAND_SIGN[name]*angle) for name,angle in zip(SERVO_ORDER,actual))
    source = MotionExecutor(initial_full12=previous)
    source.start_phase(contract.phase('P06'))
    source_frame = source.tick()
    source_mapping = deepcopy(native).advance(source_frame.full12[:8],actual_q,
                                              tracking_servo_names=source_frame.tracking_servo_names)
    results = {}
    for enabled in (False,True):
        p = provider(resources,'P05',enabled)
        p.nominal_full12=previous; p.tracking_servo_names=(SERVO_ORDER[0],)
        result=evaluate(p,'P06',3576,placed=('FR',))
        mapped=deepcopy(native).advance(result[:8],actual_q,tracking_servo_names=p.tracking_servo_names)
        results[enabled]=(result,mapped.applied_drive_command_deg,p.tracking_servo_names)
    assert results[True][0] == source_frame.full12
    assert results[True][1] == pytest.approx(source_mapping.applied_drive_command_deg,abs=1e-12)
    assert results[True][2] == source_frame.tracking_servo_names == ()
    assert results[False][0][8:] == (0.,)*4
    assert results[False][1][0] == pytest.approx(20.3)
    assert results[True][1][0] == pytest.approx(21.55)


def test_p07_held_source_request_has_one_mapper_not_nominal_60_degree_ramp(resources):
    _, contract = resources
    p = provider(resources,'P06')
    p.nominal_full12=tuple(contract.phase('P07').start_full12)
    result=evaluate(p,'P07',6648,placed=('FR','FL'))
    source=MotionExecutor(initial_full12=p.nominal_full12)
    source.start_phase(contract.phase('P07')); first=source.tick()
    assert result[0] == first.full12[0] == 37.6
    assert p.tracking_servo_names == first.tracking_servo_names == ('front_left_hip',)
    assert evaluate(p,'P07',6649,placed=('FR','FL'))[0] == 37.6
    # No residual API exists in the provider; actuator slew still resides in
    # the exact frozen mapper (max1.25degrees per120Hz tick).
    m=ServoTargetMapper({name:0. for name in SERVO_ORDER})
    before=m.advance((0.,)*8,(0.,)*8)
    after=m.advance(result[:8],(0.,)*8,tracking_servo_names=p.tracking_servo_names)
    assert max(abs(a-b) for a,b in zip(before.applied_drive_command_deg,after.applied_drive_command_deg)) <= 1.25


def test_unfinished_predecessor_remains_concurrent_and_untouched_values_not_restored(resources):
    p=provider(resources,'P07')
    values=list(p.nominal_full12); values[2]=3.2; values[11]=-.25; p.nominal_full12=tuple(values)
    first=evaluate(p,'P07',0)
    second=evaluate(p,'P08',1)
    assert len(p._continuous_layers)==2
    assert set(p.tracking_servo_names) >= {'front_left_hip','rear_left_hip'}
    assert second[0] == first[0]
    assert second[2] == 3.2 and second[11] == -.25
    assert p._continuous_layers[0]['ticks']==2


def test_completed_predecessor_does_not_keep_tracking_in_new_phase(resources):
    _,contract=resources
    p=provider(resources,'P07')
    end=round(contract.phase('P07').active_duration_s*120)
    for tick in range(end+1):evaluate(p,'P07',tick)
    assert p._continuous_layers[0]['sample'].endpoint_issued
    evaluate(p,'P08',end+1)
    assert 'front_left_hip' not in p.tracking_servo_names
    assert 'rear_left_hip' in p.tracking_servo_names


def test_source_p03_logical_correction_is_reused_not_a_policy_cap(resources):
    fsm,contract=resources
    p=provider(resources,'P03')
    source=MotionExecutor(initial_full12=contract.phase('P03').start_full12)
    s=fsm.state('P03')
    source.start_phase(contract.phase('P03'),FeedbackCorrection(s.normal_correction_fractions),time_scale=s.normal_time_scale)
    for tick in range(round(contract.phase('P03').active_duration_s*120)+2):
        actual=evaluate(p,'P03',tick)
        assert actual == source.tick().full12
    assert p.normal_drive_bias_full12 == ZERO
    assert s.normal_correction_fractions[10] == -.149


def test_source_p10_time_scale_and_finite_post_mapper_bias_window(resources):
    fsm,contract=resources
    p=provider(resources,'P10')
    state=fsm.state('P10'); phase=contract.phase('P10')
    expected=tuple(.5*a*b for a,b in zip(phase.delta_full12,state.normal_correction_fractions))
    endpoint=round(round(phase.active_duration_s*120)*state.normal_time_scale)
    biases=[]
    for tick in range(endpoint+3):
        evaluate(p,'P10',tick)
        biases.append(p.normal_drive_bias_full12)
    assert endpoint==14
    assert expected[7] == pytest.approx(.75)
    assert biases[:endpoint+1] == [expected]*(endpoint+1)
    assert biases[endpoint+1:] == [ZERO,ZERO]
    assert p._continuous_layers[0]['motion']._time_scale == state.normal_time_scale == .875


def test_equal_inputs_and_history_share_one_N_independent_of_policy_role(resources):
    a=provider(resources,'P07'); b=provider(resources,'P07')
    for tick in range(32):
        stage='P07' if tick<8 else 'P08' if tick<16 else 'P09'
        assert evaluate(a,stage,tick)==evaluate(b,stage,tick)
        assert a.tracking_servo_names==b.tracking_servo_names
        assert a.normal_drive_bias_full12==b.normal_drive_bias_full12
    assert not a.nominal_suggestion_diagnostics['successful_fsm_nominal']['residual_dependent_controller_selection']
    assert not a.nominal_suggestion_diagnostics['successful_fsm_nominal']['dormant_reference_rebound_trigger_enabled']


@pytest.mark.parametrize('mode,teacher_bias,last_bias,enabled,expected',[
    ('READY',0.,0.,True,.75),('READY',0.,0.,False,0.),
    ('TAKEOVER',3.,3.,True,1.75),('TAKEOVER',.5,0.,True,.75),
])
def test_prefix_ready_preserves_N_bias_and_takeover_does_not_add_two_biases(resources,mode,teacher_bias,last_bias,enabled,expected):
    fsm,contract=resources
    source_bias=(0.,)*7+(.75,)+(0.,)*4
    old_bias=(0.,)*7+(teacher_bias,)+(0.,)*4
    last=(0.,)*7+(last_bias,)+(0.,)*4
    frame=ControllerFrame(8,8/120,'P10',Lifecycle.EXECUTE_MOTION,ZERO,True,True,False,(),
                          ZERO,source_bias,{'mode':'test_source_N'},False,None,None,())
    supervisor=SimpleNamespace(spec=spec(enabled),evaluator=None)
    teacher=SimpleNamespace(motion=None)
    wrapper=ResetOnlyPrefixController(teacher,supervisor,fsm,contract,PrefixRequest('P10'))
    wrapper.physics_tick=8; wrapper.mode=mode; wrapper._handoff_tick=0
    wrapper._bias=old_bias; wrapper._receipt=receipt(7,nominal=ZERO,bias=last)
    wrapper._semantic=SimpleNamespace(step=lambda observation,sim_time_s:frame)
    result=wrapper.step({'physics_tick':8},sim_time_s=8/120)
    assert result.normal_drive_bias_full12[7] == pytest.approx(expected)
    assert result.drive_feedback_bias_full12 == ZERO
    assert result.full12 == ZERO
    assert wrapper.physics_tick == 9
