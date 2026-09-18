"""Bounded source-home request replay on sealed v1 measurements, no physics."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER

OUT=Path(__file__).resolve().parent
AGG=OUT/'zero_refine_3d23189_run034324/zero_review_aggregate.json'
data=json.loads(AGG.read_text(encoding='utf8'))
entry=data['first_active_stop_owner']['owner']['entry']
start=tuple(entry['issued_nominal_full12'])
home_tick=data['source_home_entry']['entry_observation_tick']
contract=load_motion_contract(ROOT/'configs/recording_motion_contract.json')
base_spec=yaml.safe_load((ROOT/'configs/ppo_non_residual_refine_v1/stage_task_spec.yaml').read_text(encoding='utf8'))


def model(version):
    spec=copy.deepcopy(base_spec)
    spec['nominal']['final_stop_owner']=f'source_home_after_physical_stop_{version}'
    p=NominalMotionProvider.from_handoff(contract,spec=spec,stage_id='P13',
        nominal_full12=start,tracking_servo_names=())
    p._final_stop_owner=copy.deepcopy(entry)
    return p


def request(p,tick,eligible=True):
    p._final_stop_diagnostic={'current_entry_eligibility':eligible,'current_observation_tick':tick}
    output,tracking,bias=p._final_stop_request()
    p.nominal_full12=output
    return output,tracking,bias


def main():
    v1,v2=model('v1'),model('v2')
    physical={r['tick']:r for r in data['last_two_seconds_physical']}
    native={r['tick']:r for r in data['last_two_seconds_native']}
    previous=[start,start]; rows=[]; maximum=[0.,0.]; v1_error=0.
    for tick in range(home_tick,data['native_tick_count']):
        r=physical[tick]
        eligible=r['body_speed_m_s']<=.05 and r['body_angular_speed_rad_s']<=.30 and max(map(abs,r['actual_full12'][8:]))<=.25
        outputs=[]
        for i,p in enumerate((v1,v2)):
            result,tracking,bias=request(p,tick,eligible)
            assert result[8:]==(0.,)*4 and tracking==() and bias==(0.,)*12
            maximum[i]=max(maximum[i],max(abs(a-b) for a,b in zip(result[:8],previous[i][:8])))
            previous[i]=result; outputs.append(result)
        expected=native[tick+1]['nominal_full12']
        v1_error=max(v1_error,max(abs(a-b) for a,b in zip(outputs[0],expected)))
        rows.append({'observation_tick':tick,'dispatch_tick':tick+1,
            'recorded_v1_measured_controlled':eligible,'v1_nominal':outputs[0],'v2_nominal':outputs[1],
            'v2_fraction':v2._final_stop_diagnostic['home_recovery']['nominal_ramp_fraction']})
    assert v1_error==0.
    assert rows[0]['v2_nominal']==start and v2._final_home_recovery['start_servo_deg']==start[:8]
    assert rows[60]['v2_nominal']==rows[0]['v1_nominal']
    # Repeat/non-adjacent observations affect neither entry nor elapsed ramp:
    # this invokes the request seam, not a fictitious full duplicate tick.
    jump=model('v2'); request(jump,home_tick)
    jumped=request(jump,home_tick+30)[0]
    repeated=request(jump,home_tick+30)[0]
    after_gap=request(jump,home_tick+90)[0]
    assert jumped==repeated==rows[30]['v2_nominal']
    assert after_gap==rows[90]['v2_nominal']
    assert jump._final_home_recovery['entry_observation_tick']==home_tick
    # Explicit synthetic mapper continuity check. Do not claim its seeded
    # internal state is an exact recorded mapper checkpoint.
    mapper=ServoTargetMapper({name:0. for name in SERVO_ORDER})
    mapper._requested={name:value for name,value in zip(SERVO_ORDER,start[:8])}
    mapper._applied=dict(mapper._requested)
    mapper._feedback_tick=500
    mapper_identity=id(mapper); applied_before=tuple(mapper._applied.values())
    max_mapper_delta=0.; prev=applied_before
    for row in rows:
        mapped=mapper.advance(row['v2_nominal'][:8],(0.,)*8,tracking_servo_names=())
        assert id(mapper)==mapper_identity
        max_mapper_delta=max(max_mapper_delta,max(abs(a-b) for a,b in zip(mapped.applied_drive_command_deg,prev)))
        prev=mapped.applied_drive_command_deg
    assert mapper.feedback_tick==500+len(rows)
    assert max_mapper_delta<=mapper.maximum_delta_deg+1e-12
    report={'schema':'wlr50_clean.terminal_v2_independent_request_review.v1',
        'source_aggregate':str(AGG),'source_aggregate_sha256':hashlib.sha256(AGG.read_bytes()).hexdigest(),
        'reviewed_supervisor_sha256':hashlib.sha256((ROOT/'src/wlr50_clean/ppo/semantic_supervisor.py').read_bytes()).hexdigest(),
        'source_home_entry_observation_tick':home_tick,'start_nominal':start,
        'source_home_target':rows[0]['v1_nominal'],'input_rows':len(rows),
        'recorded_v1_native_request_max_error_deg':v1_error,
        'v2_first_request_equal_previous_nominal':True,'ramp_duration_s':v2._final_home_recovery['ramp_duration_s'],
        'v2_target_reached_observation_tick':home_tick+60,'v2_target_first_dispatch_tick':home_tick+61,
        'maximum_requested_servo_delta_deg_per_tick':{'v1':maximum[0],'v2':maximum[1]},
        'continuous_quintic_velocity_upper_bound_deg_s':1.875*max(abs(a-b) for a,b in zip(start[:8],rows[0]['v1_nominal'][:8]))/.5,
        'duplicate_request_same_tick_idempotent':True,'skipped_ticks_use_elapsed_physical_time':True,
        'home_entry_not_restarted_on_recorded_control_loss':True,
        'all_wheel_requests_zero_no_legacy_pulse':True,'all_home_tracking_and_normal_bias_retired':True,
        'synthetic_mapper_continuity':{'same_object':True,'feedback_tick_500_to':mapper.feedback_tick,
            'maximum_applied_delta_deg':max_mapper_delta,'physical_mapper_bound_deg_per_tick':mapper.maximum_delta_deg,
            'exact_recorded_mapper_state_claim':False},
        'rows':rows,'physics_steps':0,'policy_forwards':0,'optimizer_updates':0,
        'limitations':['Real recorded v1 observations are not the closed-loop states that v2 will produce.',
            'Direct request-seam duplicate checks do not bypass production controller monotonic-time rejection.',
            'Existing mapper resets changed-target compensation/nominal-reached as usual; applied and feedback history are not rebuilt.',
            'Pre-home no-change is covered by source routing and independent tests, not inferred from this home-only replay.',
            'No claim that a 0.5s ramp leaves enough physical settling time or fixes the RR wheel oscillation.']}
    with (OUT/'terminal_v2_independent_review.json').open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
