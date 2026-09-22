"""Sealed selected windows; observed geometric decomposition, not dynamics/FK success."""
import json
import math
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats
from wlr50_clean.ppo.semantic_height_diagnostics import transform_point
from wlr50_clean.ppo.observation_schema_v2 import _quat_rotate_inverse

OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1'
PROBE=RUN/'diagnostics/FL_hip_minus1_knee_plus1_CP199168_20260921_01'
DET=RUN/'video_eval/validation/20260921T1429210894242Z_g649ccd906421_0c5fc0498bbd4acca26ca3dde5d10193/source'
B=RUN/'video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source'
definitions=json.loads((PROBE/'four_hip_geometry_startup.json').read_text())['mount_definitions']
local=definitions['FL']['value']['local_pos0_m']
assert definitions['FL']['value']['parent_body_name']=='base_link'
assert json.loads((PROBE/'run_manifest.json').read_text())['lifecycle']=='DIAGNOSTIC_SEALED'

def tail(path):
    with path.open('rb') as f:
        f.seek(0,2);end=f.tell();f.seek(max(0,end-131072));data=f.read()
    return json.loads(data.splitlines()[-1])

def selected(path,ticks,hold=False):
    result={};limit=max(t for t in ticks if t!=6555)
    for r in lines(path/'physical_observations.jsonl'):
        t=r['physics_tick']
        if t in ticks or hold and 3353<=t<=3856:result[t]=r
        if t==limit:break
    if 6555 in ticks:
        r=tail(path/'physical_observations.jsonl');assert r['physics_tick']==6555;result[6555]=r
    assert set(ticks)<=result.keys()
    return result

data=dict(B=selected(B,{0,2468,2677,2710}),det=selected(DET,{0,3190,3256,3257,3352,3856,6555},True),
          probe=selected(PROBE,{0,3190,3256,3257,3352,3856,6555},True))
initial_equal={k:data['B'][0][k]==data['probe'][0][k] for k in
    ('base','bodies','joints','wheels','body_bounds_w_m','obstacle','center_of_mass')}
assert all(initial_equal.values())

def mount(r):
    base=r['bodies']['base_link'];return transform_point(local,base['position_w_m'],base['orientation_wxyz'])

def inverse_point(point,body):
    return _quat_rotate_inverse(body['orientation_wxyz'],[a-b for a,b in zip(point,body['position_w_m'])])

initial_child_point=inverse_point(mount(data['probe'][0]),data['probe'][0]['bodies']['front_left_upper'])

def geometry(r):
    m=mount(r);w=r['wheels']['front_left_ankle'];base=r['bodies']['base_link'];bottom=w['bottom_w_m'];top=r['obstacle']['top_z_m']
    delta=[a-b for a,b in zip(bottom,m)]
    child=r['bodies']['front_left_upper'];child_mount=transform_point(initial_child_point,child['position_w_m'],child['orientation_wxyz'])
    return dict(tick=r['physics_tick'],body_collider_min_z_m=r['body_bounds_w_m']['base_link']['minimum_m'][2],
        base_link_z_m=base['position_w_m'][2],FL_mount_world_m=m,
        mount_source='offline authored joint body0/localPos0 + recorded true base_link pose; not direct historical fourhip log',
        FL_wheel_bottom_world_z_m=bottom[2],FL_wheel_center_world_z_m=w['center_w_m'][2],
        FL_bottom_relative_mount_world_z_m=delta[2],FL_bottom_relative_mount_base_m=_quat_rotate_inverse(base['orientation_wxyz'],delta),
        gap_m=bottom[2]-top,joint_connection_crosscheck_error_m=math.dist(m,child_mount),
        actual_joints_deg={n:j['position_deg'] for n,j in r['joints'].items()},
        actual_targets_deg={n:j['command_deg'] for n,j in r['joints'].items()},
        wheel_pair_contacts={leg:dict(contact_class=r['contacts'][name+'_wheel']['contact_class'],
            ground_active=r['contacts'][name+'_wheel']['ground']['active'],obstacle_active=r['contacts'][name+'_wheel']['obstacle']['active'],
            ground_normal_n=r['contacts'][name+'_wheel']['ground']['normal_force_n'],obstacle_normal_n=r['contacts'][name+'_wheel']['obstacle']['normal_force_n'])
            for leg,name in (('FL','front_left'),('FR','front_right'),('RL','rear_left'),('RR','rear_right'))})

geo={name:{t:geometry(r) for t,r in rs.items()} for name,rs in data.items()}
entry=json.loads((PROBE/'probe_entry.json').read_text())
independent=[(3256,entry['four_hip_geometry']['hip_mount_world_m']['FL']['value'])]
for r in lines(PROBE/'probe_decisions.jsonl'):
    if r['record_kind']=='step_result' and r['end_tick'] in (3352,3856):
        independent.append((r['end_tick'],r['four_hip_geometry']['hip_mount_world_m']['FL']['value']))
    if r.get('end_tick',0)==3856:break
mount_check=max(math.dist(geo['probe'][t]['FL_mount_world_m'],m) for t,m in independent)
assert mount_check<1e-10

def decomposition(a,b):
    """State B minus state A; exact geometry identity, not isolated joint causality."""
    ga,gb=geometry(a),geometry(b);qa=a['base']['orientation_wxyz'];qb=b['base']['orientation_wxyz']
    ra=ga['FL_bottom_relative_mount_base_m'];rb=gb['FL_bottom_relative_mount_base_m']
    rot=lambda q,v:transform_point(v,(0.,0.,0.),q)
    local_term=rot(qa,[x-y for x,y in zip(rb,ra)])[2]
    orient_term=rot(qb,rb)[2]-rot(qa,rb)[2]
    mount_term=gb['FL_mount_world_m'][2]-ga['FL_mount_world_m'][2]
    total=gb['gap_m']-ga['gap_m'];top_term=a['obstacle']['top_z_m']-b['obstacle']['top_z_m']
    assert abs(total-(local_term+orient_term+mount_term+top_term))<1e-12
    return dict(gap_delta_m=total,FL_mount_world_z_delta_m=mount_term,
        local_measured_wheel_geometry_at_A_orientation_m=local_term,body_orientation_term_m=orient_term,
        obstacle_top_term_m=top_term,FL_hip_actual_delta_deg=b['joints']['front_left_hip']['position_deg']-a['joints']['front_left_hip']['position_deg'],
        FL_knee_actual_delta_deg=b['joints']['front_left_knee']['position_deg']-a['joints']['front_left_knee']['position_deg'])

parts=[decomposition(data['det'][t],data['probe'][t]) for t in range(3353,3857)]
summary={k:stats(r[k] for r in parts) for k in parts[0]}
examples={name:{str(t):v for t,v in rs.items() if t==0 or not 3353<=t<3856} for name,rs in geo.items()}
for name,path in (('B',B),('det',DET),('probe',PROBE)):
    target_ticks={int(t) for t in examples[name] if t!='0'};limit=max(t for t in target_ticks if t!=6555)
    for r in lines(path/'native_tick_audit.jsonl'):
        t=r['episode_physics_tick']
        if t in target_ticks:
            h=r['native_audit']['policy_headroom_evidence'];examples[name][str(t)]['FL_dispatch']=dict(
                N=r['nominal_full12'][:2],mapped=h['baseline_native_plus_controller_full12'][:2],
                REQUEST=h['requested_policy_residual_full12'][:2],effective=h['effective_policy_residual_full12'][:2],
                source_phase=r['source_phase_id'],clipped_indices=h['clipped_servo_indices'])
        if t==limit:break
    if 6555 in target_ticks:
        r=tail(path/'native_tick_audit.jsonl');assert r['episode_physics_tick']==6555
        h=r['native_audit']['policy_headroom_evidence'];examples[name]['6555']['FL_dispatch']=dict(N=r['nominal_full12'][:2],
            mapped=h['baseline_native_plus_controller_full12'][:2],REQUEST=h['requested_policy_residual_full12'][:2],
            effective=h['effective_policy_residual_full12'][:2],source_phase=r['source_phase_id'],clipped_indices=h['clipped_servo_indices'])
result=dict(schema='CP199168.FL_observed_geometry_decomposition.v1',initial_full_physical_identity=initial_equal,
    authored_FL_mount_definition=definitions['FL'],derived_mount_vs_independent_probe_max_error_m=mount_check,
    joint_connection_error_m={name:stats(g['joint_connection_crosscheck_error_m'] for g in rs.values()) for name,rs in geo.items()},
    B_FL_mount_direct_record=None,B_FL_mount_reconstruction_caveat='Uses common authored FL frame. Initial13-body pose/joint/wheel/collider data identical, plus rigid joint connection crosscheck; B did not directly log FL mount.',
    selected_event_states=examples,hold_same_elapsed_3353_3856_decomposition=summary,
    hold_geometry={name:{key:stats(g[key] if key!='FL_mount_world_z_m' else g['FL_mount_world_m'][2]
        for t,g in geo[name].items() if 3353<=t<=3856) for key in ('gap_m','body_collider_min_z_m','FL_mount_world_z_m','FL_bottom_relative_mount_world_z_m')}
        for name in ('det','probe')},
    B_capture_to_det_terminal=decomposition(data['B'][2677],data['det'][6555]),
    B_capture_to_probe_hold_end=decomposition(data['B'][2677],data['probe'][3856]),
    exact_instantaneous_FL_hip_knee_vertical_jacobian=None,
    FK_absence='No logged FL knee joint pivot/axis/Jacobian; measured relative geometry change is not an isolated fixed-body hip/knee derivative. No inverse-pivot fit or dynamics inferred.',
    limitations='Same elapsed probe/control after identical entry still different closed-loop states. Local term includes all measured wheel geometry and cannot be uniquely assigned to selected hip/knee. B comparison is event-state geometry, not same-state counterfactual. No physical run, no PPO/AUX credit.')
with (OUT/'CP199168_FL_geometry_decomposition.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print('CHECK',mount_check,result['joint_connection_error_m'])
print('HOLD',json.dumps(summary));print('HEIGHT',json.dumps(result['hold_geometry']))
print('BCAP_TO_DET',json.dumps(result['B_capture_to_det_terminal']))
for name,rs in examples.items():
    for t,g in rs.items():
        if t=='0':continue
        print(name,t,json.dumps({k:g[k] for k in ('body_collider_min_z_m','FL_mount_world_m','FL_bottom_relative_mount_world_z_m','gap_m','actual_joints_deg','FL_dispatch','wheel_pair_contacts')}))

if __name__=='__main__':pass
