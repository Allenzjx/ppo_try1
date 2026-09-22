"""Bounded, sealed-source RR geometry; no simulator, model, fitting or learning."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'outputs/ppo_task_conditioned_hip_wheel_v1'))
from rr_probe_readonly import stats
from wlr50_clean.ppo.semantic_height_diagnostics import transform_point
from wlr50_clean.ppo.observation_schema_v2 import _quat_rotate_inverse
from wlr50_clean.ppo.semantic_observation import _quaternion, _rpy
from wlr50_clean.infrastructure.command_batch import servo_limits_deg

OUT = Path(__file__).resolve().parent
FAIL = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1228540892300Z_g5fd88852bf20_e26806a2d8b941b382160f94e1729cfc/source'
TRAIN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
NREF = ROOT / 'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source'
LEGS = dict(FL='front_left', FR='front_right', RL='rear_left', RR='rear_right')
proofs = []


def bounded(path, lo, hi, tick):
    result = []; digest = hashlib.sha256(); count = 0
    with path.open('rb') as f:
        for line in f:
            assert line.endswith(b'\n')
            r = json.loads(line); t = tick(r)
            if t > hi:
                break
            if lo <= t <= hi:
                result.append(r); digest.update(line); count += 1
    proofs.append(dict(path=str(path), window=[lo, hi], selected_complete_lines=count,
                       selected_lines_sha256=digest.hexdigest(), hash_scope='only selected original complete lines, not whole file'))
    return result


def definition(run):
    return json.loads((run / 'height_diagnostics_startup.json').read_text())['rr_hip_mount_definition']['value']


definition_fail, definition_n = definition(FAIL), definition(NREF)
assert definition_fail == definition_n
local = definition_fail['local_pos0_m']
assert definition_fail['parent_body_name'] == 'base_link'
rot = lambda q, v: transform_point(v, (0., 0., 0.), q)


def physical(r):
    b = r['bodies']['base_link']; w = r['wheels']['rear_right_ankle']
    m = transform_point(local, b['position_w_m'], b['orientation_wxyz'])
    c, bottom = w['center_w_m'], w['bottom_w_m']
    rel = _quat_rotate_inverse(b['orientation_wxyz'], [x-y for x, y in zip(c, m)])
    return dict(tick=r['physics_tick'], t_s=r['simulation_time_s'],
        gap_mm=1000*(bottom[2]-r['obstacle']['top_z_m']), RR_mount_z_mm=1000*m[2],
        RR_wheel_bottom_z_mm=1000*bottom[2], RR_center_minus_mount_base_mm=[1000*x for x in rel],
        RR_center_relative_base_m=_quat_rotate_inverse(b['orientation_wxyz'], [x-y for x,y in zip(c,b['position_w_m'])]),
        RR_bottom_minus_mount_world_z_mm=1000*(bottom[2]-m[2]),
        wheel_bottom_minus_center_z_mm=1000*(bottom[2]-c[2]), base_origin_z_mm=1000*b['position_w_m'][2],
        body_collider_min_z_mm=1000*r['body_bounds_w_m']['base_link']['minimum_m'][2],
        base_quat_wxyz=b['orientation_wxyz'], rpy_deg=[math.degrees(x) for x in _rpy(_quaternion(b['orientation_wxyz']))],
        obstacle_top_z_mm=1000*r['obstacle']['top_z_m'],
        actual_joints_deg={k:v['position_deg'] for k,v in r['joints'].items()},
        joint_targets_deg={k:v['command_deg'] for k,v in r['joints'].items()},
        contacts={leg:dict(contact_class=r['contacts'][name+'_wheel']['contact_class'],
            ground=r['contacts'][name+'_wheel']['ground']['active'], obstacle=r['contacts'][name+'_wheel']['obstacle']['active'],
            ground_force_n=r['contacts'][name+'_wheel']['ground']['normal_force_n'],
            obstacle_force_n=r['contacts'][name+'_wheel']['obstacle']['normal_force_n']) for leg,name in LEGS.items()})


def decomposition(a, b):
    qa, qb = a['base_quat_wxyz'], b['base_quat_wxyz']
    ra, rb = a['RR_center_minus_mount_base_mm'], b['RR_center_minus_mount_base_mm']
    mount = b['RR_mount_z_mm']-a['RR_mount_z_mm']
    translation = b['base_origin_z_mm']-a['base_origin_z_mm']
    local_term = rot(qa, [x-y for x,y in zip(rb,ra)])[2]
    orientation = rot(qb,rb)[2]-rot(qa,rb)[2]
    envelope = b['wheel_bottom_minus_center_z_mm']-a['wheel_bottom_minus_center_z_mm']
    top = a['obstacle_top_z_mm']-b['obstacle_top_z_mm']
    total = b['gap_mm']-a['gap_mm']
    closure = total-(mount+local_term+orientation+envelope+top)
    assert abs(closure) < 1e-10
    return dict(gap_delta_mm=total, mount_world_z_delta_mm=mount,
        mount_subterm_base_origin_translation_mm=translation,
        mount_subterm_body_orientation_mm=mount-translation,
        local_center_geometry_at_A_orientation_mm=local_term,
        body_orientation_of_relative_center_mm=orientation,
        collider_bottom_envelope_mm=envelope, obstacle_top_change_mm=top, closure_error_mm=closure)


def decision(a):
    p = a['semantic_task']['physical_evaluator']; h=a['actuator_target_effect_audit']['policy_headroom_evidence']
    joints={}; rel=None
    for role in p['transfer_roles'].values():
        ws=role['receiver_workspace_state']
        for name,m in ws['joint_range_margin_deg'].items():
            lo,hi=servo_limits_deg(name); q=m['negative_deg']+lo
            assert abs(q-(hi-m['positive_deg'])) < 1e-9
            joints[name]=q
        if role['diagonal_receiving_side']=='RR':rel=ws['wheel_relative_body_m']
    return dict(tick=a['physics_tick'], phase=a['end_phase_id'],
        gap_mm=1000*p['current_legs']['RR']['clearance_m'],
        body_collider_min_z_mm=1000*p['body_traversal_geometry']['minimum_w_m'][2],
        RR_center_relative_base_m=rel, actual_joints_deg=joints,
        joint_source='exact inversion of recorded actual-position hard-limit margins; not command readback',
        RR_mount_z_mm=None, base_quat_wxyz=None, RR_wheel_bottom_world_z_mm=None,
        missing_geometry_reason='training compact audit has no absolute base pose/quaternion or RR joint mount; no fabricated decomposition',
        contacts={leg:{k:v.get(k) for k in ('contact_mode','contact_surface','air','ground_contact','top_contact',
            'top_surface_contact','support','bearing_verified','bearing_force_n','current_lift_valid','within_top_xy')} for leg,v in p['current_legs'].items()},
        rr_dispatch=dict(nominal=a['nominal_action_full12'][6:8],mapped=h['baseline_native_plus_controller_full12'][6:8],
            REQUEST=h['requested_policy_residual_full12'][6:8],effective=h['effective_policy_residual_full12'][6:8],
            final=a['actual_drive_target_full12'][6:8], clipped=h['clipped_servo_indices']),
        original_terminal=a.get('termination_reason'), history_RR_placed=a['semantic_task']['history']['placed']['RR'])


def summarize(rows):
    return dict(rows=len(rows), ticks=[rows[0]['tick'],rows[-1]['tick']],
        **{k:stats(r[k] for r in rows) for k in ('gap_mm','body_collider_min_z_mm')},
        actual_joints_deg={k:stats(r['actual_joints_deg'][k] for r in rows) for k in rows[0]['actual_joints_deg']},
        RR_center_relative_base_z_mm=stats(1000*r['RR_center_relative_base_m'][2] for r in rows),
        contacts={leg:dict(states=dict(Counter((r['contacts'][leg].get('contact_mode') or r['contacts'][leg].get('contact_surface') or r['contacts'][leg].get('contact_class')) for r in rows)),
            verified_support_samples=sum(bool(r['contacts'][leg].get('support') and r['contacts'][leg].get('bearing_verified')) for r in rows)
            if 'support' in rows[0]['contacts'][leg] else None) for leg in LEGS})


raw_fail=bounded(FAIL/'physical_observations.jsonl',7200,9780,lambda r:r['physics_tick'])
raw_n=bounded(NREF/'physical_observations.jsonl',5915,6235,lambda r:r['physics_tick'])
f=[physical(r) for r in raw_fail]; n=[physical(r) for r in raw_n]
assert len(f)==2581 and len(n)==321
ng=next(r for r in n if r['tick']==6155)
raw_train=bounded(TRAIN/'residual_and_projection_audit.jsonl',8488,8808,lambda r:r['applied_audit']['physics_tick'])
t=[decision(r['applied_audit']) for r in raw_train]
assert len(t)==41
fd=[decision(r['step_info']) for r in bounded(FAIL/'video_policy_decisions.jsonl',7200,9780,lambda r:r['end_tick'])]
checks=[]
for run, rows,lo,hi in ((FAIL,f,7200,9780),(NREF,n,5915,6235)):
    bytick={r['tick']:r for r in rows}
    for r in bounded(run/'height_diagnostics.jsonl',lo,hi,lambda r:r['physics_tick']):
        if r['physics_tick'] in bytick:
            checks.append(abs(1000*r['rr_hip_mount_w_m']['value'][2]-bytick[r['physics_tick']]['RR_mount_z_mm']))
assert checks and max(checks)<1e-9
parts=[decomposition(ng,r) for r in f]
training_pre=[r for r in t if r['tick']<=8720];training_top=[r for r in t if 8736<=r['tick']<=8808]
assert len(training_pre)==30 and len(training_top)==10
result=dict(schema='RR_observed_height_decomposition_readonly.v1',
    role_separation={'failed':'CP213376_AUXMEAN3 deterministic P09 incomplete, stable window only',
        'learned_capture':'block03 stochastic training episode0, actual RR capture not full task success',
        'N_reference':'older accepted N video, geometric comparator only; NOT PPO demonstration or same-version control'},
    source_line_proofs=proofs, authored_RR_mount=definition_fail,
    reconstructed_vs_direct_RR_mount_max_error_mm=max(checks),
    failed_summary={**summarize(f),**{k:stats(r[k] for r in f) for k in ('RR_mount_z_mm','RR_bottom_minus_mount_world_z_mm','base_origin_z_mm')},
        'rpy_deg':{k:stats(r['rpy_deg'][i] for r in f) for i,k in enumerate(('roll','pitch','yaw'))}},
    failed_current_evaluator_contact_summary=summarize(fd)['contacts'],
    learned_pre_capture_summary=summarize(training_pre), learned_TOP_8736_8808_summary=summarize(training_top),
    learned_endpoint_states={str(r['tick']):r for r in t if r['tick'] in (8488,8720,8728,8736,8808)},
    failed_endpoint_states={str(r['tick']):r for r in f if r['tick'] in (7200,8400,9600,9780)},
    N_reference_endpoint_states={str(r['tick']):r for r in n if r['tick'] in (5915,6155,6160,6235)},
    N_capture_to_failed_window_observed_decomposition_mm={k:stats(r[k] for r in parts) for k in parts[0]},
    N_last2s_to_capture_observed_decomposition_mm=decomposition(n[0],ng),
    failed_60s_to_81_5s_observed_decomposition_mm=decomposition(f[0],f[-1]),
    learned_capture_world_mount_orientation_decomposition=None,
    instantaneous_RR_joint_vertical_jacobian=None,
    limits='No logged knee joint pivot/axis/Jacobian used. Local term is measured wheel-center configuration, not unique hip/knee cause. Decomposition order is explicit A-orientation, not an intervention. Training samples are 15Hz post-step states; input_ticks in availability report refer to the same state used by NEXT action, not this row current residual. N event is separate older nominal reference. Missing original world pose is not replaced by collider height, CoM or guessed base_z. No simulator, training, fit or auxiliary data admission.')
path=OUT/'RR_capture_height_geometry_readonly.json'
with path.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
print('OUTPUT',path)
print('FAILED',json.dumps(result['failed_summary']))
print('N_CAPTURE',json.dumps(ng))
print('DECOMPOSITION',json.dumps(result['N_capture_to_failed_window_observed_decomposition_mm']))
for r in t:
    if r['tick'] in (8488,8720,8736,8808):print('TRAIN',json.dumps(r))
print('CURRENT_CONTACTS',json.dumps(result['failed_current_evaluator_contact_summary']))
