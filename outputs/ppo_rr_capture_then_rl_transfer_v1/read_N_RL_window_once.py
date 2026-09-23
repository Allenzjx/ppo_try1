"""One bounded pass of specified sealed N physics, no runtime/model imports."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N = ROOT / 'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source'
TICKS = {6152, 6155, 6160, 6168, 6176, 6248, 6656, 6728}
LABELS = {6152:'before_RR_capture',6155:'RR_cross_and_placed_event',6160:'P10_entry',6168:'P11_entry',6176:'P12_entry',6248:'P12_early',6656:'P12_late',6728:'P13_entry'}
BASE = HERE / 'diagnose_57ae41e_sealed_fixed_com.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == 'b521aabc6ddc208ab65d2357aed97a8b2e3cadd42e2b6433e5129992a781ade1'
spec = importlib.util.spec_from_file_location('_sealed_summary', BASE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fr_base(raw):
    center = raw['wheels']['front_right_ankle']['center_w_m']
    origin = raw['base']['position_w_m']; w,x,y,z = raw['base']['orientation_wxyz']
    x,y,z=-x,-y,-z
    vx,vy,vz=[center[i]-origin[i] for i in range(3)]
    tx,ty,tz=2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)
    return [vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx]


manifest = json.loads((N/'semantic_video_source_manifest.json').read_text(encoding='utf-8'))
assert manifest['physical_task_success'] is True
path=N/'physical_observations.jsonl'; before=path.stat(); selected=[]; digest=hashlib.sha256(); scanned=0
with path.open('rb') as stream:
    for line in stream:
        scanned += 1; digest.update(line); raw=json.loads(line)
        tick=raw['physics_tick']
        if tick in TICKS:
            p=m.physical_summary(raw)
            p.update(label=LABELS[tick],FR_wheel_relative_base_frame_m=fr_base(raw),
                wheel_native_equivalent_target_from_canonical_axis_sign=[v*s for v,s in zip(raw['commanded_full12'][8:],[-1,1,-1,1])],
                wheel_native_equivalent_actual_from_canonical_axis_sign=[v*s for v,s in zip(raw['actual_full12'][8:],[-1,1,-1,1])],
                native_last_setter_not_in_selected_stream=None,
                original_source_nominal_and_touched_owners_not_in_selected_stream=None)
            selected.append(p)
        if tick>=max(TICKS): break
after=path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
assert {row['post_tick'] for row in selected}==TICKS
cdata=HERE/'video_v5_c53119a_capture_follow_review/v5_capture_follow_fixed_COM.json'
cblob=cdata.read_bytes(); assert hashlib.sha256(cblob).hexdigest()=='386b2bfa5f00f51b9fb5a5d1f5f86b8a4c8fe3ad6c760a9471c8dfa2e644463d'
c=json.loads(cblob)['milestones'][-1]
result=dict(schema='readonly.accepted_N_RL_actual_window.v1',source=str(N),new_physics=0,new_PPO=0,
    not_same_HEAD_or_paired=True,physics_prefix_rows_read=scanned,physics_prefix_sha256=digest.hexdigest(),
    physics_full_file_sha256_from_manifest=(manifest.get('artifacts',{}).get(path.name) or {}).get('sha256'),
    N_success=manifest['physical_task_success'],N_samples=selected,C_v5_terminal=c,
    limits=['Single physics-stream pass through tick6728; no Recording scan, no actor/runtime imports.',
      'Native equivalent vectors are canonical sign conversion, NOT separately observed target buffers.',
      'Original source nominal/touched owners/lastsetter absent in this stream; no invention from command differences.',
      'Current raw contacts and force are separate from historical placed; angles do not prove causality.',
      'FR relative position uses measured base origin/quaternion, not CoM.'])
out=HERE/'N_P10_P12_actual_vs_v5_RR_precontact.json'
with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
for p in selected:
    joint=p['joint_actual_command']; contact=p['current_raw_contact_pairs']
    print(json.dumps(dict(tick=p['post_tick'],label=p['label'],joints=joint,rollpitch=p['body_roll_pitch_deg_from_recorded_unit_wxyz'],
      mass_COM=p['mass_COM']['position_w_m'],FR_relative=p['FR_wheel_relative_base_frame_m'],
      wheel_target=p['commanded_full12'][8:],wheel_actual=p['actual_full12_canonical_deg_rad_s'][8:],
      contacts={leg:dict(kind=r.get('contact_class'),ground_active=(r.get('ground') or {}).get('active'),
      ground_N=(r.get('ground') or {}).get('normal_force_n'),obstacle_active=(r.get('obstacle') or {}).get('active'),
      obstacle_N=(r.get('obstacle') or {}).get('normal_force_n')) for leg,r in contact.items()})))
print(json.dumps(dict(output=str(out),rows=len(selected),python_helper_finished=True)))
