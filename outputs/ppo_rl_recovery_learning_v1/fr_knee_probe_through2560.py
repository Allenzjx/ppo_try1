"""Stdlib-only bounded live-log analysis; never waits for the writer."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
F = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1206270979619Z_g59e868f3e223_6daab84a9021442ca9d5a14aea51c95e/source/video_policy_decisions.jsonl'
P = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/CP230144_FRk_entry_minus4_v1/diagnostic_decisions.jsonl'
TICKS = (2096, 2216, 2336, 2456, 2560)

def bounded(path):
    result = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            tick = row['step_info']['physics_tick']
            if tick > 2560:
                break
            result.append(row)
            if tick == 2560:
                break
    return result

def compact(row):
    s = row['step_info']; t = s['semantic_task']; e = t['physical_evaluator']
    a = s['actuator_target_effect_audit']; roles = e['transfer_roles']
    margin = roles['RL']['receiver_workspace_state']['joint_range_margin_deg']
    # Diagnostic joint positions reconstructed from the logged current margin
    # to the original hard lower bound; no FK or target substitution.
    qh = margin['front_right_hip']['negative_deg'] - 135.
    qk = margin['front_right_knee']['negative_deg'] - 60.
    return dict(tick=s['physics_tick'], time_s=s['sim_time_s'], phase=t['stage_id'], age_s=t['stage_age_s'],
        FR_knee_source_deg=s['nominal_action_full12'][3], FR_knee_mappedN_deg=a['native_drive_target_full12'][3],
        FR_knee_filtered_REQUEST_deg=s['projected_residual_full12'][3],
        FR_knee_FINAL_deg=s['actual_drive_target_full12'][3], FR_knee_actual_deg= qk,
        FR_hip_FINAL_deg=s['actual_drive_target_full12'][2], FR_hip_actual_deg=qh,
        FR_hip_actual_minus_FINAL_deg=qh-s['actual_drive_target_full12'][2],
        FL_gap_mm=e['current_legs']['FL']['clearance_m']*1000,
        FL_front_mm=e['current_legs']['FL']['front_distance_m']*1000,
        COM_world_mm=[v*1000 for v in roles['RL']['transfer_direction_context']['mass_weighted_com_position_w_m']],
        FR_load_N=e['current_legs']['FR']['bearing_force_n'], RL_load_N=e['current_legs']['RL']['bearing_force_n'],
        FR_TOP=e['current_legs']['FR']['top_contact'], RL_ground=e['current_legs']['RL']['ground_contact'],
        probe_mode=row.get('candidate',{}).get('mode'), desired_REQUEST_deg=row.get('candidate',{}).get('desired_REQUEST_deg'),
        modified_channels=row.get('overridden_indices'), start_tick=row.get('decision_start_tick'),
        last_request_audit_tick=a['physics_tick'])

formal, probe = bounded(F), bounded(P)
fa = {r['step_info']['physics_tick']:r for r in formal}; pr={r['step_info']['physics_tick']:r for r in probe}
assert all(t in fa and t in pr for t in TICKS)
intervened = [r for r in probe if r['overridden_indices']]
assert len(intervened) == 45
assert all(r['overridden_indices'] == [3] and all(r['original_student_raw_full12'][i] == r['applied_raw_full12'][i] for i in range(12) if i != 3) for r in intervened)
window=[r for r in probe if 2096 <= r['step_info']['physics_tick'] <= 2560]
support=[]
for r in window:
    e=r['step_info']['semantic_task']['physical_evaluator']; legs=e['current_legs']
    support.append(dict(tick=r['step_info']['physics_tick'],FR_top=legs['FR']['top_contact'],
                        FR_bearing=legs['FR']['bearing_verified'],FR_force=legs['FR']['bearing_force_n'],
                        RL_bearing=legs['RL']['bearing_verified'],RL_force=legs['RL']['bearing_force_n']))
pairs = [dict(formal=compact(fa[t]), probe=compact(pr[t])) for t in TICKS]
payload=dict(schema='outputs.FR_knee_diagnostic_through2560.v1', formal_path=str(F), probe_path=str(P),
    read_limit_tick=2560, rows=pairs, modified_decisions=len(intervened), all_other11_original_raw_exact=True,
    modified_start_ticks=[intervened[0]['decision_start_tick'],intervened[-1]['decision_start_tick']],
    probe_state_at2560=pr[2560]['probe_state_after_step'],
    support_endpoint_count=len(support), support_min_FR_N=min(r['FR_force'] for r in support),
    support_min_RL_N=min(r['RL_force'] for r in support),
    all_sampled_FR_TOP_bearing=all(r['FR_top'] and r['FR_bearing'] for r in support),
    all_sampled_RL_bearing=all(r['RL_bearing'] for r in support),
    caveat='This extractor compares decision endpoints only; it does not infer live physical-file contents from directory Length/stat. A separately recorded open/seek/tell snapshot check verified all465 actual120Hz support observations2096..2560 in the published report. Actual FR joints reconstructed from current joint-margin evidence, not target. Intervention raw does not equal no-intervention baseline; no physical success/PPO/AUX claim.')
dest=OUT/'CP230144_FRk_probe_through2560.json'
md=OUT/'CP230144_FRk_probe_through2560.md'
for p in (dest,md):
    if p.exists(): raise FileExistsError(p)
dest.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
def pair(key,precision=3):
    return f"{row['formal'][key]:+.{precision}f} / {row['probe'][key]:+.{precision}f}"
lines=['# CP230144 FR-knee direction diagnostic through tick2560', '',
       '**Every cell is formal baseline / intervened probe.** Angles degrees, position mm, force N. Decision endpoints only; no claim of physical success or AUX/PPO data.', '',
       '| Tick / time s | FRk N; mapped N | FRk filtered REQUEST | FRk FINAL; actual | FRh FINAL; actual−FINAL | FL gap; front | CoM z | FR; RL force |',
       '| --- | --- | --- | --- | --- | --- | --- | --- |']
for row in pairs:
    row['formal']['COM_z']=row['formal']['COM_world_mm'][2]; row['probe']['COM_z']=row['probe']['COM_world_mm'][2]
    lines.append('| '+ ' | '.join([f"{row['formal']['tick']} / {row['formal']['time_s']:.4f}",
        pair('FR_knee_source_deg')+'; '+pair('FR_knee_mappedN_deg'), pair('FR_knee_filtered_REQUEST_deg'),
        pair('FR_knee_FINAL_deg')+'; '+pair('FR_knee_actual_deg'),
        pair('FR_hip_FINAL_deg')+'; '+pair('FR_hip_actual_minus_FINAL_deg'),
        pair('FL_gap_mm')+'; '+pair('FL_front_mm'), pair('COM_z'),pair('FR_load_N')+'; '+pair('RL_load_N')])+' |')
lines.extend(['',payload['caveat'],''])
md.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(payload,indent=2))
