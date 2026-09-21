"""Saved stochastic requests only; no forward, optimizer, simulator or GPU."""
from pathlib import Path
import json
import math
from statistics import fmean

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIOR=json.loads((OUT/'CP182528_stochastic_handoff.json').read_text())
SOURCE=Path(PRIOR['source'])
TRAIN=ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0833241304449Z_g3a50657a96c9_f4abda62b32041808fdc243bdac4e573'
def readrows(p):
    with p.open() as f:
        for line in f:yield json.loads(line)
def channel(q,i):
    c=q['current_cap_full12'][i];b=q['base_mean_full12'][i];mu=q['conditional_mean_full12'][i]
    raw=q['selected_raw_full12'][i];sd=q['effective_sigma_full12'][i];center=q['history_center_full12'][i]
    return dict(base_mean=b,conditional_mean=mu,raw_sample=raw,effective_sigma=sd,learned_sigma=q['learned_sigma_full12'][i],
        cap_deg=c,history_center=center,previous_raw=q['previous_raw_from_current_observation_full12'][i],
        cap_entry_gate=q['cap_transition_gate_full12'][i],base_weighted_contribution=.1*b,history_weighted_contribution=.9*center,
        raw_innovation=raw-mu,innovation_standard_scores=(raw-mu)/sd,
        tanh_sample=math.tanh(raw),mean_unfiltered_deg=c*math.tanh(mu),sample_unfiltered_deg=c*math.tanh(raw),
        same_center_unfiltered_innovation_deg=c*(math.tanh(raw)-math.tanh(mu)),
        local_physical_sigma_proxy_deg=c*(1-math.tanh(mu)**2)*sd,
        minus_one_sigma_displacement_deg=c*(math.tanh(mu-sd)-math.tanh(mu)),
        plus_one_sigma_displacement_deg=c*(math.tanh(mu+sd)-math.tanh(mu)),
        raw_tanh_saturated_abs_ge_point95=abs(math.tanh(raw))>=.95)
def stats(xs):
    return dict(count=len(xs),min=min(xs),max=max(xs),mean=fmean(xs),rms=math.sqrt(fmean(x*x for x in xs)))
ds=list(readrows(SOURCE/'video_policy_decisions.jsonl'))
training=[]
for r in readrows(TRAIN/'residual_and_projection_audit.jsonl'):
    training.append(r)
    if r['terminal']:break
assert len(training)==661 and training[-1]['applied_audit']['physics_tick']==5283
assert json.loads((TRAIN/'run_manifest.json').read_text())['lifecycle']=='SUCCEEDED'
window_stats={}
selected=[]
for label,records in [('CP182528_eval',ds),('block11_episode0',training)]:
    phase=lambda r:r['request_phase'] if label=='CP182528_eval' else r['applied_audit']['phase_id']
    tick=lambda r:r['end_tick'] if label=='CP182528_eval' else r['applied_audit']['physics_tick']
    p05=[r for r in records if phase(r)=='P05'][-8:]
    p06=[r for r in records if phase(r)=='P06'][:8]
    for name,rs in [('late_P05_last8',p05),('first_P06_first8',p06)]:
        summary={'decision_end_ticks':[tick(r) for r in rs],'scope':'same phase/cap window, not guaranteed motionless pose or fixed nominal'}
        for ch,i in [('FR_knee',3),('FL_hip',0)]:
            cs=[channel(r['policy_request'],i) for r in rs]
            summary[ch]={k:stats([c[k] for c in cs]) for k in ('base_mean','conditional_mean','raw_sample','effective_sigma','local_physical_sigma_proxy_deg','same_center_unfiltered_innovation_deg')}
            summary[ch]['cap_deg']=cs[0]['cap_deg'];summary[ch]['saturated_count']=sum(c['raw_tanh_saturated_abs_ge_point95'] for c in cs)
        window_stats[label+'_'+name]=summary

for s in PRIOR['representative_physical_rows']:
    if s['tick'] not in (2768,2776,2848,5167,5394,6381,6600,6699):continue
    r=next(r for r in ds if r['start_tick']<s['tick']<=r['end_tick'])
    cs={}
    for name,i in [('FR_knee',3),('FL_hip',0)]:
        cs[name]={**channel(r['policy_request'],i),**{k:s['channels'][name][k] for k in ('nominal','mapped_N','REQUEST','effective','final','actual')},
            'actual_timing':'physical post-step at selected tick','actuator_semantics':'canonical servo degrees'}
    selected.append(dict(dataset='CP182528_stochastic_eval',tick=s['tick'],phase=s['request_phase'],decision_start_tick=r['start_tick'],
        channels=cs,context={k:s[k] for k in ('body_z_m','body_linear_speed_m_s','body_angular_speed_rad_s','contacts')},
        physical_dispatch_verified=s['dispatch_verified']))
for r in training:
    a=r['applied_audit'];tick=a['physics_tick']
    if tick not in (2848,2856,2864,2880,4960,5200,5283):continue
    q=r['policy_request'];n=a['actuator_target_effect_audit'];h=n['policy_headroom_evidence'];tr=n['tracking_reference_evidence'];e=a['semantic_task']['physical_evaluator'];g=a['semantic_task']['goal_features']
    cs={}
    for name,i in [('FR_knee',3),('FL_hip',0)]:
        cs[name]={**channel(q,i),'nominal':a['nominal_action_full12'][i],'mapped_N':h['baseline_native_plus_controller_full12'][i],
            'REQUEST':a['projected_residual_full12'][i],'effective':h['effective_policy_residual_full12'][i],
            'final':a['actual_drive_target_full12'][i],
            'actual':tr['channels'][i]['nominal_deg']-tr['channels'][i]['current_actual_canonical_error_deg'],
            'actual_native_rad':tr['actual_measured_physical_rad'][i],
            'actual_timing':'pre last dispatch; canonical=logged nominal minus logged canonical error; post-step q unavailable',
            'actuator_semantics':'canonical servo degrees, with original native rad retained'}
    selected.append(dict(dataset='block11_episode0',tick=tick,global_policy_decision=r['global_policy_decision'],phase=a['phase_id'],end_phase=a['end_phase_id'],
        channels=cs,context=dict(body_z_m=None,body_linear_speed_m_s=g['body_linear_speed_m_s'],body_angular_speed_rad_s=g['body_angular_speed_rad_s'],
            contacts={l:{k:e['current_legs'][l][k] for k in ('air','support','top_contact','ground_contact','bearing_force_n','clearance_m')} for l in ('FL','FR','RL','RR')},
            termination_reason=a['termination_reason']),
        physical_dispatch_verified=n['verified'] and n['actual_mapping_matches_dispatch'] and n['setter_dispatch_targets_equal']))

result=dict(schema='wlr50_clean.saved_physical_innovation_scale.v1',source_eval=str(SOURCE),source_train=str(TRAIN),
    eval_fixed_checkpoint=PRIOR['checkpoint'],train_scope='First661-row sealed episode only; weights evolve at completed128-decision updates, not fixed CP counterfactual.',
    definition='C*tanh(raw_sample)-C*tanh(saved_conditional_mean), same saved center/cap; UNFILTERED innovation comparison only, not final-drive counterfactual.',
    sigma_proxy_definition='C*(1-tanh(mu)^2)*saved effective_sigma: local linearized physical std proxy, not measured std.',
    windows=window_stats,selected=selected,
    exact_cap_multipliers_for_identical_raw_kernel={'FR_knee_P05_to_P06':112/24,'FL_hip_P05_to_P06':32/18},
    limits=['No new actor forward, optimizer, simulator, CUDA or production change.',
        'Negative conditional means can predominantly reflect accumulated HISTORY rather than the current base head or current innovation.',
        'Filtered REQUEST/final may have different signs from the current unfiltered sample due to rate-limit/history.',
        'Body/support coupling is observed, not a noise-only causal experiment.',
        'No inference about active block12. No frequency claims or global audit.'])
with (OUT/'physical_innovation_scale_diagnosis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for k,w in window_stats.items():
    print(k,w['decision_end_ticks'])
    for c in ('FR_knee','FL_hip'):
        z=w[c];print(c,'sigma mean',z['effective_sigma']['mean'],'local physical sigma mean',z['local_physical_sigma_proxy_deg']['mean'],'innovation RMS',z['same_center_unfiltered_innovation_deg']['rms'],'range',z['same_center_unfiltered_innovation_deg']['min'],z['same_center_unfiltered_innovation_deg']['max'],'sat',z['saturated_count'])
for s in selected:
    print(s['dataset'],s['tick'],s['phase'],'bodyz',s['context']['body_z_m'])
    for n,c in s['channels'].items():print(n,*(c[k] for k in ('base_mean','history_center','conditional_mean','raw_sample','effective_sigma','cap_deg','mean_unfiltered_deg','same_center_unfiltered_innovation_deg','REQUEST','final','actual','tanh_sample')))
