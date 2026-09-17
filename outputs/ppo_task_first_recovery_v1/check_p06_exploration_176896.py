"""Bounded stored-rollout decomposition; no actor/critic/GAE recomputation."""
import json
import math
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
RUN=ROOT/'runs/ppo_task_first_recovery_v1/train/20260916T0447433367326Z_gb0438f66ec63_606bd33f006d4115bf9a8ed2ea6c23d9'
LABELS=['FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel']


def main():
    import torch
    from wlr50_clean.ppo.semantic_training import write_json
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, SERVO_COMMAND_SIGN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--rollout-name',default='rollout_001347.pt')
    parser.add_argument('--temperature',type=float,default=.5)
    parser.add_argument('--output-name',default='p06_176896_exploration_decomposition_with_actual.json')
    args=parser.parse_args()
    run=args.run.resolve(strict=True)
    torch.set_num_threads(1)
    rollout=torch.load(run/'rollouts'/args.rollout_name,map_location='cpu',weights_only=False)
    obs=rollout['observations']['policy'].reshape(-1,372).double()
    raw=rollout['actions'].reshape(-1,12).double()
    mu,sigma=[x.reshape(-1,12).double() for x in rollout['distribution_params']]
    history=obs[:,195:207]
    carry=.9*history
    base_contribution=mu-carry
    base=base_contribution/.1
    innovation=raw-mu
    z=innovation/sigma
    records=[json.loads(line) for line in (run/'residual_and_projection_audit.jsonl').read_text().splitlines()]
    assert len(records)==len(raw)==128
    terminal=[i for i,r in enumerate(records) if r['terminal']]
    starts=[0]+[i+1 for i in terminal if i+1<len(raw)]
    ends=starts[1:]+[len(raw)]
    first_episode_end=ends[0]
    parity=max(max(abs(float(a)-b) for a,b in zip(raw[i],r['raw_policy_action_full12'])) for i,r in enumerate(records))
    history_error=max(float((history[i]-raw[i-1]).abs().max()) for i in range(1,128) if i not in starts)
    def stats(x):
        return {'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),
                'rms':float(x.square().mean().sqrt()),'absmax':float(x.abs().max())}
    def snapshot(i):
        r=records[i]; a=r['applied_audit']; ev=a['semantic_task']['physical_evaluator']
        goals=ev['goal_features']
        tracking=a['actuator_target_effect_audit']['tracking_reference_evidence']
        actual_servo=[(math.degrees(q)-standing)/SERVO_COMMAND_SIGN[name]
            for name,q,standing in zip(SERVO_ORDER,tracking['actual_measured_physical_rad'],tracking['standing_pose_deg'])]
        return {'row':i,'global_decision':r['global_policy_decision'],'tick':a['physics_tick'],
            'episode_local_decision':i-max(s for s in starts if s<=i)+1,'terminal':r['terminal'],
            'reason':a['termination_reason'],'raw':raw[i].tolist(),
            'base_mean_inferred_from_stored_conditional':base[i].tolist(),
            'base_contribution':base_contribution[i].tolist(),'history_carry':carry[i].tolist(),
            'conditional_mean':mu[i].tolist(),'effective_sigma':sigma[i].tolist(),
            'innovation':innovation[i].tolist(),'innovation_standard_z':z[i].tolist(),
            'tanh_raw':raw[i].tanh().tolist(),'projected_residual':a['projected_residual_full12'],
            'N':a['nominal_action_full12'],'final_target':a['actual_drive_target_full12'],
            'actual_servo_before_last_dispatch_deg':actual_servo,
            'actual_servo_sample_episode_tick':a['physics_tick']-1,
            'actual_wheel_rad_s':ev.get('measured_wheel_velocity_rad_s'),
            'body_bounds':ev.get('body_traversal_geometry'),
            'body_linear_speed_m_s':goals['body_linear_speed_m_s'],
            'body_angular_speed_rad_s':goals['body_angular_speed_rad_s'],
            'supports':{leg:{key:value.get(key) for key in ('support','air','contact_surface',
                'bearing_force_n','load_fraction','clearance_m','front_distance_m')}
                for leg,value in ev['current_legs'].items()},
            'support_count':goals['support_count']}
    first_sat=next((i for i in range(first_episode_end) if bool((raw[i].abs()>3).any())),None)
    first_less4=next((i for i,r in enumerate(records[:first_episode_end]) if r['applied_audit']['semantic_task']['physical_evaluator']['goal_features']['support_count']<4),None)
    first_less3=next((i for i,r in enumerate(records[:first_episode_end]) if r['applied_audit']['semantic_task']['physical_evaluator']['goal_features']['support_count']<3),None)
    first_less2=next((i for i,r in enumerate(records[:first_episode_end]) if r['applied_audit']['semantic_task']['physical_evaluator']['goal_features']['support_count']<2),None)
    chosen={0,1,2,3,4,9,19,39,59,69,74,75,76,77,78,79,80,95,111,127}
    for values in (raw, sigma, innovation):
        chosen.update(int(values[:,j].abs().argmax()) for j in range(12))
    for i in (first_sat,first_less4,first_less3,first_less2):
        if i is not None:chosen.update(range(max(0,i-2),min(first_episode_end,i+3)))
    episodes={}
    for episode_index,(start,end) in enumerate(zip(starts,ends)):
        label=(('failed_episode' if episode_index==0 else f'failed_episode_{episode_index}')
               if records[end-1]['terminal'] else 'nonterminal_tail')
        channel={}
        for j,name in enumerate(LABELS):
            channel[name]={'raw':stats(raw[start:end,j]),'conditional_mean':stats(mu[start:end,j]),
                'base_mean':stats(base[start:end,j]),'base_contribution':stats(base_contribution[start:end,j]),
                'history_carry':stats(carry[start:end,j]),'effective_sigma':stats(sigma[start:end,j]),
                'innovation':stats(innovation[start:end,j]),'z':stats(z[start:end,j]),
                'raw_abs_gt3_count':int((raw[start:end,j].abs()>3).sum()),
                'mu_abs_gt3_count':int((mu[start:end,j].abs()>3).sum()),
                'abs_tanh_raw_gt095_count':int((raw[start:end,j].tanh().abs()>.95).sum())}
        wheels=raw[start:end,8:]; shared=wheels.mean(-1,keepdim=True)
        support_counts=[r['applied_audit']['semantic_task']['physical_evaluator']['goal_features']['support_count'] for r in records[start:end]]
        episodes[label]={'count':end-start,'channels':channel,'start_row':start,'end_row_exclusive':end,
            'terminal':bool(records[end-1]['terminal']),'support_count_histogram':{str(k):support_counts.count(k) for k in sorted(set(support_counts))},
            'wheel_raw_common_component':stats(shared),
            'wheel_raw_differential_component':stats(wheels-shared),
            'decisions_any_raw_abs_gt3':int((raw[start:end].abs()>3).any(-1).sum()),
            'decisions_any_servo_abs_tanh_gt095':int((raw[start:end,:8].tanh().abs()>.95).any(-1).sum()),
            'decisions_any_wheel_abs_tanh_gt095':int((raw[start:end,8:].tanh().abs()>.95).any(-1).sum())}
    output={'schema':'wlr50_clean.p06_stored_exploration_decomposition.v1','source_run':str(run),
        'rollout':str(run/'rollouts'/args.rollout_name),'temperature':args.temperature,'rho':.9,
        'raw_log_parity_max_error':parity,'history_matches_previous_raw_max_error':history_error,
        'episode_start_history':{str(i):history[i].tolist() for i in starts},
        'decomposition':'stored_raw = (stored_conditional_mean - .9*stored_history) + .9*stored_history + (stored_raw-stored_conditional_mean)',
        'base_mean_is_algebraically_inferred_not_new_network_forward':True,
        'no_actor_critic_GAE_returns_or_advantages_recomputed':True,'physics_steps':0,'optimizer_updates':0,
        'first_episode_raw_abs_gt3_row':first_sat,'terminal_rows':terminal,'first_support_below4_row':first_less4,
        'first_support_below3_row':first_less3,'first_support_below2_row':first_less2,
        'episodes':episodes,'key_samples':[snapshot(i) for i in sorted(chosen)],
        'scope_caveat':'Decision-end contact/geometry only; cannot identify sub-tick contact onset or prove one action channel caused collision. P06 never reached P09 geometry and this is not a new runtime rollout.',
        'saturation_definition':'abs(raw)>3 => abs(tanh(raw))>.995; tanh>.95 is descriptive near-scale use, not a safety threshold. Physical residual slew/headroom still applies.'}
    path=ROOT/'outputs/ppo_task_first_recovery_v1'/args.output_name
    write_json(path,output)
    print(json.dumps({'path':str(path),'parity':parity,'history_error':history_error,
        'first_rows':{k:v for k,v in output.items() if k.startswith('first_')},
        'key_samples':[{'row':r['row'],'tick':r['tick'],'FRk_target_deg':r['final_target'][3],
            'FRk_actual_one_tick_earlier_deg':r['actual_servo_before_last_dispatch_deg'][3],
            'body_min_z_m':r['body_bounds']['minimum_w_m'][2],
            'FR_support':r['supports']['FR']['support'],'FLw_target':r['final_target'][8]}
            for r in output['key_samples'] if r['row'] in (0,1,39,59,63,64,65,74,79,95,111,127)]},indent=2))


if __name__=='__main__':main()
