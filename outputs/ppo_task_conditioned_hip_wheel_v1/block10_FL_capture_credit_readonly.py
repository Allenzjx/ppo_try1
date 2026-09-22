"""Twelve fixed real capture-window inputs; no optimizer or physical calls."""
from pathlib import Path
import json,sys,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import late_state_review as late
a,c,torch=late.a,late.c,late.torch
RUN=a.ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1043344171428Z_g97ecd305afb5_0168bb1943e74402accb92ea31b10de3'
INDICES=list(range(380,386))+list(range(1475,1481))

def main():
    a.require(not torch.cuda.is_available(),'CPU-only required')
    saved=a.training.capture_training_rng_state(seed=1001)
    try:
        records={}
        with (RUN/'residual_and_projection_audit.jsonl').open() as f:
            for i,line in enumerate(f):
                if i>INDICES[-1]:break
                if i in INDICES:records[i]=json.loads(line)
        cache={};actors={};sources={};result=[]
        for i in INDICES:
            update,index=1470+i//128,i%128
            if update not in cache:
                file=RUN/'rollouts'/f'rollout_{update:06d}.pt'
                snapshot=torch.load(file,map_location='cpu',weights_only=False)
                likelihood=json.loads(file.with_name(f'update_{update:06d}_likelihood.json').read_text())
                cache[update]=(snapshot,likelihood,late.sha(file))
            snapshot,L,digest=cache[update];row=records[i];audit=row['applied_audit'];obs=snapshot['observations']['policy'][index,0]
            a.require(obs.shape==(372,) and row['global_policy_decision']==192513+i,'source decision/input mismatch')
            xs=a.tensors(obs.unsqueeze(0),device='cpu');means={}
            for label,step in [('before',192512+(update-1470)*128),('after',192512+(update-1469)*128)]:
                if step not in actors:
                    cp=HERE/'checkpoints/history'/f'checkpoint_step_{step:09d}.pt'
                    actor,metadata=late.load_actor(cp,obs.tolist());actors[step]=(actor,metadata)
                    sources[str(step)]={'checkpoint':str(cp),'sha256':late.sha(cp),'actor_sha256':metadata['actor_parameter_sha256']}
                actor,metadata=actors[step]
                a.require(metadata['runtime_contract']==snapshot['runtime_contract']
                    and metadata['policy_contract']==snapshot['policy_contract'],'same-state update boundary mismatch')
                means[label]=a.distribution(actor,xs)['mean'][0]
            old=snapshot['distribution_params'][0][index,0];raw=snapshot['actions'][index,0]
            a.require(torch.equal(raw,torch.tensor(row['raw_policy_action_full12']))
                and torch.equal(old,torch.tensor(row['old_distribution_mean_full12'])),'raw/oldmean log mismatch')
            oldV=snapshot['values'][index,0,0];ret=snapshot['returns'][index,0,0];adv=snapshot['advantages'][index,0,0]
            appearances=[]
            for batch in L['minibatches']:
                for j,ids in enumerate(batch['rollout_flat_indices']):
                    if index in ids:
                        a.require(batch['actual_advantage'][j]==float(adv),'actual optimizer advantage differs')
                        appearances.append({'minibatch':batch['minibatch_index'],'actual_normalized_advantage':batch['actual_advantage'][j],
                            'direct_loss_gradient_wrt_FL_network_mean_hip_knee':batch['loss_gradient_wrt_network_mean_full12'][j][:2],
                            'clipped_branch_strictly_active':batch['clipped_branch_strictly_active'][j]})
            a.require(len(appearances)==5,'missing actual optimizer appearances')
            fl=audit['semantic_task']['physical_evaluator']['current_legs']['FL'];rb=audit['reward_breakdown']
            result.append({'episode':0 if i<1000 else 1,'global':row['global_policy_decision'],'source_line_1based':i+1,
                'update':update,'rollout_row_0based':index,'rollout_sha256':digest,
                'input_float32_sha256':hashlib.sha256(obs.contiguous().numpy().tobytes()).hexdigest(),
                'tick':audit['physics_tick'],'phase':audit['phase_id'],'end_phase':audit['end_phase_id'],
                'FL_air':fl['air'],'FL_top_contact':fl['top_contact'],'FL_bearing_N':fl['bearing_force_n'],
                'FL_gap_mm':1000*fl['clearance_m'],'FL_placed_history':audit['semantic_task']['placed_history']['FL'],
                'reward':float(snapshot['rewards'][index,0,0]),'reward_breakdown':rb,'old_value':float(oldV),
                'return':float(ret),'raw_GAE':float(ret-oldV),'actual_normalized_advantage':float(adv),'done':bool(snapshot['dones'][index,0,0]),
                'raw_action_FL_hip_knee':raw[:2].tolist(),'old_mu_FL_hip_knee':old[:2].tolist(),
                'raw_action_minus_old_mu_FL_hip_knee':(raw-old)[:2].tolist(),
                'same_CPU_preupdate_mu_FL_hip_knee':means['before'][:2].tolist(),
                'same_CPU_postupdate_mu_FL_hip_knee':means['after'][:2].tolist(),
                'same_CPU_update_mu_difference_FL_hip_knee':(means['after']-means['before'])[:2].tolist(),
                'CPU_vs_saved_GPU_old_mu_max_abs_full12':float((means['before']-old).abs().max()),
                'actual_optimizer_appearances':appearances})
        for actor,metadata in actors.values():a.require(a.training.parameter_hash(actor)==metadata['actor_parameter_sha256'],'read-only forward changed weights')
        report={'schema':'wlr50_clean.block10_FL_capture_credit_readonly.v1','run':str(RUN),'sources':sources,'rows':result,
            'inputs_count':12,'new_optimizer_steps':0,'physical_calls':0,'PPO_credit_added':0,
            'semantics':'fixed real372/H; normalized advantages verified at actual five official minibatches; postupdate means compared on identical CPU inputs, not joint logp attribution or causal explanation'}
    finally:a.training.restore_training_rng_state(saved,expected_seed=1001)
    a.training.write_json(c.output_path(HERE/'block10_FL_capture_credit.json'),report)
    for r in result:print(json.dumps({k:r[k] for k in ('episode','global','tick','update','rollout_row_0based','FL_top_contact','reward','old_value','raw_GAE','actual_normalized_advantage','raw_action_minus_old_mu_FL_hip_knee','same_CPU_update_mu_difference_FL_hip_knee','CPU_vs_saved_GPU_old_mu_max_abs_full12')}))

if __name__=='__main__':main()
