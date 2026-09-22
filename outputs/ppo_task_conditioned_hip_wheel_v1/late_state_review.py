"""Bounded CPU same-input review; no optimizer, physics or PPO data insertion."""
from pathlib import Path
import argparse, hashlib, json, sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'candidate/finite_auxiliary'))
import finite_auxiliary_mean as a
import auxiliary_cli as c
from reviewed_data import load_reviewed_selection,sha
import torch

RUNS={
 '07':'20260921T0849405935866Z_gee5a9651591d_12b59026033c4ad08c274c684926dadc',
 '08':'20260921T0916266524903Z_gee5a9651591d_87559dc9adb340abb997ffa03fc0da89',
 '09':'20260921T0956511314318Z_g97ecd305afb5_6a28dfa361804726aa56ecb7e030410c'}
RECIPE={'P06':[('08',1456),('08',1457),('09',1469)],
 'P08':[('08',1458),('08',1463)],'P09':[('08',1458),('08',1463)],
 'P12':[('07',1450),('07',1451),('07',1453)]}


def select_states(metadata):
    cache={};runs={};result=[]
    for phase,files in RECIPE.items():
        pool=[]
        for block,update in files:
            if block not in runs:
                run=a.ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train'/RUNS[block]
                manifest=run/'run_manifest.json';m=json.loads(manifest.read_text())
                a.require(m.get('lifecycle')=='SUCCEEDED','training run not sealed')
                runs[block]={'path':str(run),'run_manifest_sha256':sha(manifest)}
            path=Path(runs[block]['path'])/'rollouts'/f'rollout_{update:06d}.pt'
            if (block,update) not in cache:
                payload=torch.load(path,map_location='cpu',weights_only=False)
                c.verify_data_compatibility(metadata,{'diagnostic_runtime_contract':payload['runtime_contract'],
                    'diagnostic_policy_contract':payload['policy_contract']},metadata['runtime_contract'])
                x=payload['observations']['policy']
                a.require(payload['schema']=='wlr50_clean.semantic_on_policy_rollout.v1'
                    and x.shape==(128,1,372) and x.dtype==torch.float32
                    and bool(torch.isfinite(x).all()),'not original complete finite role372 rollout')
                cache[block,update]=(x,sha(path))
            x,digest=cache[block,update]
            indices=(x[:,0,:13].argmax(-1)==int(phase[1:])-1).nonzero().flatten().tolist()
            for index in indices:
                if not bool(x[index,0,195:207].abs().sum()>0):continue
                pool.append({'phase':phase,'block':block,'rollout_path':str(path),'rollout_sha256':digest,
                    'row_index_0based':index,'env_index':0,'tensor_key':'observations.policy',
                    'actor_input_float32_372':x[index,0].tolist(),
                    'actor_input_float32_le_sha256':hashlib.sha256(x[index,0].contiguous().numpy().tobytes()).hexdigest()})
        a.require(pool,'no actual nonzero-H samples for requested phase')
        picks=list(range(len(pool))) if len(pool)<=8 else [round(i*(len(pool)-1)/7) for i in range(8)]
        result.extend(pool[i] for i in picks)
    return result,runs


def load_actor(checkpoint,obs):
    metadata=c.migration.checkpoint_metadata(checkpoint)
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    expected={k:v for k,v in metadata.items() if k not in ('checkpoint_path','checkpoint_sha256','save_load_round_trip')}
    a.require(p['infos']==expected,'checkpoint payload/manifest mismatch')
    runner=c.observation_runner(obs,seed=metadata['seed'],device='cpu')
    runner.alg.actor.load_state_dict(p['actor_state_dict'],strict=True);runner.alg.actor.eval()
    a.require(a.training.parameter_hash(runner.alg.actor)==metadata['actor_parameter_sha256']
        and a.training._runner_policy_contract(runner)==metadata['policy_contract'],'actor/policy hash mismatch')
    return runner.alg.actor,metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before-checkpoint',type=Path,default=HERE/'checkpoints/history/checkpoint_step_000192512.pt')
    p.add_argument('--after-checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    a.require(not torch.cuda.is_available(),'use CUDA_VISIBLE_DEVICES=-1 for read-only CPU review')
    seal=json.loads((a.HERE/'SEALED_FILES.json').read_text());a.require(c.helpers()==seal['helper_bundle_sha256'],'sealed aux helper changed')
    output=c.output_path(args.output);before=args.before_checkpoint.resolve(strict=True)
    metadata=c.migration.checkpoint_metadata(before)
    saved_rng=a.training.capture_training_rng_state(seed=metadata['seed'])
    try:
        rows,runs=select_states(metadata);xs=a.tensors([r['actor_input_float32_372'] for r in rows],device='cpu')
        actor,metadata=load_actor(before,rows[0]['actor_input_float32_372']);d=a.distribution(actor,xs)
        report={'schema':'wlr50_clean.late_state_readonly_review.v1','before_checkpoint':str(before),
            'before_checkpoint_sha256':sha(before),'run_sources':runs,'rows':rows,'script_sha256':sha(Path(__file__)),
            'sealed_aux_helper_sha256':c.helpers(),'PPO_credit':0,'auxiliary_updates':0,
            'data_inserted_into_PPO':False,'physical_trajectory_effect':None,
            'scope':'fixed saved real inputs and real encoded HISTORY; not a natural trajectory or safety guarantee'}
        if args.after_checkpoint:
            after=args.after_checkpoint.resolve(strict=True);other,am=load_actor(after,rows[0]['actor_input_float32_372'])
            a.require(am['runtime_contract']==metadata['runtime_contract'] and am['policy_contract']==metadata['policy_contract'],
                'before/after runtime or policy differs')
            post=a.distribution(other,xs);new=post['mean']
            report.update(mode='actual_saved_before_after_same_state',after_checkpoint=str(after),after_checkpoint_sha256=sha(after),
                after_auxiliary_ledger=am.get('task_conditioned_hip_wheel_branch',{}).get(a.LEDGER_KEY))
        else:
            train,target,hold,receipt=load_reviewed_selection(HERE/'FL_minus6_gap_approach_selection.json')
            c.verify_data_compatibility(metadata,receipt,metadata['runtime_contract'])
            tx=a.tensors(train,device='cpu');layer=a.mean_layer(actor)
            w=torch.nn.Parameter(layer.weight[:1].detach().clone());b=torch.nn.Parameter(layer.bias[:1].detach().clone())
            loss=.5*(a.functional_mean(actor,tx,w,b)[:,0]-torch.tensor(target)).square().mean()
            gw,gb=torch.autograd.grad(loss,(w,b));budget=a.Budget();total_lr=budget.learning_rate*(budget.max_steps+1)/2
            with torch.no_grad():new=a.functional_mean(actor,xs,w-total_lr*gw,b-total_lr*gb)
            report.update(mode='frozen_first_gradient_linear_prediction_only_not_optimized',
                selection_sha256=receipt['selection_sha256'],budget=a.asdict(budget),sum_learning_rates=total_lr,
                first_gradient_norm=float(torch.cat((gw.flatten(),gb.flatten())).norm()))
            post={'sigma':d['sigma']}
        for index,row in enumerate(rows):
            row.update(real_encoded_HISTORY_195_267=xs['policy'][index,195:267].tolist(),
                before_network_mean_full12=d['network_mean'][index].tolist(),before_HISTORY_center_full12=d['history'][index].tolist(),
                before_conditional_mean_full12=d['mean'][index].tolist(),before_sigma_full12=d['sigma'][index].tolist(),
                compared_conditional_mean_full12=new[index].tolist(),compared_sigma_full12=post['sigma'][index].tolist(),
                FL_REQUEST_difference_deg=float(d['caps'][index,0]*(new[index,0].tanh()-d['mean'][index,0].tanh())))
        report['phase_summary']={phase:{'count':sum(r['phase']==phase for r in rows),
            'max_abs_FL_REQUEST_difference_deg':max(abs(r['FL_REQUEST_difference_deg']) for r in rows if r['phase']==phase)} for phase in RECIPE}
        a.require(a.training.parameter_hash(actor)==metadata['actor_parameter_sha256'],'read-only review changed original actor')
    finally:a.training.restore_training_rng_state(saved_rng,expected_seed=metadata['seed'])
    a.training.write_json(output,report);print(json.dumps({'output':str(output),'mode':report['mode'],'phases':report['phase_summary']}))


if __name__=='__main__':main()
