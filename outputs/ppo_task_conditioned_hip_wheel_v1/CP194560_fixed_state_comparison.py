"""One CPU-only fixed-observation checkpoint comparison; zero training credit."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import late_state_review as late
a,c,torch=late.a,late.c,late.torch
from reviewed_data import load_reviewed_selection,sha

def stats(x):return {'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max())}

def main():
    a.require(not torch.cuda.is_available(),'CPU-only visibility required')
    seal=json.loads((a.HERE/'SEALED_FILES.json').read_text())
    a.require(c.helpers()==seal['helper_bundle_sha256'],'sealed helper mismatch')
    names={'before':'checkpoint_step_000192512.pt','aux':'checkpoint_aux_flmean_CP192512_budget16_01.pt',
        'after_PPO':'checkpoint_step_000194560.pt'}
    paths={k:HERE/'checkpoints/history'/v for k,v in names.items()}
    base=c.migration.checkpoint_metadata(paths['before'])
    rng=a.training.capture_training_rng_state(seed=base['seed'])
    try:
        train,target,hold,data=load_reviewed_selection(HERE/'FL_minus6_gap_approach_selection.json')
        rows,runs=late.select_states(base)
        observations=train+hold+[r['actor_input_float32_372'] for r in rows]
        a.require(len(observations)==125,'expected original75+24+26 states')
        xs=a.tensors(observations,device='cpu');ds={};sources={};metadata={}
        for label,path in paths.items():
            actor,m=late.load_actor(path,observations[0]);metadata[label]=m
            a.require(m['runtime_contract']==base['runtime_contract'] and m['policy_contract']==base['policy_contract'],
                'comparison cannot change MDP, observation semantics or policy kernel')
            ds[label]=a.distribution(actor,xs)
            a.require(a.training.parameter_hash(actor)==m['actor_parameter_sha256'],'read-only forward changed weights')
            sources[label]={'path':str(path),'checkpoint_sha256':sha(path),'manifest_sha256':sha(path.with_name(path.stem+'_manifest.json')),
                'actor_parameter_sha256':m['actor_parameter_sha256'],
                'PPO_counters':{k:m[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')}}
        for d in ds.values():
            a.require(torch.equal(d['history'],ds['before']['history']) and torch.equal(d['caps'],ds['before']['caps']),
                'fixed numeric HISTORY/caps changed between checkpoints')
        a.require(torch.equal(ds['before']['sigma'],ds['aux']['sigma'])
            and torch.equal(ds['before']['mean'][:,1:],ds['aux']['mean'][:,1:]),'immediate aux unexpectedly changed sigma or other11')
        ledger=metadata['aux']['task_conditioned_hip_wheel_branch'][a.LEDGER_KEY]
        a.require(metadata['after_PPO']['task_conditioned_hip_wheel_branch'][a.LEDGER_KEY]==ledger,'aux lineage did not persist')
        cohorts={'P05_all75':list(range(75)),'P05_hold63':list(range(12,75)),
            'front_holdout24':list(range(75,99)),'late_all26':list(range(99,125))}
        cohorts.update({phase:[99+i for i,r in enumerate(rows) if r['phase']==phase] for phase in late.RECIPE})
        groups={}
        for group,indices in cohorts.items():
            values={};deltas={}
            for label,d in ds.items():
                values[label]={'FL_network_mean_raw':stats(d['network_mean'][indices,0]),
                    'FL_conditional_mean_raw':stats(d['mean'][indices,0]),
                    'FL_REQUEST_deg':stats(d['caps'][indices,0]*d['mean'][indices,0].tanh()),
                    'FL_effective_sigma_raw':stats(d['sigma'][indices,0])}
            for name,older,newer in [('aux_minus_before','before','aux'),('PPO_minus_aux','aux','after_PPO'),('final_minus_before','before','after_PPO')]:
                x,y=ds[older],ds[newer];request=y['caps'][indices]*(y['mean'][indices].tanh()-x['mean'][indices].tanh())
                deltas[name]={'FL_network_mean_raw':stats(y['network_mean'][indices,0]-x['network_mean'][indices,0]),
                    'FL_conditional_mean_raw':stats(y['mean'][indices,0]-x['mean'][indices,0]),'FL_REQUEST_deg':stats(request[:,0]),
                    'mean_raw_difference_mean_full12':(y['mean'][indices]-x['mean'][indices]).mean(0).tolist(),
                    'REQUEST_difference_mean_full12':request.mean(0).tolist(),
                    'sigma_ratio_mean_full12':(y['sigma'][indices]/x['sigma'][indices]).mean(0).tolist(),
                    'sigma_ratio_min_full12':(y['sigma'][indices]/x['sigma'][indices]).min(0).values.tolist(),
                    'sigma_ratio_max_full12':(y['sigma'][indices]/x['sigma'][indices]).max(0).values.tolist()}
            groups[group]={'count':len(indices),'FL_fixed_HISTORY_raw':stats(ds['before']['history'][indices,0]),'values':values,'changes':deltas}
        output=c.output_path(HERE/'CP194560_fixed_state_comparison.json')
        report={'schema':'wlr50_clean.fixed_state_three_checkpoint_comparison.v1','sources':sources,'script_sha256':sha(Path(__file__)),
            'groups':groups,'data_selection_sha256':data['selection_sha256'],'late_rows':rows,'late_run_sources':runs,
            'canonical_order':['FLhip','FLknee','FRhip','FRknee','RLhip','RLknee','RRhip','RRknee','FLwheel','FRwheel','RLwheel','RRwheel'],
            'REQUEST_units':['deg']*8+['rad_per_s']*4,'immediate_aux_other11_means_and_all12_sigma_exactly_unchanged':True,
            'auxiliary_ledger_unchanged':True,'comparison_forward_inputs':125,'optimizer_steps':0,'PPO_credit':0,'random_policy_draws':0,
            'full12_original_numeric_outputs':{k:{field:d[field].tolist() for field in ('network_mean','mean','sigma','history','caps')} for k,d in ds.items()},
            'physical_trajectory_effect':None,'limitation':'same original numeric372 inputs with fixed real HISTORY; offline means/REQUEST are not actual actuator or natural-trajectory outcomes'}
    finally:a.training.restore_training_rng_state(rng,expected_seed=base['seed'])
    a.training.write_json(output,report)
    print(json.dumps({'output':str(output),'P05_hold63':groups['P05_hold63'],'front_holdout24':groups['front_holdout24']['changes']['PPO_minus_aux']['FL_REQUEST_deg']},indent=2))

if __name__=='__main__':main()
