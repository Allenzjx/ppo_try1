"""Sealed continuous gain10 block review, each rollout against its own collector.

Imports/dry tests are stdlib only. Requires explicit Isaac exit and one real
COMPLETE run manifest. Never manufactures subruns, writes checkpoints or updates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import analyze_rr_learning_signal_update8 as prior

common,previous,update7 = prior.common,prior.previous,prior.update7
require = prior.require
RUN = common.ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_gain10_continuous2048_from_CP229376_0d0f894'
HISTORY = common.OUTPUT/'checkpoints/history'
SOURCE_NAME = 'checkpoint_CP229376_local004096_aux000064_lineage448_v2_g0d0f89489992.pt'
SOURCE_SHA = 'cd6d724af6901d03ee235f3b5d3160b4f9db86195d0cf423b3e1e159625368da'
SOURCE_MANIFEST_SHA = 'da5917b6a2affeb9fb0d1d52597d89e0aa15928221de060d5acdc157646f989c'


def checked_record(cp,sidecar,runtime):
    meta = json.loads(sidecar.read_text())
    require(meta.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2' and
            meta.get('runtime_contract') == runtime and meta.get('checkpoint_sha256') == common.sha(cp) and
            Path(meta.get('checkpoint','')).resolve() == cp.resolve() and
            sidecar.resolve() == cp.with_name(cp.stem+'_manifest.json').resolve() and
            meta.get('rollout_empty') is True and meta.get('save_load_round_trip') is True and
            meta.get('front_FL_assist') is True and meta.get('rear_task_assists') is False,
            'checkpoint bytes/sidecar/runtime/complete boundary mismatch')
    return cp,sidecar,meta


def find_update_records(run,runtime,updates,history=HISTORY):
    """Select by exact actual update counters + run identity, never expected names."""
    candidates=[]
    for sidecar in history.glob('*_g'+prior.HEAD[:12]+'_manifest.json'):
        meta=json.loads(sidecar.read_text())
        if meta.get('runtime_contract') == runtime and Path(meta.get('source_run','')).resolve() == run.resolve():
            candidates.append((sidecar,meta))
    result=[]
    for update in updates:
        matches=[(path,meta) for path,meta in candidates if meta.get('counts') == update.get('counts')]
        require(len(matches) == 1, 'each actual update must resolve to exactly one source-bound checkpoint sidecar')
        sidecar,meta=matches[0]
        result.append(checked_record(Path(meta['checkpoint']),sidecar,runtime))
    return result


def sealed_inputs(run):
    require(run.resolve() == RUN.resolve(), 'only the declared continuous run is in scope')
    path=run/'run_manifest.json'
    require(path.is_file(), 'continuous run is not sealed COMPLETE yet')
    manifest=json.loads(path.read_text())
    require(manifest.get('lifecycle') == 'COMPLETE' and manifest.get('mode') == 'train',
            'requires the real COMPLETE training manifest, not checkpoint existence')
    started=json.loads((run/'run_manifest.started.json').read_text());runtime=started['runtime_contract']
    require(runtime.get('source_git_commit') == prior.HEAD and
            runtime.get('experiment_id') == 'ppo_rr_capture_first_cp225280_v1' and
            runtime['local_contract'].get(prior.GAIN_KEY) == prior.GAIN and
            runtime['local_contract'].get('observation_dimension') == 448 and
            runtime['local_contract'].get('source_tracking_owner_revision') == previous.REVISION,
            'exact current448/gain10 source runtime required')
    cp=Path(started['checkpoint']);cp=cp if cp.is_absolute() else common.ROOT/cp
    sidecar=cp.with_name(cp.stem+'_manifest.json')
    require(cp.name == SOURCE_NAME and common.sha(cp) == SOURCE_SHA and
            common.sha(sidecar) == SOURCE_MANIFEST_SHA, 'not the exact latest CP229376 collector source')
    first=checked_record(cp,sidecar,runtime)
    updates=list(common.json_rows(run/'updates.jsonl'))
    require(1 <= len(updates) <= 4, 'review supports one to four actual complete updates, not planned files')
    records=[first]+find_update_records(run,runtime,updates)
    accounting=[]
    for index,update in enumerate(updates):
        item=prior.validate_pair(records[index][2],records[index+1][2])
        require(update['counts'] == records[index+1][2]['counts'] and
                update['optimizer_steps'] == item['delta']['local_optimizer_steps'], 'actual update receipt mismatch')
        accounting.append(item)
    pointer=manifest['result'];final=records[-1]
    require(Path(pointer['checkpoint']).resolve() == final[0].resolve() and
            Path(pointer['manifest']).resolve() == final[1].resolve() and
            pointer['checkpoint_sha256'] == common.sha(final[0]) and
            pointer['manifest_sha256'] == common.sha(final[1]) and pointer['counts'] == final[2]['counts'],
            'COMPLETE run final pointer differs from the actual last update checkpoint')
    receipt=records[0][2][prior.LEDGER_KEY][0]['receipt'];p=Path(receipt['path'])
    require(common.sha(p) == receipt['sha256'], 'inherited coordinate receipt hash changed')
    proof=json.loads(p.read_text())
    require(proof.get('status') == 'PASS' and proof.get('flags') and
            all(v is True for v in proof['flags'].values()), 'inherited coordinate migration checks failed')
    return records,updates,accounting


def rollout_rows(samples,index):
    """Keep episode IDs/history, reset only local storage indices for this update."""
    rows=[dict(row,index=offset) for offset,row in enumerate(samples[index*512:(index+1)*512])]
    require(len(rows) == 512, 'incomplete per-update sample slice')
    return rows


def cohort_windows(samples,slices):
    """Keep prior TOP memory across updates; only reindex rows for storage lookup."""
    whole=update7.physical_cohorts(samples)
    windows=[]
    for index,rows in enumerate(slices):
        start=index*512;end=start+len(rows)
        windows.append({name:[rows[r['index']-start] for r in group if start <= r['index'] < end]
                        for name,group in whole.items()})
    whole['all_on_policy']=whole.pop('all512')
    return whole,windows


def rollout_boundaries(samples,accounting):
    """Observe real terminal flags; neither update nor budget boundary implies done."""
    result=[]
    for index,item in enumerate(accounting):
        position=(index+1)*512-1;tail=samples[position]
        terminated=tail['local_reward']['terminated']
        require(type(terminated) is bool, 'actual boolean tail termination required')
        following=samples[position+1] if position+1 < len(samples) else None
        if following is not None:
            if not terminated:
                require(following['episode'] == tail['episode'] and
                        following['before']['tick'] == tail['after']['tick'],
                        'nonterminal update boundary unexpectedly reset or skipped physical history')
                relation='same_live_episode_continues_with_next_collector'
            else:
                require(following['episode'] > tail['episode'], 'actual terminal did not produce new prefix episode')
                relation='new_real_prefix_after_actual_terminal'
        else:
            relation='actual_terminal_at_budget_boundary' if terminated else 'nonterminal_final_budget_tail'
        result.append(dict(absolute_update=item['absolute_update'],episode=tail['episode'],tick=tail['tick'],
            actual_terminal=terminated,local_success=tail['local_success'],relation=relation,
            gap_mm=1000*tail['after']['gap_m'],next_episode=None if following is None else following['episode'],
            bootstrap_required_by_nonterminal=not terminated,
            bootstrap_recurrence_independently_recomputed=False))
    return result


def tensor_pair(run,rows,records,accounting,checks,groups):
    """Called only after analyze's explicit-exit and complete-source guards."""
    import torch
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import state_hash
    number=accounting['absolute_update'];rollout=run/'rollouts'/f'rollout_{number:04d}.pt'
    storage=torch.load(rollout,map_location='cpu',weights_only=False)
    obs=storage['observations']['policy'].reshape(512,448)
    comparisons=dict(observations=(obs,[r['observation'] for r in rows]),
        raw=(storage['actions'].reshape(512,12),[r['policy']['selected_raw_full12'] for r in rows]),
        old_logp=(storage['actions_log_prob'].reshape(512),[r['old_logp'] for r in rows]),
        mean=(storage['distribution_params'][0].reshape(512,12),[r['policy']['conditional_mean_full12'] for r in rows]),
        std=(storage['distribution_params'][1].reshape(512,12),[r['policy']['active_conditional_std_full12'] for r in rows]),
        reward=(storage['rewards'].reshape(512),[r['local_reward']['reward'] for r in rows]),
        done=(storage['dones'].reshape(512).float(),[float(r['local_reward']['terminated']) for r in rows]))
    for key,(actual,expected) in comparisons.items():
        checks[key]=common.close_tensor(torch,actual,torch.tensor(expected,dtype=actual.dtype),key)
    mean,std=storage['distribution_params']
    logp=torch.distributions.Normal(mean,std).log_prob(storage['actions']).sum(-1).reshape(512)
    checks['Gaussian_logp']=common.close_tensor(torch,logp,storage['actions_log_prob'].reshape(512),'raw logp',1e-4)
    gae=(storage['returns']-storage['values']).reshape(512)
    checks['per512_normalized_GAE']=common.close_tensor(torch,(gae-gae.mean())/(gae.std()+1e-8),
        storage['advantages'].reshape(512),'per512 GAE')
    for i,row in enumerate(rows):
        for field,key in (('reward','rewards'),('return','returns'),('value','values'),('advantage','advantages')):
            row[field]=float(storage[key].reshape(512)[i])
        row['raw_gae']=float(gae[i])
    outputs,receipts=[],[]
    for cp,sidecar,meta in records:
        data=torch.load(cp,map_location='cpu',weights_only=False)
        require(all(meta.get(k) == v for k,v in data['infos'].items()) and
                all(state_hash(data[k]) == value for k,value in meta['state_hashes'].items()) and
                data.get('iter') == meta['counts']['local_ppo_updates'], 'embedded state/metadata mismatch')
        lrs=[g['lr'] for g in data['optimizer_state_dict']['param_groups']]
        require(lrs and all(lr == meta['learning_rate'] for lr in lrs), 'actual Adam LR mismatch')
        cfg=dict(meta['runner_config']['actor']);cfg.pop('class_name')
        actor=SemanticRRCaptureLocalHistoryMLPModel({'policy':obs[:1]},{'actor':['policy']},'actor',12,**cfg)
        require(list(actor.local_mean_coordinate_gain_full12) == prior.GAIN, 'actor gain binding lost')
        actor.load_state_dict(data['actor_state_dict'],strict=True);actor.eval()
        with torch.inference_mode():
            mu=actor({'policy':obs},stochastic_output=False).clone();ev=actor._last_forward_evidence
            outputs.append(dict(mean=mu,local=ev['local_raw_mean_delta'].clone(),std=ev['head_log_std'].exp().clone()))
        actor.assert_frozen_state()
        receipts.append(dict(path=str(cp),sha256=common.sha(cp),manifest_sha256=common.sha(sidecar),
            counts=meta['counts'],LR=meta['learning_rate'],actual_Adam_group_LRs=lrs,prior_sha256=actor.expected_prior_state_sha256))
        del actor,data
    require(receipts[0]['prior_sha256'] == receipts[1]['prior_sha256'], 'frozen prior drift')
    # outputs[0] is THIS rollout's collector, not the whole run's initial actor.
    for key,index in (('mean',0),('std',1)):
        checks['actual_collector_'+key]=common.close_tensor(torch,outputs[0][key],
            storage['distribution_params'][index].reshape(512,12),'actual collector '+key,1e-4)
    checks['actual_collector_effective_local']=common.close_tensor(torch,outputs[0]['local'],
        torch.tensor([r['policy']['local_raw_mean_delta_full12'] for r in rows]),'actual collector local',1e-4)
    fixed={}
    for name,group in groups.items():
        ids=[r['index'] for r in group]
        fixed[name]=dict(n=len(ids),RR={joint:{field:dict(before=common.stats(outputs[0][field][ids,ch].tolist()),
            after=common.stats(outputs[1][field][ids,ch].tolist()),
            shift=common.stats((outputs[1][field][ids,ch]-outputs[0][field][ids,ch]).tolist()))
            for field in ('mean','local','std')} for joint,ch in common.RR.items()})
    return dict(absolute_update=number,collector=receipts[0],updated=receipts[1],verification=checks,
        rollout=dict(path=str(rollout),sha256=common.sha(rollout)),fixed_input_RR=fixed,
        physical_events=update7.contact_summary(rows,groups),
        cohort_GAE_and_response={name:common.metric_summary(group) for name,group in groups.items()},
        knee_headroom_and_actual_response=previous.knee_response(rows))


def analyze(run=RUN, *, isaac_stopped=False):
    require(isaac_stopped, 'Explicit Isaac exit required before sealed reads or Tensor work')
    records,updates,accounting=sealed_inputs(run)
    runtime=records[0][2]['runtime_contract'];versions=runtime['local_runtime_versions']
    require(Path(sys.executable).resolve() == Path(versions['python_executable']).resolve() and
            sys.version.split()[0] == versions['python'] and
            all(importlib.metadata.version(name) == version for name,version in versions['distributions'].items()),
            'use the actual sealed Python and package versions')
    previous.validate_loaded_source(runtime)
    for relative in ('src/wlr50_clean/ppo/semantic_rr_mean_coordinates.py',
                     'src/wlr50_clean/ppo/semantic_rr_capture_local.py',
                     'src/wlr50_clean/ppo/semantic_rr_capture_local_task.py',
                     'configs/ppo_rr_capture_first_cp225280_v1/local_training.json'):
        require(common.sha(common.ROOT/relative) == runtime['files'].get(relative),'sealed gain10 source changed: '+relative)
    offset=records[0][2]['counts']['local_ppo_updates']
    samples,collected_updates,prefix=common.collect(run,len(updates),update_offset=offset)
    require(collected_updates == updates and prefix == sum(a['delta']['prefix_decisions'] for a in accounting),
            'full-block actual prefix/update accounting differs')
    execution=previous.validate_execution_rows(common.json_rows(run/'decisions.jsonl'),expected_count=len(samples))
    slices=[rollout_rows(samples,i) for i in range(len(updates))];checks=[]
    whole_groups,window_groups=cohort_windows(samples,slices)
    boundaries=rollout_boundaries(samples,accounting)
    for rows,update,account in zip(slices,updates,accounting):
        check=dict(raw_likelihood_history=prior.validate_gain_rows(rows),
            actual_minibatch_exposure=prior.minibatch_evidence(run,rows,update,account))
        require(update['actual_phase_counts'] == dict(Counter(r['phase'] for r in rows)), 'actual per-update phase counts mismatch')
        checks.append(check)
    # All real-run completion, checkpoint chain and JSON checks precede imports.
    import torch
    torch.set_num_threads(1);sys.path.insert(0,str(common.ROOT/'src'))
    reports=[tensor_pair(run,rows,records[i:i+2],accounting[i],checks[i],window_groups[i]) for i,rows in enumerate(slices)]
    return dict(schema='wlr50_clean.rr_gain10_continuous_block_learning_signal.v1',run=str(run),
        counts=dict(new_PPO_decisions=len(samples),prefix_credit0=prefix,new_PPO_updates=len(updates),
            new_PPO_Adam_steps=sum(a['delta']['local_optimizer_steps'] for a in accounting),
            inherited_AUX_optimizer_steps=64,new_AUX_optimizer_steps=0,requested_four_updates_met=len(updates)==4),
        source_counts=records[0][2]['counts'],final_counts=records[-1][2]['counts'],execution=execution,
        actual_phase_counts=dict(Counter(r['phase'] for r in samples)),
        physical_events_whole_run=update7.contact_summary(samples,whole_groups),rollout_boundaries=boundaries,
        updates=reports,channel_KL=common.channel_kl_report(run,samples,updates,update_offset=offset),
        source_json_sha256={name:common.sha(run/name) for name in
            ('run_manifest.json','run_manifest.started.json','updates.jsonl','decisions.jsonl')},
        inherited_coordinate_migrations=records[0][2][prior.LEDGER_KEY],
        limits=['Each512 rollout is matched to its actual immediately preceding checkpoint; no initial-actor reuse.',
            'GAE normalization is independently checked per512, never across2048; tail recurrence is not independently reconstructed.',
            'Per-update physical cohorts cover a window but preserve whole-run TOP/reacquisition history; all_on_policy is the whole-block total.',
            'An update boundary keeps a live episode; the final nonterminal budget tail is not relabelled done/failure/success.',
            'No hidden subrun manifest, checkpoint fabrication, model update, AUX or physical run is performed.',
            'Raw/local means retain one HISTORY and explicit gain10; offline mean shifts are not deterministic physical success.',
            'One to four actual COMPLETE updates may be reported; requested_four_updates_met discloses any shortfall.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-stopped',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();target=args.output.resolve()
    require(target.is_relative_to(common.OUTPUT) and not target.exists(),'write only a NEW isolated outputs report')
    result=analyze(isaac_stopped=args.isaac_stopped)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(target),counts=result['counts'],phase_counts=result['actual_phase_counts'])))


if __name__=='__main__':main()
