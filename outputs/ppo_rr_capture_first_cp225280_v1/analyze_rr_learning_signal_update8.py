"""Read-only sealed gain10 update8 review; no Torch/PXR at import or dry checks.

Both actors must use the exact current448 gain10 constructor and saved config.
No old0ff actor fallback, parameter migration, optimizer or simulation is run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import analyze_rr_learning_signal as common
import analyze_rr_learning_signal_update6 as previous
import analyze_rr_learning_signal_update7 as update7

RUN = common.ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_rr_mean10_fresh512_0d0f894'
HEAD = '0d0f8948999222f9b8710b0eb913a9419537be83'
SOURCE_NAME = 'checkpoint_CP228864_local003584_aux000064_lineage448_v2_g0d0f89489992.pt'
SOURCE_SHA = 'ea6a93db6342a5eaf09cf119c914ebc6e3ddba730e384b7751914cfdfb443101'
SOURCE_MANIFEST_SHA = '6c02b9d5c099fc3a8a7332e2df726a174b72acbf87359d61441b597aacbadc48'
GAIN_KEY = 'local_mean_coordinate_gain_full12'
GAIN = [1.,1.,1.,1.,1.,1.,10.,10.,1.,1.,1.,1.]
LEDGER_KEY = 'local_mean_coordinate_migrations'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pair(before, after):
    """Derive actual counts; this analyzer covers exactly one complete512 block."""
    old,new = before['counts'],after['counts']
    require(set(old) == set(new) and all(type(old[k]) is int and type(new[k]) is int and
            0 <= old[k] <= new[k] for k in old), 'counts invalid/regressed across update')
    delta = {k:new[k]-old[k] for k in old}
    cfg = before['runner_config']; profile = before['runtime_contract']['local_contract']
    epochs,batches = cfg['algorithm']['num_learning_epochs'],cfg['algorithm']['num_mini_batches']
    require((epochs,batches,profile['rollout_length'],cfg['num_steps_per_env']) == (5,4,512,512),
            'this review uses the actual five-epoch/four-minibatch complete512 recipe')
    require(delta['local_policy_decisions'] == profile['rollout_length'] and
            delta['local_ppo_updates'] == 1 and delta['local_optimizer_steps'] == epochs*batches and
            delta['task_v2_policy_decisions'] == delta['local_policy_decisions'] and
            delta['task_v2_ppo_updates'] == delta['local_ppo_updates'] and
            old['auxiliary_updates'] == new['auxiliary_updates'] == 64,
            'actual block must be +512/+1/+20, same taskv2 credit, no new AUX')
    require(before['runtime_contract'] == after['runtime_contract'] and cfg == after['runner_config'],
            'pre/post runtime or full saved runner config changed')
    require(cfg['actor'].get(GAIN_KEY) == profile.get(GAIN_KEY) == GAIN and
            cfg['actor'].get('observation_layout') == previous.OBSERVATION_LAYOUT and
            cfg['actor'].get('legacy447_migration_only',False) is False,
            'explicit actual448 gain10 config required, not old0ff reconstruction')
    for key in ('local_auxiliary_events', LEDGER_KEY, 'local_migration', 'local_control_rebinds', 'prior'):
        require(before.get(key) == after.get(key), key+' changed during fresh PPO')
    auxiliary = before.get('local_auxiliary_events')
    require(isinstance(auxiliary,list) and auxiliary and
            all(type(e.get('actual_optimizer_steps')) is int and e['actual_optimizer_steps'] > 0 for e in auxiliary) and
            sum(e['actual_optimizer_steps'] for e in auxiliary) == 64, 'unchanged real AUX64 ledger required')
    events = before.get(LEDGER_KEY)
    require(isinstance(events,list) and len(events) == 1, 'single unchanged coordinate receipt required')
    event = events[0]
    require(event.get('source_head') == '0ff03eafeeb75ba8505a99478cea2a6243cf93f5' and
            event.get('destination_head') == HEAD and event.get('target_gain') == GAIN and
            event.get('source_gain') == [1.]*12 and
            event.get('destination_runtime_sha256') == before['runtime_contract']['runtime_content_sha256'] and
            event.get('source_counts') == event.get('destination_counts') and
            all(type(event.get(k)) is int and event[k] == 0 for k in
                ('new_policy_decisions','new_PPO_updates','new_PPO_Adam_steps','new_AUX_steps')),
            'coordinate lineage is not the declared zero-credit gain10 migration')
    return dict(delta=delta,rows=delta['local_policy_decisions'],epochs=epochs,minibatches=batches,
                update_offset=old['local_ppo_updates'],absolute_update=new['local_ppo_updates'])


def sealed_inputs(run):
    require(run.resolve() == RUN.resolve(), 'only the declared gain10 update8 run is in scope')
    path = run/'run_manifest.json'
    require(path.is_file(), 'update8 is not sealed COMPLETE yet')
    manifest = json.loads(path.read_text())
    require(manifest.get('lifecycle') == 'COMPLETE' and manifest.get('mode') == 'train',
            'requires sealed COMPLETE update8 training block')
    started = json.loads((run/'run_manifest.started.json').read_text())
    runtime = started['runtime_contract']; profile = runtime.get('local_contract',{})
    require(runtime.get('source_git_commit') == HEAD and
            runtime.get('experiment_id') == 'ppo_rr_capture_first_cp225280_v1' and
            profile.get('observation_dimension') == 448 and profile.get(GAIN_KEY) == GAIN and
            profile.get('source_tracking_owner_revision') == previous.REVISION and
            profile.get('capture_source_dispatch') == 'rr_local_defer_p09_late_and_new_p12_until_terminal_v2' and
            profile.get('rear_task_assist') is False and profile.get('rear_owner_projection') is False,
            'requires exact0d0 gain10/current448 no-rear-assist runtime')
    pointer,records = manifest['result'],[]
    for value in (started['checkpoint'],pointer['checkpoint']):
        cp = Path(value); cp = cp if cp.is_absolute() else common.ROOT/cp
        sidecar = cp.with_name(cp.stem+'_manifest.json'); meta = json.loads(sidecar.read_text())
        require(meta.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2' and
                meta.get('runtime_contract') == runtime and meta.get('checkpoint_sha256') == common.sha(cp) and
                Path(meta.get('checkpoint','')).resolve() == cp.resolve() and
                meta.get('rollout_empty') is True and meta.get('save_load_round_trip') is True and
                meta.get('front_FL_assist') is True and meta.get('rear_task_assists') is False,
                'sealed compatible gain10 checkpoint metadata/hash mismatch')
        records.append((cp,sidecar,meta))
    require(records[0][0].name == SOURCE_NAME and common.sha(records[0][0]) == SOURCE_SHA and
            common.sha(records[0][1]) == SOURCE_MANIFEST_SHA, 'not the exact published gain10 source CP')
    require(common.sha(records[1][0]) == pointer['checkpoint_sha256'] and
            common.sha(records[1][1]) == pointer['manifest_sha256'] and
            Path(records[1][2]['source_run']).resolve() == run.resolve(), 'sealed result pointer/source binding mismatch')
    accounting = validate_pair(records[0][2],records[1][2])
    receipt = records[0][2][LEDGER_KEY][0]['receipt']; receipt_path = Path(receipt['path'])
    require(common.sha(receipt_path) == receipt['sha256'], 'coordinate receipt changed')
    proof = json.loads(receipt_path.read_text())
    require(proof.get('status') == 'PASS' and proof.get('flags') and
            all(value is True for value in proof['flags'].values()) and
            proof.get('source_gain') == [1.]*12 and proof.get('target_gain') == GAIN,
            'coordinate migration receipt lacks passed actual function/Adam checks')
    return records,accounting


def validate_gain_rows(rows):
    evidence = previous.validate_row_evidence(rows); largest = 0.
    for r in rows:
        p = r['policy']
        require(p.get(GAIN_KEY) == GAIN and p.get('rho') == .9 and p.get('extra_model_forwards') == 0,
                'request gain/one-HISTORY model evidence differs')
        for j in range(12):
            prior,local = p['prior_raw_mean_full12'][j],p['local_raw_mean_delta_full12'][j]
            center = p['history_center_full12'][j]
            expected = (1-p['rho'])*(prior+local)+p['rho']*center
            error = max(abs(p['combined_raw_mean_full12'][j]-(prior+local)),
                abs(p['conditional_mean_full12'][j]-expected),
                abs(p['applied_local_raw_mean_delta_full12'][j]-local),
                abs(p['selected_tanh_full12'][j]-math.tanh(p['selected_raw_full12'][j])))
            require(math.isfinite(error) and error <= 2e-5, 'actual effective raw mean/HISTORY/tanh arithmetic differs')
            largest = max(largest,error)
    return dict(**evidence,gain_full12=GAIN,maximum_raw_HISTORY_tanh_error=largest,
        local_mean_semantics='already gain-applied raw delta before one HISTORY; never multiply logged local delta by10 again')


def minibatch_evidence(run,rows,update,accounting):
    path = run/'rollouts'/f"likelihood_{accounting['absolute_update']:04d}.json"
    batches = json.loads(path.read_text())['minibatches']; seen = Counter()
    expected_steps = accounting['epochs']*accounting['minibatches']; n = accounting['rows']
    require(len(batches) == update['optimizer_steps'] == expected_steps, 'actual optimizer minibatch count differs')
    for batch in batches:
        ids,logps = batch['rollout_flat_indices'],batch['old_log_probability']
        require(len(ids) == len(logps) == n//accounting['minibatches'], 'actual minibatch width differs')
        for identity,logp in zip(ids,logps):
            require(len(identity) == 1 and type(identity[0]) is int and 0 <= identity[0] < n,
                    'invalid actual storage identity')
            index = identity[0]
            require(abs(float(logp)-rows[index]['old_logp']) <= 1e-5, 'actual minibatch old likelihood mismatch')
            seen[index] += 1
    require(set(seen) == set(range(n)) and set(seen.values()) == {accounting['epochs']},
            'actual rows not exposed exactly once per epoch')
    return dict(unique_actual_PPO_rows=n,optimizer_minibatches=len(batches),row_exposures=sum(seen.values()),
                phase_rows=dict(Counter(r['phase'] for r in rows)))


def analyze(run=RUN, *, isaac_stopped=False):
    require(isaac_stopped, 'Explicit Isaac exit required before sealed reads or tensor analysis')
    records,accounting = sealed_inputs(run)
    versions = records[0][2]['runtime_contract']['local_runtime_versions']
    require(Path(sys.executable).resolve() == Path(versions['python_executable']).resolve() and
            sys.version.split()[0] == versions['python'], 'analysis must use the sealed Python runtime')
    require(all(importlib.metadata.version(name) == version for name,version in versions['distributions'].items()),
            'installed distribution versions differ from sealed runtime')
    n,offset = accounting['rows'],accounting['update_offset']
    rows,updates,prefix = common.collect(run,1,update_offset=offset); update = updates[0]
    require(len(rows) == n and update['counts'] == records[1][2]['counts'] and
            prefix == accounting['delta']['prefix_decisions'] and
            update['optimizer_steps'] == accounting['delta']['local_optimizer_steps'],
            'actual storage/update/prefix counts disagree with sealed checkpoint deltas')
    runtime = records[0][2]['runtime_contract']; previous.validate_loaded_source(runtime)
    for relative in ('src/wlr50_clean/ppo/semantic_rr_mean_coordinates.py',
                     'src/wlr50_clean/ppo/semantic_rr_capture_local.py',
                     'src/wlr50_clean/ppo/semantic_rr_capture_local_task.py',
                     'configs/ppo_rr_capture_first_cp225280_v1/local_training.json'):
        require(common.sha(common.ROOT/relative) == runtime['files'].get(relative),
                'current gain10 source/config differs from sealed runtime: '+relative)
    groups = update7.physical_cohorts(rows)
    eligible_air = groups['request_eligible_AIR_inside_top']
    groups['eligible_AIR_actual_gap_decreasing'] = [r for r in eligible_air
        if not r['after']['ground_contact'] and r['before']['gap_m']-r['after']['gap_m'] > 1e-6]
    groups['eligible_AIR_gap_not_decreasing_or_ground'] = [r for r in eligible_air
        if r['after']['ground_contact'] or r['before']['gap_m']-r['after']['gap_m'] <= 1e-6]
    checks = dict(raw_likelihood_history=validate_gain_rows(rows),
        execution=previous.validate_execution_rows(common.json_rows(run/'decisions.jsonl'),expected_count=n),
        optimizer_storage_exposure=minibatch_evidence(run,rows,update,accounting))
    # No Torch/model/Isaac import above; current actor is built from actual gain10 config below.
    import torch
    torch.set_num_threads(1); sys.path.insert(0,str(common.ROOT/'src'))
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import state_hash
    rollout = run/'rollouts'/f"rollout_{accounting['absolute_update']:04d}.pt"
    storage = torch.load(rollout,map_location='cpu',weights_only=False)
    obs = storage['observations']['policy'].reshape(n,448)
    comparisons = dict(observations=(obs,[r['observation'] for r in rows]),
        raw=(storage['actions'].reshape(n,12),[r['policy']['selected_raw_full12'] for r in rows]),
        old_logp=(storage['actions_log_prob'].reshape(n),[r['old_logp'] for r in rows]),
        mean=(storage['distribution_params'][0].reshape(n,12),[r['policy']['conditional_mean_full12'] for r in rows]),
        std=(storage['distribution_params'][1].reshape(n,12),[r['policy']['active_conditional_std_full12'] for r in rows]),
        reward=(storage['rewards'].reshape(n),[r['local_reward']['reward'] for r in rows]),
        done=(storage['dones'].reshape(n).float(),[float(r['local_reward']['terminated']) for r in rows]))
    for key,(actual,expected) in comparisons.items():
        checks[key] = common.close_tensor(torch,actual,torch.tensor(expected,dtype=actual.dtype),key)
    mean,std = storage['distribution_params']
    logp = torch.distributions.Normal(mean,std).log_prob(storage['actions']).sum(-1).reshape(n)
    checks['Gaussian_logp'] = common.close_tensor(torch,logp,storage['actions_log_prob'].reshape(n),'logp',1e-4)
    gae = (storage['returns']-storage['values']).reshape(n)
    checks['normalized_GAE'] = common.close_tensor(torch,(gae-gae.mean())/(gae.std()+1e-8),storage['advantages'].reshape(n),'GAE')
    for i,r in enumerate(rows):
        for field,key in (('reward','rewards'),('return','returns'),('value','values'),('advantage','advantages')):
            r[field] = float(storage[key].reshape(n)[i])
        r['raw_gae'] = float(gae[i])
    outputs,receipts = [],[]
    for cp,sidecar,meta in records:
        data = torch.load(cp,map_location='cpu',weights_only=False)
        require(all(meta.get(k) == v for k,v in data['infos'].items()) and
                all(state_hash(data[k]) == h for k,h in meta['state_hashes'].items()) and
                data.get('iter') == meta['counts']['local_ppo_updates'], 'embedded metadata/state/iteration mismatch')
        actual_lrs = [group['lr'] for group in data['optimizer_state_dict']['param_groups']]
        require(actual_lrs and all(lr == meta['learning_rate'] for lr in actual_lrs), 'actual Adam LR differs from sidecar')
        cfg = dict(meta['runner_config']['actor']); cfg.pop('class_name')
        actor = SemanticRRCaptureLocalHistoryMLPModel({'policy':obs[:1]},{'actor':['policy']},'actor',12,**cfg)
        require(list(actor.local_mean_coordinate_gain_full12) == GAIN, 'constructed actor lost gain10')
        actor.load_state_dict(data['actor_state_dict'],strict=True); actor.eval()
        with torch.inference_mode():
            mu = actor({'policy':obs},stochastic_output=False).clone(); ev = actor._last_forward_evidence
            outputs.append(dict(mean=mu,local=ev['local_raw_mean_delta'].clone(),std=ev['head_log_std'].exp().clone()))
        actor.assert_frozen_state()
        receipts.append(dict(path=str(cp),sha256=common.sha(cp),manifest_sha256=common.sha(sidecar),counts=meta['counts'],
            LR=meta['learning_rate'],actual_Adam_group_LRs=actual_lrs,gain_full12=GAIN,
            prior_sha256=actor.expected_prior_state_sha256,state_hashes=meta['state_hashes']))
        del actor,data
    require(receipts[0]['prior_sha256'] == receipts[1]['prior_sha256'], 'frozen prior drift')
    for key,index in (('mean',0),('std',1)):
        checks['loaded_collection_'+key] = common.close_tensor(torch,outputs[0][key],storage['distribution_params'][index].reshape(n,12),key,1e-4)
    checks['loaded_collection_effective_local'] = common.close_tensor(torch,outputs[0]['local'],
        torch.tensor([r['policy']['local_raw_mean_delta_full12'] for r in rows]),'effective local',1e-4)
    fixed = {}
    for label,group in groups.items():
        indices = [r['index'] for r in group]
        fixed[label] = dict(n=len(indices),RR={joint:{field:dict(
            before=common.stats(outputs[0][field][indices,ch].tolist()),
            after=common.stats(outputs[1][field][indices,ch].tolist()),
            shift=common.stats((outputs[1][field][indices,ch]-outputs[0][field][indices,ch]).tolist()))
            for field in ('mean','local','std')} for joint,ch in common.RR.items()})
    return dict(schema='wlr50_clean.rr_update8_gain10_learning_signal.v1',run=str(run),
        counts=dict(new_PPO_decisions=n,prefix_credit0=prefix,new_PPO_updates=accounting['delta']['local_ppo_updates'],
            new_PPO_Adam_steps=accounting['delta']['local_optimizer_steps'],inherited_AUX_optimizer_steps=64,
            new_AUX_optimizer_steps=0,actual_count_deltas=accounting['delta']),
        checkpoints=receipts,verification=checks,gain_full12=GAIN,
        unchanged_coordinate_migrations=records[0][2][LEDGER_KEY],
        source_rollout=dict(path=str(rollout),sha256=common.sha(rollout)),
        source_learning_audit_files={p.name:common.sha(p) for p in (run/'run_manifest.json',run/'run_manifest.started.json',
            run/'decisions.jsonl',run/'updates.jsonl',run/'rollouts'/f"likelihood_{accounting['absolute_update']:04d}.json")},
        physical_events=update7.contact_summary(rows,groups),fixed_input_RR=fixed,
        fixed_input_all12=[dict(channel=j,mean_shift=common.stats((outputs[1]['mean'][:,j]-outputs[0]['mean'][:,j]).tolist()),
            std_before=common.stats(outputs[0]['std'][:,j].tolist()),std_after=common.stats(outputs[1]['std'][:,j].tolist())) for j in range(12)],
        cohort_GAE_and_response={k:common.metric_summary(v) for k,v in groups.items()},
        knee_headroom_and_actual_response=previous.knee_response(rows),
        channel_KL=common.channel_kl_report(run,rows,updates,update_offset=offset),
        actual_source_tracking=previous.tracking_summary(rows),optimizer_receipt=update,
        limits=['Only the declared complete512 block; counts are derived and cross-checked, not inferred from filename.',
            'Local deltas are gain-applied raw means before the one existing HISTORY, not parameter-coordinate rows.',
            'Before-state cohorts identify input states; endpoint/drop cohorts are outcomes, not causal labels.',
            'Raw hip-negative/knee-positive is a candidate direction, not a universal successful target.',
            'Stored returns-values and normalized advantages checked; independent tail-bootstrap recurrence not claimed.',
            'Fixed-input mean/std movement and KL are not deterministic physical success or gradient causal attribution.',
            'AUX64 and coordinate publication are inherited; no new AUX, optimizer, simulation, or hidden teacher ran.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-stopped',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); target=args.output.resolve()
    require(target.is_relative_to(common.OUTPUT) and not target.exists(), 'write only a NEW isolated outputs report')
    result=analyze(isaac_stopped=args.isaac_stopped)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(target),counts=result['counts'],events=result['physical_events'])))


if __name__ == '__main__':main()
