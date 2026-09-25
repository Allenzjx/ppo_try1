"""Small sealed update7 review: AUX64 parent -> one fresh512 PPO update.

Imports are stdlib only. Tensor work requires explicit Isaac exit AND a sealed
COMPLETE run; no optimizer, checkpoint write, simulation or success assumption.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import analyze_rr_learning_signal as common
import analyze_rr_learning_signal_update6 as previous

RUN = common.ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_after_aux64_fresh512_0ff03ea'
HEAD = '0ff03eafeeb75ba8505a99478cea2a6243cf93f5'


def validate_pair(before, after):
    keys = ('local_policy_decisions', 'local_ppo_updates', 'local_optimizer_steps',
            'task_v2_policy_decisions', 'task_v2_ppo_updates', 'auxiliary_updates')
    for meta, expected in ((before, (3072,6,120,1024,2,64)),
                           (after, (3584,7,140,1536,3,64))):
        if tuple(meta['counts'].get(k) for k in keys) != expected:
            raise ValueError('requires update6 to7: +512/+1/+20; AUX remains64')
    old, new = before.get('local_auxiliary_events'), after.get('local_auxiliary_events')
    if (not isinstance(old, list) or not old or old != new or
            any(type(e.get('actual_optimizer_steps')) is not int or
                e['actual_optimizer_steps'] <= 0 for e in old) or
            sum(e['actual_optimizer_steps'] for e in old) != 64):
        raise ValueError('AUX64 training ledger must remain unchanged during fresh PPO')


def sealed_inputs(run):
    if run.resolve() != RUN.resolve():
        raise ValueError('only the declared update7 run is in scope')
    path = run/'run_manifest.json'
    if not path.is_file():
        raise ValueError('update7 is not sealed COMPLETE yet')
    manifest = json.loads(path.read_text())
    if manifest.get('lifecycle') != 'COMPLETE' or manifest.get('mode') != 'train':
        raise ValueError('requires sealed COMPLETE update7 training block')
    started = json.loads((run/'run_manifest.started.json').read_text())
    runtime = started['runtime_contract']
    profile = runtime.get('local_contract', {})
    if (runtime.get('source_git_commit') != HEAD or
            runtime.get('experiment_id') != 'ppo_rr_capture_first_cp225280_v1' or
            profile.get('source_tracking_owner_revision') != previous.REVISION or
            profile.get('observation_dimension') != 448 or profile.get('rollout_length') != 512 or
            profile.get('capture_source_dispatch') != 'rr_local_defer_p09_late_and_new_p12_until_terminal_v2'):
        raise ValueError('requires exact0ff runtime and corrected tracking448/512')
    pointer, records = manifest['result'], []
    for value in (started['checkpoint'], pointer['checkpoint']):
        cp = Path(value)
        cp = cp if cp.is_absolute() else common.ROOT/cp
        sidecar = cp.with_name(cp.stem+'_manifest.json')
        meta = json.loads(sidecar.read_text())
        if (meta.get('schema') != 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2' or
                meta.get('runtime_contract') != runtime or meta.get('checkpoint_sha256') != common.sha(cp) or
                meta.get('rollout_empty') is not True or meta.get('save_load_round_trip') is not True or
                meta.get('runner_config', {}).get('actor', {}).get('observation_layout') != previous.OBSERVATION_LAYOUT):
            raise ValueError('sealed compatible448 checkpoint metadata/hash mismatch')
        records.append((cp, sidecar, meta))
    if (common.sha(records[1][0]) != pointer['checkpoint_sha256'] or
            common.sha(records[1][1]) != pointer['manifest_sha256']):
        raise ValueError('sealed pointer hashes differ')
    validate_pair(records[0][2], records[1][2])
    return records


def physical_cohorts(rows):
    """Cohorts describe actual states, never label a whole episode successful."""
    groups = {name: [] for name in ('all512', 'request_AIR', 'request_eligible_AIR_inside_top',
        'request_TOP_bearing', 'endpoint_TOP_bearing', 'first_TOP_transition',
        'TOP_reacquisition', 'bearing_drop_transition', 'after_prior_TOP_now_unborne', 'local_success')}
    seen_top = {}
    for r in rows:
        b, a, episode = r['before'], r['after'], r['episode']
        seen = seen_top.get(episode, False) or b['current_top_contact']
        groups['all512'].append(r)
        if b['free_air']: groups['request_AIR'].append(r)
        if b['free_air'] and b['within_top_xy'] and b['current_attempt_capture_eligible']:
            groups['request_eligible_AIR_inside_top'].append(r)
        if b['current_top_bearing']: groups['request_TOP_bearing'].append(r)
        if a['current_top_bearing']: groups['endpoint_TOP_bearing'].append(r)
        if a['current_top_contact'] and not b['current_top_contact']:
            groups['TOP_reacquisition' if seen else 'first_TOP_transition'].append(r)
        if b['current_top_bearing'] and not a['current_top_bearing']:
            groups['bearing_drop_transition'].append(r)
        if seen and not a['current_top_bearing']: groups['after_prior_TOP_now_unborne'].append(r)
        if r['local_success']:
            if not (a['current_top_contact'] and a['current_top_bearing'] and
                    a['current_attempt_capture_eligible'] and not a['ground_contact'] and
                    a['within_top_xy'] and a['placed'] and a['crossed'] and r['hold_s'] >= .5):
                raise ValueError('reported local success lacks current eligible TOP hold')
            groups['local_success'].append(r)
        seen_top[episode] = seen or a['current_top_contact']
    return groups


def contact_summary(rows, groups):
    episodes = []
    for ep in sorted({r['episode'] for r in rows}):
        actual = [r for r in rows if r['episode'] == ep]
        top = [r for r in actual if r['after']['current_top_contact']]
        episodes.append(dict(episode=ep, PPO_rows=len(actual),
            first_TOP_endpoint=top[0]['tick'] if top else None,
            TOP_endpoints=len(top), bearing_endpoints=sum(r['after']['current_top_bearing'] for r in actual),
            maximum_native_observer_hold_s=max(r['hold_s'] for r in actual),
            success_endpoint_ticks=[r['tick'] for r in actual if r['local_success']],
            last_tick=actual[-1]['tick'], last_local_terminated=actual[-1]['local_reward']['terminated'],
            last_gap_mm=1000*actual[-1]['after']['gap_m']))
    return dict(episodes=episodes, counts={k:len(v) for k,v in groups.items()},
        evidence_scope='15Hz endpoints plus stored native-observer hold; no independent120Hz scan; no TOP is not proof that no subdecision contact occurred.')


def minibatch_evidence(run, rows, update):
    batches = json.loads((run/'rollouts/likelihood_0007.json').read_text())['minibatches']
    if len(batches) != 20 or update['optimizer_steps'] != 20:
        raise ValueError('requires20 actual optimizer minibatches')
    seen = Counter()
    for batch in batches:
        ids, logps = batch['rollout_flat_indices'], batch['old_log_probability']
        if len(ids) != 128 or len(logps) != 128:
            raise ValueError('requires128 actual rows per minibatch')
        for identity, logp in zip(ids, logps):
            if len(identity) != 1 or type(identity[0]) is not int or not 0 <= identity[0] < 512:
                raise ValueError('invalid actual storage identity')
            index = identity[0]
            if abs(float(logp)-rows[index]['old_logp']) > 1e-5:
                raise ValueError('actual minibatch old likelihood mismatch')
            seen[index] += 1
    if set(seen) != set(range(512)) or set(seen.values()) != {5}:
        raise ValueError('512 rows not each exposed five times')
    return dict(unique_actual_PPO_rows=512, optimizer_minibatches=20, row_exposures=2560,
                phase_rows=dict(Counter(r['phase'] for r in rows)),
                phase_row_exposures=dict(Counter(r['phase'] for r in rows for _ in range(seen[r['index']]))))


def analyze(run=RUN, *, isaac_stopped=False):
    if not isaac_stopped:
        raise ValueError('Explicit Isaac exit required before tensor analysis')
    records = sealed_inputs(run)
    rows, updates, prefix = common.collect(run, 1, update_offset=6)
    update = updates[0]
    if update['counts'] != records[1][2]['counts']:
        raise ValueError('actual update receipt differs from saved counts')
    groups = physical_cohorts(rows)
    previous.validate_loaded_source(records[0][2]['runtime_contract'])
    checks = dict(raw_likelihood_history=previous.validate_row_evidence(rows),
        execution=previous.validate_execution_rows(common.json_rows(run/'decisions.jsonl')),
        optimizer_storage_exposure=minibatch_evidence(run, rows, update))
    # All sealed/runtime/accounting guards precede these imports.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(common.ROOT/'src'))
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import state_hash
    rollout = run/'rollouts/rollout_0007.pt'
    storage = torch.load(rollout, map_location='cpu', weights_only=False)
    obs = storage['observations']['policy'].reshape(512,448)
    comparisons = dict(observations=(obs,[r['observation'] for r in rows]),
        raw=(storage['actions'].reshape(512,12),[r['policy']['selected_raw_full12'] for r in rows]),
        old_logp=(storage['actions_log_prob'].reshape(512),[r['old_logp'] for r in rows]),
        mean=(storage['distribution_params'][0].reshape(512,12),[r['policy']['conditional_mean_full12'] for r in rows]),
        std=(storage['distribution_params'][1].reshape(512,12),[r['policy']['active_conditional_std_full12'] for r in rows]),
        reward=(storage['rewards'].reshape(512),[r['local_reward']['reward'] for r in rows]),
        done=(storage['dones'].reshape(512).float(),[float(r['local_reward']['terminated']) for r in rows]))
    for key,(actual,expected) in comparisons.items():
        checks[key] = common.close_tensor(torch,actual,torch.tensor(expected,dtype=actual.dtype),key)
    mean,std = storage['distribution_params']
    logp = torch.distributions.Normal(mean,std).log_prob(storage['actions']).sum(-1).reshape(512)
    checks['Gaussian_logp'] = common.close_tensor(torch,logp,storage['actions_log_prob'].reshape(512),'logp',1e-4)
    gae = (storage['returns']-storage['values']).reshape(512)
    checks['normalized_GAE'] = common.close_tensor(torch,(gae-gae.mean())/(gae.std()+1e-8),storage['advantages'].reshape(512),'GAE')
    for i,r in enumerate(rows):
        for field,key in (('reward','rewards'),('return','returns'),('value','values'),('advantage','advantages')):
            r[field] = float(storage[key].reshape(512)[i])
        r['raw_gae'] = float(gae[i])
    outputs, receipts = [], []
    for cp,sidecar,meta in records:
        data = torch.load(cp,map_location='cpu',weights_only=False)
        if any(meta.get(k) != v for k,v in data['infos'].items()) or any(
                state_hash(data[k]) != h for k,h in meta['state_hashes'].items()):
            raise ValueError('embedded metadata or serialized actor/critic/Adam mismatch')
        cfg = dict(meta['runner_config']['actor']); cfg.pop('class_name')
        actor = SemanticRRCaptureLocalHistoryMLPModel({'policy':obs[:1]}, {'actor':['policy']},'actor',12,**cfg)
        actor.load_state_dict(data['actor_state_dict'],strict=True); actor.eval()
        with torch.inference_mode():
            mu = actor({'policy':obs},stochastic_output=False).clone()
            ev = actor._last_forward_evidence
            outputs.append(dict(mean=mu,local=ev['local_raw_mean_delta'].clone(),std=ev['head_log_std'].exp().clone()))
        actor.assert_frozen_state()
        receipts.append(dict(path=str(cp),sha256=common.sha(cp),manifest_sha256=common.sha(sidecar),
            counts=meta['counts'],LR=meta['learning_rate'],prior_sha256=actor.expected_prior_state_sha256))
        del actor,data
    if receipts[0]['prior_sha256'] != receipts[1]['prior_sha256']:
        raise ValueError('frozen prior drift')
    for key,index in (('mean',0),('std',1)):
        checks['loaded_collection_'+key] = common.close_tensor(torch,outputs[0][key],storage['distribution_params'][index].reshape(512,12),key,1e-4)
    fixed = {}
    for label,group in groups.items():
        indices = [r['index'] for r in group]
        fixed[label] = dict(n=len(indices), RR={joint:{field:dict(
            before=common.stats(outputs[0][field][indices,ch].tolist()),
            after=common.stats(outputs[1][field][indices,ch].tolist()),
            shift=common.stats((outputs[1][field][indices,ch]-outputs[0][field][indices,ch]).tolist()))
            for field in ('mean','local','std')} for joint,ch in common.RR.items()})
    return dict(schema='wlr50_clean.rr_update7_after_aux64_learning_signal.v1',run=str(run),
        counts=dict(new_PPO_decisions=512,prefix_credit0=prefix,new_PPO_updates=1,new_PPO_Adam_steps=20,
                    inherited_AUX_optimizer_steps=64,new_AUX_optimizer_steps=0),
        checkpoints=receipts,verification=checks,source_rollout=dict(path=str(rollout),sha256=common.sha(rollout)),
        physical_events=contact_summary(rows,groups),fixed_input_RR=fixed,
        cohort_GAE_and_response={k:common.metric_summary(v) for k,v in groups.items()},
        knee_headroom_and_actual_response=previous.knee_response(rows),
        channel_KL=common.channel_kl_report(run,rows,updates,update_offset=6),
        actual_source_tracking=previous.tracking_summary(rows),optimizer_receipt=update,
        limits=['Cohorts overlap; all512 is the unique on-policy total. Phase uses saved step_info.phase_id.',
            'Before-state cohorts are actor-input conditions; endpoint/drop cohorts describe outcomes, not causal action labels.',
            'Stored returns-values and normalization checked; no independent tail-bootstrap recurrence claimed.',
            'Fixed-input mean changes are not physical replay, deterministic success or clipped-gradient attribution.',
            'AUX64 is inherited training lineage, not extra PPO samples or realtime rear assistance. No optimizer or physical process was run.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-stopped',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(common.OUTPUT) or target.exists():
        raise ValueError('write only a NEW isolated outputs report')
    result = analyze(isaac_stopped=args.isaac_stopped)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=str(target),counts=result['counts'],events=result['physical_events'])))


if __name__ == '__main__':
    main()
