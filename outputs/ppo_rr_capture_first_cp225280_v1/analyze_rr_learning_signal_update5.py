"""One sealed v2/448 RR update: fixed-input means, GAE and tracking evidence.

Shares the completed-block analyzer's bucket/KL/log-likelihood checks. Import is
stdlib-only. Tensor work requires explicit --isaac-stopped AND COMPLETE run.
No PPO/AUX update, control change, physical replay or checkpoint publication.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import analyze_rr_learning_signal as common

RUN = common.ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_v2_fresh512_5f8487b'
HEAD = '5f8487b76f27eb165a329a0b6f3096f54ebb4d48'
OUTPUT = common.OUTPUT


def knee_signal(rows):
    """Collection score diagnostic, not the actual clipped multi-epoch gradient."""
    groups = defaultdict(list)
    for r in rows:
        p = r['policy']
        residual = p['selected_raw_full12'][7] - p['conditional_mean_full12'][7]
        sign = 'above_collection_mean' if residual > 0 else 'below_collection_mean' if residual < 0 else 'exact_mean'
        groups[sign].append(r)
    result = {}
    for label, group in groups.items():
        result[label] = dict(n=len(group), reward=common.stats(r['reward'] for r in group),
            raw_gae=common.stats(r['raw_gae'] for r in group),
            normalized_advantage=common.stats(r['advantage'] for r in group),
            returns=common.stats(r['return'] for r in group),
            mean_score_times_advantage=common.stats(
                r['advantage'] * (r['policy']['selected_raw_full12'][7] -
                    r['policy']['conditional_mean_full12'][7]) /
                r['policy']['active_conditional_std_full12'][7]**2 for r in group),
            potential_shaping=common.stats(r['local_reward']['potential_shaping'] for r in group),
            terminal_event=common.stats(r['local_reward']['terminal_event'] for r in group),
            gap_descent_mm=common.stats(1000*(r['before']['gap_m']-r['after']['gap_m']) for r in group),
            actual_knee_delta_deg=common.stats(r['after']['actual_rr_hip_knee_deg'][1]-
                r['before']['actual_rr_hip_knee_deg'][1] for r in group))
    return dict(groups=result,
        meaning='Above mean is relative to the same collected conditional Gaussian, not positive absolute joint angle. Positive score-times-advantage suggests an unclipped mean-increase direction only; not causal action attribution or actual PPO gradient.')


def tracking_summary(rows):
    pending = [r for r in rows if any(
        x.get('pending_event_tick') == 648 and x.get('pending_event_consumed') is False
        for x in r['source_deferred'].get('layers', []))]
    tracking = [r for r in pending if 'rear_right_knee' in r['tracking'].get('tracking_servo_names', [])]
    frozen_bias = [r for r in pending if r not in tracking]
    def summarize(group):
        result = dict(n=len(group))
        for j, channel in common.RR.items():
            result[j] = dict(source_nominal_deg=common.stats(r['source_nominal'][channel] for r in group),
                mapped_N_deg=common.stats(r['native_nominal'][channel] for r in group),
                mapped_minus_source_deg=common.stats(r['native_nominal'][channel]-r['source_nominal'][channel] for r in group),
                controller_bias_deg=common.stats(r['controller_bias'][channel] for r in group),
                final_deg=common.stats(r['final'][channel] for r in group))
        return result
    proof_path = OUTPUT/'v2_episode1_tracking_owner_followup.json'
    proof = json.loads(proof_path.read_text())
    transitions, previous = [], None
    for r in rows:
        names = tuple(r['tracking'].get('tracking_servo_names', []))
        key = (r['episode'], names)
        if key != previous:
            transitions.append(dict(index=r['index'], episode=r['episode'], endpoint_tick=r['tick'],
                time_s=r['time_s'], tracking=list(names),
                source_RR=r['source_nominal'][6:8], mapped_N_RR=r['native_nominal'][6:8],
                final_RR=r['final'][6:8], actual_RR=r['after']['actual_rr_hip_knee_deg'],
                gap_m=r['after']['gap_m']))
        previous = key
    return dict(pending_all=summarize(pending), pending_RR_knee_tracking=summarize(tracking),
        pending_after_or_without_RR_tracking=summarize(frozen_bias), tracking_transitions=transitions,
        previous_source_owner_evidence=dict(path=str(proof_path), sha256=common.sha(proof_path),
            finding=proof['finding'], old_pending_offset=proof['old_pending_RR_mapper_offset'],
            new_after_stop_offset=proof['new_after_stop_RR_mapper_offset'],
            causal_limit=proof['causal_limit']),
        causal_limit='Measured mapping difference is a source/tracking contribution. Different policy/body/contact feedback also exists; this is not a paired physical intervention or evidence that the bug alone caused failure.')


def sealed_inputs(run):
    manifest_path = run/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('lifecycle') != 'COMPLETE' or manifest.get('mode') != 'train':
        raise ValueError('requires the sealed COMPLETE training block')
    started = json.loads((run/'run_manifest.started.json').read_text())
    if started['runtime_contract']['source_git_commit'] != HEAD:
        raise ValueError('this bounded review accepts only the declared 5f source run')
    pointer = manifest['result']
    before_path, after_path = Path(started['checkpoint']), Path(pointer['checkpoint'])
    before_path = before_path if before_path.is_absolute() else common.ROOT/before_path
    after_path = after_path if after_path.is_absolute() else common.ROOT/after_path
    records = []
    for path in (before_path, after_path):
        sidecar = path.with_name(path.stem+'_manifest.json')
        metadata = json.loads(sidecar.read_text())
        if (metadata['schema'] != 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2'
                or metadata['runtime_contract'] != started['runtime_contract']
                or metadata['runtime_contract']['local_contract']['observation_dimension'] != 448
                or metadata['checkpoint_sha256'] != common.sha(path)
                or metadata['rollout_empty'] is not True or metadata['save_load_round_trip'] is not True):
            raise ValueError('complete compatible448 checkpoint metadata/hash mismatch')
        records.append((path, sidecar, metadata))
    if common.sha(after_path) != pointer['checkpoint_sha256'] or common.sha(records[1][1]) != pointer['manifest_sha256']:
        raise ValueError('sealed run pointer mismatch')
    old, new = (r[2]['counts'] for r in records)
    if (new['local_ppo_updates'] - old['local_ppo_updates'] != 1
            or new['local_policy_decisions'] - old['local_policy_decisions'] != 512
            or new['auxiliary_updates'] != old['auxiliary_updates']):
        raise ValueError('bounded review requires one real512 PPO update, no AUX')
    return records, old['local_ppo_updates']


def analyze(run=RUN, *, isaac_stopped=False):
    if not isaac_stopped:
        raise ValueError('Explicit Isaac exit confirmation is required before tensor analysis')
    records, offset = sealed_inputs(run)
    rows, updates, prefix = common.collect(run, 1, update_offset=offset)
    if updates[0]['counts'] != records[1][2]['counts']:
        raise ValueError('saved update counters differ from sealed checkpoint')
    # No Torch/model import until all guards and completed-update checks above.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(common.ROOT/'src'))
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import state_hash
    rollout = run/'rollouts'/f'rollout_{offset+1:04d}.pt'
    storage = torch.load(rollout, map_location='cpu', weights_only=False)
    obs = storage['observations']['policy'].reshape(512, 448)
    checks = {}
    comparisons = dict(
        observations=(obs, [r['observation'] for r in rows]),
        raw=(storage['actions'].reshape(512,12), [r['policy']['selected_raw_full12'] for r in rows]),
        old_logp=(storage['actions_log_prob'].reshape(512), [r['old_logp'] for r in rows]),
        stored_mean=(storage['distribution_params'][0].reshape(512,12), [r['policy']['conditional_mean_full12'] for r in rows]),
        stored_std=(storage['distribution_params'][1].reshape(512,12), [r['policy']['active_conditional_std_full12'] for r in rows]),
        reward=(storage['rewards'].reshape(512), [r['local_reward']['reward'] for r in rows]),
        dones=(storage['dones'].reshape(512).float(), [float(r['local_reward']['terminated']) for r in rows]))
    for name,(actual,expected) in comparisons.items():
        checks[name] = common.close_tensor(torch,actual,torch.tensor(expected,dtype=actual.dtype),name)
    means, sigmas = storage['distribution_params']
    logp = torch.distributions.Normal(means,sigmas).log_prob(storage['actions']).sum(-1).reshape(512)
    checks['gaussian_logp'] = common.close_tensor(torch,logp,storage['actions_log_prob'].reshape(512),'logp',1e-4)
    raw_gae = (storage['returns']-storage['values']).reshape(512)
    normalized = (raw_gae-raw_gae.mean())/(raw_gae.std()+1e-8)
    checks['normalized_GAE'] = common.close_tensor(torch,normalized,storage['advantages'].reshape(512),'GAE')
    for i,r in enumerate(rows):
        for field,key in (('reward','rewards'),('return','returns'),('value','values'),('advantage','advantages')):
            r[field] = float(storage[key].reshape(512)[i])
        r['raw_gae'] = float(raw_gae[i])
    outputs, checkpoint_receipts = [], []
    for path,sidecar,meta in records:
        data = torch.load(path,map_location='cpu',weights_only=False)
        if any(meta.get(k) != v for k,v in data['infos'].items()):
            raise ValueError('embedded checkpoint metadata differs')
        if any(state_hash(data[k]) != h for k,h in meta['state_hashes'].items()):
            raise ValueError('serialized actor/critic/Adam hash differs')
        cfg = dict(meta['runner_config']['actor'])
        cfg.pop('class_name')
        if cfg['observation_layout'] != 'role439_rr_capture_local_v2':
            raise ValueError('same-state review must not adapt/migrate an old layout')
        actor = SemanticRRCaptureLocalHistoryMLPModel({'policy':obs[:1]},
            {'actor':['policy']},'actor',12,**cfg)
        actor.load_state_dict(data['actor_state_dict'],strict=True)
        actor.eval()
        with torch.inference_mode():
            mu = actor({'policy':obs},stochastic_output=False).clone()
            evidence = actor._last_forward_evidence
            outputs.append(dict(mean=mu, local=evidence['local_raw_mean_delta'].clone(),
                                std=evidence['head_log_std'].exp().clone()))
        actor.assert_frozen_state()
        checkpoint_receipts.append(dict(path=str(path),sha256=common.sha(path),
            manifest_sha256=common.sha(sidecar),counts=meta['counts'],LR=meta['learning_rate'],
            prior_sha256=actor.expected_prior_state_sha256))
        del actor,data
    if checkpoint_receipts[0]['prior_sha256'] != checkpoint_receipts[1]['prior_sha256']:
        raise ValueError('frozen prior drift')
    checks['loaded_collection_mean'] = common.close_tensor(
        torch,outputs[0]['mean'],storage['distribution_params'][0].reshape(512,12),'collection mean',1e-4)
    groups = defaultdict(list)
    for r in rows:
        groups[r['bucket']].append(r)
    cohorts = {'all512_same_inputs':rows, **groups}
    cohorts['eligible_AIR_inside_top'] = [r for r in rows if r['before']['free_air']
        and r['before']['within_top_xy'] and r['before']['current_attempt_capture_eligible']]
    fixed = {}
    for label,group in cohorts.items():
        if not group:
            fixed[label] = dict(n=0)
            continue
        indices = [r['index'] for r in group]
        item = dict(n=len(indices),RR={})
        for joint,ch in common.RR.items():
            shift = outputs[1]['mean'][indices,ch]-outputs[0]['mean'][indices,ch]
            item['RR'][joint] = dict(
                conditional_mean_before=common.stats(outputs[0]['mean'][indices,ch].tolist()),
                conditional_mean_after=common.stats(outputs[1]['mean'][indices,ch].tolist()),
                same_input_mean_shift=common.stats(shift.tolist()),
                local_head_mean_before=common.stats(outputs[0]['local'][indices,ch].tolist()),
                local_head_mean_after=common.stats(outputs[1]['local'][indices,ch].tolist()),
                std_after=common.stats(outputs[1]['std'][indices,ch].tolist()))
        fixed[label] = item
    return dict(schema='wlr50_clean.rr_completed_block_learning_signal.v2_single448',
        run=str(run),counts=dict(on_policy=512,prefix_credit0=prefix,updates=1,
            optimizer_steps=updates[0]['optimizer_steps'],AUX=0,source_counts=records[0][2]['counts'],
            destination_counts=records[1][2]['counts']),
        verification=checks,checkpoints=checkpoint_receipts,
        source_rollout=dict(path=str(rollout),sha256=common.sha(rollout)),
        optimizer_receipt=updates[0],fixed_state_mean_changes=fixed,
        aggregate=common.metric_summary(rows),
        physical_buckets={k:common.metric_summary(v) for k,v in groups.items()},
        knee_GAE_signal={k:knee_signal(v) for k,v in cohorts.items()},
        channel_KL=common.channel_kl_report(run,rows,updates,update_offset=offset),
        pending_tracking=tracking_summary(rows),
        limits=['Same448 inputs from this actual512 rollout reused across pre/post networks; no fabricated447→448 feature.',
            'GAE/score associations are observational, not a causal intervention or exact clipped-PPO gradient.',
            'Native hold is logged evidence; endpoint buckets cannot reconstruct all120Hz contacts.',
            'Tracking defect provenance is measured separately; its exact physical failure contribution is unproven.',
            'No optimizer/update/model write/physics run; fixed-input mean progress is not task success.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--isaac-stopped',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run.resolve(),isaac_stopped=args.isaac_stopped)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()),'counts':result['counts'],
        'LR':result['checkpoints'][-1]['LR'],'fixed':result['fixed_state_mean_changes']['eligible_AIR_inside_top']}))


if __name__ == '__main__':
    main()
