"""Read-only update6 tensor analysis, to run ONLY after explicit Isaac exit.

Import/tests are standard-library only. The exact COMPLETE 1e10d39 block,
CP227840 source bytes, compatible448 pre/post sidecars and update6 must all
validate before any Torch/model import. No optimizer or simulation is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import analyze_rr_learning_signal as common
import analyze_rr_learning_signal_update5 as update5

RUN = common.ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_tracking_fixed512_1e10d39'
HEAD = '1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81'
REVISION = 'pending_source_tracking_inheritance_v1'
BEFORE_SHA = '81aa94a31039730c7ad87a45b53de22a12361047d324d4d5700b835d7cddbdb5'
OBSERVATION_LAYOUT = 'role439_rr_capture_local_v2'


def sealed_inputs(run):
    path = run/'run_manifest.json'
    if not path.is_file():
        raise ValueError('update6 is not sealed COMPLETE yet')
    manifest = json.loads(path.read_text())
    if manifest.get('lifecycle') != 'COMPLETE' or manifest.get('mode') != 'train':
        raise ValueError('requires sealed COMPLETE update6 training block')
    started = json.loads((run/'run_manifest.started.json').read_text())
    runtime = started['runtime_contract']
    profile = runtime.get('local_contract', {})
    if (runtime.get('source_git_commit') != HEAD
            or runtime.get('experiment_id') != 'ppo_rr_capture_first_cp225280_v1'
            or profile.get('source_tracking_owner_revision') != REVISION
            or profile.get('observation_dimension') != 448
            or profile.get('rollout_length') != 512
            or profile.get('capture_source_dispatch') != 'rr_local_defer_p09_late_and_new_p12_until_terminal_v2'):
        raise ValueError('requires exact1e runtime and corrected tracking448/512 contract')
    pointer, records = manifest['result'], []
    for value in (started['checkpoint'], pointer['checkpoint']):
        checkpoint = Path(value)
        checkpoint = checkpoint if checkpoint.is_absolute() else common.ROOT/checkpoint
        sidecar = checkpoint.with_name(checkpoint.stem+'_manifest.json')
        meta = json.loads(sidecar.read_text())
        if (meta.get('schema') != 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2'
                or meta.get('runtime_contract') != runtime
                or meta.get('checkpoint_sha256') != common.sha(checkpoint)
                or meta.get('rollout_empty') is not True
                or meta.get('save_load_round_trip') is not True
                or meta.get('runner_config', {}).get('actor', {}).get('observation_layout') != OBSERVATION_LAYOUT):
            raise ValueError('sealed compatible448 checkpoint metadata/hash mismatch')
        records.append((checkpoint, sidecar, meta))
    if records[0][2]['checkpoint_sha256'] != BEFORE_SHA:
        raise ValueError('source is not the exact rebound CP227840 checkpoint')
    if (common.sha(records[1][0]) != pointer['checkpoint_sha256']
            or common.sha(records[1][1]) != pointer['manifest_sha256']):
        raise ValueError('sealed pointer hash mismatch')
    keys = ('local_policy_decisions','local_ppo_updates','local_optimizer_steps',
            'task_v2_policy_decisions','task_v2_ppo_updates','auxiliary_updates')
    expected = ((2560,5,100,512,1,0),(3072,6,120,1024,2,0))
    for record, values in zip(records, expected):
        if tuple(record[2]['counts'].get(k) for k in keys) != values:
            raise ValueError('requires exact update5 to6:512 fresh decisions/20 Adam/no AUX')
    return records, 5


def validate_loaded_source(runtime):
    for relative in ('src/wlr50_clean/ppo/semantic_rr_capture_local_actor.py',
                     'src/wlr50_clean/ppo/semantic_training.py',
                     'src/wlr50_clean/ppo/semantic_history_actor.py',
                     'src/wlr50_clean/ppo/semantic_p05_capture_actor.py',
                     'src/wlr50_clean/ppo/semantic_rear_owner_actor.py'):
        if common.sha(common.ROOT/relative) != runtime.get('files',{}).get(relative):
            raise ValueError('analysis model implementation no longer matches sealed source: '+relative)


def success_cohorts(rows):
    if len(rows) != 512 or [r['index'] for r in rows] != list(range(512)):
        raise ValueError('requires512 exact ordered storage rows')
    first = [r for r in rows if r['episode'] == 1]
    if (len(first) != 41 or first != rows[:41] or first[-1]['tick'] != 8312
            or not first[-1]['local_success'] or first[-1]['hold_s'] < .5):
        raise ValueError('first successful41-row episode binding differs')
    m = first[-1]['after']
    if not (m['current_top_contact'] and m['current_top_bearing']
            and m['current_attempt_capture_eligible'] and not m['ground_contact']
            and m['within_top_xy'] and m['placed'] and m['crossed']):
        raise ValueError('first41 terminal is not current qualified TOP hold')
    return dict(all512=rows, first_successful_episode41=first,
        other471=rows[41:],
        first41_before_contact=[r for r in first if not r['after']['current_top_contact']],
        first41_contact_and_hold=[r for r in first if r['after']['current_top_contact']],
        eligible_AIR_inside_top=[r for r in rows if r['before']['free_air']
            and r['before']['within_top_xy'] and r['before']['current_attempt_capture_eligible']])


def validate_row_evidence(rows):
    for r in rows:
        p = r['policy']
        if (r['old_logp'] != p['selected_raw_log_probability']
                or p['sampling_draws'] != 1 or p['history_kernel_applications'] != 1
                or p['extra_random_draws'] != 0 or p.get('independent_diagnostic')
                or len(r['observation']) != 448):
            raise ValueError('actual raw likelihood/HISTORY/448 evidence mismatch')
    return dict(rows=len(rows),old_logp_equals_actual_sample_receipt=True,
                sampling_draws=1,HISTORY_applications=1,extra_draws=0)


def validate_minibatch_old_logp(run, rows, update):
    data = json.loads((run/'rollouts/likelihood_0006.json').read_text())
    batches = data['minibatches']
    if len(batches) != update['optimizer_steps'] or len(batches) != 20:
        raise ValueError('update6 must contain20 actual minibatches')
    exposures = Counter()
    for batch in batches:
        identities, old = batch['rollout_flat_indices'], batch['old_log_probability']
        if len(identities) != len(old) or len(old) != 128:
            raise ValueError('actual128 minibatch identity/logp mismatch')
        for indices, logp in zip(identities, old):
            if len(indices) != 1 or type(indices[0]) is not int or not 0 <= indices[0] < 512:
                raise ValueError('ambiguous actual storage identity')
            index = indices[0]
            if abs(float(logp)-rows[index]['old_logp']) > 1e-5:
                raise ValueError('minibatch old_logp differs from actual sampled row')
            exposures[index] += 1
    if set(exposures) != set(range(512)) or set(exposures.values()) != {5}:
        raise ValueError('actual storage exposure is not five epochs')
    return dict(rows=512,optimizer_minibatches=20,actual_row_exposures=sum(exposures.values()),
                exposures_per_row=5,old_logp_matches=True)


def validate_execution_rows(source_rows, expected_count=512):
    count = 0
    for row in source_rows:
        if row.get('kind') == 'frozen_prior_prefix':
            if row.get('PPO_credit') != 0:
                raise ValueError('prefix has PPO credit')
            continue
        info, policy = row['step_info'], row['policy_request']
        audit = info['actuator_target_effect_audit']
        if (row.get('kind') != 'activated_on_policy' or row.get('PPO_credit') != 1
                or policy['selected_raw_full12'] != info['raw_policy_action_full12']
                or audit['all12_policy_channels_unmodified_at_actuator'] is not True
                or audit['phase_mask_full12'] != [1]*12):
            raise ValueError('raw sample/issued action/all12 execution evidence differs')
        count += 1
    if count != expected_count:
        raise ValueError('execution receipt count differs from storage')
    return dict(rows=count,selected_raw_equals_issued=True,all12_policy_permission=True)


def knee_response(rows):
    result = {}
    for label, group in (('all',rows),
            ('headroom_clipped',[r for r in rows if 7 in r['headroom']['clipped_servo_indices']]),
            ('not_headroom_clipped',[r for r in rows if 7 not in r['headroom']['clipped_servo_indices']])):
        result[label] = dict(n=len(group),
            requested_deg=common.stats(r['headroom']['requested_policy_residual_full12'][7] for r in group),
            effective_deg=common.stats(r['headroom']['effective_policy_residual_full12'][7] for r in group),
            request_minus_effective_deg=common.stats(r['headroom']['requested_policy_residual_full12'][7]-
                r['headroom']['effective_policy_residual_full12'][7] for r in group),
            final_deg=common.stats(r['final'][7] for r in group),
            actual_deg=common.stats(r['after']['actual_rr_hip_knee_deg'][1] for r in group),
            actual_knee_delta_deg=common.stats(r['after']['actual_rr_hip_knee_deg'][1]-
                r['before']['actual_rr_hip_knee_deg'][1] for r in group),
            gap_descent_mm=common.stats(1000*(r['before']['gap_m']-r['after']['gap_m']) for r in group),
            positive_raw_gae=sum(r['raw_gae'] > 0 for r in group),
            negative_raw_gae=sum(r['raw_gae'] < 0 for r in group))
    return result


def tracking_summary(rows):
    """Current receipts only; no import/read of old tracking-defect evidence."""
    transitions, previous, pending = [], None, []
    for r in rows:
        tr = r['tracking']
        if not isinstance(tr.get('tracking_servo_names'),list):
            raise ValueError('actual source tracking receipt unavailable')
        layers = r['source_deferred'].get('layers')
        if not isinstance(layers,list):
            raise ValueError('current deferred source receipt unavailable')
        carrier = []
        for layer in layers:
            if layer.get('pending_event_tick') != 648 or layer.get('pending_event_consumed') is not False:
                continue
            names = layer.get('deferred_source_tracking_servo_names')
            if (layer.get('deferred_tracking_source') != 'previous_sample_before_unconsumed_late'
                    or not isinstance(names,list)):
                raise ValueError('current previous-source tracking inheritance evidence differs')
            carrier.append(tuple(names))
        if carrier:
            pending.append(r)
        actual = tuple(tr['tracking_servo_names'])
        key = (r['episode'], tuple(carrier), actual,
               tuple(r['source_nominal'][6:8]),tuple(r['controller_bias'][6:8]))
        if key != previous:
            transitions.append(dict(index=r['index'],episode=r['episode'],tick=r['tick'],phase=r['phase'],
                inherited_carrier_tracking=[list(x) for x in carrier],actual_tracking=list(actual),
                previous_ACK_tracking_active=tr.get('mapper_pre_state',{}).get('tracking_active'),
                previous_ACK_tick=tr.get('previous_ack_physics_tick'),
                reference_semantics=tr.get('reference_semantics'),
                source_RR=r['source_nominal'][6:8],mapped_RR=r['native_nominal'][6:8],
                generic_controller_RR=r['controller_bias'][6:8],actual_RR=r['after']['actual_rr_hip_knee_deg']))
        previous = key
    return dict(actual_rows=len(rows),pending_rows=len(pending),
        pending_with_actual_RR_tracking=sum('rear_right_knee' in r['tracking']['tracking_servo_names'] for r in pending),
        transitions=transitions,old_bug_report_consulted=False,
        independent_pre_gate_source647_tracking='N/A in learner-only rows; receipt provenance is explicitly reported, not independently reconstructed',
        distinction='Carrier inheritance and other legal source layers have separate tracking responsibility; P10 tracking is not automatically a regression.')


def analyze(run=RUN, *, isaac_stopped=False):
    if not isaac_stopped:
        raise ValueError('Explicit Isaac exit confirmation required before any tensor analysis')
    records, offset = sealed_inputs(run)
    rows, updates, prefix = common.collect(run,1,update_offset=offset)
    if updates[0]['counts'] != records[1][2]['counts'] or updates[0]['optimizer_steps'] != 20:
        raise ValueError('sealed update6 receipt differs from checkpoint')
    success_cohorts(rows)
    validate_row_evidence(rows)
    tracking_summary(rows)
    validate_loaded_source(records[0][2]['runtime_contract'])
    execution_checks = validate_execution_rows(common.json_rows(run/'decisions.jsonl'))
    # No Torch/model import until all guards and completed-update checks above.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(common.ROOT/'src'))
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    from wlr50_clean.ppo.semantic_training import state_hash
    rollout = run/'rollouts'/f'rollout_{offset+1:04d}.pt'
    storage = torch.load(rollout, map_location='cpu', weights_only=False)
    obs = storage['observations']['policy'].reshape(512, 448)
    checks = {'raw_execution':execution_checks}
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
    checks['loaded_collection_std'] = common.close_tensor(
        torch,outputs[0]['std'],storage['distribution_params'][1].reshape(512,12),'collection std',1e-4)
    checks['raw_old_likelihood_history'] = validate_row_evidence(rows)
    checks['official_minibatch_old_logp'] = validate_minibatch_old_logp(run, rows, updates[0])
    cohorts = success_cohorts(rows)
    fixed = {}
    for label,group in cohorts.items():
        indices = [r['index'] for r in group]
        item = dict(n=len(indices),RR={})
        if indices:
            for joint,ch in common.RR.items():
                item['RR'][joint] = dict(
                    conditional_mean_before=common.stats(outputs[0]['mean'][indices,ch].tolist()),
                    conditional_mean_after=common.stats(outputs[1]['mean'][indices,ch].tolist()),
                    same_input_mean_shift=common.stats((outputs[1]['mean'][indices,ch]-outputs[0]['mean'][indices,ch]).tolist()),
                    local_head_mean_before=common.stats(outputs[0]['local'][indices,ch].tolist()),
                    local_head_mean_after=common.stats(outputs[1]['local'][indices,ch].tolist()),
                    local_head_mean_shift=common.stats((outputs[1]['local'][indices,ch]-outputs[0]['local'][indices,ch]).tolist()),
                    std_before=common.stats(outputs[0]['std'][indices,ch].tolist()),
                    std_after=common.stats(outputs[1]['std'][indices,ch].tolist()))
        fixed[label] = item
    return dict(schema='wlr50_clean.rr_update6_tracking_fixed_learning_signal.v1',
        run=str(run),source_tracking_owner_revision=REVISION,
        counts=dict(on_policy=512,prefix_credit0=prefix,new_updates=1,absolute_update=6,
            optimizer_steps=updates[0]['optimizer_steps'],AUX=0,
            source_counts=records[0][2]['counts'],destination_counts=records[1][2]['counts']),
        verification=checks,checkpoints=checkpoint_receipts,
        source_rollout=dict(path=str(rollout),sha256=common.sha(rollout)),
        optimizer_receipt=updates[0],fixed_state_mean_changes=fixed,
        cohort_GAE_and_mean_score={k:common.metric_summary(v) for k,v in cohorts.items()},
        knee_GAE_signal={k:update5.knee_signal(v) for k,v in cohorts.items()},
        knee_headroom_and_actual_response={k:knee_response(v) for k,v in cohorts.items()},
        channel_KL=common.channel_kl_report(run,rows,updates,update_offset=offset),
        actual_current_source_tracking=tracking_summary(rows),
        local_success_rows=[dict(index=r['index'],episode=r['episode'],tick=r['tick'],
            hold_s=r['hold_s'],metrics=r['after'],reward=r['reward'],raw_gae=r['raw_gae'],
            normalized_advantage=r['advantage']) for r in rows if r['local_success']],
        limits=[
            'First successful episode41 is an episode cohort, NOT41 successful action labels; other471 may include additional successes.',
            'Same actual448 observations and HISTORY inputs reused across frozen-prior pre/post networks; not physical replay.',
            'Raw GAE is saved returns-values; saved whole-rollout advantage normalization checked without sign relabelling. No independent tail-bootstrap reconstruction is claimed.',
            'Mean-score times normalized advantage is collection Gaussian evidence, not the clipped multi-epoch PPO gradient.',
            'Tracking evidence comes only from this1e run receipts; no old5f tracking-bug report imported or treated as current evidence.',
            'A valid independent P10 tracker can coexist with empty deferred-carrier tracking. Empty carrier does not mean every layer tracking must remain empty.',
            'Headroom and actual response are decision-endpoint evidence, not attribution to a single joint or every120Hz contact measurement.',
            'No optimizer, checkpoint publication, physical process or task-capability claim.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--isaac-stopped',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(common.OUTPUT) or target.exists():
        raise ValueError('write only a NEW report under the isolated outputs directory')
    result = analyze(args.run.resolve(),isaac_stopped=args.isaac_stopped)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(target),'counts':result['counts'],
        'first41_fixed_mean':result['fixed_state_mean_changes']['first_successful_episode41']}))


if __name__ == '__main__':
    main()
