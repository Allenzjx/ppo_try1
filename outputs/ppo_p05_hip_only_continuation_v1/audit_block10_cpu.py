"""Bounded read-only block10/11 audit; explicit actual run/source/target, never polls.

Prepared before training. No planned counts are treated as accomplished work.
Uses established block08/09 primitives; reports only under this outputs directory.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import re
import subprocess

import torch
import audit_block08_first_carry_cpu as h
from wlr50_clean.ppo.semantic_training import state_hash

OUT = Path(__file__).resolve().parent
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
BRANCHES = ('p05_capture_assist', 'capture_feedback_semantics', 'rr_postcross_workspace',
            'rr_receiver_retirement_v2', 'p05_preedge_approach_recovery')
BASE_SOURCE_SHA = '05c0b58bb73c05740d9deae389b86de0f12af525b3c248f09757c36d12c9bf57'
RUNTIME = '6ac7b553d79280eddb72f4d0dece586aa2657db7'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'target', 'source-sha', 'report-name'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--mode', choices=('first', 'sealed'), required=True)
    result.add_argument('--initialization', choices=('full_episode', 'checkpoint_policy'), required=True)
    result.add_argument('--prefix-phase', choices=('P04','P07'), default='P04')
    result.add_argument('--expected-decisions', type=int)
    return result


def parse_prefixes(run, episodes, source_meta, source_sha, target_meta, target_phase):
    """Read no further than the prefix belonging to the last audited episode."""
    path = run / 'prefix_evidence.jsonl'
    starts = []; counts = Counter(); phases = Counter(); last_native = None
    for row in h.lines(path):
        assert row['policy_credit'] is False
        if row['kind'] == 'checkpoint_prefix_decision':
            counts['decisions'] += 1; counts['physics_ticks'] += row['physics_ticks']; phases[row['phase_id']] += 1
            assert row['actuator_target_effect_audit_summary']['all_ticks_verified'] and row['no_in_episode_state_writes_verified']
            last_native = row['actuator_target_effect_audit']
        elif row['kind'] == 'checkpoint_prefix_result':
            assert row['accepted'] is True
        elif row['kind'] == 'policy_credit_start':
            start = row['start']; p = start['prefix_policy_provenance']
            assert start['actual_phase'] == target_phase and p['checkpoint_sha256'] == source_sha
            assert p['source_global_policy_decisions'] == source_meta['global_policy_decisions']
            assert p['frozen_actor_parameter_sha256'] == source_meta['actor_parameter_sha256']
            assert p['frozen_for_entire_training_block'] and p['independent_parameter_and_buffer_storage_verified']
            assert p['source_policy_contract'] == p['effective_policy_contract'] == target_meta['policy_contract']
            assert p['source_runtime_content_sha256'] == p['effective_runtime_content_sha256'] == target_meta['runtime_contract']['runtime_content_sha256']
            assert last_native is not None
            starts.append({'counts': dict(counts), 'phase_counts': dict(phases), 'start_tick': start['physics_tick'],
                           'start_time_s': start['sim_time_s'], 'last_native': last_native})
            counts = Counter(); phases = Counter(); last_native = None
            if len(starts) == episodes:
                break
    assert len(starts) == episodes, 'actual prefix evidence is not complete; stop, do not poll'
    return starts


def audit(args):
    assert not torch.cuda.is_available(), 'CPU-only process required'
    torch.set_num_threads(1)
    assert re.fullmatch(r'block(?:10|11)[a-zA-Z0-9_]*', args.report_name)
    run, source_path, target_path = (Path(value).resolve(strict=True) for value in (args.run, args.source, args.target))
    sm, tm = (h.read(p.with_name(p.stem + '_manifest.json')) for p in (source_path, target_path))
    assert h.sha(source_path) == sm['checkpoint_sha256'] == args.source_sha
    assert h.sha(target_path) == tm['checkpoint_sha256'] and Path(tm['source_run']).resolve() == run
    assert tm['save_load_round_trip'] is True
    assert sm['runtime_contract']['source_git_commit'] == tm['runtime_contract']['source_git_commit'] == RUNTIME
    delta = {k: tm[k] - sm[k] for k in COUNTERS}; decisions = delta[COUNTERS[0]]; updates_n = delta[COUNTERS[1]]
    assert decisions > 0 and decisions % 128 == 0 and updates_n == decisions // 128 and delta[COUNTERS[2]] == updates_n * 20
    if args.mode == 'first':
        assert decisions == 128
    if args.expected_decisions is not None:
        assert decisions == args.expected_decisions
    migration = tm['p05_preedge_approach_recovery_migration']
    assert migration == sm['p05_preedge_approach_recovery_migration']
    assert h.sha(migration['plan_path']) == migration['plan_sha256']
    assert migration['schema'] == 'wlr50_clean.p05_preedge_approach_recovery_same389.v1'
    assert migration['target_git_commit'] == RUNTIME
    origins = {}
    for name in BRANCHES:
        assert tm[name + '_branch'] == sm[name + '_branch'] and tm[name + '_migration'] == sm[name + '_migration']
        origin = tm[name + '_branch']['counter_origin']; actual = tm[name + '_branch_counts']
        assert actual == {k: tm[k] - origin[k] for k in COUNTERS}
        assert {k: actual[k] - sm[name + '_branch_counts'][k] for k in COUNTERS} == delta
        origins[name] = {'counter_origin': origin, 'actual_branch_counts': actual}
    assert origins['p05_preedge_approach_recovery']['counter_origin'] == dict(zip(COUNTERS, (216448, 1656, 33120)))
    if sm['global_policy_decisions'] == 216448:
        assert args.source_sha == BASE_SOURCE_SHA
    preserved = [k for k in sm if k != 'resume_migration' and k.endswith(('_branch', '_migration'))]
    preserved += ['runtime_contract', 'policy_contract', 'runner_config', 'normalization', 'normalizer_state_sha256']
    assert all(sm[k] == tm[k] for k in preserved)
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events']) == 4 and [e['event_index'] for e in ledger['events']] == [1, 2, 3, 4]
    assert [(e['fit_report']['accepted_auxiliary_updates'], e['fit_report']['attempted_auxiliary_optimizer_steps']) for e in ledger['events']] == [(32,32),(32,32),(32,32),(7,8)]
    assert [ledger[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [103,104]
    old_aux = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert [old_aux[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [7,8]
    rows = h.take(run / 'residual_and_projection_audit.jsonl', decisions)
    episode_count = sum(r['applied_audit']['decision_count'] == 1 for r in rows)
    prefixes = parse_prefixes(run, episode_count, sm, args.source_sha, tm, args.prefix_phase) if args.initialization == 'checkpoint_policy' else []
    if args.initialization == 'full_episode':
        path = run / 'prefix_evidence.jsonl'
        assert not path.exists() or next(h.lines(path), None) is None
    updates = h.take(run / 'optimizer_updates.jsonl', updates_n)
    assert [u['ppo_update'] for u in updates] == list(range(sm['ppo_updates'] + 1, tm['ppo_updates'] + 1))
    assert updates[0]['actor_parameter_sha256_before'] == sm['actor_parameter_sha256']
    assert updates[-1]['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
    assert all(a['actor_parameter_sha256_after'] == b['actor_parameter_sha256_before'] for a,b in zip(updates,updates[1:]))
    assert all(u['optimizer_steps'] == 20 and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] for u in updates)
    phases = Counter(); endpoints = Counter(); execution = Counter(); rr_inputs = Counter(); episodes = []; max_logp = 0.
    previous = None; previous_native = None; rollout_summary = []
    for block, update in enumerate(updates):
        number = update['ppo_update']; selected = rows[block*128:(block+1)*128]
        batch = torch.load(run / f'rollouts/rollout_{number:06}.pt', map_location='cpu', weights_only=False)
        obs = batch['observations']['policy']; actions = batch['actions']; means,stds = batch['distribution_params']
        assert obs.shape == (128,1,389) and actions.shape == (128,1,12) and torch.equal(obs,batch['observations']['critic'])
        assert batch['runtime_contract'] == tm['runtime_contract'] and batch['policy_contract'] == tm['policy_contract']
        if prefixes:
            assert batch['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256'] == args.source_sha
            assert batch['curriculum_epoch']['prefix_request']['target_phase'] == args.prefix_phase
        batch_phases = Counter()
        for offset, row in enumerate(selected):
            index = block*128 + offset; a = row['applied_audit']; p = row['policy_request']; native = a['actuator_target_effect_audit']
            assist = native['capture_assist_evidence']; ev = a['semantic_task']['physical_evaluator']
            if a['decision_count'] == 1:
                assert not episodes or episodes[-1]['terminal']
                prefix = prefixes[len(episodes)] if prefixes else {'counts': {'decisions':0,'physics_ticks':0}, 'last_native':None}
                episodes.append({'episode_index':len(episodes),'learner_decisions':0,'learner_physics_ticks':0,'input_phases':Counter(),
                    'prefix_decisions':prefix['counts']['decisions'],'prefix_physics_ticks':prefix['counts']['physics_ticks'],
                    'assist_owned_endpoints':0, 'first_RR_placed_tick':None, 'first_RR_regrounded_endpoint':None})
                previous = None; previous_native = prefix['last_native']
                assert a['phase_id'] == (args.prefix_phase if prefixes else 'P01')
                if not prefixes:
                    assert a['physics_tick'] - a['physics_ticks'] == 0, 'natural P01 learner must begin at real reset'
            ep = episodes[-1]
            assert row['global_policy_decision'] == sm['global_policy_decisions'] + index + 1
            assert a['decision_count'] == ep['learner_decisions'] + 1
            if prefixes:
                assert a['physical_core_decision_count_including_prefix'] == ep['prefix_decisions'] + a['decision_count']
                assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
                assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
            else:
                assert not a.get('prefix_teacher_data_in_ppo_storage', False) and not a.get('prefix_checkpoint_policy_data_in_ppo_storage', False)
            assert p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
            assert row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12']
            assert row['old_distribution_mean_full12'] == p['conditional_mean_full12'] and row['old_distribution_std_full12'] == p['effective_sigma_full12']
            assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1]*12
            assert native['capture_assist_state_transition_independently_reconstructed']
            assert assist['owner_indices'] in ([],[0,1]) and (not assist['owner_indices'] or assist['knee_hold_final_verified'])
            assert assist['candidate_before_assist_full12'][2:] == assist['candidate_after_assist_full12'][2:]
            assert a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified']
            ticks = a['actuator_target_effect_audit_ticks']; assert len(ticks) == a['physics_ticks'] and all(t['verified'] for t in ticks)
            if previous_native is not None:
                assert torch.equal(obs[offset,0,372:384],torch.tensor(h.capture_assist_features(previous_native['capture_assist_evidence']['state_after']),dtype=obs.dtype))
            if previous is not None:
                assert not previous['terminal'] and previous['applied_audit']['end_phase_id'] == a['phase_id']
                assert previous['applied_audit']['physics_tick'] == a['physics_tick'] - a['physics_ticks']
            if a['phase_id'] != a['end_phase_id']:
                execution['ordinary_phase_handoffs'] += 1; assert not row['terminal']
            ep['learner_decisions'] += 1; ep['learner_physics_ticks'] += a['physics_ticks']; ep['input_phases'][a['phase_id']] += 1
            ep['assist_owned_endpoints'] += bool(assist['owner_indices'])
            rr = ev['current_legs']['RR']; placed = ev['history']['placed']['RR']
            if placed and ep['first_RR_placed_tick'] is None: ep['first_RR_placed_tick'] = ev['history']['event_ticks']['placed']['RR']
            if placed and rr['ground_contact'] and ep['first_RR_regrounded_endpoint'] is None: ep['first_RR_regrounded_endpoint'] = a['physics_tick']
            ep.update(physical_duration_s=a['sim_time_s'],last_phase=a['end_phase_id'],terminal=bool(row['terminal']),
                termination_reason=a['termination_reason'],full_task_success=bool(a['full_task_success']),
                placed_history=ev['history']['placed'],event_ticks=ev['history']['event_ticks'],
                final_RR={k:rr[k] for k in ('clearance_m','front_distance_m','top_contact','ground_contact','current_lift_valid')},
                final_RL={k:ev['current_legs']['RL'].get(k) for k in ('clearance_m','front_distance_m','top_contact','ground_contact','air','support')})
            phases[a['phase_id']] += 1; batch_phases[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
            execution['learner_physics_ticks'] += a['physics_ticks']; execution['native_verified_ticks'] += len(ticks)
            execution['assist_owned_endpoints'] += bool(assist['owner_indices']); execution['assist_initialized_endpoints'] += bool(assist['state_after']['initialized'])
            execution['terminal_samples'] += bool(row['terminal'])
            for key,column in (('qualified_current',149),('crossed_history',153),('placed_history',157)):
                rr_inputs[key] += int(obs[offset,0,column] == 1)
            previous = row; previous_native = native
        for tensor,key in ((actions,'raw_policy_action_full12'),(means,'old_distribution_mean_full12'),(stds,'old_distribution_std_full12'),
            (batch['actions_log_prob'],'old_log_probability'),(batch['values'],'old_value'),(batch['rewards'],'reward'),(batch['dones'],'terminal')):
            assert torch.isfinite(tensor).all() and torch.equal(tensor,torch.tensor([r[key] for r in selected],dtype=tensor.dtype).reshape(tensor.shape))
        assert [h.PHASES[i] for i in obs[:,0,:13].argmax(-1)] == [r['applied_audit']['phase_id'] for r in selected]
        for begin,end,key in ((372,384,'capture_assist_observed_features'),(384,389,'capture_continuation_observed_features')):
            assert torch.equal(obs[:,0,begin:end],torch.tensor([r['policy_request'][key] for r in selected],dtype=obs.dtype))
        assert torch.equal(obs[:,0,17],torch.tensor([r['applied_audit']['reward_breakdown']['potential_before'] for r in selected],dtype=obs.dtype))
        likelihood = h.read(run / f'rollouts/update_{number:06}_likelihood.json')
        uses = Counter(i for mini in likelihood['minibatches'] for ids in mini['rollout_flat_indices'] for i in ids)
        assert len(likelihood['minibatches']) == 20 and uses == Counter({i:5 for i in range(128)})
        error = float((torch.distributions.Normal(means,stds).log_prob(actions).sum(-1).unsqueeze(-1) - batch['actions_log_prob']).abs().max())
        max_logp = max(max_logp,error); assert max_logp <= 1e-5
        rollout_summary.append({'update':number,'phase_counts':dict(batch_phases),'effective_LR':update['optimizer_learning_rate']})
    manifest = None; outer_lifecycle = None; postflight = None
    if args.mode == 'sealed':
        manifest = h.read(run / 'training_manifest.json')
        assert manifest['lifecycle'] in ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')
        assert manifest['actual_policy_decisions'] == decisions
        assert manifest['requested_policy_decisions'] == decisions and manifest['rounding_overrun'] == 0
        assert manifest['planned_requested_policy_decisions'] - manifest['unconsumed_requested_policy_decisions'] == decisions
        assert manifest['ppo_updates_this_run'] == updates_n and manifest['optimizer_steps_this_run'] == delta['optimizer_steps']
        if manifest['lifecycle'] == 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY':
            stop = manifest['stop_after_update']
            assert stop['schema'] == 'wlr50_clean.semantic_stop_after_update.v1'
            assert Path(stop['run_dir']).resolve() == run and stop['source_git_commit'] == RUNTIME
            assert h.sha(stop['request_path']) == stop['request_sha256']
            assert tm['stop_after_update'] == stop
        else:
            assert manifest['stop_after_update'] is None
        assert next(itertools.islice(h.lines(run/'residual_and_projection_audit.jsonl'),decisions,None),None) is None
        assert len(list(h.lines(run/'optimizer_updates.jsonl'))) == updates_n
        ended = [ep for ep in episodes if ep['terminal']]; complete = list(h.lines(run/'completed_episodes.jsonl'))
        assert len(ended) == len(complete)
        for ep,item in zip(ended,complete):
            assert (ep['learner_decisions'],ep['termination_reason'],ep['physical_duration_s'],ep['full_task_success']) == (item['policy_decisions'],item['termination_reason'],item['duration_s'],item['full_task_success'])
        assert manifest['telemetry']['core']['physics_ticks'] == execution['learner_physics_ticks']
        outer = h.read(run / 'run_manifest.json'); outer_lifecycle = outer['lifecycle']
        assert outer['runtime_contract'] == tm['runtime_contract']
        if outer_lifecycle == 'FAILED':
            # Preserve the outer failure. Only classify this known AFTER-training
            # pinned-HEAD error, never waive a load/runtime/checkpoint contract.
            assert manifest['lifecycle'] == 'SUCCEEDED'
            assert outer.get('error') == 'semantic run HEAD differs from its pinned revision'
            assert 'runtime_contract(**contract_options)' in outer['traceback']
            root = OUT.parents[1]
            def git(*arguments):
                return subprocess.run(['git','-C',str(root),*arguments],capture_output=True,text=True,check=True).stdout.strip()
            head = git('rev-parse','HEAD'); assert head != RUNTIME
            changed = git('diff','--name-only',RUNTIME+'..'+head).splitlines()
            assert changed and all(p == '.gitignore' or p.startswith('outputs/') for p in changed)
            production_paths = ('src','scripts','configs','pyproject.toml')
            assert not git('diff','--name-only',RUNTIME+'..'+head,'--',*production_paths)
            assert not git('diff','--name-only','--',*production_paths)
            assert all(h.sha(root/path)==expected for path,expected in tm['runtime_contract']['files'].items())
            postflight = {'classification':'completed_training_saved_checkpoint_then_failed_pinned_HEAD_postflight',
                'outer_run_lifecycle_preserved':'FAILED','outer_error':outer['error'],
                'pinned_runtime_commit':RUNTIME,'current_commit':head,
                'committed_changed_paths_count':len(changed),'committed_changes_only_gitignore_and_outputs':True,
                'production_git_diff_empty':True,'production_worktree_diff_empty':True,
                'all_pinned_runtime_file_bytes_still_match':True,
                'original_manifests_edited':False,'checkpoint_contract_bypass_performed':False,
                'recovery_scope':'valid preserved training source; explicit current-HEAD/new-semantic contract resolution is still required before resume'}
        else:
            assert outer_lifecycle == manifest['lifecycle']
    payloads = [torch.load(path,map_location='cpu',weights_only=False) for path in (source_path,target_path)]
    for payload,meta in zip(payloads,(sm,tm)):
        assert h.parameter_sha(payload['actor_state_dict']) == meta['actor_parameter_sha256']
        assert h.parameter_sha(payload['critic_state_dict']) == meta['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == meta['optimizer_state_sha256']
        assert all(payload['infos'][k] == v for k,v in meta.items() if k in payload['infos'])
    adam = [float(payloads[1]['optimizer_state_dict']['state'][k]['step'] - v['step']) for k,v in payloads[0]['optimizer_state_dict']['state'].items()]
    assert len(adam) == 12 and adam == [float(delta['optimizer_steps'])]*12
    assert sm['actor_parameter_sha256'] != tm['actor_parameter_sha256'] and sm['optimizer_state_sha256'] != tm['optimizer_state_sha256']
    assert updates[-1]['optimizer_learning_rate'] == tm['optimizer_learning_rate'] and 'identity_RSL_normalizer' in tm['normalization']
    srng,trng = sm['training_rng_state'],tm['training_rng_state']
    assert set(srng) == set(trng) and srng['seed'] == trng['seed'] == 1001
    assert srng['torch_cuda_device_count'] == trng['torch_cuda_device_count'] == len(trng['torch_cuda']) == 1
    assert state_hash(srng) != state_hash(trng)
    for ep in episodes: ep['input_phases'] = {p:ep['input_phases'][p] for p in h.PHASES}
    for prefix in prefixes: del prefix['last_native']
    report = {'schema':'wlr50_clean.block10_actual_PPO_preedge_carry_audit.v1','result':'PASS','audit_mode':args.mode,
        'run':str(run),'source_checkpoint':str(source_path),'source_sha256':args.source_sha,'checkpoint':str(target_path),'checkpoint_sha256':tm['checkpoint_sha256'],
        'initialization':args.initialization,'actual_added_counts':delta,'actual_lifetime_counts':{k:tm[k] for k in COUNTERS},
        'prefix_target_phase':args.prefix_phase if prefixes else None,
        'training_lifecycle':manifest['lifecycle'] if manifest else 'not_audited_first_batch_only',
        'outer_run_lifecycle':outer_lifecycle,'postflight_classification':postflight,
        'planned_decisions':manifest['planned_requested_policy_decisions'] if manifest else None,
        'unconsumed_decisions':manifest['unconsumed_requested_policy_decisions'] if manifest else None,
        'unconsumed_decisions_receive_no_learning_credit':True,
        'verified_update_boundary_stop':manifest['stop_after_update'] if manifest else None,
        'phase_input_counts':{p:phases[p] for p in h.PHASES},'phase_endpoint_counts':dict(endpoints),'execution':dict(execution),
        'prefixes':prefixes,'prefix_decisions':sum(p['counts']['decisions'] for p in prefixes),'prefix_PPO_credit':0,
        'episodes':episodes,'ended_episodes':sum(e['terminal'] for e in episodes),
        'partial_episode_scope':'sampling boundary' if args.mode == 'sealed' else 'audit cutoff, training may continue',
        'RR_input_counts':dict(rr_inputs),'all_five_origins':origins,'all_source_branch_migration_fields_exact':preserved,
        'front_AUX':[96,96],'RR_AUX':[7,8],'mixed_AUX':[103,104],'old_separate_AUX':[7,8],'AUX_added':0,
        'raw389_full12_mu_sigma_logp_value_reward_done_exact':True,'each_sample_PPO_uses':5,'CPU_logp_max_error':max_logp,
        'all_updates_actor_changed_finite_gradient':True,'all12_Adam_step_deltas':adam,'Identity_preserved':True,
        'RNG_full_schema_and_CUDA_count_preserved_state_advanced':True,'RNG_source_hash':state_hash(srng),'RNG_target_hash':state_hash(trng),
        'per_rollout':rollout_summary,'final_effective_LR':tm['optimizer_learning_rate'],
        'checkpoint_payload_sidecar_hashes_verified':True,'official_save_reload_recorded':True,
        'physical_success_inferred_from_training_lifecycle':False,'audit_fit_physics_checkpoint_writes':0}
    report_path = OUT / (args.report_name + '.json'); report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    table = '\n'.join(f"| {e['episode_index']} | {e['learner_decisions']} | {e['physical_duration_s']:.6f} | {e['last_phase']} | {e['termination_reason'] or report['partial_episode_scope']} |" for e in episodes)
    md = f'''# {args.report_name}

PASS — actual new decisions/PPO/Adam: **{decisions}/{updates_n}/{delta['optimizer_steps']}**; cumulative **{tm['global_policy_decisions']}/{tm['ppo_updates']}/{tm['optimizer_steps']}**. Initialization: `{args.initialization}`. Training lifecycle is not physical task success.

Lifecycle: `{report['training_lifecycle']}`. Planned {report['planned_decisions']}; unconsumed {report['unconsumed_decisions']} receive no learning credit. A legal update-boundary stop does not make a nonterminal partial episode a physical failure or success.

Outer run lifecycle: `{outer_lifecycle}`. {('Original FAILED run manifest is retained: '+postflight['outer_error']+'. Training manifest is independently SUCCEEDED and all completed updates/checkpoint payloads are verified. Current commit '+postflight['current_commit']+' changes only .gitignore/outputs; production Git diff is empty and every pinned runtime file still matches. The checkpoint remains a valid preserved training source, but current HEAD/new-semantic resume requires explicit contract resolution; no bypass or manifest rewrite was performed.') if postflight else 'No postflight HEAD-failure classification is required for this audited scope.'}

Phase inputs: {dict(phases)}. Frozen-prefix decisions: {report['prefix_decisions']}, all zero learning credit. Native-verified learner ticks: {execution['native_verified_ticks']}; FL-assist-owned endpoints: {execution['assist_owned_endpoints']}. RR current-qualified/crossed-history/placed-history input counts: {dict(rr_inputs)}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
{table}

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error {max_logp:.9g}. Actual actor and Adam updates verify, all12 Adam states advance{delta['optimizer_steps']}; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final{tm['optimizer_learning_rate']:.9g}. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `{source_path}` ({args.source_sha}). Target: `{target_path}` ({tm['checkpoint_sha256']}). Run: `{run}`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. {'Only the first sealed update was audited; later work is not credited here.' if args.mode == 'first' else 'A nonterminal final episode is only a sampling-boundary partial, not an invented success/failure.'}
'''
    (OUT / (args.report_name + '.md')).write_text(md,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('result','actual_added_counts','actual_lifetime_counts','phase_input_counts','prefix_decisions','ended_episodes','CPU_logp_max_error')},indent=2))
    return report


if __name__ == '__main__':
    audit(parser().parse_args())
