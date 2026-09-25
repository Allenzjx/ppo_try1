"""Sealed-course physical coverage, streaming JSON only; never imports robot/Torch.

Usage (after each requested run seals):
  python inspect_course.py --run ABS_RUN [--run ABS_RUN] --report course_coverage
Inputs are read-only. Reports are new files beside this script; existing files
are never overwritten. A pre-update advantage record alone earns no credit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PHASES = tuple(f'P{i:02d}' for i in range(1, 14))
WINDOWS = (
    'RR_reachable_AIR_preparation', 'RR_actual_bearing_front_preparation',
    'positive_FR_axis_CoM_body_projection_with_RL_fraction_decline', 'qualified_RL_edge_recovery',
    'RL_qualified_AIR_capture_region', 'RL_actual_TOP_bearing', 'RL_placed_history',
    'P13_actual_stage', 'all_task_complete')
FR_PROJECTION_WINDOW = WINDOWS[2]
FR_PROJECTION_SEMANTICS = (
    'Positive local fixed-FR-axis projections of both CoM and body displacement, '
    'plus declining RL load fraction while RR and a front leg currently bear. '
    'NOT verified lateral/rightward transfer, NOT absolute RL force decline, '
    'NOT qualified RL unloading. Forward movement alone can make the projection positive; '
    'a load fraction can decline because other legs load more. World xyz components '
    'and current absolute forces are reported separately; yaw/body-frame decomposition '
    'is not inferred from the projection.')


def require(test, message):
    if not test:
        raise ValueError(message)


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def small_json(path, limit=2_000_000):
    with path.open('r', encoding='utf-8-sig') as stream:
        text = stream.read(limit + 1)
    require(len(text) <= limit, f'compact JSON size limit exceeded: {path}')
    return json.loads(text)


def rows(path, optional=False):
    if optional and not path.exists():
        return
    with path.open('r', encoding='utf-8-sig') as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'unsealed/invalid JSONL {path}:{n}') from error


def minibatches(path):
    """Decode one actual likelihood minibatch at a time, not the whole file."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8-sig') as stream:
        buf = ''
        while True:
            part = stream.read(65536)
            require(part, f'no minibatches array: {path}')
            buf += part
            found = re.search(r'"minibatches"\s*:\s*\[', buf)
            if found:
                buf = buf[found.end():]
                break
            require(len(buf) < 1_000_000, f'unexpected large likelihood header: {path}')
        while True:
            buf = buf.lstrip()
            if buf.startswith(','):
                buf = buf[1:].lstrip()
            if buf.startswith(']'):
                return
            try:
                value, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                part = stream.read(65536)
                require(part, f'incomplete likelihood minibatch: {path}')
                buf += part
                require(len(buf) <= 8_000_000, f'one minibatch exceeds bounded decoder: {path}')
                continue
            require(isinstance(value, dict), 'unexpected minibatch record')
            yield value
            buf = buf[end:]


def bound_support(contract):
    entry = contract['selected_configuration']['stage_task_spec.yaml']
    head = contract['source_git_commit']
    require(re.fullmatch(r'[0-9a-f]{40}', head), 'invalid frozen source HEAD')
    blob = subprocess.check_output(['git', 'show', f"{head}:{entry['path']}"], cwd=ROOT)
    require(hashlib.sha256(blob).hexdigest() == entry['sha256'], 'frozen task bytes differ')
    text = blob.decode('utf-8-sig')
    values = {}
    for key in ('force_noise_floor_n', 'unloaded_leg_maximum_load_fraction'):
        matches = re.findall(r'^  ' + key + r':\s*([0-9.eE+-]+)\s*$', text, re.M)
        require(len(matches) == 1, f'cannot uniquely read existing support setting {key}')
        values[key] = float(matches[0])
    require(values['force_noise_floor_n'] > 0 and
            0 < values['unloaded_leg_maximum_load_fraction'] < 1, 'invalid support settings')
    return dict(values, frozen_task_sha256=entry['sha256'], frozen_head=head)


def bearing(leg, floor, top=False):
    yes = (leg.get('air') is False and leg.get('support') is True
           and leg.get('bearing_verified') is True and finite(leg.get('bearing_force_n'))
           and leg['bearing_force_n'] >= floor
           and (leg.get('ground_contact') is True or leg.get('top_surface_contact') is True))
    if top:
        yes = (yes and leg.get('ground_contact') is False and leg.get('top_contact') is True
               and leg.get('top_surface_contact') is True and leg.get('obstacle_pair_active') is True
               and leg.get('within_top_xy') is True and leg.get('contact_surface') == 'TOP')
    return bool(yes)


def qualified(leg):
    return (leg.get('current_lift_valid') is True and leg.get('motion_continuation_allowed') is True
            and leg.get('active_attempt') is True and leg.get('ground_contact') is False)


def physical_windows(task, support):
    """Nonexclusive observed windows, NOT new task predicates or action targets."""
    ev = task.get('physical_evaluator', {})
    if ev.get('valid') is not True or ev.get('termination_reason') is not None:
        return set(), {'physical_status': 'invalid_or_unavailable'}
    legs = ev['current_legs']; rr, rl = legs['RR'], legs['RL']
    hist = ev['history']; prep = task.get('cooperative_preparation', {})
    floor = support['force_noise_floor_n']
    rr_bearing = bearing(rr, floor, top=True)
    front_bearing = [leg for leg in ('FL', 'FR') if bearing(legs[leg], floor)]
    relevant = prep.get('valid') is True and prep.get('relevant') is True
    flags = set()
    if (relevant and prep.get('capturable_AIR_preparation') is True and qualified(rr)
            and rr.get('air') is True and rr.get('within_top_xy') is True
            and rr.get('within_lateral_span') is True and hist['front_edge_crossed']['RR']):
        flags.add('RR_reachable_AIR_preparation')
    if relevant and rr_bearing and front_bearing:
        flags.add('RR_actual_bearing_front_preparation')
    # Existing transfer_roles window: direction fixed at that window's start,
    # not total transfer displacement and not proof FR actually bears load.
    role = task.get('transfer_roles', {}).get('RL', {})
    motion = role.get('transfer_direction_context', {})
    direction = motion.get('fixed_direction_world')
    displacement = motion.get('body_world_displacement_m')
    body_toward = None
    if (isinstance(direction, (list, tuple)) and isinstance(displacement, (list, tuple))
            and len(direction) == len(displacement) == 3 and all(map(finite, tuple(direction) + tuple(displacement)))):
        body_toward = sum(a*b for a, b in zip(direction, displacement))
    com_toward = motion.get('com_toward_receiver_m')
    load_drop = motion.get('load_fraction_change')
    if (rr_bearing and front_bearing and role.get('valid') is True
            and role.get('diagonal_receiving_side') == 'FR'
            and role.get('load_change_valid') is True and finite(load_drop) and load_drop > 0
            and finite(com_toward) and com_toward > 0 and finite(body_toward) and body_toward > 0):
        flags.add(FR_PROJECTION_WINDOW)
    tick, qt = ev.get('physics_tick'), rl.get('current_lift_qualified_tick')
    if (qualified(rl) and type(qt) is int and type(tick) is int and 0 <= qt <= tick
            and rl.get('air') is False and rl.get('obstacle_pair_active') is True
            and rl.get('contact_reaction') is True and finite(rl.get('contact_reaction_force_n'))
            and rl['contact_reaction_force_n'] >= 0 and rl.get('top_contact') is False
            and rl.get('top_surface_contact') is False
            and rl.get('contact_mode') in ('FRONT_WALL', 'OBSTACLE_AMBIGUOUS')
            and rl.get('contact_surface') == rl.get('contact_mode')):
        flags.add('qualified_RL_edge_recovery')
    if (qualified(rl) and rl.get('air') is True and rl.get('within_top_xy') is True
            and rl.get('within_lateral_span') is True and hist['front_edge_crossed']['RL']):
        flags.add('RL_qualified_AIR_capture_region')
    if bearing(rl, floor, top=True): flags.add('RL_actual_TOP_bearing')
    if hist['placed']['RL']: flags.add('RL_placed_history')
    if task.get('stage_id') == 'P13': flags.add('P13_actual_stage')
    if ev.get('task_completed_controlled') is True: flags.add('all_task_complete')
    leg_keys = ('clearance_m','front_distance_m','air','ground_contact','within_top_xy',
                'current_lift_valid','active_attempt','contact_surface','bearing_force_n','bearing_verified')
    return flags, dict(physics_tick=tick, sim_time_s=ev.get('simulation_time_s'),
        RR={k:rr.get(k) for k in leg_keys}, RL={k:rl.get(k) for k in leg_keys},
        RR_actual_bearing=rr_bearing, front_actual_bearing=front_bearing,
        RL_current_load_fraction=rl.get('load_fraction') if rl.get('load_fraction_valid') is True else None,
        RL_below_existing_unload_fraction=(rl['load_fraction'] <= support['unloaded_leg_maximum_load_fraction']
            if rl.get('load_fraction_valid') is True and finite(rl.get('load_fraction')) else None),
        FR_direction_window_reference_tick=motion.get('reference_tick'),
        FR_direction_window_s=motion.get('window_s'), FR_body_toward_m=body_toward,
        FR_CoM_toward_m=com_toward, RL_load_fraction_drop=load_drop,
        FR_direction_window_fixed_axis_world=direction,
        FR_window_body_world_displacement_xyz_m=displacement,
        FR_window_CoM_world_displacement_xyz_m=motion.get('com_world_displacement_m'),
        RL_current_bearing_force_n=rl.get('bearing_force_n'),
        RL_current_bearing_force_verified=rl.get('bearing_verified'),
        RL_current_contact_reaction_force_n=rl.get('contact_reaction_force_n'),
        RL_current_qualified_lift=qualified(rl),
        RL_current_ground_contact=rl.get('ground_contact'),
        RL_current_absolute_force_drop_over_FR_window_n=None,
        RL_absolute_force_drop_status='reference_force_not_stored_in_transfer_direction_context',
        fl_range_credit=prep.get('fl_range_credit'), rl_space_credit=prep.get('rl_space_credit'))


def scalar(x):
    while isinstance(x, list) and len(x) == 1: x = x[0]
    require(finite(x), 'nonfinite scalar learning evidence')
    return float(x)


def gaussian_logp(raw, mean, sigma):
    require(len(raw) == len(mean) == len(sigma) == 12 and all(map(finite, raw+mean+sigma))
            and min(sigma) > 0, 'invalid raw Gaussian evidence')
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi) for x,m,s in zip(raw,mean,sigma))


def inspect_run(run):
    run = run.resolve(strict=True)
    manifest = small_json(run/'run_manifest.json')
    require(manifest.get('completed_at_utc') and manifest.get('lifecycle') != 'RUNNING',
            'run must be sealed; this tool never watches live journals')
    training = small_json(run/'training_manifest.json')
    support = bound_support(manifest['runtime_contract'])
    completed = {}
    for item in rows(run/'optimizer_updates.jsonl'):
        number = item['ppo_update']
        require(number not in completed and item.get('optimizer_steps', 0) > 0, 'invalid completed update')
        completed[number] = item
    require(completed, 'no completed optimizer update')
    audits = []
    for item in rows(run/'advantage_audit.jsonl'):
        number = item['ppo_update_intended']
        if number not in completed: continue
        require(item['teacher_prefix_samples_included'] is False and item['num_envs'] == 1,
                'requires genuine N1 learner-only rollout')
        require(item['last_global_policy_decision'] == completed[number]['global_policy_decisions'],
                'pre-update audit not bound to completed update')
        audits.append(item)
    require(len(audits) == len(completed), 'missing/duplicate completed rollout audit')
    audits.sort(key=lambda x:x['first_global_policy_decision'])
    for left, right in zip(audits, audits[1:]):
        require(left['last_global_policy_decision']+1 == right['first_global_policy_decision'], 'update ranges overlap/gap')
    phase_counts, endpoint_counts, input_counts, events = Counter(), Counter(), Counter(), Counter()
    examples = {name: [] for name in WINDOWS}; terminal_examples = []
    compact_updates = []
    stream = iter(rows(run/'residual_and_projection_audit.jsonl'))
    previous = None; input_missing = 0; seen = 0; first = audits[0]['first_global_policy_decision']
    max_logp_error = 0.; owner_channels = Counter(); owner_input_channels = Counter(); policy_checks = Counter()
    owner_endpoint_receipt_missing = 0
    last_task = {}; last_row = None
    for advantage in audits:
        number = advantage['ppo_update_intended']; n = advantage['sample_count']; batch = []
        phases = Counter()
        terminals = {r['global_policy_decision']:r for r in advantage['terminal_samples']}
        while len(batch) < n:
            row = next(stream, None); require(row is not None, 'missing optimized decision row')
            ident = row['global_policy_decision']
            if ident < first: continue
            require(ident == first+seen, 'decision range gap/duplicate')
            seen += 1
            a = row['applied_audit']; task = a['semantic_task']; request = row['policy_request']
            require(a.get('prefix_teacher_data_in_ppo_storage') is False
                    and a.get('prefix_checkpoint_policy_data_in_ppo_storage') is False, 'prefix in PPO')
            require(request['selected_raw_full12'] == row['raw_policy_action_full12']
                    and request['conditional_mean_full12'] == row['old_distribution_mean_full12']
                    and request['effective_sigma_full12'] == row['old_distribution_std_full12']
                    and request['selected_raw_log_probability'] == row['old_log_probability'], 'raw request mismatch')
            require(request['rear_task_assists_enabled'] is False and request['sampling_draws'] == 1,
                    'not the declared rear-assist-OFF stochastic learner')
            native = a['actuator_target_effect_audit']
            require(native['verified'] is True and native['phase_mask_full12'] == [1]*12, 'native/mask audit differs')
            max_logp_error = max(max_logp_error, abs(gaussian_logp(row['raw_policy_action_full12'],
                row['old_distribution_mean_full12'],row['old_distribution_std_full12'])-row['old_log_probability']))
            policy_checks['raw12_mean_sigma_logp_equal'] += 1
            owner_features = request.get('rear_owner_observed_features', [])
            if len(owner_features) == 17:
                policy_checks['public_owner17'] += 1
                require(all(value in (0., 1.) for value in owner_features[8:17]), 'invalid public owner bits')
                for j, active in enumerate(owner_features[8:12]):
                    if active: owner_input_channels[str((0,1,4,5)[j])] += 1
            owner = native.get('rear_owner_recovery_evidence', {}).get('state_after')
            if owner is None:
                owner_endpoint_receipt_missing += 1
            else:
                require(len(owner['active']) == 4, 'invalid endpoint owner receipt')
                for j, active in enumerate(owner['active']):
                    if active: owner_channels[str((0,1,4,5)[j])] += 1
            flags, physical = physical_windows(task, support)
            endpoint_counts.update(flags)
            start = a['physics_tick']-a['physics_ticks']
            if previous is not None and previous['tick'] == start and previous['terminal'] is False:
                inp, _ = physical_windows(previous['task'], support); input_counts.update(inp)
            else: input_missing += 1
            previous = {'tick':a['physics_tick'],'task':task,'terminal':row['terminal']}
            phase_counts[a['phase_id']] += 1; phases[a['phase_id']] += 1
            ev = task['physical_evaluator']; hist = ev.get('history', {})
            for leg in ('RR','RL'):
                legrow = ev['current_legs'][leg]
                events[leg+'_current_qualified'] += int(qualified(legrow))
                for kind in ('front_edge_crossed','placed'):
                    events[leg+'_'+kind+'_history'] += int(hist.get(kind,{}).get(leg) is True)
            rb = a['reward_breakdown']; terminal = terminals.get(ident)
            entry = dict(global_policy_decision=ident,ppo_update=number,request_phase=a['phase_id'],
                end_phase=a['end_phase_id'],episode_start_tick=start,episode_end_tick=a['physics_tick'],
                reward=row['reward'],old_value=row['old_value'],terminal=row['terminal'],
                old_log_probability=row['old_log_probability'],
                termination_reason=a.get('termination_reason'),full_task_success=a.get('full_task_success'),
                raw_GAE=(terminal['raw_gae_returns_minus_old_values'] if terminal else None),
                raw_GAE_source=('exact_terminal_advantage_audit' if terminal else 'not_saved_per_row_in_JSON'),
                reward_breakdown={k:rb.get(k) for k in ('families','potential_before','potential_after',
                    'potential_shaping','terminal_event','cooperative_counterroll_cost')},
                effective_sigma_full12=request['effective_sigma_full12'],physical=physical,
                endpoint_windows=sorted(flags),standardized_advantage=None)
            batch.append(entry); last_task=task; last_row=entry
        expected = {k:v['sample_count'] for k,v in advantage['by_request_phase'].items()}
        require(dict(phases) == expected, 'request phase count != completed advantage mapping')
        exposures = Counter()
        for mb in minibatches(run/'rollouts'/f'update_{number:06d}_likelihood.json'):
            for j, indices in enumerate(mb['rollout_flat_indices']):
                require(len(indices) == 1 and 0 <= indices[0] < n, 'ambiguous raw row association')
                i=indices[0]; exposures[i] += 1; value=scalar(mb['actual_advantage'][j])
                entry=batch[i]
                if entry['standardized_advantage'] is None: entry['standardized_advantage']=value
                else: require(entry['standardized_advantage'] == value, 'per-minibatch normalization needs separate handling')
                require(scalar(mb['old_log_probability'][j]) == entry['old_log_probability'],
                        'likelihood row not bound to collected raw sample logp')
        require(exposures == Counter({i:5 for i in range(n)}), 'not five official optimizer exposures per sample')
        require(advantage['stored_advantage_semantics'] == 'official_whole_rollout_standardized_GAE',
                'unexpected advantage normalization; do not reinterpret signs')
        for entry in batch:
            if entry['terminal']: terminal_examples.append(entry)
            for name in entry['endpoint_windows']:
                if len(examples[name]) < 2: examples[name].append(entry)
        compact_updates.append(dict(ppo_update=number,
            first_global_policy_decision=advantage['first_global_policy_decision'],
            last_global_policy_decision=advantage['last_global_policy_decision'],sample_count=n,
            optimizer_steps=completed[number]['optimizer_steps'],
            learning_rate=completed[number]['optimizer_learning_rate'],
            gamma=advantage['gamma'],gae_lambda=advantage['lambda'],tail_bootstrap=advantage['tail_bootstrap'],
            by_request_phase=advantage['by_request_phase']))
    require(max_logp_error < 5e-5, 'raw Gaussian old likelihood disagreement')
    require(seen == training['actual_policy_decisions'] == sum(a['sample_count'] for a in audits), 'completed count mismatch')
    require(len(completed) == training['ppo_updates_this_run'], 'PPO completion count mismatch')
    require(sum(x['optimizer_steps'] for x in completed.values()) == training['optimizer_steps_this_run'], 'Adam count mismatch')
    prefixes=Counter()
    for row in rows(run/'prefix_evidence.jsonl',optional=True):
        require(row.get('policy_credit') is False, 'prefix credited')
        if row.get('kind') in ('checkpoint_prefix_decision','prefix_decision'):
            prefixes['decisions']+=1; prefixes['physics_ticks']+=row['physics_ticks']
        if row.get('kind') == 'policy_credit_start': prefixes['learner_starts']+=1
    # All trailing collection is deliberately not consumed or credited.
    return dict(run=str(run),lifecycle=manifest['lifecycle'],completed_at_utc=manifest['completed_at_utc'],
        bound_support=support,optimized_decisions=seen,ppo_updates=len(completed),
        optimizer_steps=sum(x['optimizer_steps'] for x in completed.values()),
        actual_request_phase_counts={p:phase_counts[p] for p in PHASES},
        input_physical_window_counts={k:input_counts[k] for k in WINDOWS},
        input_physical_window_missing_alignment=input_missing,
        endpoint_physical_window_counts={k:endpoint_counts[k] for k in WINDOWS},
        physical_event_history_counts=dict(events),
        public_owner_active_input_counts={str(i):owner_input_channels[str(i)] for i in (0,1,4,5)},
        public_owner_active_endpoint_counts=(dict(owner_channels) if owner_endpoint_receipt_missing == 0 else None),
        public_owner_endpoint_receipt_missing=owner_endpoint_receipt_missing,
        public_owner_clock_semantics='actual public17 at policy input; omitted native endpoint receipt is unavailable, not inactive',
        prefix={**dict(prefixes),'PPO_credit':0},raw_policy_checks=dict(policy_checks),
        maximum_old_raw_logp_error=max_logp_error,examples=examples,terminal_examples=terminal_examples,
        final_optimized_endpoint=last_row,ends_nonterminal_partial=not last_row['terminal'],
        ordinary_phase_change_is_not_terminal=True,updates=compact_updates,
        planned_unconsumed_no_credit=training['unconsumed_requested_policy_decisions'],
        final_global_policy_decisions=training['global_policy_decisions'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',type=Path,required=True)
    parser.add_argument('--report',default='course_coverage')
    args=parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.report), 'report must be a simple new basename')
    resolved=[p.resolve(strict=True) for p in args.run]
    require(len(set(resolved)) == len(resolved), 'duplicate physical run cannot be credited twice')
    target=OUT/(args.report+'.json'); md=OUT/(args.report+'.md')
    require(not target.exists() and not md.exists(), 'report exists; use a new basename')
    reports=[inspect_run(p) for p in resolved]
    result=dict(schema='readonly.optimized_physical_course_coverage.v2',
        semantics='nonexclusive real endpoint and separately aligned input windows; not phase-label substitution',
        window_semantics={FR_PROJECTION_WINDOW:FR_PROJECTION_SEMANTICS},
        superseded_window_label={'FR_directed_body_CoM_motion_with_RL_unload':
            'Renamed without changing its numeric predicate; older reports are preserved, '
            'but their label does not establish real lateral motion or qualified unloading.'},
        limitations=['No Torch/checkpoint/tensor/model/physics access.',
            'No per-tick duration claim: endpoint counts are optimized policy intervals, not 120Hz samples.',
            'Window CoM motion is local fixed-window displacement, not total transfer or receiving-leg bearing.',
            FR_PROJECTION_SEMANTICS,
            'Nonterminal per-row raw GAE is not serialized in JSON: null is intentional, not zero.',
            'Actual standardized optimizer advantages retain their original signs; no advantage relabeling.',
            'Physical progress/capture does not imply complete P01 success; prefix has zero learning credit.'],
        runs=reports,total_optimized_decisions=sum(r['optimized_decisions'] for r in reports),
        total_ppo_updates=sum(r['ppo_updates'] for r in reports),
        total_optimizer_steps=sum(r['optimizer_steps'] for r in reports))
    with target.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    text=['# Optimizer-completed physical coverage','',
        f"Actual new decisions / PPO / Adam: {result['total_optimized_decisions']} / {result['total_ppo_updates']} / {result['total_optimizer_steps']}.",'',
        '| Run | Window | aligned input | endpoint |','| --- | --- | ---: | ---: |']
    for report in reports:
        for name in WINDOWS:
            text.append(f"| {Path(report['run']).name} | {name} | {report['input_physical_window_counts'][name]} | {report['endpoint_physical_window_counts'][name]} |")
    text += ['', 'Counts are nonexclusive; missing first-prefix/episode input alignment is explicitly N/A.',
        FR_PROJECTION_SEMANTICS,
        'JSON contains original reward/value/standardized advantage examples, exact terminal raw GAE, and per-phase raw-GAE statistics. Nonterminal raw GAE is not guessed.',
        'Nonterminal final rollout boundaries are partial episodes, not failures or successes. No tensor, policy, reward, or runtime was modified.']
    with md.open('x',encoding='utf-8') as stream: stream.write('\n'.join(text)+'\n')
    print(json.dumps({'json':str(target),'markdown':str(md),'optimized_decisions':result['total_optimized_decisions']}))


if __name__ == '__main__':
    main()
