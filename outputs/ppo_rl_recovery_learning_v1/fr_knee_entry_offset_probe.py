"""NOT EXECUTED: CP230144 natural-P01 single-FR-knee direction diagnostic.

No production/model imports on module import. A reviewed explicit flag and the
ordinary single-Isaac lock precede all numeric/physics imports. This is not PPO,
not AUX, not a teacher prefix, and not formal learned-policy success.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

import student_entry_direction_probe as common

ROOT = common.ROOT
HEAD = '59e868f3e223c589e7645a0f5d63f91fa6119fb6'
CHECKPOINT = ROOT / 'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000230144.pt'
CP_SHA = '376ab4b6fe6f2a6a82bedd1190ba0814dc5e5b396127c48c34cf7b1d4a8265f1'
MANIFEST_SHA = '30dc90ece977823762eb7f19b79b944f41b20e14c5231c45014ab58d17db3ba9'
LAYOUT = 'role422_rear_owner_recovery_v1'
ZERO = dict({key: value for key, value in common.ZERO_CREDIT.items()
             if key != 'teacher_actions_deployed'},
            teacher_prefix_sample_credit=0, policy_credit=False,
            production_realtime_teacher_deployed=False,
            isolated_diagnostic_intervention_enabled=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def finite12(values):
    require(isinstance(values, (tuple, list)) and len(values) == 12 and
            all(type(v) in (int, float) and math.isfinite(v) for v in values), 'finite Full12 required')
    return tuple(values)


def smooth(x):
    x = min(1., max(0., x))
    return x * x * (3. - 2. * x)


def replace_fr_request(original, cap, request):
    original = finite12(original)
    require(math.isfinite(cap) and cap > 0 and math.isfinite(request) and abs(request) < cap,
            'finite unsaturated FR REQUEST required')
    if request == cap * math.tanh(original[3]):
        return original  # exact no-change control; do not round-trip tanh/atanh
    result = list(original)
    result[3] = math.atanh(request / cap)
    return tuple(result)


def eligibility(task, *, phase, tick, force_floor):
    ev = task.get('physical_evaluator') or {}
    legs = ev.get('current_legs') or {}
    blockers = []
    if phase != 'P05' or task.get('stage_id') != 'P05':
        blockers.append('not_P05')
    if (ev.get('valid') is not True or ev.get('physics_tick') != tick or
            task.get('termination_reason') is not None or ev.get('termination_reason') is not None):
        blockers.append('no_fresh_nonterminal_physical_state')
    for leg in ('FR', 'RL'):
        item = legs.get(leg) or {}
        force = item.get('bearing_force_n')
        if (item.get('air') is not False or item.get('support') is not True or
                item.get('bearing_verified') is not True or type(force) not in (int, float) or
                not math.isfinite(force) or force < force_floor):
            blockers.append(leg + '_current_bearing_absent')
    fr = legs.get('FR') or {}
    if not (fr.get('top_contact') is True and fr.get('within_top_xy') is True and
            fr.get('within_lateral_span') is True and fr.get('ground_contact') is False):
        blockers.append('FR_current_legal_TOP_absent')
    return dict(eligible=not blockers, blockers=blockers, tick=tick,
                phase=phase, stage_age_s=task.get('stage_age_s'),
                FR=deepcopy(fr), RL=deepcopy(legs.get('RL') or {}))


class EntryOffset:
    """Fixed -4 degree entry REQUEST; no repeated residual subtraction.

    1 s smooth ramp, 1 s hold, 1 s smooth blend to the CURRENT conditional
    student request. The ordinary projector still applies its 60 deg/s rate,
    masks, cap, same-tick actuator headroom and final slew. Early loss cancels
    new intervention at the next decision; no manipulations inside core.step.
    """
    def __init__(self):
        self.state = dict(status='WAIT', start_tick=None, entry_filtered_request_deg=None,
                          entry_cap_deg=None, release_requested_tick=None,
                          release_effective_tick=None, release_reason=None,
                          modified_decisions=0, trigger=None, last_command=None)

    def observe(self, tick, evidence):
        if self.state['status'] == 'ACTIVE' and not evidence['eligible']:
            self.state.update(status='RELEASE_PENDING', release_requested_tick=tick,
                              release_reason='current_eligibility_lost:' + ','.join(evidence['blockers']))

    def choose(self, original, *, tick, cap, filtered_request, evidence):
        original = finite12(original)
        require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0, 'finite positive cap required')
        require(math.isfinite(filtered_request), 'finite prior filtered REQUEST required')
        state = self.state
        self.observe(tick, evidence)
        if state['status'] == 'RELEASE_PENDING':
            state.update(status='RELEASED', release_effective_tick=tick)
        age = evidence.get('stage_age_s')
        if state['status'] == 'WAIT' and evidence.get('phase') == 'P05':
            require(type(age) in (int, float) and math.isfinite(age), 'P05 age required')
            if age > .75:
                state.update(status='NOT_TRIGGERED', release_reason='early_P05_window_missed')
            elif age >= .5 and evidence['eligible']:
                # Fail closed instead of clipping an unreachable anchor into a
                # deceptively different diagnostic. This is not a task gate.
                if abs(filtered_request) >= cap or abs(filtered_request - 4.) >= cap:
                    state.update(status='NOT_TRIGGERED', release_reason='entry_minus4_outside_current_cap')
                else:
                    state.update(status='ACTIVE', start_tick=tick,
                                 entry_filtered_request_deg=filtered_request, entry_cap_deg=cap,
                                 trigger=deepcopy(evidence))
        if state['status'] == 'ACTIVE':
            if abs(cap - state['entry_cap_deg']) > 1e-9:
                state.update(status='RELEASED', release_requested_tick=tick,
                             release_effective_tick=tick, release_reason='cap_changed')
            elif tick - state['start_tick'] >= 360:
                state.update(status='RELEASED', release_requested_tick=tick,
                             release_effective_tick=tick, release_reason='three_second_window_complete')
        applied = list(original)
        audit = dict(student_request_deg=cap * math.tanh(original[3]),
                     prior_filtered_request_deg=filtered_request, diagnostic_active=False)
        if state['status'] == 'ACTIVE':
            elapsed = (tick - state['start_tick']) / 120.
            anchor = state['entry_filtered_request_deg']
            if elapsed < 1.:
                request = anchor - 4. * smooth(elapsed)
                mode = 'RAMP_FROM_ENTRY'
            elif elapsed < 2.:
                request = anchor - 4.
                mode = 'HOLD_ENTRY_MINUS4'
            else:
                weight = smooth(elapsed - 2.)
                request = (1. - weight) * (anchor - 4.) + weight * audit['student_request_deg']
                mode = 'BLEND_TO_CURRENT_STUDENT'
            require(abs(request) < cap, 'finite unsaturated inverse required')
            applied = list(replace_fr_request(original, cap, request))
            audit.update(diagnostic_active=True, mode=mode, elapsed_s=elapsed,
                         desired_REQUEST_deg=request, fixed_entry_delta_deg=request-anchor,
                         raw_replacement=applied[3])
        changed = [i for i, (a, b) in enumerate(zip(original, applied)) if a != b]
        require(changed in ([], [3]), 'only FR knee raw may change')
        state['modified_decisions'] += bool(changed)
        state['last_command'] = audit
        return tuple(applied), changed, audit

    def finish(self, tick, reason):
        if self.state['status'] in ('ACTIVE', 'RELEASE_PENDING'):
            if self.state['release_requested_tick'] is None:
                self.state.update(release_requested_tick=tick, release_reason=reason)
            self.state.update(status='RELEASED', release_effective_tick=tick)


def original_density(original, request):
    mean = finite12(request['conditional_mean_full12'])
    sigma = finite12(request['effective_sigma_full12'])
    require(all(s > 0 for s in sigma), 'positive logged sigma required')
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi)
               for x, m, s in zip(original, mean, sigma))


def validate_step(info, applied):
    audit = info.get('actuator_target_effect_audit') or {}
    summary = info.get('actuator_target_effect_audit_summary') or {}
    require(tuple(info.get('raw_policy_action_full12', ())) == applied and
            tuple(audit.get('raw_policy_action_full12', ())) == applied and
            summary.get('all_ticks_verified') is True and
            summary.get('physics_ticks') == info.get('physics_ticks') and
            info.get('no_in_episode_state_writes_verified') is True,
            'ordinary applied raw/one-write/no-state-injection audit failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-decisions', type=int, default=900)
    parser.add_argument('--execute-reviewed-real-diagnostic', action='store_true')
    args = parser.parse_args(argv)
    require(args.execute_reviewed_real_diagnostic, 'NOT EXECUTED: explicit reviewed flag required')
    require(1 <= args.max_decisions <= 900, 'bounded1..900 diagnostic decisions required')
    args.checkpoint = CHECKPOINT.resolve(strict=True)
    sidecar = args.checkpoint.with_name(args.checkpoint.stem + '_manifest.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    require(common.sha256(args.checkpoint) == CP_SHA and common.sha256(sidecar) == MANIFEST_SHA
            and metadata.get('save_load_round_trip') is True
            and metadata.get('global_policy_decisions') == 230144, 'exact sealed CP230144 required')
    args.expected_head, args.semantic_version = HEAD, 'v3'
    args.experiment_id, args.device = 'rr_rl_timing_policy_learning_v1', 'cuda:0'
    common._policy_args(args, metadata)
    require(metadata['policy_contract']['observation_dimension'] == 439 and
            metadata['policy_contract']['observation_layout'] == LAYOUT, 'owner439 required')
    run = args.run_dir.resolve()
    allowed = (ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe').resolve()
    require(run.is_relative_to(allowed) and run != allowed and not run.exists(), 'new isolated diagnostic child required')
    probe = EntryOffset()
    manifest = dict(schema='outputs.fr_knee_entry_offset_probe.v1', lifecycle='RUNNING', **ZERO,
        run_role='NATURAL_P01_CP230144_SINGLE_FR_KNEE_DIRECTION_DIAGNOSTIC',
        formal_policy_or_success_claim=False, natural_P01_student_prefix=True,
        successful_N_prefix_used=False, state_injection_or_teleport=False,
        checkpoint=str(args.checkpoint), checkpoint_sha256=CP_SHA, checkpoint_manifest_sha256=MANIFEST_SHA,
        expected_head=HEAD, seed=4001, deterministic=True,
        runner_sha256=common.sha256(Path(__file__)), shared_helper_sha256=common.sha256(Path(common.__file__)),
        maximum_intervention_s=3., maximum_diagnostic_decisions=args.max_decisions,
        maximum_diagnostic_seconds=args.max_decisions/15., global_native_cap_s=200.,
        candidate_semantics='raw[3] inverse-tanh of fixed entry filtered REQUEST minus up to4deg; 1s ramp/1s hold/1s release; other11 current student raw untouched',
        history_semantics='ordinary actual-applied raw and filtered-request HISTORY retained; intervention influences subsequent student conditioning; no imaginary baseline history or instant release equivalence',
        log_probability_semantics='read-only original conditional Gaussian density only; applied intervention is not an on-policy sample',
        cutoff_semantics='independent zero-credit diagnostic ends at reviewed decision budget (default900/60s), or earlier ordinary terminal; neither a cutoff nor reached physical event is formal PPO success',
        release_latency='first physical eligibility loss latched; no next modified decision, <=7 remaining physics ticks; native safety remains immediate',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    lock = (ROOT / 'runs/ppo_semantic_v2/.single_process.lock').open('r+b')
    app = core = physical = unchanged = None
    created = False
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        sys.path.insert(0, str(ROOT / 'src'))
        from wlr50_clean.ppo.semantic_cli import runtime_contract, _checkpoint_collection_options
        contract = runtime_contract(expected_head=HEAD, semantic_version='v3', experiment_id=args.experiment_id)
        require(metadata['runtime_contract'] == contract, 'exact frozen59e runtime required; no diagnostic migration')
        collection = _checkpoint_collection_options(args)
        require(collection and metadata['runner_config']['num_steps_per_env'] == 512, 'explicit512 checkpoint options required')
        manifest.update(runtime_contract=contract, collection_loader_options=collection)
        run.mkdir(parents=True, exist_ok=False)
        created = True
        common.exclusive_json(run / 'run_manifest.started.json', manifest)
        import torch  # noqa: F401; runtime only after sole-Isaac lock
        import tensordict  # noqa: F401; retain proven Windows native import order
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_training import seed_training_rngs, jsonable
        from wlr50_clean.ppo.semantic_video_cli import build_video_core, checkpoint_loader
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder, physical_json
        seed_training_rngs(4001)
        core = build_video_core(app, role='C', semantic_version='v3', experiment_id=args.experiment_id)
        observation = tuple(core.reset(seed=4001))
        require(core.frame.state_id == 'P01' and core.frame.physics_tick == 0 and len(observation) == 439,
                'natural same-policy P01 initialization required')
        assists = common._rear_assists_off(core.backend, core.frame)
        require(assists['front_capture_assist_present'], 'existing FL capture assist must remain ON')
        manifest['assist_receipt'] = assists
        for phase in (f'P{i:02d}' for i in range(1, 14)):
            require(tuple(core.projector.config.mask_for(phase)) == (1,)*12, 'ordinary Full12 permission required')
        action, proof, unchanged = checkpoint_loader(args, contract)(observation)
        manifest['official_checkpoint_load_proof'] = proof
        config = ROOT / 'configs/ppo_rr_rl_timing_policy_learning_v1'
        physical = PhysicalEvaluationRecorder(run, task_spec_path=config/'stage_task_spec.yaml', quality_score_path=config/'quality_score.yaml')
        physical.start(core.frame)
        force_floor = core.backend._controller.supervisor.spec['support']['force_noise_floor_n']
        decision = {}
        def evidence(frame):
            return eligibility(frame.info['semantic_task'], phase=frame.state_id,
                               tick=frame.physics_tick, force_floor=force_floor)
        with ExitStack() as streams:
            ticks = streams.enter_context((run/'diagnostic_physics.jsonl').open('x', encoding='utf-8'))
            decisions = streams.enter_context((run/'diagnostic_decisions.jsonl').open('x', encoding='utf-8'))
            def emit(stream, row):
                stream.write(json.dumps(common.diagnostic_json_payload(row, physical_json=physical_json, jsonable=jsonable), allow_nan=False)+'\n')
                stream.flush()
            def observe(before, after, projection):
                physical.observe(before, after, projection)
                ack, audit = after.info['atomic_ack'], after.info['actuator_target_effect_audit']
                require(audit.get('verified') is True and ack.get('articulation_writes_this_call') == 1,
                        'single ordinary audited articulation write required')
                current = evidence(after)
                probe.observe(after.physics_tick, current)
                # Compact explicit action chain plus full measured raw observation.
                emit(ticks, {**ZERO, 'episode_physics_tick': after.physics_tick,
                    'sim_time_s': after.sim_time_s, 'decision_start_tick': decision.get('decision_start_tick'),
                    'source_nominal_full12': before.nominal_action_full12,
                    'projected_residual_full12': projection.safe_projected_residual_full12,
                    'actual_drive_target_full12': after.info['drive_target_full12'],
                    'atomic_ACK': ack, 'native_readback_audit': audit,
                    'raw_observation': after.info.get('raw_observation'),
                    'physical_evaluator': after.info['semantic_task']['physical_evaluator'],
                    'eligibility': current, 'probe_state': deepcopy(probe.state)})
            core.tick_observer = observe
            cutoff = None
            while not core.done and core.frame.sim_time_s < 200.-1e-10:
                if core.decision_count >= args.max_decisions:
                    cutoff = 'reviewed_zero_credit_diagnostic_decision_budget'
                    break
                original = tuple(action(observation, core.decision_count))
                request = deepcopy(action.last_request)
                require(tuple(request['selected_raw_full12']) == original and request['sampling_draws'] == 0,
                        'unmodified deterministic original request required')
                cap = core.projector.config.scale_for(core.frame.state_id)[3] * core.projector.config.physical_residual_scale_full12[3]
                require(abs(cap-request['current_cap_full12'][3]) < 1e-6, 'actor/projector cap mismatch')
                filtered = core.bridge.previous_projected_residual_full12[3]
                require(abs(filtered-request['previous_filtered_request_full12'][3]) < 2e-5,
                        'actor observation/real filtered REQUEST mismatch')
                current = evidence(core.frame)
                applied, changed, candidate = probe.choose(original, tick=core.frame.physics_tick,
                    cap=cap, filtered_request=filtered, evidence=current)
                decision = dict(decision_index=core.decision_count, decision_start_tick=core.frame.physics_tick,
                    decision_start_time_s=core.frame.sim_time_s, phase=core.frame.state_id,
                    policy_observation_vector=list(observation), original_student_raw_full12=list(original),
                    original_student_request_audit=request, original_gaussian_log_density_readonly=original_density(original, request),
                    applied_raw_full12=list(applied), applied_log_probability_for_PPO=None,
                    overridden_indices=changed, intervention_issued_for_this_decision=bool(changed),
                    eligibility=current, candidate=candidate)
                step = core.step(applied)
                observation = tuple(step.observation)
                validate_step(step.info, applied)
                emit(decisions, {**ZERO, **decision, 'decision_end_tick': core.frame.physics_tick,
                    'step_info': step.info, 'probe_state_after_step': deepcopy(probe.state)})
        probe.finish(core.frame.physics_tick, 'ordinary_terminal_or_diagnostic_cutoff')
        unchanged()
        require(runtime_contract(expected_head=HEAD, semantic_version='v3', experiment_id=args.experiment_id) == contract
                and common.sha256(args.checkpoint) == CP_SHA and common.sha256(sidecar) == MANIFEST_SHA,
                'runtime/checkpoint changed during diagnostic')
        manifest.update(lifecycle='DIAGNOSTIC_SEALED', probe=deepcopy(probe.state), physical_summary=physical.summary(),
            physical_seconds=core.frame.sim_time_s, final_phase=core.frame.state_id,
            actual_task_termination_reason=core.frame.info['semantic_task'].get('termination_reason'),
            diagnostic_cutoff=cutoff, native_episode_terminal=core.done,
            task_success_claim=False, frozen_learned_state_unchanged=True)
    except BaseException as exc:
        manifest.update(lifecycle='DIAGNOSTIC_ERROR', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if unchanged is not None:
            try:
                unchanged()
                manifest['frozen_learned_state_unchanged'] = True
            except BaseException as exc:
                manifest.update(lifecycle='DIAGNOSTIC_ERROR', frozen_learned_state_unchanged=False, integrity_error=repr(exc))
        if physical is not None:
            physical.close()
        if core is not None and core.frame is not None:
            probe.finish(core.frame.physics_tick, 'shutdown_no_further_dispatch')
            manifest.update(last_episode_tick=core.frame.physics_tick, last_simulation_time_s=core.frame.sim_time_s)
        manifest.update(probe=deepcopy(probe.state), completed_at_utc=datetime.now(timezone.utc).isoformat())
        if created:
            common.exclusive_json(run/'run_manifest.json', manifest)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        lock.close()


if __name__ == '__main__':
    main()
