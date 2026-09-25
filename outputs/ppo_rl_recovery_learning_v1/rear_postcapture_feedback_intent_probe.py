"""NOT EXECUTED: isolated P10 single-channel desired-FINAL feedback intent.

This is NOT an exact FINAL intervention. One raw value is held for eight native
ticks while the ordinary mapper, owner and limits evolve. An immutable entry
FINAL defines the intent; the previous committed native baseline gives only a
bounded REQUEST approximation. Actual errors and ordinary HISTORY are logged.
No Torch/Isaac imports occur before the explicit flag and sole-resource lock.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import traceback

import fr_knee_entry_offset_probe as front

common = front.common
ROOT = common.ROOT
CHANNELS = {'FL_knee': 1, 'FR_knee': 3, 'RR_hip': 6}
BRANCH = 'cp225280_front_preserved_v1'
IDENTITY = 'front_preservation439_branch_identity'
LAYOUT = 'role422_rear_owner_recovery_v1'
ZERO = dict(front.ZERO, diagnostic_rows_entered_in_on_policy_storage=False,
            formal_deterministic_policy_result=False)
require, finite12, smooth = front.require, front.finite12, front.smooth


def replace_request(raw, *, index, cap, request):
    raw = finite12(raw)
    require(index in CHANNELS.values(), 'allowlisted single channel required')
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0,
            'finite positive cap required')
    require(type(request) in (int, float) and math.isfinite(request) and abs(request) < cap,
            'bounded finite inverse-tanh request required')
    if request == cap * math.tanh(raw[index]):
        return raw
    result = list(raw)
    result[index] = math.atanh(request / cap)
    return tuple(result)


def eligibility(task, *, phase, tick, dependency):
    """Use the EXISTING rear_dependency result, plus a fresh legal TOP record."""
    ev = task.get('physical_evaluator') or {}
    rr = (ev.get('current_legs') or {}).get('RR') or {}
    blockers = []
    if phase not in ('P10', 'P11', 'P12') or task.get('stage_id') != phase:
        blockers.append('outside_P10_P11_P12')
    if (ev.get('valid') is not True or ev.get('physics_tick') != tick or
            task.get('termination_reason') is not None or ev.get('termination_reason') is not None):
        blockers.append('stale_invalid_or_terminal_physics')
    if not (dependency.get('rr_top_contact') is True and
            dependency.get('rr_current_bearing') is True and
            dependency.get('support_transfer_permitted') is True):
        blockers.append('current_rear_dependency_permission_absent')
    if not (rr.get('top_contact') is True and rr.get('top_surface_contact') is True and
            rr.get('contact_surface') == 'TOP' and rr.get('within_top_xy') is True and
            rr.get('ground_contact') is False and rr.get('air') is False and
            rr.get('support') is True and rr.get('bearing_verified') is True):
        blockers.append('RR_current_legal_TOP_bearing_absent')
    return dict(eligible=not blockers, blockers=blockers, tick=tick, phase=phase,
                dependency=deepcopy(dependency), RR=deepcopy(rr))


def committed_snapshot(frame, *, index):
    """Read only the latest real ACK; never advance or clone a live mapper."""
    q = frame.info['actuator_target_effect_audit']
    ack = frame.info['atomic_ack']
    h = q['policy_headroom_evidence']
    tracking = q['tracking_reference_evidence']
    bootstrap = tracking.get('bootstrap_physics_tick')
    final = finite12(frame.info['drive_target_full12'])
    baseline = finite12(h['baseline_native_plus_controller_full12'])
    require(type(bootstrap) is int and bootstrap > 0 and
            q.get('physics_tick') == bootstrap + frame.physics_tick - 1 and
            tracking.get('dispatch_physics_tick') == q.get('physics_tick') and
            q.get('verified') is True and q.get('actual_mapping_matches_dispatch') is True and
            q.get('physics_tick') == ack.get('physics_tick') and
            tuple(ack.get('drive_target_full12', ())) == final and
            tuple(q.get('native_drive_target_full12', ())) == tuple(ack.get('native_drive_target_full12', ())),
            'current committed native ACK provenance required')
    lo, hi = h['servo_safety_limits_deg'][index]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (lo, hi)) and lo < hi,
            'ordinary reserved physical band required')
    return dict(episode_tick=frame.physics_tick, native_dispatch_tick=q['physics_tick'],
                bootstrap_native_dispatch_tick=bootstrap,
                final_deg=final[index], native_baseline_deg=baseline[index],
                safety_band_deg=[lo, hi], selected_index=index,
                baseline_semantics='previous_committed_native_plus_controller_before_policy_owner_final_slew',
                exact_future_FINAL_prediction=False)


class FeedbackIntent:
    """One P10 handoff only; fixed entry FINAL intent, not cumulative offsets.

    Nominal 1s smooth ramp + 1s hold + 1s smooth release. Loss is latched at
    120Hz: the next decision has no override (at most 7 remaining native ticks).
    No tick observer modifies raw, targets, physical state or HISTORY.
    """
    def __init__(self, *, channel, offset_deg, handoff_tick):
        require(channel in CHANNELS, 'allowlisted channel required')
        require(type(offset_deg) in (int, float) and offset_deg in (-4., 4.),
                'reviewed finite +4 or -4 degree entry offset required')
        require(type(handoff_tick) is int and handoff_tick > 0, 'real P10 handoff tick required')
        self.index, self.offset, self.handoff_tick = CHANNELS[channel], float(offset_deg), handoff_tick
        self.state = dict(status='WAIT', channel=channel, selected_index=self.index,
            requested_offset_deg=self.offset, start_tick=None, entry_final_deg=None,
            entry_filtered_request_deg=None, entry_cap_deg=None, release_requested_tick=None,
            release_effective_tick=None, release_reason=None, modified_decisions=0,
            trigger=None, last_command=None)

    def observe(self, tick, evidence, *, cap=None):
        if self.state['status'] != 'ACTIVE':
            return
        reason = None
        if not evidence['eligible']:
            reason = 'eligibility_lost:' + ','.join(evidence['blockers'])
        elif cap is not None and abs(cap - self.state['entry_cap_deg']) > 1e-9:
            reason = 'selected_cap_changed'
        if reason is not None:
            self.state.update(status='RELEASE_PENDING', release_requested_tick=tick,
                              release_reason=reason)

    def _release(self, tick, reason):
        if self.state['release_requested_tick'] is None:
            self.state.update(release_requested_tick=tick, release_reason=reason)
        self.state.update(status='RELEASED', release_effective_tick=tick)

    def choose(self, original, *, tick, cap, filtered_request, committed, evidence):
        original = finite12(original)
        require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0,
                'finite selected cap required')
        require(type(filtered_request) in (int, float) and math.isfinite(filtered_request),
                'finite real filtered REQUEST required')
        state = self.state
        fresh = (committed.get('episode_tick') == tick and
                 committed.get('selected_index') == self.index)
        values = [committed.get(k) for k in ('final_deg', 'native_baseline_deg')]
        fresh = fresh and all(type(v) in (int, float) and math.isfinite(v) for v in values)
        self.observe(tick, evidence, cap=cap)
        if state['status'] == 'RELEASE_PENDING':
            self._release(tick, state['release_reason'])
        if state['status'] == 'WAIT':
            reason = None
            if tick != self.handoff_tick or evidence.get('phase') != 'P10':
                reason = 'first_P10_handoff_missed'
            elif not evidence['eligible']:
                reason = 'first_P10_handoff_not_supported'
            elif not fresh:
                reason = 'stale_or_invalid_committed_baseline'
            else:
                lo, hi = committed['safety_band_deg']
                target = committed['final_deg'] + self.offset
                required = target - committed['native_baseline_deg']
                if not (lo <= target <= hi and abs(filtered_request) < cap and abs(required) < cap):
                    reason = 'entry_intent_outside_current_cap_or_reserved_band'
            if reason:
                state.update(status='NOT_TRIGGERED', release_reason=reason)
            else:
                state.update(status='ACTIVE', start_tick=tick,
                    entry_final_deg=committed['final_deg'], entry_filtered_request_deg=filtered_request,
                    entry_cap_deg=cap, trigger=deepcopy(evidence))
        if state['status'] == 'ACTIVE':
            if not fresh:
                self._release(tick, 'stale_or_invalid_committed_baseline')
            elif tick - state['start_tick'] >= 360:
                self._release(tick, 'three_second_intent_window_complete')
        student_request = cap * math.tanh(original[self.index])
        audit = dict(diagnostic_active=False, student_unfiltered_request_deg=student_request,
                     previous_filtered_request_deg=filtered_request,
                     committed=deepcopy(committed), exact_FINAL_intervention=False)
        applied = original
        if state['status'] == 'ACTIVE':
            elapsed = (tick - state['start_tick']) / 120.
            anchor = state['entry_final_deg']
            goal = anchor + self.offset
            baseline = committed['native_baseline_deg']
            if elapsed < 1.:
                desired = anchor + self.offset * smooth(elapsed)
                mode = 'RAMP_FIXED_ENTRY_FINAL_INTENT'
            elif elapsed < 2.:
                desired = goal
                mode = 'HOLD_FIXED_ENTRY_FINAL_INTENT'
            else:
                weight = smooth(elapsed - 2.)
                desired = (1.-weight) * goal + weight * (baseline + student_request)
                mode = 'BLEND_INTENT_TO_CURRENT_STUDENT_REQUEST'
            unbounded_request = desired - baseline
            # At age zero, retain actual committed filtered REQUEST exactly;
            # do not jump to a fictitious inversion of owner/slew history.
            request = filtered_request if elapsed == 0. else unbounded_request
            bound = math.nextafter(cap, 0.)
            bounded = min(bound, max(-bound, request))
            applied = replace_request(original, index=self.index, cap=cap, request=bounded)
            audit.update(diagnostic_active=True, elapsed_s=elapsed, mode=mode,
                fixed_entry_final_deg=anchor, fixed_goal_final_deg=goal,
                desired_FINAL_intent_deg=desired,
                committed_FINAL_intent_error_deg=desired-committed['final_deg'],
                inferred_REQUEST_before_cap_deg=unbounded_request,
                issued_REQUEST_before_native_filter_deg=bounded,
                request_cap_clipped=bounded != request,
                first_step_retains_committed_filtered_REQUEST=elapsed == 0.,
                baseline_estimator_has_no_owner_or_future_mapper_guarantee=True)
        changed = [i for i, (a, b) in enumerate(zip(original, applied)) if a != b]
        require(changed in ([], [self.index]), 'other11 original student raw channels changed')
        state['modified_decisions'] += bool(changed)
        state['last_command'] = deepcopy(audit)
        return applied, changed, audit

    def finish(self, tick, reason):
        if self.state['status'] in ('ACTIVE', 'RELEASE_PENDING'):
            self._release(tick, reason)


class NoOverrideControl:
    """True same-prefix control: all12 current student raw pass unchanged.

    Zero offset with an entry-target hold would not be a control. This class
    never calls the inverse replacement, nor holds a joint/request/FINAL.
    """
    def __init__(self, *, channel, handoff_tick):
        require(channel in CHANNELS and type(handoff_tick) is int and handoff_tick > 0,
                'allowlisted diagnostic channel and real handoff required')
        self.index, self.handoff_tick = CHANNELS[channel], handoff_tick
        self.state = dict(status='WAIT_CONTROL', channel=channel, selected_index=self.index,
            no_override_control=True, requested_offset_deg=None, entry_final_deg=None,
            start_tick=None, modified_decisions=0, trigger=None, last_command=None,
            first_eligibility_loss_tick=None, first_eligibility_loss_blockers=None)

    def observe(self, tick, evidence, *, cap=None):
        if (self.state['status'] == 'CONTROL_ACTIVE' and not evidence['eligible'] and
                self.state['first_eligibility_loss_tick'] is None):
            self.state.update(first_eligibility_loss_tick=tick,
                              first_eligibility_loss_blockers=list(evidence['blockers']))

    def choose(self, original, *, tick, cap, filtered_request, committed, evidence):
        original = finite12(original)
        if self.state['status'] == 'WAIT_CONTROL':
            accepted = (tick == self.handoff_tick and evidence.get('phase') == 'P10' and
                        evidence['eligible'] and committed.get('episode_tick') == tick and
                        committed.get('selected_index') == self.index)
            self.state.update(status='CONTROL_ACTIVE' if accepted else 'CONTROL_ENTRY_INELIGIBLE',
                start_tick=tick, entry_final_deg=committed.get('final_deg'), trigger=deepcopy(evidence),
                entry_comparable=accepted)
        self.observe(tick, evidence, cap=cap)
        audit = dict(diagnostic_active=False, no_override_control=True,
            student_unfiltered_request_deg=cap*math.tanh(original[self.index]),
            previous_filtered_request_deg=filtered_request, committed=deepcopy(committed),
            desired_FINAL_intent_deg=None, exact_FINAL_intervention=False,
            mode='ALL12_CURRENT_STUDENT_RAW_UNMODIFIED')
        self.state['last_command'] = deepcopy(audit)
        return original, [], audit

    def finish(self, tick, reason):
        if self.state['status'] != 'CONTROL_SEALED':
            self.state.update(status='CONTROL_SEALED', finished_tick=tick, finish_reason=reason)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--checkpoint-manifest-sha256', required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--channel', choices=sorted(CHANNELS), required=True)
    arm = p.add_mutually_exclusive_group(required=True)
    arm.add_argument('--entry-final-offset-deg', type=float, choices=(-4., 4.))
    arm.add_argument('--no-override-control', action='store_true',
        help='True all12 raw pass-through control; NOT a zero-offset entry hold')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-student-decisions', type=int, default=120)
    p.add_argument('--execute-reviewed-real-diagnostic', action='store_true')
    return p


def checkpoint_binding(args):
    require(re.fullmatch('[0-9a-f]{40}', args.expected_head) is not None and
            all(re.fullmatch('[0-9a-f]{64}', x) is not None for x in
                (args.checkpoint_sha256, args.checkpoint_manifest_sha256)), 'exact head/hash CLI binding required')
    args.checkpoint = args.checkpoint.resolve(strict=True)
    allowed = (ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches'/BRANCH/'checkpoints/history').resolve()
    require(args.checkpoint.parent == allowed, 'explicit front-preserved branch history checkpoint required')
    sidecar = args.checkpoint.with_name(args.checkpoint.stem+'_manifest.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    require(common.sha256(args.checkpoint) == args.checkpoint_sha256 and
            common.sha256(sidecar) == args.checkpoint_manifest_sha256 and
            metadata.get('checkpoint_sha256') == args.checkpoint_sha256 and
            metadata.get('save_load_round_trip') is True, 'sealed checkpoint/sidecar hash or roundtrip mismatch')
    require(metadata.get(IDENTITY, {}).get('schema') == 'wlr50_clean.cp225280_front_preserved439.v1' and
            metadata['policy_contract']['observation_dimension'] == 439 and
            metadata['policy_contract']['observation_layout'] == LAYOUT and
            metadata['runner_config']['num_steps_per_env'] == 512,
            'front-preserved owner439/collection512 identity required; official loader validates full receipt')
    return metadata, sidecar


def independent_cleanup(manifest, actions, *, primary_error=None, manifest_writer=None,
                        error_receipt_writer=None, error_reporter=None):
    """Attempt every cleanup; never replace an already-pending main exception.

    Resource closes precede the final manifest, so it can contain close errors.
    A failed/partial exclusive manifest is not overwritten: an independent
    error receipt is attempted, then stderr if that receipt cannot be written.
    Without a primary error, re-raise the FIRST cleanup error only after all
    resource and evidence attempts. All callbacks are independently protected.
    """
    errors = []
    first = None
    manifest['cleanup_errors'] = errors
    manifest['primary_exception_preserved'] = None if primary_error is None else repr(primary_error)

    def attempt(name, callback):
        nonlocal first
        if callback is None:
            return True
        try:
            callback()
            return True
        except BaseException as exc:
            if first is None:
                first = (exc, exc.__traceback__)
            errors.append(dict(step=name, error=repr(exc), traceback=traceback.format_exc()))
            manifest['lifecycle'] = 'DIAGNOSTIC_ERROR'
            return False

    for name, callback in actions:
        attempt(name, callback)
    manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    attempt('final_manifest_write', manifest_writer)
    if errors:
        def receipt():
            return dict(schema='outputs.rear_feedback_intent_cleanup_errors.v1',
                        primary_exception=manifest['primary_exception_preserved'],
                        cleanup_errors=deepcopy(errors), lifecycle='DIAGNOSTIC_ERROR')
        written = False
        if error_receipt_writer is not None:
            written = attempt('cleanup_error_receipt_write', lambda: error_receipt_writer(receipt()))
        if not written:
            reporter = error_reporter or (lambda row: print(json.dumps(row, allow_nan=False), file=sys.stderr))
            attempt('cleanup_error_stderr_report', lambda: reporter(receipt()))
    if first is not None and primary_error is None:
        raise first[0].with_traceback(first[1])
    return errors


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.execute_reviewed_real_diagnostic, 'NOT EXECUTED: explicit reviewed flag required')
    require(1 <= args.max_student_decisions <= 900, 'bounded1..900 post-prefix diagnostic decisions required')
    metadata, sidecar = checkpoint_binding(args)
    args.semantic_version, args.experiment_id, args.device = 'v3', 'rr_rl_timing_policy_learning_v1', 'cuda:0'
    common._policy_args(args, metadata)
    args.seed = 1001  # Match the real nominal-prefix course, not locked video seed4001.
    run = args.run_dir.resolve()
    allowed = (ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe').resolve()
    require(run.is_relative_to(allowed) and run != allowed and not run.exists(), 'new isolated diagnostic child required')
    credits = dict(ZERO, isolated_diagnostic_intervention_enabled=not args.no_override_control,
        diagnostic_arm='NO_OVERRIDE_CONTROL' if args.no_override_control else 'ENTRY_FINAL_FEEDBACK_INTENT')
    manifest = dict(schema='outputs.rear_postcapture_feedback_intent_probe.v1', lifecycle='RUNNING', **credits,
        run_role=('DIAGNOSTIC_NOMINAL_P10_PREFIX_ALL12_RAW_CONTROL' if args.no_override_control
                  else 'DIAGNOSTIC_NOMINAL_P10_PREFIX_SINGLE_CHANNEL_FEEDBACK_INTENT'),
        formal_policy_or_success_claim=False, exact_FINAL_intervention=False,
        successful_nominal_prefix_used=True, natural_P01_full_policy_evaluation=False,
        state_injection_or_teleport=False, deterministic=True, seed=1001,
        checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256, expected_head=args.expected_head,
        source_global_policy_decisions=metadata['global_policy_decisions'], channel=args.channel,
        fixed_entry_final_offset_deg=args.entry_final_offset_deg,
        maximum_intervention_s=0. if args.no_override_control else 3., comparison_window_s=3.,
        maximum_student_decisions=args.max_student_decisions, maximum_prefix_decisions=1800,
        global_native_cap_s=200., runner_sha256=common.sha256(Path(__file__)),
        shared_helper_sha256=common.sha256(Path(common.__file__)),
        front_helper_sha256=common.sha256(Path(front.__file__)),
        intent_semantics='immutable entry FINAL plus offset; previous committed native baseline approximates REQUEST; not exact final tracking',
        history_semantics='ordinary actual applied raw/filtered REQUEST/FINAL HISTORY retained; no baseline history fiction or reset at release',
        likelihood_semantics='original deterministic student Gaussian density only; applied intervention has no PPO likelihood or credit',
        prefix_adapter_count_semantics='adapter credited_decisions count diagnostic suffix dispatches only; no on-policy storage or optimizer exists',
        raw_step_outcome_semantics='retained ordinary physical/suffix labels are not formal learned-policy success',
        entry_anchor_semantics='current committed FINAL at first real P10 student handoff; not an earlier within-decision native stage-transition tick',
        causal_scope='entryFINAL offset may counteract large student drift; not offset relative to moving student FINAL; compare actual same-checkpoint/seed/prefix no-override control',
        release_latency='loss latched each120Hz tick; no next modified decision; <=7 remaining native ticks; existing physical safety remains immediate',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    app = core = physical = unchanged = probe = prefix = None
    primary_error = None
    created = False
    lock = (ROOT/'runs/ppo_semantic_v2/.single_process.lock').open('r+b')
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        sys.path.insert(0, str(ROOT/'src'))
        from wlr50_clean.ppo.semantic_cli import runtime_contract, _checkpoint_collection_options
        contract = runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id=args.experiment_id)
        require(metadata['runtime_contract'] == contract, 'exact runtime required; no implicit migration')
        collection = _checkpoint_collection_options(args)
        require(collection, 'official explicit collection512 identity route required')
        manifest.update(runtime_contract=contract, collection_loader_options=collection)
        run.mkdir(parents=True, exist_ok=False)
        created = True
        common.exclusive_json(run/'run_manifest.started.json', manifest)
        import torch  # noqa: F401; only after explicit review flag and sole-Isaac lock
        import tensordict  # noqa: F401; proven Windows DLL import order
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_training import seed_training_rngs, jsonable
        from wlr50_clean.ppo.semantic_video_cli import build_video_core, checkpoint_loader
        from wlr50_clean.ppo.semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, CheckpointPolicyPrefixRslAdapter
        from wlr50_clean.ppo.semantic_rear_policy_timing import rear_dependency
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder, physical_json
        seed_training_rngs(args.seed)
        core = build_video_core(app, role='C', semantic_version='v3', experiment_id=args.experiment_id)
        config = ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1'
        physical = PhysicalEvaluationRecorder(run, task_spec_path=config/'stage_task_spec.yaml', quality_score_path=config/'quality_score.yaml')
        decision = {}
        def selected_cap(frame):
            return core.projector.config.scale_for(frame.state_id)[CHANNELS[args.channel]] * core.projector.config.physical_residual_scale_full12[CHANNELS[args.channel]]
        def evidence(frame):
            supervisor = core.backend._controller.supervisor
            task = frame.info['semantic_task']
            dep = rear_dependency(task, supervisor.spec['support'], mode=supervisor.spec['nominal']['rear_policy_timing'])
            return eligibility(task, phase=frame.state_id, tick=frame.physics_tick, dependency=dep)
        with ExitStack() as streams:
            ticks = streams.enter_context((run/'diagnostic_physics.jsonl').open('x', encoding='utf-8'))
            decisions = streams.enter_context((run/'diagnostic_decisions.jsonl').open('x', encoding='utf-8'))
            prefix_stream = streams.enter_context((run/'prefix_evidence.jsonl').open('x', encoding='utf-8'))
            def emit(stream, row):
                stream.write(json.dumps(common.diagnostic_json_payload(row, physical_json=physical_json, jsonable=jsonable), allow_nan=False)+'\n')
                stream.flush()
            def observe(before, after, projection):
                physical.observe(before, after, projection)
                audit, ack = after.info['actuator_target_effect_audit'], after.info['atomic_ack']
                require(audit.get('verified') is True and ack.get('articulation_writes_this_call') == 1,
                        'one ordinary audited articulation write required')
                ev = evidence(after)
                if probe is not None:
                    probe.observe(after.physics_tick, ev, cap=selected_cap(after))
                desired = (decision.get('candidate') or {}).get('desired_FINAL_intent_deg')
                applied_final = after.info['drive_target_full12'][CHANNELS[args.channel]]
                emit(ticks, {**credits, 'episode_physics_tick': after.physics_tick, 'sim_time_s': after.sim_time_s,
                    'diagnostic_student_decision': decision.get('student_decision'),
                    'desired_FINAL_intent_held_for_decision_deg': desired,
                    'actual_selected_FINAL_deg': applied_final,
                    'actual_FINAL_minus_desired_intent_deg': None if desired is None else applied_final-desired,
                    'source_nominal_full12': before.nominal_action_full12,
                    'projected_residual_full12': projection.safe_projected_residual_full12,
                    'actual_drive_target_full12': after.info['drive_target_full12'],
                    'atomic_ACK': ack, 'native_readback_audit': audit,
                    'raw_observation': after.info.get('raw_observation'),
                    'physical_evaluator': after.info['semantic_task']['physical_evaluator'],
                    'eligibility': ev, 'probe_state': None if probe is None else deepcopy(probe.state)})
            core.tick_observer = observe
            prefix = CheckpointPolicyPrefixRslAdapter(core, seed=args.seed, device=args.device,
                evidence_sink=lambda row: emit(prefix_stream, row),
                request=CheckpointPolicyPrefixRequest(target_phase='P10', teacher_offset_decisions=0,
                                                     maximum_prefix_decisions=1800, source='successful_nominal'))
            require(core.frame.state_id == 'P01' and core.frame.physics_tick == 0 and len(core.observation) == 439,
                    'untouched natural P01 bootstrap required')
            physical.start(core.frame)
            assists = common._rear_assists_off(core.backend, core.frame)
            require(assists['front_capture_assist_present'], 'existing FL capture assist must remain ON')
            manifest['assist_receipt'] = assists
            for phase in (f'P{i:02d}' for i in range(1, 14)):
                require(tuple(core.projector.config.mask_for(phase)) == (1,)*12, 'ordinary all12 residual permission required')
            action, proof, unchanged = checkpoint_loader(args, contract)(tuple(core.observation))
            manifest['official_checkpoint_load_proof'] = proof
            start = prefix.install_prefix_policy(lambda observation: (0.,)*12, {
                'schema': 'wlr50_clean.successful_nominal_prefix.v1', 'source': 'successful_nominal',
                'raw_action_full12': [0.]*12, 'policy_credit': False,
                'execution_profile_sha256': common.sha256(config/'execution_profile.yaml'),
                'stage_task_spec_sha256': common.sha256(config/'stage_task_spec.yaml'),
                'runtime_content_sha256': contract['runtime_content_sha256'],
                'interface_contract': {'observation_dimension':439, 'observation_layout':LAYOUT, 'action_dimension':12}})
            manifest['prefix_handoff'] = start
            require(start.get('mode') == 'successful_nominal_initialized_suffix' and
                    start.get('actual_phase') == 'P10' and core.frame.state_id == 'P10' and
                    start.get('requested_phase_still_active_at_credit') is True,
                    'real first P10 handoff required; no diagnostic on fallback')
            probe = (NoOverrideControl(channel=args.channel, handoff_tick=core.frame.physics_tick)
                     if args.no_override_control else
                     FeedbackIntent(channel=args.channel, offset_deg=args.entry_final_offset_deg,
                                    handoff_tick=core.frame.physics_tick))
            observation = tuple(core.observation)
            cutoff = None
            for student_index in range(args.max_student_decisions):
                if core.done or core.frame.sim_time_s >= 200.-1e-10:
                    break
                original = tuple(action(observation, core.decision_count))
                request = deepcopy(action.last_request)
                require(tuple(request['selected_raw_full12']) == original and request['sampling_draws'] == 0,
                        'unmodified deterministic current student request required')
                cap = selected_cap(core.frame)
                i = CHANNELS[args.channel]
                filtered = core.bridge.previous_projected_residual_full12[i]
                require(abs(cap-request['current_cap_full12'][i]) < 1e-6 and
                        abs(filtered-request['previous_filtered_request_full12'][i]) < 2e-5,
                        'actor/projector/current HISTORY mismatch')
                applied, changed, candidate = probe.choose(original, tick=core.frame.physics_tick,
                    cap=cap, filtered_request=filtered, committed=committed_snapshot(core.frame, index=i),
                    evidence=evidence(core.frame))
                decision = dict(student_decision=student_index+1, decision_start_tick=core.frame.physics_tick,
                    phase=core.frame.state_id, policy_observation_vector=list(observation),
                    original_student_raw_full12=list(original), original_student_request_audit=request,
                    original_gaussian_log_density_readonly=front.original_density(original, request),
                    applied_raw_full12=list(applied), applied_log_probability_for_PPO=None,
                    overridden_indices=changed, candidate=candidate)
                step = prefix.core.step(applied)
                observation = tuple(step.observation)
                front.validate_step(step.info, applied)
                emit(decisions, {**credits, **decision, 'decision_end_tick':core.frame.physics_tick,
                    'step_info':step.info, 'probe_state_after_step':deepcopy(probe.state)})
            else:
                cutoff = 'reviewed_post_prefix_zero_credit_decision_budget'
        probe.finish(core.frame.physics_tick, 'ordinary_terminal_or_diagnostic_cutoff')
        unchanged()
        require(runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id=args.experiment_id) == contract and
                common.sha256(args.checkpoint) == args.checkpoint_sha256 and
                common.sha256(sidecar) == args.checkpoint_manifest_sha256, 'runtime/checkpoint changed during diagnostic')
        manifest.update(lifecycle='DIAGNOSTIC_SEALED', probe=deepcopy(probe.state), physical_summary=physical.summary(),
            physical_seconds=core.frame.sim_time_s, final_phase=core.frame.state_id,
            actual_task_termination_reason=core.frame.info['semantic_task'].get('termination_reason'),
            diagnostic_cutoff=cutoff, native_episode_terminal=core.done, task_success_claim=False,
            frozen_learned_state_unchanged=True, prefix_and_diagnostic_counts=prefix.core.telemetry_summary())
    except BaseException as exc:
        primary_error = exc
        manifest.update(lifecycle='DIAGNOSTIC_ERROR', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        def check_learned_state():
            try:
                unchanged()
                manifest['frozen_learned_state_unchanged'] = True
            except BaseException:
                manifest['frozen_learned_state_unchanged'] = False
                raise
        def capture_final_state():
            if core is not None and core.frame is not None:
                if probe is not None:
                    probe.finish(core.frame.physics_tick, 'shutdown_no_further_dispatch')
                manifest.update(last_episode_tick=core.frame.physics_tick, last_simulation_time_s=core.frame.sim_time_s)
            manifest['probe'] = None if probe is None else deepcopy(probe.state)
        independent_cleanup(manifest, (
            ('learned_state_integrity', check_learned_state if unchanged is not None else None),
            ('physical_recorder_close', physical.close if physical is not None else None),
            ('capture_final_probe_state', capture_final_state),
            ('app_close', (lambda: app.close(wait_for_replicator=False, skip_cleanup=True)) if app is not None else None),
            ('lock_close', lock.close)), primary_error=primary_error,
            manifest_writer=(lambda: common.exclusive_json(run/'run_manifest.json', manifest)) if created else None,
            error_receipt_writer=(lambda row: common.exclusive_json(run/'cleanup_errors.json', row)) if created else None)


if __name__ == '__main__':
    main()
