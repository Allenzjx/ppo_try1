"""NOT EXECUTED: one-shot FL-wheel residual diagnostic, ordinary same439 core.

Successful nominal physically runs P01->P10 with zero sample credit. Then the
saved student's ordinary deterministic kernel runs. At P12, only after an
explicit source four-wheel stop has an actual dispatch receipt and RR/FL both
currently bear on legal TOP, a negative FL-wheel tanh request is halved once
per fresh student decision (absolute raw replacement, never accumulated).

No production imports at module import, no optimizer calls, no AUX labels.
Safety still runs every physics tick. Loss of eligibility is latched at its
first observed physics tick; the immutable eight-tick decision finishes unless
ordinary safety ends it, then no new intervention is issued (<=7 tick delay).
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
WHEELS = ('front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle')
ZERO_CREDIT = dict(common.ZERO_CREDIT, teacher_prefix_sample_credit=0,
    teacher_sample_credit=0, policy_credit=False,
    teacher_actions_deployed=True, teacher_semantics='explicit successful nominal initialization only; no learned teacher or AUX labels')
LAYOUT = 'role422_rear_owner_recovery_v1'


def half_negative_fl_request(original):
    """Absolute substitution from THIS fresh raw request, not previous output."""
    if len(original) != 12 or any(type(x) not in (int, float) or not math.isfinite(x) for x in original):
        raise ValueError('finite raw Full12 required')
    applied = list(original)
    if original[8] < 0.:
        applied[8] = math.atanh(.5 * math.tanh(original[8]))
    return tuple(applied)


def scheduled_wheel_event(provider, frame):
    """Read, never tick/mutate, the nominal provider's current future sample.

    ``after`` frame nominal is next dispatch's source; this is NOT an ACK.
    Fail closed for coincident wheel events from different source layers.
    """
    found = []
    for layer in provider._continuous_layers:
        sample = layer.get('sample')
        if sample is None or layer.get('advanced_this_tick') is not True:
            continue
        for group in sample.atomic_groups:
            names = set(group.channels)
            if names.intersection(WHEELS):
                found.append(dict(stage=layer['stage'],channels=sorted(names.intersection(WHEELS)),
                    all_wheels=set(WHEELS).issubset(names),
                    stop=all(x == 0. for x in sample.full12[8:]),
                    source_motion_tick=sample.tick_index,source_time_s=group.time_s))
    if not found:
        return None
    return dict(source_frame_tick=frame.physics_tick, events=found,
        explicit_P12_fourwheel_stop=(len(found)==1 and found[0]['stage']=='P12'
            and found[0]['all_wheels'] and found[0]['stop']),
        nominal_wheels=list(frame.nominal_action_full12[8:]))


class SourceStopReceipt:
    """One real dispatch later: stop permission never comes from zeros alone."""
    def __init__(self):
        self.pending = None
        self.receipt = None
        self.active = False
        self.last_command_tick = None

    def observe_dispatch(self, *, before_tick, after_tick, source_nominal, ack, audit):
        if (after_tick != before_tick+1 or audit.get('verified') is not True
                or ack.get('articulation_writes_this_call') != 1
                or audit.get('setter_dispatch_targets_equal') is not True
                or audit.get('actual_mapping_matches_dispatch') is not True
                or audit.get('same_tick_counterfactual') is not True
                or type(ack.get('physics_tick')) is not int
                or ack.get('physics_tick') != audit.get('physics_tick')
                or (self.last_command_tick is not None
                    and ack['physics_tick'] != self.last_command_tick+1)):
            raise ValueError('adjacent verified ordinary dispatch required')
        self.last_command_tick=ack['physics_tick']
        event = self.pending
        self.pending = None
        if event is not None:
            if event['source_frame_tick'] != before_tick:
                raise ValueError('stale source wheel event, not an issued stop')
            self.active = bool(event['explicit_P12_fourwheel_stop']
                and list(source_nominal[8:]) == [0.]*4
                and event['nominal_wheels'] == [0.]*4)
            self.receipt = dict(event, issued_episode_tick=after_tick,
                command_physics_tick=ack['physics_tick'], verified_dispatch=True) if self.active else None
        if list(source_nominal[8:]) != [0.]*4:
            self.active = False

    def snapshot(self):
        return dict(active=self.active,receipt=deepcopy(self.receipt),pending=deepcopy(self.pending))


def eligibility(task, *, phase, tick, nominal, stop, force_floor, top_gap):
    ev=task.get('physical_evaluator') or {}; legs=ev.get('current_legs') or {}; blockers=[]
    if (phase != 'P12' or task.get('stage_id') != 'P12'):
        blockers.append('not_P12')
    if (ev.get('valid') is not True or ev.get('physics_tick') != tick
            or task.get('termination_reason') is not None or ev.get('termination_reason') is not None):
        blockers.append('no_fresh_nonterminal_physical_state')
    if (not stop.active or stop.receipt is None or stop.receipt['issued_episode_tick'] > tick):
        blockers.append('P12_explicit_stop_not_yet_dispatched')
    if list(nominal[8:]) != [0.]*4:
        blockers.append('current_nominal_wheels_not_stopped')
    selected={}
    for name in ('RR','FL'):
        leg=legs.get(name) or {}; selected[name]=deepcopy(leg)
        force=leg.get('bearing_force_n'); gap=leg.get('clearance_m')
        valid = (leg.get('air') is False and leg.get('ground_contact') is False
            and leg.get('support') is True and leg.get('bearing_verified') is True
            and leg.get('top_contact') is True and leg.get('top_surface_contact') is True
            and leg.get('obstacle_pair_active') is True and leg.get('contact_surface')=='TOP'
            and leg.get('within_top_xy') is True and leg.get('within_lateral_span') is True
            and type(force) in (int,float) and math.isfinite(force) and force >= force_floor
            and type(gap) in (int,float) and math.isfinite(gap) and top_gap[0] <= gap <= top_gap[1])
        if not valid:
            blockers.append(name+'_current_legal_TOP_bearing_absent')
    return dict(eligible=not blockers,blockers=blockers,phase=phase,physics_tick=tick,
        current=selected,stop=stop.snapshot(),history_placed_is_not_current_support=True)


class OneShot:
    def __init__(self, maximum_seconds=3.):
        if not math.isfinite(maximum_seconds) or not 1/15 <= maximum_seconds <= 3.:
            raise ValueError('probe duration must be in [1/15,3] s')
        self.maximum_ticks=int(math.floor(maximum_seconds*120+1e-9))
        self.state=dict(status='WAIT',start_tick=None,release_requested_tick=None,
            release_effective_tick=None,release_reason=None,trigger=None,issued_override_decisions=0)

    def observe(self, tick, evidence):
        if self.state['status']=='ACTIVE' and not evidence['eligible']:
            self.state.update(status='RELEASE_PENDING',release_requested_tick=tick,
                release_reason='current_eligibility_lost:'+','.join(evidence['blockers']))

    def choose(self, original, tick, evidence):
        # Validate even in WAIT/released states; never issue malformed requests.
        candidate=half_negative_fl_request(original)
        state=self.state
        self.observe(tick,evidence)
        if state['status']=='RELEASE_PENDING':
            state.update(status='RELEASED',release_effective_tick=tick)
        if state['status']=='WAIT' and evidence['eligible'] and original[8] < 0.:
            state.update(status='ACTIVE',start_tick=tick,trigger=deepcopy(evidence))
        if state['status']=='ACTIVE' and tick+8-state['start_tick'] > self.maximum_ticks:
            state.update(status='RELEASED',release_requested_tick=tick,release_effective_tick=tick,
                release_reason='finite_duration_no_next_full_decision')
        applied=candidate if state['status']=='ACTIVE' else tuple(original)
        changed=[i for i,(a,b) in enumerate(zip(original,applied)) if a != b]
        assert changed in ([],[8])
        state['issued_override_decisions'] += bool(changed)
        return applied,changed

    def finish(self, tick, reason):
        """No further command exists at a terminal/error boundary."""
        if self.state['status'] in ('ACTIVE','RELEASE_PENDING'):
            if self.state['release_requested_tick'] is None:
                self.state.update(release_requested_tick=tick,release_reason=reason)
            self.state.update(status='RELEASED',release_effective_tick=tick)


def verified_suffix_step(info, applied):
    """Audited ordinary dispatch, not an independently rebuilt mapper result."""
    audit=info.get('actuator_target_effect_audit') or {}
    summary=info.get('actuator_target_effect_audit_summary') or {}
    if (tuple(info.get('raw_policy_action_full12',())) != tuple(applied)
            or tuple(audit.get('raw_policy_action_full12',())) != tuple(applied)
            or summary.get('all_ticks_verified') is not True
            or summary.get('physics_ticks') != info.get('physics_ticks')
            or info.get('no_in_episode_state_writes_verified') is not True):
        raise ValueError('diagnostic suffix must retain ordinary verified raw/actual dispatch')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-sha256','checkpoint-manifest-sha256','expected-head'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--experiment-id',default='rr_rl_timing_policy_learning_v1',choices=sorted(common.SUPPORTED_EXPERIMENTS))
    p.add_argument('--semantic-version',default='v3',choices=['v3'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=1001)
    p.add_argument('--maximum-probe-seconds',type=float,default=3.)
    p.add_argument('--execute-reviewed-real-diagnostic',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv);probe=OneShot(args.maximum_probe_seconds)
    if not args.execute_reviewed_real_diagnostic:
        raise ValueError('NOT EXECUTED candidate; explicit reviewed flag required')
    if len(args.expected_head)!=40:
        raise ValueError('full frozen HEAD required')
    checkpoint=args.checkpoint.resolve(strict=True)
    cp_manifest=checkpoint.with_name(checkpoint.stem+'_manifest.json').resolve(strict=True)
    metadata=json.loads(cp_manifest.read_text(encoding='utf-8'))
    if (checkpoint.name=='checkpoint_last.pt' or common.sha256(checkpoint)!=args.checkpoint_sha256
            or common.sha256(cp_manifest)!=args.checkpoint_manifest_sha256
            or metadata.get('checkpoint_sha256')!=args.checkpoint_sha256
            or metadata.get('save_load_round_trip') is not True):
        raise ValueError('immutable saved/reloaded checkpoint binding required')
    policy=metadata['policy_contract']
    if policy.get('observation_dimension')!=439 or policy.get('observation_layout')!=LAYOUT or policy.get('action_dimension',12)!=12:
        raise ValueError('this isolated candidate only supports exact same439/Full12')
    sys.path.insert(0,str(ROOT/'src'))
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    contract=runtime_contract(expected_head=args.expected_head,semantic_version=args.semantic_version,experiment_id=args.experiment_id)
    if metadata.get('runtime_contract')!=contract:
        raise ValueError('checkpoint/runtime mismatch; no migration in a diagnostic')
    seed=args.seed;common._policy_args(args,metadata);args.seed=seed
    args.checkpoint=checkpoint
    run=args.run_dir.resolve();allowed=(ROOT/'runs'/f'ppo_{args.experiment_id}'/'direction_probe').resolve()
    if not run.is_relative_to(allowed) or run==allowed:
        raise ValueError('new isolated direction_probe child required')
    run.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='outputs.fl_wheel_residual_half_probe.v1',lifecycle='RUNNING',**ZERO_CREDIT,
        run_role='SUCCESSFUL_NOMINAL_P10_PREFIX_SINGLE_CHANNEL_DIAGNOSTIC',formal_success_claim=False,
        successful_N_prefix_used=True,natural_P01_student_prefix=False,state_injection_or_teleport=False,
        checkpoint=str(checkpoint),checkpoint_sha256=args.checkpoint_sha256,runtime_contract=contract,
        checkpoint_manifest=str(cp_manifest),checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        runner_sha256=common.sha256(Path(__file__)),shared_helper_sha256=common.sha256(Path(common.__file__)),
        seed=seed,deterministic=True,maximum_probe_seconds=args.maximum_probe_seconds,
        intervention='raw[8]=atanh(0.5*tanh(original_raw[8])) iff original_raw[8]<0; other11 identical',
        intervention_credit='always zero; not an AUX target or formal policy success',
        release_latency='first physical loss is latched; no next modified decision, <=7 remaining physics ticks; native safety can end immediately',
        duration_semantics='maximum 3 seconds of modified raw decisions; ordinary mapper/filter/HISTORY response may persist after release',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    common.exclusive_json(run/'run_manifest.started.json',manifest)
    app=core=physical=unchanged=None
    lock=(ROOT/'runs/ppo_semantic_v2/.single_process.lock').open('r+b')
    stop=SourceStopReceipt();decision={};prefix_ready=False
    try:
        import msvcrt
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        import torch  # runtime only, after lock
        import tensordict  # preserve proven Windows import order
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=False).app;app.update()
        from wlr50_clean.ppo.semantic_training import seed_training_rngs,jsonable
        from wlr50_clean.ppo.semantic_video_cli import build_video_core,checkpoint_loader
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder,physical_json
        from wlr50_clean.ppo.semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest,_CheckpointPolicyCreditCore
        seed_training_rngs(seed)
        core=build_video_core(app,role='C',semantic_version=args.semantic_version,experiment_id=args.experiment_id)
        config=ROOT/'configs'/f'ppo_{args.experiment_id}'
        physical=PhysicalEvaluationRecorder(run,task_spec_path=config/'stage_task_spec.yaml',quality_score_path=config/'quality_score.yaml')
        with ExitStack() as streams:
            ticks=streams.enter_context((run/'diagnostic_physics.jsonl').open('x',encoding='utf-8'))
            decisions=streams.enter_context((run/'diagnostic_decisions.jsonl').open('x',encoding='utf-8'))
            prefixes=streams.enter_context((run/'prefix_evidence.jsonl').open('x',encoding='utf-8'))
            def emit(stream,row):
                payload=common.diagnostic_json_payload(row,physical_json=physical_json,jsonable=jsonable)
                stream.write(json.dumps(payload,allow_nan=False)+'\n');stream.flush()
            def prefix_sink(row):
                emit(prefixes,{**row,**ZERO_CREDIT})
                if row.get('kind')=='checkpoint_prefix_result' and row.get('accepted') is not True:
                    raise RuntimeError('diagnostic prefix unavailable; no automatic fallback episode')
            def evidence(frame):
                spec=core.backend._controller.supervisor.spec
                return eligibility(frame.info['semantic_task'],phase=frame.state_id,tick=frame.physics_tick,
                    nominal=frame.nominal_action_full12,stop=stop,force_floor=spec['support']['force_noise_floor_n'],
                    top_gap=(spec['geometry']['top_gap_min_m'],spec['geometry']['top_gap_max_m']))
            def observe(before,after,projection):
                physical.observe(before,after,projection)
                ack=after.info['atomic_ack'];audit=after.info['actuator_target_effect_audit']
                stop.observe_dispatch(before_tick=before.physics_tick,after_tick=after.physics_tick,
                    source_nominal=before.nominal_action_full12,ack=ack,audit=audit)
                provider=core.backend._controller.nominal_provider
                stop.pending=scheduled_wheel_event(provider,after)
                current=evidence(after)
                if prefix_ready:probe.observe(after.physics_tick,current)
                emit(ticks,{**ZERO_CREDIT,'episode_physics_tick':after.physics_tick,'sim_time_s':after.sim_time_s,
                    'prefix_initialization':not prefix_ready,'decision':decision,'eligibility':current,
                    'source_nominal_full12':list(before.nominal_action_full12),
                    'actual_drive_target_full12':after.info['drive_target_full12'],
                    'atomic_ACK':ack,'native_readback_audit':audit,'raw_observation':after.info.get('raw_observation'),
                    'physical_evaluator':after.info['semantic_task']['physical_evaluator'],
                    'rear_owner_recovery_evidence':after.info.get('rear_owner_recovery_evidence'),
                    'projected_residual_full12':projection.safe_projected_residual_full12,'probe_state':deepcopy(probe.state)})
            core.tick_observer=observe
            # The prefix wrapper owns this one core's observer before reset; do
            # not assign core.tick_observer after constructing the wrapper.
            prefix=_CheckpointPolicyCreditCore(core,request=CheckpointPolicyPrefixRequest(target_phase='P10',source='successful_nominal'),evidence_sink=prefix_sink)
            observation=tuple(prefix.reset(seed=seed))
            if len(observation)!=439 or core.frame.physics_tick!=0 or core.frame.state_id!='P01':
                raise ValueError('untouched natural P01 bootstrap required')
            physical.start(core.frame)
            manifest['rear_assist_receipt']=common._rear_assists_off(core.backend,core.frame)
            for phase in (f'P{i:02d}' for i in range(1,14)):
                if tuple(core.projector.config.mask_for(phase))!=(1,)*12:
                    raise ValueError('ordinary all12 permission required')
            manifest['all12_residual_permission_verified_all_phases']=True
            action,proof,unchanged=checkpoint_loader(args,contract)(observation)
            manifest['official_checkpoint_load_proof']=proof
            observation=tuple(prefix.install(lambda obs:(0.,)*12,dict(schema='wlr50_clean.successful_nominal_prefix.v1',
                source='successful_nominal',raw_action_full12=[0.]*12,policy_credit=False,
                execution_profile_sha256=common.sha256(config/'execution_profile.yaml'),stage_task_spec_sha256=common.sha256(config/'stage_task_spec.yaml'),
                runtime_content_sha256=contract['runtime_content_sha256'],interface_contract=dict(observation_dimension=439,observation_layout=LAYOUT,action_dimension=12)),seed=seed))
            prefix_ready=True;manifest['prefix_handoff']=deepcopy(prefix.start_record)
            while not core.done and core.frame.sim_time_s<200.-1e-10:
                original=tuple(action(observation,core.decision_count));request=deepcopy(action.last_request)
                current=evidence(core.frame);applied,changed=probe.choose(original,core.frame.physics_tick,current)
                decision=dict(decision_start_tick=core.frame.physics_tick,decision_start_time_s=core.frame.sim_time_s,
                    phase=core.frame.state_id,original_student_raw_full12=list(original),original_student_request_audit=request,
                    applied_raw_full12=list(applied),overridden_indices=changed,
                    original_FL_tanh=math.tanh(original[8]),applied_FL_tanh=math.tanh(applied[8]),
                    intervention_issued_for_this_decision=bool(changed),eligibility=current,
                    policy_observation_vector=list(observation),log_probability_semantics='original request audit only; intervened action never enters PPO')
                # Ordinary core.step, no altered source/N, mapper, HISTORY, or
                # substep callback. All diagnostic suffix credit remains zero.
                step=core.step(applied);observation=tuple(step.observation)
                verified_suffix_step(step.info,applied)
                emit(decisions,{**ZERO_CREDIT,**decision,'decision_end_tick':core.frame.physics_tick,
                    'step_info':step.info,'probe_state_after_step':deepcopy(probe.state),
                    'native_core_task_labels_scope':'physical episode including nominal prefix and declared intervention; never formal policy success'})
        probe.finish(core.frame.physics_tick,'ordinary_episode_terminal_or_200s_cap')
        unchanged()
        if runtime_contract(expected_head=args.expected_head,semantic_version=args.semantic_version,experiment_id=args.experiment_id)!=contract:
            raise RuntimeError('runtime changed during diagnostic')
        if common.sha256(checkpoint)!=args.checkpoint_sha256 or common.sha256(cp_manifest)!=args.checkpoint_manifest_sha256:
            raise RuntimeError('checkpoint changed during diagnostic')
        manifest.update(lifecycle='DIAGNOSTIC_SEALED',probe=deepcopy(probe.state),physical_summary=physical.summary(),
            actual_task_termination_reason=core.frame.info['semantic_task'].get('termination_reason'),
            physical_seconds=core.frame.sim_time_s,same_episode_continued_to_natural_result=bool(core.done),
            diagnostic_200s_cap_without_native_terminal=not core.done)
    except BaseException as exc:
        manifest.update(lifecycle='DIAGNOSTIC_ERROR',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if unchanged is not None:
            try:unchanged();manifest['frozen_learned_state_unchanged']=True
            except BaseException as exc:manifest.update(lifecycle='DIAGNOSTIC_ERROR',integrity_error=repr(exc),frozen_learned_state_unchanged=False)
        if physical is not None:physical.close()
        if core is not None and core.frame is not None:
            probe.finish(core.frame.physics_tick,'diagnostic_shutdown_no_further_dispatch')
            manifest.update(last_episode_tick=core.frame.physics_tick,last_simulation_time_s=core.frame.sim_time_s)
        manifest.update(probe=deepcopy(probe.state),completed_at_utc=datetime.now(timezone.utc).isoformat())
        common.exclusive_json(run/'run_manifest.json',manifest)
        if app is not None:app.close(wait_for_replicator=False,skip_cleanup=True)
        lock.close()


if __name__=='__main__':main()
