"""OPTIONAL, NOT RUN: finite FL diagnostic with durable pre-action evidence.

Reuses the existing FiniteDirection/trigger and official checkpoint loader.
No optimizer, teacher deployment, source-controller changes, or PPO credit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT / 'src'))
from direction_probe import FiniteDirection, trigger, sha, write
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER

EXPERIMENT = 'task_conditioned_hip_wheel_v1'
POLICY = 'task_conditioned_hip_wheel_sigma_v1'
LAYOUT = 'diagonal_transfer_state_v1'
CONFIG = ROOT / 'configs' / ('ppo_' + EXPERIMENT)
CHANNELS = tuple(SERVO_ORDER) + tuple(WHEEL_ORDER)
UNITS = ('deg',) * 8 + ('rad/s',) * 4
HISTORY_NAMES = ('previous_raw_full12', 'previous_residual_full12',
                 'previous_previous_residual_full12', 'previous_applied_full12',
                 'previous_previous_applied_full12', 'previous_nominal_full12')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, size, label):
    result = tuple(float(x) for x in value)
    require(len(result) == size and all(math.isfinite(x) for x in result), label)
    return result


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def schema_offsets(schema):
    offset, result = 0, {}
    for group in schema['feature_groups']:
        name, size = group['name'], group['size']
        require(name not in result and type(size) is int and size > 0, 'invalid observation group')
        result[name] = (offset, offset + size, group['scale'])
        offset += size
    require(offset == 372 and schema['transfer_role_features_version'] == LAYOUT,
            'requires actual current 372 observation schema')
    return result


def validate_checkpoint_metadata(metadata, checkpoint, current_contract):
    """No migration guessing: require an exact current task runtime and real branch."""
    checkpoint = Path(checkpoint).resolve()
    require(Path(metadata['checkpoint_path']).resolve() == checkpoint, 'checkpoint path binding differs')
    require(metadata.get('save_load_round_trip') is True, 'checkpoint lacks verified roundtrip')
    require(metadata['runtime_contract'] == current_contract
            and current_contract.get('experiment_id') == EXPERIMENT,
            'checkpoint needs exact current runtime; use reviewed migration elsewhere')
    policy = metadata['policy_contract']
    require(policy['version'] == POLICY and policy['observation_layout'] == LAYOUT,
            'wrong checkpoint policy/layout')
    selected = current_contract['selected_configuration']
    require(set(selected) == {'execution_profile.yaml', 'observation_schema.json', 'quality_score.yaml',
                              'reward_config.yaml', 'action_schema.json', 'stage_task_spec.yaml'},
            'checkpoint must bind the six current experiment configs')
    for name, binding in selected.items():
        require(Path(binding['path']) == Path('configs') / ('ppo_' + EXPERIMENT) / name,
                'wrong selected experiment configuration')
    branch = metadata.get('task_conditioned_hip_wheel_branch', {})
    origin = branch.get('counter_origin', {})
    counts = metadata.get('task_conditioned_hip_wheel_branch_counts', {})
    require(branch.get('branch_id') == EXPERIMENT, 'wrong training branch')
    for name in ('global_policy_decisions', 'ppo_updates', 'optimizer_steps'):
        require(type(metadata.get(name)) is int and type(origin.get(name)) is int
                and type(counts.get(name)) is int and counts[name] > 0
                and metadata[name] - origin[name] == counts[name], 'branch counter binding differs')


def pre_action_receipt(*, decision, tick, sim_time, phase, observation, schema, history,
                       request, baseline, injected, caps, phase_mask, runtime_mask,
                       safety_mask, intervention, nominal, previous_ack):
    """Pure snapshot, called and flushed BEFORE core.step; never executes an action."""
    obs = vector(observation, 372, 'pre-action observation must contain 372 finite values')
    actor_input = tuple(f32(x) for x in obs)  # same conversion as official loader's torch.float32 tensor
    offsets = schema_offsets(schema)
    require(all(abs(x) <= schema['clip'] for x in obs), 'observation exceeds schema clipping')
    require(phase in tuple(f'P{i:02d}' for i in range(1, 14)), 'unknown phase')
    phase_index = int(phase[1:])-1
    require(actor_input[:13] == tuple(float(i == phase_index) for i in range(13))
            and request.get('stage_index') == phase_index, 'phase/request/pre-observation differ')
    history_copy = {name: vector(history[name], 12, 'invalid live history') for name in HISTORY_NAMES}
    for name, values in history_copy.items():
        lo, hi, scale = offsets[name]
        scales = (float(scale),) * 12 if isinstance(scale, (int, float)) else vector(scale, 12, 'history scale')
        encoded = tuple(f32(max(-schema['clip'], min(schema['clip'], x/s)))
                        for x, s in zip(values, scales, strict=True))
        require(actor_input[lo:hi] == encoded, 'pre-observation differs from actual live history: ' + name)
    require(request.get('mode') == 'deterministic_conditional_mean'
            and request.get('policy_version') == POLICY
            and request.get('sampling_draws') == 0
            and request.get('extra_model_forwards') == 0
            and request.get('extra_random_draws') == 0, 'requires one actual unmodified deterministic forward')
    baseline = vector(baseline, 12, 'baseline raw')
    injected = vector(injected, 12, 'intervened raw')
    caps = vector(caps, 12, 'physical caps')
    require(all(x > 0 for x in caps), 'caps must be positive')
    require(baseline == tuple(request['selected_raw_full12'])
            == tuple(request['conditional_mean_full12']), 'baseline is not the actual deterministic conditional mean')
    lo, hi, _ = offsets['previous_raw_full12']
    require(actor_input[lo:hi] == tuple(request['previous_raw_from_current_observation_full12']),
            'actor request history does not match this pre-action observation')
    lo, hi, scales = offsets['previous_residual_full12']
    decoded_request = tuple(f32(a*f32(b)) for a, b in zip(actor_input[lo:hi], scales, strict=True))
    require(decoded_request == tuple(request['previous_filtered_request_full12']),
            'request audit filtered HISTORY differs from actor input')
    require(all(abs(a-b) < 1e-5 for a, b in zip(caps, request['current_cap_full12'], strict=True)),
            'policy audit and projector caps differ')
    masks = [vector(mask, 12, 'residual permission mask') for mask in (phase_mask, runtime_mask, safety_mask)]
    require(all(x in (0., 1.) for mask in masks for x in mask), 'mask is not binary')
    delta = tuple(b-a for a, b in zip(baseline, injected, strict=True))
    require(all(x == 0 for x in delta[1:]), 'finite FL diagnostic changed an unselected channel')
    candidate_before = tuple(c*math.tanh(x) for c, x in zip(caps, baseline, strict=True))
    candidate_after = tuple(c*math.tanh(x) for c, x in zip(caps, injected, strict=True))
    selector = (1,) + (0,) * 11 if intervention and intervention.get('requested_selected_deg') else (0,) * 12
    require(not any(delta) or selector[0] == 1, 'changed action lacks intervention receipt')
    require(not selector[0] or intervention.get('case') == 'FL_minus3', 'wrong finite intervention case')
    effective_mask = tuple(math.prod(x) for x in zip(*masks, strict=True))
    receipt = dict(schema='wlr50_clean.direction_probe_pre_action.v1', record_kind='pre_action',
        decision=decision, start_tick=tick, start_sim_time_s=sim_time, phase=phase,
        source='same_live_core_observation_before_one_real_deterministic_forward_and_step',
        observation_encoded_372=obs, actor_input_float32_372=actor_input,
        actor_input_float32_le_sha256=hashlib.sha256(struct.pack('<372f', *actor_input)).hexdigest(),
        observation_history_groups={name: dict(start=offsets[name][0], stop_exclusive=offsets[name][1],
                                               scale=offsets[name][2]) for name in HISTORY_NAMES},
        actual_live_history=history_copy,
        history_semantics='raw=dimensionless; residual=previous_filtered_REQUEST; applied=final_drive; no resets',
        original_policy_request_same_live_state=copy.deepcopy(request),
        policy_baseline_raw_full12=baseline, manually_selected_raw_full12=injected,
        manual_raw_delta_full12=delta, current_physical_caps_full12=caps,
        candidate_physical_before_permission_and_slew_full12=candidate_before,
        candidate_physical_after_intervention_before_permission_and_slew_full12=candidate_after,
        candidate_difference_full12=tuple(b-a for a, b in zip(candidate_before, candidate_after, strict=True)),
        phase_residual_permission_mask_full12=masks[0], runtime_residual_permission_mask_full12=masks[1],
        safety_residual_permission_mask_full12=masks[2], combined_residual_permission_mask_full12=effective_mask,
        manual_override_selector_full12=selector,
        mask_semantics='permissions apply to residual only; override selector is not an actuator/nominal mask',
        canonical_channels=CHANNELS, physical_units_full12=UNITS,
        pre_source_nominal_full12=vector(nominal, 12, 'pre nominal'),
        previous_actual_ACK=copy.deepcopy(previous_ack), intervention=copy.deepcopy(intervention),
        actual_execution_source='subsequent same-decision step_info and native_tick_audit; candidates are not final targets',
        counterfactual_scope='same_actual_intervened_state; not an untreated deterministic trajectory after intervention',
        diagnostic_only=True, new_PPO_decisions=0, new_PPO_updates=0, new_optimizer_steps=0)
    encoded = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    receipt['pre_action_receipt_sha256'] = hashlib.sha256(encoded).hexdigest()
    return receipt


def seal_step_exception(manifest, *, physical, decision, receipt_hash, start_tick,
                        last_core_frame_tick, exception):
    """Cache one exclusive-writing summary; never replace the original step error."""
    if not manifest.get('physical_summary_attempted'):
        manifest['physical_summary_attempted'] = True
        try:
            manifest['physical_summary'] = physical.summary()
        except BaseException as summary_error:
            manifest['physical_summary'] = None
            manifest['physical_summary_error'] = repr(summary_error)
    observed = physical._last_frame  # recorder's actual last observation, not possibly lagging core.frame
    endpoint = None if observed is None else dict(physics_tick=observed.physics_tick,
        sim_time_s=observed.sim_time_s, phase=observed.state_id,
        source='PhysicalEvaluationRecorder._last_frame; last_observed_not_inferred_simulator_state')
    fields = dict(physical_summary=manifest.get('physical_summary'),
        observed_physical_endpoint=endpoint,
        endpoint_tick=None if endpoint is None else endpoint['physics_tick'],
        endpoint_sim_time_s=None if endpoint is None else endpoint['sim_time_s'],
        final_phase=None if endpoint is None else endpoint['phase'],
        last_core_frame_tick=last_core_frame_tick, core_frame_may_lag_observed_endpoint=True,
        environment_step_returned=False, fabricated_terminal_step_info=False,
        failed_decision=decision, failed_pre_action_receipt_sha256=receipt_hash)
    manifest.update(fields)
    return dict(record_kind='step_exception', decision=decision,
        pre_action_receipt_sha256=receipt_hash, start_tick=start_tick,
        exception=repr(exception), physical_summary_error=manifest.get('physical_summary_error'), **fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('control', 'FL_minus3'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--enable-physical-diagnostic', action='store_true')
    args = parser.parse_args()
    require(args.enable_physical_diagnostic, 'candidate does not run without explicit diagnostic enablement')
    require(15 <= args.max_seconds <= 85, 'unchanged finite diagnostic budget is 15..85 seconds')
    cp = args.checkpoint.resolve(strict=True)
    require(cp.is_relative_to((OUT/'checkpoints/history').resolve()), 'checkpoint must be from this task branch')
    metadata_path = cp.with_name(cp.stem + '_manifest.json')
    metadata = json.loads(metadata_path.read_text())
    require(sha(cp) == metadata['checkpoint_sha256'], 'checkpoint hash differs')
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    contract = runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id=EXPERIMENT)
    validate_checkpoint_metadata(metadata, cp, contract)
    schema = json.loads((CONFIG/'observation_schema.json').read_text())
    schema_offsets(schema)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='wlr50_clean.task_direction_probe_preaction_candidate.v1',
        case=args.case, baseline='policy', experiment_id=EXPERIMENT, checkpoint=str(cp),
        checkpoint_sha256=sha(cp), checkpoint_manifest_sha256=sha(metadata_path), runtime_contract=contract,
        policy_contract=metadata['policy_contract'], baseline_policy_sampling='deterministic_conditional_mean',
        new_PPO_decisions=0, new_PPO_updates=0, new_optimizer_steps=0, diagnostic_only=True,
        source_N_changed=False, state_injection=False, max_seconds=args.max_seconds,
        finite_intervention='unchanged FL_minus3: -3deg, 12-decision quintic ramp, old hold/release/tail',
        harness_sha256=sha(__file__), reused_probe_sha256=sha(OUT/'direction_probe.py'),
        probe_geometry_sha256=sha(OUT/'probe_geometry.py'), observation_schema_sha256=sha(CONFIG/'observation_schema.json'),
        role='independent_human_intervention_diagnostic_not_formal_PPO_success',
        training_data_eligible=False, auxiliary_updates=False, teacher_deployment=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(), lifecycle='RUNNING')
    write(args.run_dir/'run_manifest.started.json', manifest)
    app = physical = height = core = unchanged = lock = None
    issued = completed = 0
    try:
        import msvcrt
        lock = (ROOT/'runs/ppo_semantic_v2/.single_process.lock').open('r+b')
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        # Keep the existing Windows diagnostic preload order. Importing the old
        # module alone does not execute these imports inside its main().
        import torch
        from tensordict import TensorDict
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
        from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
        from wlr50_clean.ppo.semantic_height_diagnostics import HeightDiagnostics
        from wlr50_clean.ppo.semantic_training import seed_training_rngs, jsonable
        from wlr50_clean.ppo.semantic_video_cli import checkpoint_loader
        from probe_geometry import ProbeGeometry
        seed_training_rngs(4001)
        backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True,
            execution_profile=CONFIG/'execution_profile.yaml', task_spec_path=CONFIG/'stage_task_spec.yaml')
        core = SemanticEpisodeEnv(backend, collect_trace=False, action_config=CONFIG/'execution_profile.yaml',
            reward_config_path=CONFIG/'reward_config.yaml', observation_schema_path=CONFIG/'observation_schema.json')
        observation = core.reset(seed=4001)
        require(core.frame.physics_tick == 0 and core.frame.state_id == 'P01', 'requires natural P01 reset')
        load_args = argparse.Namespace(checkpoint=cp, device='cuda:0', semantic_version='v3',
            experiment_id=EXPERIMENT, seed=4001, stochastic_policy=False, policy_seed=None,
            _policy_version=POLICY, _observation_layout=LAYOUT)
        action, proof, unchanged = checkpoint_loader(load_args, contract)(observation)
        manifest['checkpoint_load_provenance'] = proof
        write(args.run_dir/'checkpoint_load_provenance.json', proof)
        physical = PhysicalEvaluationRecorder(args.run_dir, task_spec_path=CONFIG/'stage_task_spec.yaml',
                                              quality_score_path=CONFIG/'quality_score.yaml')
        physical.start(core.frame)
        height = HeightDiagnostics(args.run_dir, backend)
        height.start(core.frame)
        geometry = ProbeGeometry(height)
        write(args.run_dir/'four_hip_geometry_startup.json', geometry.sample(core.frame))
        def observer(before, after, projection):
            physical.observe(before, after, projection)
            height.sample(after, terminal=bool(after.info['semantic_task'].get('termination_reason')))
        core.tick_observer = observer
        probe = FiniteDirection(args.case)  # unchanged finite implementation; never wrap/add recursively
        with (args.run_dir/'probe_decisions.jsonl').open('x', encoding='utf-8') as stream:
            def emit(record):
                stream.write(json.dumps(jsonable(record), allow_nan=False)+'\n')
                stream.flush()
            while not core.done and core.frame.sim_time_s < args.max_seconds-1e-10:
                tick, phase, sim_time = core.frame.physics_tick, core.frame.state_id, core.frame.sim_time_s
                pre_observation, pre_history = tuple(observation), copy.deepcopy(core._history)
                baseline = action(pre_observation, issued)  # exactly one official audited det forward
                request = copy.deepcopy(action.last_request)
                require((tick, phase, sim_time) == (core.frame.physics_tick, core.frame.state_id, core.frame.sim_time_s)
                        and pre_observation == tuple(core.observation) and pre_history == core._history,
                        'policy forward changed live pre-action state/history')
                ev = core.frame.info['semantic_task']['physical_evaluator']
                entered = probe.anchor is None and trigger(args.case, phase, ev,
                    backend._controller.nominal_provider.endpoint_issued)
                if entered:
                    probe.start(pre_history['previous_residual_full12'])
                caps = tuple(a*b for a, b in zip(core.projector.config.scale_for(phase),
                    core.projector.config.physical_residual_scale_full12, strict=True))
                raw, intervention = probe.apply(baseline, caps, phase, ev)
                safety = core.frame.safety_projection
                receipt = pre_action_receipt(decision=issued, tick=tick, sim_time=sim_time, phase=phase,
                    observation=pre_observation, schema=schema, history=pre_history, request=request,
                    baseline=baseline, injected=raw, caps=caps, phase_mask=core.projector.config.mask_for(phase),
                    runtime_mask=core.frame.action_mask_full12,
                    safety_mask=(1,)*12 if safety is None else safety.channel_mask_full12,
                    intervention=intervention, nominal=core.frame.nominal_action_full12,
                    previous_ack=jsonable(backend._adapter.last_ack))
                emit(receipt)  # survives a terminal exception from this decision's physical step
                if entered:
                    write(args.run_dir/'probe_entry.json', dict(pre_action_receipt=receipt,
                        evaluator=ev, four_hip_geometry=geometry.sample(core.frame)))
                issued += 1
                try:
                    step = core.step(raw)
                except BaseException as exc:
                    error_record = seal_step_exception(manifest, physical=physical, decision=issued-1,
                        receipt_hash=receipt['pre_action_receipt_sha256'], start_tick=tick,
                        last_core_frame_tick=core.frame.physics_tick, exception=exc)
                    try:
                        emit(error_record)
                    except BaseException as log_error:
                        manifest['step_exception_log_error'] = repr(log_error)
                    raise
                completed += 1
                observation = tuple(step.observation)
                emit(dict(record_kind='step_result', decision=issued-1,
                    pre_action_receipt_sha256=receipt['pre_action_receipt_sha256'],
                    start_tick=tick, end_tick=core.frame.physics_tick, environment_step_returned=True,
                    step_info=jsonable(step.info), four_hip_geometry=geometry.sample(core.frame)))
        unchanged()
        manifest.update(lifecycle='DIAGNOSTIC_SEALED', physical_summary=physical.summary(),
            endpoint_tick=core.frame.physics_tick, final_phase=core.frame.state_id,
            original_task_reason=core.frame.info['semantic_task'].get('termination_reason'),
            probe_started=probe.anchor is not None, probe_complete=probe.complete,
            external_budget_stop=not core.done, learned_state_unchanged=True)
    except BaseException as exc:
        manifest.update(lifecycle='DIAGNOSTIC_ERROR', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if unchanged is not None:
            try:
                unchanged()
                manifest['learned_state_unchanged'] = True
            except BaseException as exc:
                manifest.update(lifecycle='DIAGNOSTIC_ERROR', learned_state_unchanged=False,
                                learned_state_check_error=repr(exc))
        manifest.update(actual_diagnostic_decisions_issued=issued,
            actual_diagnostic_decisions_completed=completed, completed_at_utc=datetime.now(timezone.utc).isoformat())
        if physical is not None:
            physical.close()
        if height is not None:
            manifest['height_diagnostics'] = height.close(core.frame)
        write(args.run_dir/'run_manifest.json', manifest)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
