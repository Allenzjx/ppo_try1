"""Finite, labelled physical interventions. Never a PPO rollout or deployment aid."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
CASES = {'control': {}, 'FR_RL_minus3': {4: -3.}, 'FL_minus3': {0: -3.},
         'RR_prepare_RL_minus3': {4: -3.}, 'RR_lift_FL_RL_minus3': {0: -3., 4: -3.}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)

def quintic(x):
    x = min(1., max(0., x))
    return x*x*x*(10+x*(-15+6*x))

def trigger(case, phase, ev, endpoint):
    if ev.get('valid') is not True or ev.get('termination_reason'):
        return False
    legs, history = ev['current_legs'], ev['history']
    if case == 'FR_RL_minus3':
        fr = legs['FR']
        return (phase == 'P02' and history['active_lift']['FR'] and fr['air']
                and fr['clearance_m'] >= .015 and fr['consecutive_air_samples'] >= 2)
    if case == 'FL_minus3':
        fl = legs['FL']
        return (phase == 'P05' and endpoint and history['front_edge_crossed']['FL']
                and not history['placed']['FL'] and fl['air'] and fl['within_top_xy'])
    if case == 'RR_prepare_RL_minus3':
        return phase == 'P08' and not legs['RR'].get('current_lift_valid', False)
    if case == 'RR_lift_FL_RL_minus3':
        return phase in ('P08', 'P09') and legs['RR'].get('current_lift_valid') is True
    return False

class FiniteDirection:
    def __init__(self, case):
        self.case, self.offsets = case, CASES[case]
        self.anchor = None
        self.age = 0
        self.release_age = None
        self.last = None
        self.capture_age = None
    def start(self, residual):
        assert self.anchor is None and len(residual) == 12
        self.anchor = tuple(residual)
    def apply(self, baseline, caps, phase, ev):
        raw = list(baseline)
        if self.anchor is None:
            return tuple(raw), None
        if self.complete:
            return tuple(raw), {'kind': 'labelled_direction_probe_not_PPO',
                                'case': self.case, 'state': 'intervention_finished_live_policy_continues'}
        placed = ev['history']['placed']
        reached = (placed['FR'] if self.case.startswith('FR_') else placed['FL']
                   if self.case.startswith('FL_') else placed['RR'])
        if reached and self.capture_age is None:
            self.capture_age = self.age
        # Contact is followed for a finite live interval, not used as a new guard.
        duration = 120 if self.case.startswith('RR_') else 75
        if self.release_age is None and (self.age >= duration or
                self.capture_age is not None and self.age-self.capture_age >= 24):
            self.release_age, self.release_start = self.age, dict(self.last)
        requested = {}
        for i, offset in self.offsets.items():
            nominal_policy = caps[i]*math.tanh(baseline[i])
            if self.release_age is None:
                # For the two-hip recovery case RL follows FL by two decisions;
                # no shared-clock change to the actual nominal source.
                delay = 2 if self.case == 'RR_lift_FL_RL_minus3' and i == 4 else 0
                value = self.anchor[i]+quintic((self.age+1-delay)/12)*offset
            else:
                mix = quintic((self.age-self.release_age+1)/12)
                value = (1-mix)*self.release_start[i]+mix*nominal_policy
            if not math.isfinite(value) or abs(value/caps[i]) >= 1:
                raise ValueError('finite intervention exceeds existing residual capacity')
            requested[i] = value
            raw[i] = math.atanh(value/caps[i])
        receipt = {'kind': 'labelled_direction_probe_not_PPO', 'case': self.case,
                   'index': self.age, 'anchor_full12': self.anchor, 'delta_deg': self.offsets,
                   'requested_selected_deg': requested, 'release_age': self.release_age,
                   'capture_age': self.capture_age, 'not_added_recursively_to_HISTORY': True}
        self.last = requested
        self.age += 1
        return tuple(raw), receipt
    @property
    def complete(self):
        return self.release_age is not None and self.age >= self.release_age+12+30

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--baseline', choices=('policy', 'nominal'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    if not 15 <= args.max_seconds <= 85:
        raise ValueError('bounded diagnostic requires 15..85 seconds')
    cp = args.checkpoint.resolve(strict=True)
    metadata = json.loads(cp.with_name(cp.stem+'_manifest.json').read_text())
    if sha(cp) != metadata['checkpoint_sha256']:
        raise ValueError('checkpoint hash differs')
    actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if actual_head != args.expected_head:
        raise ValueError('unexpected HEAD')
    contract = metadata['runtime_contract']
    # This diagnostic supports an archive-only HEAD change, not any source change.
    differences = [p for p, digest in contract['files'].items() if sha(ROOT/p) != digest]
    if differences:
        raise ValueError('diagnostic source differs from checkpoint: '+repr(differences))
    config = ROOT/'configs/ppo_fl_capture_quality_v1'
    for name, binding in contract['selected_configuration'].items():
        if Path(binding['path']) != Path('configs/ppo_fl_capture_quality_v1')/name:
            raise ValueError('wrong checkpoint experiment')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'wlr50_clean.task_direction_probe.v1', 'case': args.case,
        'baseline': args.baseline, 'checkpoint': str(cp), 'checkpoint_sha256': sha(cp),
        'actual_head': actual_head, 'source_head': contract['source_git_commit'],
        'all_checkpoint_runtime_files_identical': True, 'new_PPO_decisions': 0,
        'new_PPO_updates': 0, 'new_optimizer_steps': 0, 'diagnostic_only': True,
        'source_N_changed': False, 'state_injection': False, 'max_seconds': args.max_seconds,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'lifecycle': 'RUNNING'}
    write(args.run_dir/'run_manifest.started.json', manifest)
    app = physical = height = None
    lock = (ROOT/'runs/ppo_semantic_v2/.single_process.lock').open('r+b')
    try:
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        import torch
        from tensordict import TensorDict
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
        from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
        from wlr50_clean.ppo.semantic_height_diagnostics import HeightDiagnostics
        from wlr50_clean.ppo.semantic_training import (seed_training_rngs, construct_semantic_runner,
            load_semantic_checkpoint, parameter_hash, state_hash, _normalizers, jsonable,
            audited_history_policy_request)
        seed_training_rngs(4001)
        backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True,
            execution_profile=config/'execution_profile.yaml', task_spec_path=config/'stage_task_spec.yaml')
        core = SemanticEpisodeEnv(backend, collect_trace=False, action_config=config/'execution_profile.yaml',
            reward_config_path=config/'reward_config.yaml', observation_schema_path=config/'observation_schema.json')
        observation = core.reset(seed=4001)
        assert core.frame.physics_tick == 0 and core.frame.state_id == 'P01'
        class ObservationEnv:
            num_envs, num_actions = 1, 12
            cfg = {'evaluation': True, 'semantic_version': 'v3'}
            def get_observations(self):
                tensor = torch.tensor([observation], dtype=torch.float32, device='cuda:0')
                return TensorDict({'policy': tensor, 'critic': tensor.clone()}, batch_size=[1], device='cuda:0')
        policy = metadata['policy_contract']
        runner, _ = construct_semantic_runner(ObservationEnv(), seed=metadata['seed'], device='cuda:0',
            initialize_actor=False, policy_version=policy['version'], observation_layout=policy['observation_layout'])
        load_semantic_checkpoint(runner, cp, contract=contract, seed=metadata['seed'])
        runner.alg.eval_mode()
        def hashes():
            return [parameter_hash(runner.alg.actor), parameter_hash(runner.alg.critic),
                    state_hash(runner.alg.optimizer.state_dict()), state_hash(_normalizers(runner))]
        initial_hashes = hashes()
        physical = PhysicalEvaluationRecorder(args.run_dir, task_spec_path=config/'stage_task_spec.yaml',
                                              quality_score_path=config/'quality_score.yaml')
        physical.start(core.frame)
        height = HeightDiagnostics(args.run_dir, backend)
        height.start(core.frame)
        def observer(before, after, projection):
            physical.observe(before, after, projection)
            height.sample(after, terminal=bool(after.info['semantic_task'].get('termination_reason')))
        core.tick_observer = observer
        probe = FiniteDirection(args.case)
        decisions = 0
        with (args.run_dir/'probe_decisions.jsonl').open('x', encoding='utf-8') as stream:
            while not core.done and core.frame.sim_time_s < args.max_seconds-1e-10:
                tensor = torch.tensor([observation], dtype=torch.float32, device='cuda:0')
                inputs = TensorDict({'policy': tensor, 'critic': tensor.clone()}, batch_size=[1], device='cuda:0')
                with torch.inference_mode():
                    selected, request = audited_history_policy_request(runner.alg.actor, inputs,
                        lambda: runner.alg.actor(inputs, stochastic_output=False), stochastic=False)
                baseline = tuple(selected[0].cpu().tolist()) if args.baseline == 'policy' else (0.,)*12
                ev = core.frame.info['semantic_task']['physical_evaluator']
                if probe.anchor is None and trigger(args.case, core.frame.state_id, ev,
                        backend._controller.nominal_provider.endpoint_issued):
                    probe.start(core._history['previous_residual_full12'])
                    write(args.run_dir/'probe_entry.json', {'tick': core.frame.physics_tick,
                        'observation': observation, 'evaluator': ev, 'anchor': probe.anchor,
                        'nominal': core.frame.nominal_action_full12,
                        'previous_ACK': jsonable(backend._adapter.last_ack)})
                caps = tuple(a*b for a,b in zip(core.projector.config.scale_for(core.frame.state_id),
                                               core.projector.config.physical_residual_scale_full12))
                raw, intervention = probe.apply(baseline, caps, core.frame.state_id, ev)
                tick = core.frame.physics_tick
                step = core.step(raw)
                decisions += 1
                observation = tuple(step.observation)
                stream.write(json.dumps({'start_tick': tick, 'end_tick': core.frame.physics_tick,
                    'policy_request_not_necessarily_applied': jsonable(request), 'baseline_raw': baseline,
                    'injected_raw': raw, 'intervention': intervention, 'step_info': jsonable(step.info)}, allow_nan=False)+'\n')
                stream.flush()
        assert hashes() == initial_hashes, 'diagnostic mutated learned state'
        manifest.update(lifecycle='DIAGNOSTIC_SEALED', physical_summary=physical.summary(),
            endpoint_tick=core.frame.physics_tick, final_phase=core.frame.state_id,
            original_task_reason=core.frame.info['semantic_task'].get('termination_reason'),
            probe_started=probe.anchor is not None, probe_complete=probe.complete,
            actual_diagnostic_decisions=decisions,
            external_budget_stop=not core.done, learned_state_unchanged=True)
    except BaseException as exc:
        manifest.update(lifecycle='DIAGNOSTIC_ERROR', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if physical is not None:
            physical.close()
        if height is not None:
            manifest['height_diagnostics'] = height.close(core.frame)
        manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.run_dir/'run_manifest.json', manifest)
        print(json.dumps({k: manifest.get(k) for k in ('lifecycle', 'case', 'endpoint_tick',
            'final_phase', 'original_task_reason', 'probe_started', 'probe_complete')}), flush=True)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        lock.close()

if __name__ == '__main__':
    main()
