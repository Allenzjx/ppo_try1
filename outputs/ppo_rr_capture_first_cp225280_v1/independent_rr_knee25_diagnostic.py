"""Independent hip-25/knee+25 direction diagnostic; NOT formal C or training.

Import and --help are stdlib-only. Physical execution requires the explicit
--run-independent-diagnostic flag AFTER the parent's normal safe boundary.
No production file is patched. The production route's diagnostic callback is
temporarily bound ONLY inside this independent process and restored in finally.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
NAME = 'ppo_rr_capture_first_cp225280_v1'
MODE = 'INDEPENDENT_DIRECTION_DIAGNOSTIC'
PARAMETERS = dict(schema='wlr50_clean.independent_rr_knee25_parameters.v1',
    hip_delta_from_entry_FINAL_deg=-25., knee_delta_from_entry_FINAL_deg=25.,
    ramp_s=1.5, raw_inverse_tanh_ratio_limit=.985, overridden_channels=[6,7],
    capture='first_native_measured_TOP_FINAL_latched_then_hold_from_next_policy_decision',
    activation='existing_local_task_active', maximum_global_time_s=200.,
    PPO_credit=0, auxiliary_credit=0, auto_teacher_dataset=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters_sha():
    return hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_selected_counts(counts, *, expected_local_decisions, expected_ppo_updates,
                             expected_optimizer_steps):
    expected = dict(local_policy_decisions=expected_local_decisions,
        local_ppo_updates=expected_ppo_updates,local_optimizer_steps=expected_optimizer_steps)
    if any(type(v) is not int or v < 0 for v in expected.values()):
        raise ValueError('explicit expected counters must be nonnegative integers')
    if any(counts.get(k) != v for k,v in expected.items()):
        raise ValueError('selected sealed checkpoint differs from explicit expected counters')
    if counts.get('auxiliary_updates') != 0:
        raise ValueError('unexpected new auxiliary lineage for this diagnostic')


class Knee25Probe:
    def __init__(self, driver):
        self.driver = driver
        self.reset()

    def reset(self):
        self.pending = None
        self.first_top = None
        self.overridden_decisions = 0

    def observe_native(self, frame, snapshot):
        metrics = snapshot['metrics']
        if (self.first_top is None and snapshot['active'] and metrics['current_top_contact']):
            final = list(frame.info['drive_target_full12'])
            self.first_top = dict(native_physics_tick=frame.physics_tick,
                episode_tick=metrics['tick'], time_s=frame.sim_time_s,
                actual_measured_TOP=True, current_bearing=metrics['current_top_bearing'],
                current_attempt_capture_eligible=metrics['current_attempt_capture_eligible'],
                captured_FINAL_RR_deg={6:final[6],7:final[7]},
                hold_begins='next_existing_policy_decision_no_extra_actuator_write',
                PPO_credit=0, automatic_teacher_label=False)

    def __call__(self, core, selected, entry, elapsed, captured_targets=None):
        # captured_targets is the production evaluate's endpoint latch. Use the
        # native-tick latch instead, avoiding substitution of a later target.
        baseline = list(core.frame.info['atomic_ack']['policy_headroom_evidence'][
            'baseline_native_plus_controller_full12'])
        caps = list(core.backend.execution_profile['residual']['phase_caps_full12'][core.frame.state_id])
        if len(selected) != 12 or len(entry) != 12 or len(baseline) != 12 or len(caps) != 12:
            raise ValueError('full12 selected/entry/ACK/caps required')
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('diagnostic elapsed time must be finite and nonnegative')
        fraction = min(1.,elapsed/PARAMETERS['ramp_s'])
        targets = ({int(k):v for k,v in self.first_top['captured_FINAL_RR_deg'].items()}
                   if self.first_top else
                   {6:entry[6]-25.*fraction,7:entry[7]+25.*fraction})
        issued = list(selected)
        ratios, clipped = {}, []
        for index,target in targets.items():
            if not all(math.isfinite(v) for v in (target,baseline[index],caps[index])) or caps[index] <= 0:
                raise ValueError('finite target/baseline and positive physical residual cap required')
            ratio = (target-baseline[index])/caps[index]
            bound = PARAMETERS['raw_inverse_tanh_ratio_limit']
            bounded = max(-bound,min(bound,ratio))
            if ratio != bounded:
                clipped.append(index)
            issued[index] = math.atanh(bounded)
            ratios[index] = dict(unbounded=ratio,issued=bounded)
        intervention = dict(schema='wlr50_clean.independent_rr_knee25_request.v1',
            mode=MODE, intervention='RR_entry_FINAL_hip_minus25_knee_plus25_ramped1.5s',
            PPO_credit=0, auxiliary_credit=0, is_on_policy_action=False,
            script_sha256=self.driver['script_sha256'], parameters_sha256=parameters_sha(),
            overridden_channels=[6,7], other10_selected_policy_values_unchanged=True,
            selected_policy_raw_full12=list(selected), actually_issued_raw_full12=issued,
            entry_FINAL_full12=list(entry), elapsed_s=elapsed, ramp_fraction=fraction,
            candidate_absolute_targets=targets, previous_committed_baseline_full12=baseline,
            previous_committed_frame_tick=core.frame.physics_tick,
            phase_caps_full12=caps, inverse_tanh_ratios=ratios,
            inverse_tanh_limited_channels=clipped,
            actual_final_targets_require_native_dispatch_evidence=True,
            not_same_tick_counterfactual=True, first_native_TOP_latch=copy.deepcopy(self.first_top),
            automatic_teacher_label=False, rear_task_assist_enabled=False)
        self.pending = intervention
        self.overridden_decisions += 1
        return issued, intervention


class RecordingCore:
    """Transparent core wrapper: one original step, no extra physics/write."""
    def __init__(self, inner, probe, run, driver, write_json, write_line):
        self.inner, self.probe, self.run, self.driver = inner,probe,run,driver
        self.write_json, self.write_line = write_json,write_line
        self.stream = None
        self.decisions = 0
        self._observer = None

    def __getattr__(self, name):
        return getattr(self.inner,name)

    def reset(self, *args, **kwargs):
        self.probe.reset()
        return self.inner.reset(*args,**kwargs)

    @property
    def tick_observer(self):
        return self._observer

    @tick_observer.setter
    def tick_observer(self, observer):
        self._observer = observer
        def combined(before,after,projection):
            # CaptureCore observes its task before invoking this callback.
            self.probe.observe_native(after,self.inner.task.snapshot())
            if observer is not None:
                observer(before,after,projection)
        self.inner.tick_observer = combined

    def step(self, issued):
        if self.stream is None:
            source = self.run/'source'
            if not source.is_dir():
                raise RuntimeError('production evaluate must create the source directory first')
            self.write_json(source/'independent_diagnostic_driver.json',self.driver)
            self.stream = (source/'independent_probe_steps.jsonl').open('x',encoding='utf-8')
        observation = list(self.inner.observation)
        if len(observation) != 448 or not all(math.isfinite(v) for v in observation):
            raise ValueError('actual full448 pre-step observation required')
        before = copy.deepcopy(self.inner.task.snapshot())
        active = bool(before['active'])
        intervention = copy.deepcopy(self.probe.pending)
        if active != bool(intervention):
            raise RuntimeError('independent override/request lifecycle mismatch')
        if intervention and list(issued) != intervention['actually_issued_raw_full12']:
            raise RuntimeError('issued raw differs from declared independent override')
        selected = intervention['selected_policy_raw_full12'] if intervention else list(issued)
        pre_tick = self.inner.frame.physics_tick
        step = self.inner.step(issued)  # exactly one native production step
        self.decisions += 1
        info = step.info
        self.write_line(self.stream,dict(
            schema='wlr50_clean.independent_rr_knee25_observed_transition.v1',
            mode=MODE, decision=self.decisions, PPO_credit=0, auxiliary_credit=0,
            automatic_teacher_label=False, provisional_data_only=True,
            script_sha256=self.driver['script_sha256'],parameters_sha256=parameters_sha(),
            checkpoint_sha256=self.driver['checkpoint_sha256'],
            start_tick=pre_tick,end_tick=self.inner.frame.physics_tick,
            request_phase=before['metrics']['phase_id'],
            observation_full448=observation,
            selected_policy_raw_full12=selected,actually_issued_raw_full12=list(issued),
            independent_override=intervention, next_observation_full448=list(step.observation),
            before_local_task=before,after_local_task=self.inner.task.snapshot(),
            current_physical_evaluator=info['semantic_task']['physical_evaluator'],
            final_target_full12=info['actual_drive_target_full12'],
            actuator_target_effect_audit=info['actuator_target_effect_audit'],
            terminal=bool(step.terminated),termination_reason=info.get('termination_reason'),
            first_native_TOP_latch=copy.deepcopy(self.probe.first_top)))
        self.probe.pending = None
        return step

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def execute(args):
    if not args.run_independent_diagnostic:
        raise ValueError('not running: require explicit --run-independent-diagnostic after safe boundary')
    sys.path.insert(0,str(ROOT/'src'))
    from wlr50_clean.ppo import semantic_rr_capture_local as route
    runtime = route.contract(args.expected_head)
    checkpoint = args.checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem+'_manifest.json')
    if sha(checkpoint) != args.checkpoint_sha256 or sha(sidecar) != args.manifest_sha256:
        raise ValueError('explicit checkpoint/sidecar hashes differ')
    metadata = json.loads(sidecar.read_text())
    expected_counts = dict(expected_local_decisions=args.expected_local_decisions,
        expected_ppo_updates=args.expected_ppo_updates,
        expected_optimizer_steps=args.expected_optimizer_steps)
    validate_selected_counts(metadata['counts'],**expected_counts)
    if metadata['runtime_contract'] != runtime or metadata['rollout_empty'] is not True:
        raise ValueError('same strict production runtime and complete checkpoint required')
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME):
        raise ValueError('independent run must remain in the existing isolated namespace')
    run.mkdir(parents=True,exist_ok=False)
    driver = dict(schema='wlr50_clean.independent_rr_knee25_driver.v1', mode=MODE,
        script=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__)),
        parameters=PARAMETERS,parameters_sha256=parameters_sha(),runtime_contract=runtime,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        checkpoint_manifest=str(sidecar),manifest_sha256=sha(sidecar),
        counts=metadata['counts'],expected_counts=expected_counts,PPO_credit=0,auxiliary_credit=0,
        prior_and_local_weights_not_modified=True,production_files_not_modified=True,
        process_local_callback_override='semantic_rr_capture_local.diagnostic_request ONLY',
        no_extra_policy_forward=True,no_extra_actuator_write=True,
        no_extra_physics_step=True,rear_task_assists=False,
        existing_FL_assist='p05_hip_only_continuation_v1',
        purpose='direction/physical capture evidence only; never an automatic teacher dataset')
    route.write(run/'run_manifest.started.json',dict(runtime_contract=runtime,mode=MODE,
        checkpoint=str(checkpoint),independent_diagnostic_driver=driver,
        started_utc=datetime.now(timezone.utc).isoformat()))
    app,core = None,None
    old_callback = route.diagnostic_request
    try:
        # Same native import/launch order as the production route.
        import torch
        import tensordict
        from wlr50_clean.ppo.rl_library_wrapper import seed_training_rngs
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=False,enable_cameras=False).app
        app.update()
        seed_training_rngs(1001)
        runner = route.make_runner(args.device,1001)
        prior,counts = route.load(runner,checkpoint,runtime)
        runner.alg.eval_mode()
        probe = Knee25Probe(driver)
        core = RecordingCore(route.build_core(app),probe,run,driver,route.write,route.line)
        # This is a separate diagnostic process, not a hot edit in any live run.
        route.diagnostic_request = probe
        result = route.evaluate(core,runner,runtime,prior,counts,run,diagnostic=True)
        core.close()
        if (result['mode'] != MODE or result['diagnostic_intervention'] is not True
                or result['PPO_updates'] != 0 or result['checkpoint_model_unchanged'] is not True):
            raise RuntimeError('independent diagnostic mislabeled or modified learned state')
        if sha(checkpoint) != args.checkpoint_sha256 or sha(sidecar) != args.manifest_sha256:
            raise RuntimeError('diagnostic altered its immutable source checkpoint')
        route.write(run/'run_manifest.json',dict(lifecycle='COMPLETE',mode=MODE,result=result,
            independent_diagnostic_driver=driver,recorded_decisions=core.decisions,
            overridden_decisions=probe.overridden_decisions,
            first_native_TOP_latch=probe.first_top,new_policy_decisions=0,new_PPO_updates=0,
            new_optimizer_steps=0,new_AUX_updates=0,automatic_teacher_dataset_created=False))
        print(json.dumps({'run':str(run),'lifecycle':'COMPLETE','mode':MODE,
                          'local_RR_success':result['local_RR_success'],
                          'PPO_credit':0,'overridden_decisions':probe.overridden_decisions}),flush=True)
    except BaseException:
        failure = dict(lifecycle='FAILED',mode=MODE,PPO_credit=0,traceback=traceback.format_exc())
        route.write(run/'failure.json',failure)
        raise
    finally:
        route.diagnostic_request = old_callback
        if core is not None:
            core.close()
        if app is not None:
            app.close(wait_for_replicator=False,skip_cleanup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-independent-diagnostic',action='store_true')
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--expected-local-decisions',type=int,required=True)
    parser.add_argument('--expected-ppo-updates',type=int,required=True)
    parser.add_argument('--expected-optimizer-steps',type=int,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    execute(parser.parse_args())


if __name__=='__main__':
    main()
