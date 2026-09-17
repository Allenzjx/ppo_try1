"""Replay retained physical inputs through old/new P13 ownership computations.

No simulator, optimizer, actuator, observation edits or new physical outcome.
"""
import copy
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
OLD_COMMIT='ad0c1328f1772c440755f3b6a6622c35e464396e'
RELATIVE='src/wlr50_clean/ppo/semantic_supervisor.py'
SOURCE=ROOT/'runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
    from wlr50_clean.ppo.semantic_training import write_json
    before_bytes=subprocess.check_output(['git','show',f'{OLD_COMMIT}:{RELATIVE}'],cwd=ROOT)
    previous=types.ModuleType('wlr50_clean.ppo._retained_zero_stop_before')
    previous.__file__=str(ROOT/RELATIVE)
    previous.__package__='wlr50_clean.ppo'
    sys.modules[previous.__name__]=previous
    exec(compile(before_bytes,previous.__file__,'exec'),previous.__dict__)
    old=previous.SemanticControllerAdapter.from_paths(ROOT/'configs/fsm_states.yaml',
        ROOT/'configs/recording_motion_contract.json',
        task_spec_path=ROOT/'configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml')
    expected={r['episode_physics_tick']:r for r in rows(SOURCE/'native_tick_audit.jsonl')
        if r['source_phase_id']=='P13'}
    first=min(expected)-1
    new=None
    count=0
    max_error=max_recorded_error=0.
    stage_mismatches=tracking_mismatches=bias_mismatches=0
    first_post=first_new_owner=first_old_owner=None
    first_difference=None
    last=None
    for observation in rows(SOURCE/'physical_observations.jsonl'):
        tick=observation['physics_tick']
        if tick==first:
            # Frozen source contracts contain mappingproxy values. Share only
            # immutable source/spec trees; clone all mutable runtime history.
            memo={}
            def share_source(value):
                if id(value) in memo:
                    return
                memo[id(value)]=value
                if dataclasses.is_dataclass(value):
                    for field in dataclasses.fields(value):
                        share_source(getattr(value,field.name))
                elif isinstance(value,Mapping):
                    for key,item in value.items():
                        share_source(key); share_source(item)
                elif isinstance(value,(tuple,list)):
                    for item in value:
                        share_source(item)
            share_source(old.spec); share_source(old.contract)
            new=copy.deepcopy(old,memo)
            # Exactly the sole modified production method; all source clocks,
            # layers and measured history are copied from complete old replay.
            new.nominal_provider._observe_final_stop_owner=types.MethodType(
                NominalMotionProvider._observe_final_stop_owner,new.nominal_provider)
        old_frame=old.step(observation,sim_time_s=observation['simulation_time_s'])
        if new is None:
            continue
        new_frame=new.step(observation,sim_time_s=observation['simulation_time_s'])
        count+=1
        error=max(abs(a-b) for a,b in zip(old_frame.full12,new_frame.full12))
        max_error=max(max_error,error)
        if error and first_difference is None:
            first_difference={'observation_tick':tick,'old':old_frame.full12,'new':new_frame.full12}
        stage_mismatches+=old_frame.state_id!=new_frame.state_id
        tracking_mismatches+=old_frame.tracking_servo_names!=new_frame.tracking_servo_names
        bias_mismatches+=old.nominal_provider.normal_drive_bias_full12!=new.nominal_provider.normal_drive_bias_full12
        record=expected.get(tick+1)
        if record:
            max_recorded_error=max(max_recorded_error,
                max(abs(a-b) for a,b in zip(old_frame.full12,record['nominal_full12'])))
        ev=old.task_snapshot['physical_evaluator']
        if first_post is None and ev['post_completion_observation_started']:
            first_post=tick
        if first_new_owner is None and new.nominal_provider._final_stop_owner is not None:
            first_new_owner=tick
        if first_old_owner is None and old.nominal_provider._final_stop_owner is not None:
            first_old_owner=tick
        last={'tick':tick,'old_result':old.task_snapshot['termination_reason'],
              'new_result':new.task_snapshot['termination_reason'],
              'old_wheels':old_frame.full12[8:],'new_wheels':new_frame.full12[8:]}
    checks={'old_new_nominal_full12_exact':max_error==0.,
        'old_replay_matches_recorded_nominal_exact':max_recorded_error==0.,
        'stage_identity':stage_mismatches==0,'tracking_identity':tracking_mismatches==0,
        'controller_bias_identity':bias_mismatches==0,'post_window_starts_at_original_tick':first_post==8737,
        'owner_acquisition_same_tick':first_old_owner==first_new_owner,
        'same_replayed_existing_result':last['old_result']==last['new_result']}
    receipt={'schema':'wlr50_clean.retained_zero_p13_stop_replay.v1','source':str(SOURCE),
        'baseline_source_manifest':str(SOURCE/'semantic_video_source_manifest.json'),
        'baseline_source_manifest_sha256':hashlib.sha256((SOURCE/'semantic_video_source_manifest.json').read_bytes()).hexdigest(),
        'old_runtime_commit':OLD_COMMIT,'old_supervisor_sha256':hashlib.sha256(before_bytes).hexdigest(),
        'new_supervisor_sha256':hashlib.sha256((ROOT/RELATIVE).read_bytes()).hexdigest(),
        'physical_inputs_unmodified':True,'all_pre_P13_ticks_replayed_to_construct_history':first,
        'first_P13_observation_tick':first,'complete_P13_input_rows':count,
        'last_P13_observation_tick':last['tick'],'recorded_P13_dispatch_rows':len(expected),
        'max_old_new_nominal_error':max_error,'max_old_recorded_nominal_error':max_recorded_error,
        'first_post_window_tick':first_post,'first_new_owner_tick':first_new_owner,
        'first_old_owner_tick':first_old_owner,'first_difference':first_difference,
        'checks':checks,'all_checks_passed':all(checks.values()),'last':last,
        'not_a_new_physical_run':True,'physics_steps':0,'optimizer_updates':0}
    path=ROOT/'outputs/ppo_task_first_recovery_v1/retained_zero_p13_stop_replay_receipt.json'
    write_json(path,receipt)
    print(json.dumps({'path':str(path),**receipt},indent=2))
    if not all(checks.values()):
        raise RuntimeError('retained zero P13 replay did not establish exact identity')


if __name__=='__main__':
    main()
