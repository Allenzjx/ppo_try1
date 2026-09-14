"""Bounded offline control computation on immutable Trial043 telemetry; no Isaac/Torch."""
from __future__ import annotations
import copy
import inspect
import json
import math
from pathlib import Path
import sys

from wlr50_clean.fsm.controller import SensorFsmController
from wlr50_clean.fsm.drive_feedback import ReferenceBoundedDriveFeedback
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.fsm.recovery import RecoveryPlanner
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, SERVO_COMMAND_SIGN
from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.sensing.sensor_reader import SensorReader
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider, SemanticControllerAdapter, load_task_spec
from wlr50_clean.ppo.semantic_physical_sensing import SemanticSensorReader

ROOT = Path.cwd()
SOURCE = Path('C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1')
RUN = SOURCE / 'runs/trial_043_20260902_clean_v010'
manifest = json.loads((RUN / 'trial_manifest.json').read_text())
standing = {r['joint_name']: r['standing_pose_deg'] for r in manifest['environment_initialization']['records']}
mapper = ServoTargetMapper(standing)
# 180 zero/no-tracking settle writes leave all commands/biases zero and advance the clock.
mapper._feedback_tick = manifest['settle_ticks']
controller = SensorFsmController.from_paths(ROOT/'configs/fsm_states.yaml', ROOT/'configs/recording_motion_contract.json')
spec_path = ROOT/'configs/ppo_all_stage_acceptance_v1/stage_task_spec.yaml'
spec = load_task_spec(spec_path)
probes = {}
source_max_error = 0.0
source_controller_error = 0.0
source_tracking_mismatches = 0
previous = None
rows = 0
leg_names = {'FL':'front_left_ankle','FR':'front_right_ankle','RL':'rear_left_ankle','RR':'rear_right_ankle'}

def q_for(observation):
    return tuple(math.radians(standing[n] + SERVO_COMMAND_SIGN[n] * observation['actual_full12'][i])
                 for i,n in enumerate(SERVO_ORDER))

def task_for(observation, phase):
    obstacle = observation['obstacle']
    legs = {}
    for leg,name in leg_names.items():
        wheel = observation['wheels'][name]
        center = wheel['center_w_m']; bottom = wheel['bottom_w_m']
        legs[leg] = {'front_distance_m': center[0]-obstacle['front_x_m'],
                     'within_lateral_span': obstacle['right_y_m'] <= center[1] <= obstacle['left_y_m'],
                     'clearance_m':bottom[2]-obstacle['top_z_m']}
    return {'stage_id':phase,'termination_reason':None,
            'physical_evaluator':{'valid':True,'termination_reason':None,
                'physics_tick':observation['physics_tick'],'simulation_time_s':observation['simulation_time_s'],
                'history':{'placed':{leg:bool(observation['guards']['leg_top_loaded_latched:'+leg]['passed']) for leg in leg_names},'active_lift':{}},
                'current_legs':legs}}

with (RUN/'observation_120hz.jsonl').open() as observations, (RUN/'full12_commands_120hz.jsonl').open() as commands:
    for obs_line, cmd_line in zip(observations, commands):
        observation = json.loads(obs_line); row = json.loads(cmd_line)
        tick = row['control_physics_tick']
        if tick > 6711:
            break
        assert tick == observation['physics_tick'] == rows
        prestate = copy.deepcopy(mapper)
        actual_q = q_for(observation)
        frame = controller.step(observation, sim_time_s=row['sim_time_s'])
        source_controller_error = max(source_controller_error, max(abs(a-b) for a,b in zip(frame.full12,row['nominal_full12'])))
        source_tracking_mismatches += frame.tracking_servo_names != tuple(row['tracking_servo_names'])
        mapping = mapper.advance(row['nominal_full12'][:8],actual_q,tracking_servo_names=row['tracking_servo_names'])
        source_max_error = max(source_max_error,max(abs(a-b) for a,b in zip(mapping.applied_drive_command_deg,row['native_drive_target_full12'][:8])))
        phase = row['state_id']
        if phase in ('P06','P07') and (previous is None or phase != previous['state_id']):
            provider = NominalMotionProvider(controller.contract,spec=spec)
            provider.nominal_full12 = tuple(previous['nominal_full12'])
            provider.state_id = previous['state_id']
            provider.tracking_servo_names = tuple(previous['tracking_servo_names'])
            probes[phase] = {'provider':provider,'mapper':prestate,'samples':[],
                             'start_tick':tick,'same_prestate_and_observation':True,
                             'scope':'isolated current owner with measured Trial043 geometry; not a counterfactual live trajectory or full semantic replay',
                             'first_nominal_difference':None,'first_native_difference':None}
        if phase in probes and tick-probes[phase]['start_tick'] < 64:
            p = probes[phase]
            nominal = p['provider'].evaluate(task_for(observation,phase),observation)
            m = p['mapper'].advance(nominal[:8],actual_q,tracking_servo_names=p['provider'].tracking_servo_names)
            native = tuple(m.applied_drive_command_deg)+tuple(nominal[8:])
            nominal_difference = max(abs(a-b) for a,b in zip(nominal,row['nominal_full12']))
            native_difference = max(abs(a-b) for a,b in zip(native,row['native_drive_target_full12']))
            evidence = {'tick':tick,'sim_time_s':row['sim_time_s'],
                        'source_nominal':row['nominal_full12'],'N_nominal':nominal,
                        'source_native':row['native_drive_target_full12'],'N_native':native,
                        'source_tracking':row['tracking_servo_names'],'N_tracking':p['provider'].tracking_servo_names,
                        'source_actual_canonical_deg':observation['actual_full12'][:8],
                        'source_compensation':mapping.tracking_compensation_deg,'N_compensation':m.tracking_compensation_deg}
            if nominal_difference > 1e-9 and p['first_nominal_difference'] is None:p['first_nominal_difference']=evidence
            if native_difference > 1e-9 and p['first_native_difference'] is None:p['first_native_difference']=evidence
            if tick-p['start_tick'] in (0,1,2,12,24,32,63):p['samples'].append(evidence)
        previous=row; rows+=1

for probe in probes.values():
    del probe['provider'];del probe['mapper']
modules={}
for cls in (SensorFsmController,SemanticControllerAdapter,NominalMotionProvider,MotionExecutor,
            ReferenceBoundedDriveFeedback,RecoveryPlanner,RobotAdapter,ServoTargetMapper,SensorReader,SemanticSensorReader):
    modules[cls.__name__]={'module':cls.__module__,'file':str(Path(inspect.getfile(cls)).resolve())}
print(json.dumps({'schema':'wlr50_clean.successful_reference_offline_control_probe.v1',
    'python_executable':sys.executable,'cwd':str(ROOT),'sys_path':sys.path,
    'imported_classes':modules,'Isaac_run':False,'Torch_imported':'torch' in sys.modules,
    'source_command_observation_pairs':rows,'source_mapper_max_abs_deg_error':source_max_error,
    'source_controller_nominal_max_abs_error':source_controller_error,
    'source_controller_tracking_mismatch_count':source_tracking_mismatches,
    'input_task_spec':str(spec_path),'probes':probes},allow_nan=False,indent=2))
