"""One historical deterministic P01 decision, read-only candidate preparation.

No reconstruction of reset/decision1, fitting, budget, teacher deployment,
on-policy credit, simulator, GPU or checkpoint write.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
import yaml

from wlr50_clean.ppo.semantic_observation import (
    SemanticObservationBuilder, load_semantic_observation_schema, _quaternion,
    _multiply, _rpy, _quat_rotate_inverse,
)
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_p05_capture_actor import p05_capture_request_history
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_history_actor import history_conditioned_head, HISTORY_RHO
from wlr50_clean.ppo.semantic_training import construct_semantic_runner, parameter_hash
from wlr50_clean.ppo.semantic_migration import _version_bytes
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc/source'
CP = OUT / 'checkpoints/history/checkpoint_step_000201728.pt'
CURRENT_CP = OUT / 'checkpoints/history/checkpoint_aux_detP02_step_000211968_v2.pt'
SOURCE_SHA = 'a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6'
CURRENT_SHA = '27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55'
SCHEMA_PATH = 'configs/ppo_p05_hip_only_continuation_v1/observation_schema.json'
ATOL = 1e-6


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(value, label):
    if not bool(value):
        raise ValueError(label)


def prefix(name, count):
    rows, row_shas, digest = [], [], hashlib.sha256()
    with (SOURCE / name).open('rb') as stream:
        for raw in itertools.islice(stream, count):
            rows.append(json.loads(raw)); row_shas.append(hashlib.sha256(raw).hexdigest()); digest.update(raw)
    require(len(rows) == count, 'source prefix absent: ' + name)
    return rows, row_shas, {'path': str(SOURCE / name), 'rows_read': count,
        'prefix_bytes_sha256': digest.hexdigest(), 'whole_source_file_rehashed': False}


def compare(actual, expected, label, *, exact=False):
    actual = actual.detach().cpu()
    expected = torch.tensor(expected, dtype=actual.dtype)
    require(actual.shape == expected.shape, label + ' shape mismatch')
    error = float((actual.double() - expected.double()).abs().max())
    require(torch.equal(actual, expected) if exact else error <= ATOL, label + f' mismatch: {error}')
    return error


def main():
    require(not torch.cuda.is_available(), 'must run CPU-only')
    torch.set_num_threads(1)
    for name in ('candidate.npz', 'candidate_manifest.json', 'README.md'):
        require(not (HERE / name).exists(), 'refusing artifact overwrite: ' + name)
    sm, cm = (read(path.with_name(path.stem + '_manifest.json')) for path in (CP, CURRENT_CP))
    require(sha(CP) == sm['checkpoint_sha256'] == SOURCE_SHA, 'source CP mismatch')
    require(sha(CURRENT_CP) == cm['checkpoint_sha256'] == CURRENT_SHA, 'current semantic reference CP mismatch')
    require(sm['policy_contract'] == cm['policy_contract'] and sm['normalization'] == cm['normalization'], 'policy/encoder normalization changed')
    core_paths = (SCHEMA_PATH, 'src/wlr50_clean/ppo/semantic_observation.py', 'src/wlr50_clean/ppo/semantic_env.py',
        'src/wlr50_clean/ppo/semantic_p05_capture_actor.py', 'src/wlr50_clean/ppo/semantic_history_actor.py',
        'src/wlr50_clean/ppo/semantic_receiving_wheel_sigma.py')
    for path in core_paths:
        require(sha(ROOT / path) == sm['runtime_contract']['files'][path] == cm['runtime_contract']['files'][path], 'source/current runtime bytes differ: ' + path)
    # Reuse the completed structural two-migration audit, not its P02 row-level
    # conclusion. This one P01 state's trigger/potential is verified below.
    admission_path = OUT / 'det_P02_current_semantics_admission.json'
    require(sha(admission_path) == '08a44852a26b98aaddaded707cb6f52ebd76f25e14cee69ce510e851f8c7915b', 'structural migration audit changed')
    structural = read(admission_path)
    require(structural['both_immutable_plans_and_pure_strict_factors_validated'] is True
            and structural['source_runtime'] == sm['runtime_contract']['source_git_commit']
            and structural['current_runtime'] == cm['runtime_contract']['source_git_commit'], 'migration structural evidence does not cover runtimes')
    for key, plan in (('feedback', cm['capture_feedback_semantics_migration']), ('RR_potential', cm['rr_postcross_workspace_migration'])):
        require(sha(plan['plan_path']) == plan['plan_sha256'] == structural['migration_plan_sha256'][key], 'migration plan binding differs')
    decisions, decision_shas, decision_receipt = prefix('video_policy_decisions.jsonl', 259)
    before, decision = decisions[:2]
    require(before['decision'] == 1 and decision['decision'] == 2 and before['end_tick'] == decision['start_tick'] == 8
            and decision['end_tick'] == 16 and decision['request_phase'] == 'P01', 'reviewed one-decision window differs')
    p = decision['policy_request']; task = before['step_info']['semantic_task']; evaluation = task['physical_evaluator']
    require(task['stage_id'] == 'P01' and evaluation['physics_tick'] == 8 and evaluation['valid'] is True
            and evaluation['termination_reason'] is None and task['termination_reason'] is None, 'input is not valid source P01')
    require(p['mode'] == 'deterministic_conditional_mean' and p['sampling_draws'] == p['extra_random_draws'] == 0, 'source was not deterministic')
    require(decision['raw_policy_action_full12'] == p['selected_raw_full12'] == p['conditional_mean_full12'], 'label is not actual deterministic raw')
    maps, receipts = [], [decision_receipt]
    for name, key, count in (('physical_observations.jsonl','physics_tick',10),
                             ('capture_assist_ticks.jsonl','episode_physics_tick',16),
                             ('native_tick_audit.jsonl','episode_physics_tick',16)):
        values, _, receipt = prefix(name, count)
        maps.append({value[key]: value for value in values}); receipts.append(receipt)
    physical, capture, native = maps
    current, prior, following = capture[8], capture[7], capture[9]
    require(current['phase'] == 'P01', 'capture input phase mismatch')
    evidence = following['dispatch']['tracking_reference_evidence']
    require(evidence == native[9]['native_audit']['tracking_reference_evidence'], 'next mapper prestate sources disagree')
    require(evidence['previous_ack_physics_tick'] == current['dispatch']['physics_tick']
            and evidence['mapper_advances'] == evidence['articulation_writes'] == 0, 'mapper ACK/provenance mismatch')
    mapper = {**evidence['mapper_pre_state'], 'feedback_tick': evidence['mapper_feedback_tick'],
        'final_drive_servo_deg': native[9]['native_audit']['previous_final_drive_servo_deg']}
    require(mapper['final_drive_servo_deg'] == current['dispatch']['drive_target_full12'][:8], 'previous final actuator target mismatch')
    history = {'previous_raw_full12': before['raw_policy_action_full12'],
        'previous_residual_full12': native[8]['projected_residual_full12'],
        'previous_previous_residual_full12': native[7]['projected_residual_full12'],
        'previous_applied_full12': current['dispatch']['drive_target_full12'],
        'previous_previous_applied_full12': prior['dispatch']['drive_target_full12'],
        'previous_nominal_full12': prior['nominal_full12']}
    require(history['previous_residual_full12'] == current['dispatch']['independent_policy_residual_requested_full12'], 'request history sources disagree')
    schema = load_semantic_observation_schema(ROOT / SCHEMA_PATH)
    frame = SimpleNamespace(state_id='P01', sim_time_s=current['sim_time_s'], nominal_action_full12=current['nominal_full12'],
        info={'semantic_task': task, 'raw_observation': physical[8], 'mapper_state_summary': mapper,
            'mapped_nominal_full12': current['dispatch']['native_drive_target_full12'], 'capture_assist': current['capture_assist']})
    builder = SemanticObservationBuilder(schema)
    q = _quaternion(physical[7]['base']['orientation_wxyz'])
    builder.previous_time = prior['sim_time_s']
    builder.previous_rpy = _rpy(_quaternion(_multiply(q, schema.fixed_chassis_to_body_wxyz)))
    builder.previous_omega = _quat_rotate_inverse(q, physical[7]['base']['angular_velocity_w_rad_s'])
    x = torch.tensor([schema.encode(builder.build(frame, history).groups)], dtype=torch.float32)
    require(x.shape == (1,389) and torch.isfinite(x).all() and x[0,:13].tolist() == [1.] + [0.]*12, 'invalid reconstructed P01')
    compare(x[0,195:207], p['previous_raw_from_current_observation_full12'], 'previous raw', exact=True)
    compare(x[0,372:384], p['capture_assist_observed_features'], 'assist fields', exact=True)
    compare(x[0,384:389], p['capture_continuation_observed_features'], 'pending fields', exact=True)
    require(torch.count_nonzero(x[0,372:389]) == 0 and current['capture_assist']['mode'] == 0
            and current['capture_assist']['initialized'] == 0, 'WAIT state not explicit')
    for tick in range(9,17):
        audit = native[tick]['native_audit']
        require(audit['verified'] is True and audit['raw_policy_action_full12'] == decision['raw_policy_action_full12']
                and audit['policy_request_phase'] == 'P01' and audit['phase_mask_full12'] == [1]*12, 'execution differs from raw P01 label')
        require(audit['capture_assist_evidence']['owner_indices'] == [] and audit['capture_assist_evidence']['assist_correction_full12'] == [0.]*12, 'source action was assist-owned')
    runner, _ = construct_semantic_runner(_ObservationOnlyEnv(389), seed=sm['seed'], device='cpu',
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    payload = torch.load(CP, map_location='cpu', weights_only=False)
    actor = runner.alg.actor; actor.load_state_dict(payload['actor_state_dict'], strict=True); actor.eval()
    require(parameter_hash(actor) == sm['actor_parameter_sha256'], 'source actor hash differs')
    with torch.no_grad():
        raw_head = actor.mlp(x); center, hist = p05_capture_request_history(x)
        head = history_conditioned_head(raw_head, center, HISTORY_RHO)
        log_std, _ = receiving_wheel_effective_log_std(head[:,1,:], x[:,:372], .25)
        mean = actor(TensorDict({'policy':x, 'critic':x.clone()}, batch_size=[1]), stochastic_output=False)[0]
    errors = {'conditional_mean': compare(mean, p['conditional_mean_full12'], 'conditional mean'),
        'base_mean': compare(raw_head[0,0], p['base_mean_full12'], 'base mean'),
        'history_center': compare(center[0], p['history_center_full12'], 'history center', exact=True),
        'learned_sigma': compare(raw_head[0,1].exp(), p['learned_sigma_full12'], 'learned sigma'),
        'effective_sigma': compare(log_std[0].exp(), p['effective_sigma_full12'], 'effective sigma'),
        'effective_log_std': compare(log_std[0], p['effective_log_std_full12'], 'effective logstd')}
    require(hist['gate_full12'][0].tolist() == p['cap_transition_gate_full12'], 'history cap gate differs')
    require(parameter_hash(actor) == sm['actor_parameter_sha256'], 'actor changed during inspection')
    task_path = 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    old_spec = yaml.safe_load(_version_bytes(ROOT, sm['runtime_contract'], task_path))
    new_spec = yaml.safe_load((ROOT / task_path).read_bytes())
    require(sha(ROOT / task_path) == cm['runtime_contract']['files'][task_path], 'current task specification changed')
    current_without_mode = dict(new_spec); current_without_mode.pop('rr_postcross_workspace_semantics')
    require(current_without_mode == old_spec, 'undeclared current task semantics change')
    require(all(evaluation['history'][name]['RR'] is False for name in ('active_lift','front_edge_crossed','placed'))
            and evaluation['current_legs']['RR']['current_lift_valid'] is False, 'RR predicate prerequisites present')
    require(not _current_rr_receiver_preparation_retired(new_spec,'RR',evaluation), 'RR retirement activates')
    old_sup = object.__new__(TaskStageSupervisor); old_sup.spec = old_spec
    new_sup = object.__new__(TaskStageSupervisor); new_sup.spec = new_spec
    old_phi, new_phi = old_sup.physical_potential(evaluation), new_sup.physical_potential(evaluation)
    require(old_phi == new_phi == task['task_progress_potential'] and np.float32(new_phi) == x[0,17].item(), 'current potential differs')
    endpoint = decision['step_info']['semantic_task']
    require(endpoint['stage_id'] == 'P02' and 'P01' in endpoint['completed_stage_ids'], 'source P01 did not hand off')
    placed_task = decisions[258]['step_info']['semantic_task']; fr = placed_task['physical_evaluator']['current_legs']['FR']
    require(placed_task['history']['event_ticks']['placed']['FR'] == 2072
            and placed_task['placed_history']['FR'] is True and fr['contact_surface'] == 'TOP'
            and fr['top_contact'] and fr['top_surface_contact'] and fr['within_top_xy'] and fr['within_lateral_span']
            and fr['support'] and fr['bearing_verified'] and not fr['ground_contact'], 'source downstream real FR placement absent')
    arrays = {'X389_reconstructed': x.numpy(), 'raw12_actual_deterministic': np.asarray([decision['raw_policy_action_full12']],dtype=np.float32),
        'source_conditional_mean12': np.asarray([p['conditional_mean_full12']],dtype=np.float32),
        'source_effective_sigma12': np.asarray([p['effective_sigma_full12']],dtype=np.float32),
        'source_decision': np.asarray([2],dtype=np.int64), 'input_tick': np.asarray([8],dtype=np.int64)}
    require(arrays['raw12_actual_deterministic'][0].tolist() == decision['raw_policy_action_full12'], 'raw label altered by serialization')
    np.savez(HERE / 'candidate.npz', **arrays)
    with np.load(HERE / 'candidate.npz', allow_pickle=False) as loaded:
        require(all(np.array_equal(loaded[key], value) for key,value in arrays.items()), 'candidate serialization mismatch')
    result = {'schema':'wlr50_clean.det_P01_second_decision_candidate.v1',
        'status':'ONE_ROW_CANDIDATE_FEASIBILITY_NOT_ADMITTED_TO_FIT', 'rows':1,
        'source_checkpoint':{'path':str(CP),'sha256':SOURCE_SHA,'actor_sha256':sm['actor_parameter_sha256']},
        'current_semantic_reference':{'path':str(CURRENT_CP),'sha256':CURRENT_SHA,'runtime':cm['runtime_contract']['source_git_commit']},
        'source_prefixes':receipts, 'decision1_read_as_recorded_history_not_reconstructed_as_training_input':True,
        'source_decision2_line_sha256':decision_shas[1], 'source_prior_decision1_line_sha256':decision_shas[0],
        'source_FR_placement_decision259_line_sha256':decision_shas[258],
        'dataset':{'path':str(HERE/'candidate.npz'),'sha256':sha(HERE/'candidate.npz'),'observation_shape':[1,389],'raw_shape':[1,12]},
        'helper':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
        'original_389_directly_saved':False,'field_supported_reconstruction_no_guessed_state':True,
        'source_mean_history_sigma_errors':errors,'fixed_CPU_CUDA_numeric_atol':ATOL,
        'same_physical_tick_input_sources':{'task_and_previous_raw':'decision1 endpoint tick8',
            'two_REQUEST_history_levels':'native8/native7','applied_nominal_history':'capture8/capture7',
            'mapper_current_state':'capture9 prestate tied to capture8 ACK and previous final drive',
            'finite_difference_previous_sample':'physical7, not reset0 or a guessed zero derivative'},
        'all8_action_physics_ticks_verified_raw_all12_permission_no_assist':True,
        'current_semantic_compatibility':{'P01_WAIT_does_not_trigger_hold_air_revision':True,
            'RR_qualified_crossed_placed_all_false':True,'RR_retirement_active':False,
            'old_phi':old_phi,'current_phi':new_phi,'X17':float(x[0,17]),'X17_float32_equal':True,
            'same_codec_policy_caps_sigma_nominal_physics_contract':True,
            'structural_two_migration_audit_sha256':sha(admission_path)},
        'source_local_front_evidence':{'input_tick':8,'input_time_s':8/120,'input_phase':'P01',
            'decision2_endpoint_tick':16,'decision2_endpoint_phase':'P02','P01_completed_at_endpoint':True,
            'FR_qualified_event_tick':placed_task['history']['event_ticks']['active_lift']['FR'],
            'FR_placed_event_tick':2072,'FR_placement_endpoint_decision':259,'FR_current_TOP_support_evidence':fr},
        'limitations':['One historical state is not representative P01 training coverage and gives no independent train/validation split.',
            'P01 reset/decision1 input has not been reconstructed or admitted; no synthetic reset row was created.',
            'P01 source scheduling completed at16; FR qualification occurred later at23 in P02, so no early qualification is invented.',
            'Same numeric input semantic compatibility is not current actor equality or future trajectory equivalence.',
            'Only a candidate for possible separately authorized off-policy supervised use; not new PPO data or current learner success.',
            'No fitting, budget, teacher, on-policy credit or old value/reward-target admission.'],
        'AUX_updates':0,'PPO_decisions_added':0,'PPO_updates_added':0,'checkpoint_written':False}
    (HERE/'candidate_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (HERE/'README.md').write_text(f'''# One-row historical P01 decision2 candidate

**PASS for bounded candidate feasibility only.** Source CP201728 actual deterministic decision2 uses input tick8 (0.066667 s); it runs8 ticks and hands off to P02 at tick16. Reset/decision1 input was not reconstructed. Its actual recorded action is used only as the previous-raw history required by decision2.

X389 was not directly stored by the original video. This single row is reconstructed from explicitly aligned task/physics, two-level HISTORY, next-dispatch mapper prestate/current ACK and previous physical-sample derivatives. No missing state was guessed. Original CPU replay errors: μ {errors['conditional_mean']:.12g}, base μ {errors['base_mean']:.12g}, history {errors['history_center']:.12g}, σ {errors['effective_sigma']:.12g}; fixed tolerance1e-6 accommodates original CUDA versus CPU float arithmetic and is not bitwise-input proof.

Current5fd compatibility holds for this recorded state: assist WAIT is inactive, RR qualification/cross/placed false, receiver-retirement0. Old/current potential both {new_phi:.12g}, exactly equal to X17 after float32 encoding. Existing two-migration structural proof and unchanged relevant codec/kernel/config bytes are bound in the manifest. This is not current actor or future-trajectory equivalence.

Source-local evidence: real P01 task handoff at16; FR qualification tick23 (later in P02, not prematurely credited to P01); FR actual placed tick2072 with current legal TOP/bearing/support evidence at decision259. These are historical source outcomes, not current learner success or a full-task success claim. All8 action ticks have verified actual raw, all12 permission and no assist ownership.

`candidate.npz` contains exactly1×389 reconstructed input and1×12 unchanged actually executed deterministic raw action (the recorded conditional mean), plus source mean/sigma and IDs. A single state provides no representative P01 coverage and no independent train/validation split. No fitting, budget, AUX/PPO credit, teacher or checkpoint was created; old bound candidate/v1 helpers remain untouched. The CPU helper exits after this preparation.
''',encoding='utf-8')
    print(json.dumps({'status':result['status'],'rows':1,'errors':errors,'current_phi':new_phi,
        'X17_float32_exact':True,'source_FR_placed_tick':2072,'dataset_sha256':result['dataset']['sha256'],
        'manifest':str(HERE/'candidate_manifest.json'),'fit_performed':False},indent=2))


if __name__ == '__main__':
    main()
