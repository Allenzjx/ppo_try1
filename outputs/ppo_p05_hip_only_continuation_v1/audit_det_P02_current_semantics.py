"""Read-only current-semantics admission check; never fits or mutates data.

Scope:254 already reconstructed CP201728 P02 inputs, two recorded migrations,
and current CP211968. No trajectory replay, reward retuning or checkpoint write.
"""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from wlr50_clean.ppo.semantic_migration import _contract, _version_bytes, digest
from wlr50_clean.ppo.semantic_capture_feedback_migration import capture_feedback_factor
from wlr50_clean.ppo.semantic_rr_workspace_migration import rr_workspace_factor
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired
from wlr50_clean.ppo.semantic_capture_assist import capture_assist_features

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / 'det_front_rehearsal_data_v1'
HISTORY = OUT / 'checkpoints/history'
SOURCE_CP = HISTORY / 'checkpoint_step_000201728.pt'
CURRENT_CP = HISTORY / 'checkpoint_step_000211968.pt'
CURRENT_SHA = '5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881'
TASK_PATH = 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
CAPTURE_PATH = 'src/wlr50_clean/ppo/semantic_capture_assist.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def meta(cp):
    return read(cp.with_name(cp.stem + '_manifest.json'))


def require(value, label):
    if not bool(value):
        raise ValueError(label)


def pure_function_ast(raw, name):
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, 'function definition is not unambiguous: ' + name)
    return ast.dump(nodes[0], include_attributes=False)


def recorded_migration(current, key, before, after):
    migration = current[key]
    supplied = read(migration['plan_path'])
    require(sha(migration['plan_path']) == migration['plan_sha256'], key + ' immutable plan hash differs')
    require({k: v for k, v in migration.items() if k not in ('plan_path', 'plan_sha256')} == supplied,
            key + ' metadata differs from immutable plan')
    require(migration['source_contract_sha256'] == digest(before)
            and migration['target_contract_sha256'] == digest(after), key + ' contract binding differs')
    require(migration['source_runtime_content_sha256'] == before['runtime_content_sha256']
            and migration['target_runtime_content_sha256'] == after['runtime_content_sha256'], key + ' runtime chain differs')
    source = Path(migration['source_checkpoint'])
    require(sha(source.with_name(source.stem + '_manifest.json')) == migration['source_manifest_sha256'], key + ' source sidecar differs')
    require(meta(source)['checkpoint_sha256'] == migration['source_checkpoint_sha256'], key + ' source checkpoint recorded binding differs')
    delta = sorted(path for path in set(before['files']) | set(after['files']) if before['files'].get(path) != after['files'].get(path))
    require(delta == migration['allowed_changed_files'], key + ' scope differs')
    require(migration['changed_file_hashes'] == {path: {'before': before['files'].get(path), 'after': after['files'].get(path)} for path in delta},
            key + ' exact changed hashes differ')
    return migration


def main():
    reader_spec = importlib.util.spec_from_file_location('det_source_reader', DATA / 'read_candidate.py')
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)
    arrays, data_receipt = reader.load_candidate(DATA / 'candidate_manifest.json')
    old_meta, current_meta = meta(SOURCE_CP), meta(CURRENT_CP)
    require(sha(CURRENT_CP) == current_meta['checkpoint_sha256'] == CURRENT_SHA, 'current CP211968 identity differs')
    old, current = _contract(old_meta['runtime_contract']), _contract(current_meta['runtime_contract'])
    fb_source = meta(HISTORY / 'checkpoint_step_000203776.pt')
    rr_source = meta(HISTORY / 'checkpoint_step_000207872.pt')
    middle = _contract(rr_source['runtime_contract'])
    require(old == _contract(fb_source['runtime_contract']), 'source CP201728 is not the same0001 runtime as feedback migration source')
    require(old_meta['policy_contract'] == current_meta['policy_contract'], 'policy dimensions/kernel/caps/sigma contract changed')
    require(old_meta['normalization'] == current_meta['normalization'], 'normalization semantics changed')
    unchanged_top_fields = {key: value for key, value in old.items()
        if key not in ('files', 'source_git_commit', 'runtime_content_sha256', 'selected_configuration')}
    require(unchanged_top_fields == {key: value for key, value in current.items()
        if key not in ('files', 'source_git_commit', 'runtime_content_sha256', 'selected_configuration')},
        'physics/rate/topology/runtime profile changed outside declared source files')
    fb = recorded_migration(current_meta, 'capture_feedback_semantics_migration', old, middle)
    fb_factor = capture_feedback_factor(fb_source, old, middle, reason=fb['reason'],
        reviewed_code_sha256=fb['capture_feedback_semantics_factor']['reviewed_code_sha256'])
    require(digest(fb_factor) == digest(fb['capture_feedback_semantics_factor']), 'feedback pure strict factor no longer validates')
    rr = recorded_migration(current_meta, 'rr_postcross_workspace_migration', middle, current)
    old_spec = yaml.safe_load(_version_bytes(ROOT, old, TASK_PATH))
    middle_spec = yaml.safe_load(_version_bytes(ROOT, middle, TASK_PATH))
    current_spec = yaml.safe_load((ROOT / TASK_PATH).read_bytes())
    require(old_spec == middle_spec, 'feedback migration changed task specification')
    rr_factor = rr_workspace_factor(rr_source, middle, current, reason=rr['reason'],
        reviewed_code_sha256=rr['rr_postcross_workspace_factor']['reviewed_code_sha256'],
        source_task_spec=middle_spec, target_task_spec=current_spec)
    require(digest(rr_factor) == digest(rr['rr_postcross_workspace_factor']), 'RR pure strict factor no longer validates')
    all_delta = sorted(path for path in set(old['files']) | set(current['files']) if old['files'].get(path) != current['files'].get(path))
    require(set(all_delta) == set(fb['allowed_changed_files']) | set(rr['allowed_changed_files']), 'unexplained runtime file difference')
    for path in all_delta:
        require(sha(ROOT / path) == current['files'][path], 'current reviewed changed file differs: ' + path)
    stable_paths = sorted(path for path in old['files'] if old['files'][path] == current['files'].get(path))
    # Check actual disk for the encoding/kernel/execution inputs relevant to the
    # candidate; do not rehash unrelated historical recordings or datasets.
    stable_relevant = [path for path in stable_paths if path.startswith('configs/ppo_p05_hip_only_continuation_v1/')
        or any(token in path for token in ('semantic_observation.py', 'semantic_history_actor.py', 'semantic_p05_capture_actor.py',
            'semantic_receiving_wheel_sigma.py', 'semantic_policy_distribution.py', 'semantic_transfer_roles.py',
            'semantic_nominal', 'semantic_residual_adapter.py', 'semantic_env.py', 'semantic_backend.py', 'actuator_target_effect.py'))]
    for path in stable_relevant:
        require(sha(ROOT / path) == current['files'][path], 'same-hash relevant runtime file changed on disk: ' + path)
    require(pure_function_ast(_version_bytes(ROOT, old, CAPTURE_PATH), 'capture_assist_features')
            == pure_function_ast((ROOT / CAPTURE_PATH).read_bytes(), 'capture_assist_features'),
            'capture numeric feature encoder changed')
    old_supervisor = object.__new__(TaskStageSupervisor); old_supervisor.spec = old_spec
    current_supervisor = object.__new__(TaskStageSupervisor); current_supervisor.spec = current_spec
    source = Path(data_receipt['source_video_manifest']['path']).parent
    decisions, decision_shas = [], []
    with (source / 'video_policy_decisions.jsonl').open('rb') as stream:
        for raw in itertools.islice(stream, 256):
            decisions.append(json.loads(raw)); decision_shas.append(hashlib.sha256(raw).hexdigest())
    checks = [json.loads(line) for line in (DATA / 'row_checks.jsonl').read_text(encoding='utf-8').splitlines()]
    capture = {}
    with (source / 'capture_assist_ticks.jsonl').open('rb') as stream:
        for raw in itertools.islice(stream, 2040):
            row = json.loads(raw)
            if row['episode_physics_tick'] in set(arrays['input_tick'].tolist()):
                capture[row['episode_physics_tick']] = row
    rows = []
    for index, (decision_id, tick, x) in enumerate(zip(arrays['source_decision'], arrays['input_tick'], arrays['X389_reconstructed'], strict=True)):
        try:
            decision, predecessor = decisions[int(decision_id) - 1], decisions[int(decision_id) - 2]
            task = predecessor['step_info']['semantic_task']; evaluation = task['physical_evaluator']
            require(decision_shas[int(decision_id) - 1] == checks[index]['source_decision_line_sha256'], 'source action row differs from reconstructed candidate')
            require(decision['start_tick'] == predecessor['end_tick'] == evaluation['physics_tick'] == int(tick), 'input evaluator timestamp mismatch')
            require(decision['request_phase'] == task['stage_id'] == capture[int(tick)]['phase'] == 'P02', 'input stage differs')
            require(evaluation['valid'] is True and evaluation['termination_reason'] is None, 'input physical evaluator invalid')
            rr_current = evaluation['current_legs']['RR']; rr_history = evaluation['history']
            require(rr_history['active_lift']['RR'] is False and rr_history['front_edge_crossed']['RR'] is False
                    and rr_history['placed']['RR'] is False and rr_current['current_lift_valid'] is False,
                    'RR prerequisite active; old/new potential cannot be presumed equal')
            require(x[149] == x[153] == x[157] == 0, 'encoded RR facts disagree')
            assist = capture[int(tick)]['capture_assist']
            require(assist['mode'] == 0. and assist['initialized'] == 0. and assist['active'] is False,
                    'source capture is not inactive WAIT')
            require(np.array_equal(np.asarray(capture_assist_features(assist), dtype=np.float32), x[372:384]), 'current capture encoder differs')
            require(np.all(x[372:389] == 0), 'source pending/assist state unexpectedly active')
            require(not _current_rr_receiver_preparation_retired(current_spec, 'RR', evaluation), 'current RR retirement activates')
            before = digest(evaluation)
            old_phi = old_supervisor.physical_potential(evaluation)
            new_phi = current_supervisor.physical_potential(evaluation)
            require(digest(evaluation) == before, 'pure potential mutated recorded evaluation')
            stored_phi = task['task_progress_potential']
            require(abs(old_phi - stored_phi) <= 1e-12 and abs(new_phi - old_phi) <= 1e-12, 'physical potential semantic mismatch')
            require(np.float32(old_phi) == np.float32(new_phi) == x[17], 'current float32 index17 differs from candidate')
            rows.append({'candidate_index': index, 'source_decision': int(decision_id), 'input_tick': int(tick),
                'input_evaluator_from_previous_decision': predecessor['decision'],
                'RR_historical_qualified': False, 'RR_current_qualified': False, 'RR_crossed': False, 'RR_placed': False,
                'assist_WAIT': True, 'feedback_hold_to_air_change_can_trigger': False, 'RR_retirement_can_trigger': False,
                'source_task_phi': stored_phi, 'old_spec_phi': old_phi, 'current_spec_phi': new_phi,
                'X17': float(x[17]), 'source_phi_abs_error': abs(old_phi - stored_phi),
                'old_current_phi_abs_difference': abs(new_phi - old_phi),
                'current_phi_to_X17_rounding_error': abs(new_phi - float(x[17])), 'X17_float32_exact': True})
        except Exception as exc:
            raise ValueError(f'BLOCK ADMISSION at candidate {index}, decision {decision_id}, input tick {tick}: {exc}') from exc
    report = {'schema': 'wlr50_clean.det_front_current_semantics_admission.v1',
        'result': 'PASS_BOUNDED_P02_HISTORICAL_SUPERVISED_DATA_SEMANTICS_ONLY',
        'rows_checked': len(rows), 'source_checkpoint': str(SOURCE_CP), 'source_checkpoint_sha256': old_meta['checkpoint_sha256'],
        'current_checkpoint': str(CURRENT_CP), 'current_checkpoint_sha256': CURRENT_SHA,
        'candidate_manifest_sha256': data_receipt['read_validation']['manifest_sha256'],
        'candidate_npz_sha256': data_receipt['dataset']['sha256'], 'source_runtime': old['source_git_commit'],
        'intermediate_runtime': middle['source_git_commit'], 'current_runtime': current['source_git_commit'],
        'migration_plan_sha256': {'feedback': fb['plan_sha256'], 'RR_potential': rr['plan_sha256']},
        'both_immutable_plans_and_pure_strict_factors_validated': True, 'exact_explained_runtime_delta': all_delta,
        'unchanged_runtime_files_count': len(stable_paths), 'same_hash_relevant_files_disk_checked': stable_relevant,
        'policy_kernel_action_caps_sigma_contract_equal': True, 'normalization_semantics_equal': True,
        'capture_numeric_encoder_AST_equal': True, 'physical_runtime_profile_equal': True,
        'task_spec_diff_only': {'rr_postcross_workspace_semantics': current_spec['rr_postcross_workspace_semantics']},
        'maximum_source_phi_abs_error': max(row['source_phi_abs_error'] for row in rows),
        'maximum_old_current_phi_abs_difference': max(row['old_current_phi_abs_difference'] for row in rows),
        'maximum_float64_phi_to_float32_X17_rounding': max(row['current_phi_to_X17_rounding_error'] for row in rows),
        'all_float32_X17_exact': True, 'RR_retirement_active_inputs': 0, 'capture_HOLD_to_AIR_eligible_inputs': 0,
        'scope': 'Only these254 P02 reconstructed states and unchanged actual deterministic raw labels; suggested85/85 split is not independent trajectory validation.',
        'restrictions': ['Historical off-policy supervised targets only; not on-policy PPO samples or new-reward return/value ground truth.',
            '389 was reconstructed with explicit source-field provenance, not directly stored or proven bitwise by12 outputs.',
            'No claim CP211968 matches the old actor or will traverse the same states/actions in a current rollout.',
            'No P01/reset admission, no other stages admitted, no pure-policy FL/full-task success label.',
            'This check authorizes no optimization; any bounded AUX needs separate explicit execution approval and latest-checkpoint binding.'],
        'no_data_or_existing_helper_mutation': True, 'AUX_updates': 0, 'PPO_updates': 0,
        'row_checks': rows}
    (OUT / 'det_P02_current_semantics_admission.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    md = f'''# CP201728 P02 candidate → current CP211968 semantic admission

**PASS, narrowly scoped to254 historical P02 inputs/actions for possible off-policy supervised rehearsal; no optimization authorized or executed.**

- Source runtime0001 → feedback-v2 a802 → current5fd: both existing immutable migration plans and their pure strict factor validators pass. Actual runtime inventory differences are only their declared7-file union. Policy/kernel/caps/sigma and normalization contracts are equal; physical rates/assets/profile and all other file hashes are unchanged. Relevant unchanged codec/actor/nominal/adapter/config files were checked against current disk.
- Every input uses the immediately preceding real decision endpoint, exactly aligned to its source start tick. All254 RR historical qualification/cross/placed bits and current qualification are false; current retirement predicate activates0 times. No unrecorded state was filled.
- All254 capture observations are uninitialized WAIT with zero12 assist+5 pending features. The changed P05 HOLD→AIR edge cannot trigger in this P02 window; numeric feature-encoder AST is unchanged (the added feedback revision is nonnumeric provenance).
- Current pure `physical_potential` and old-spec pure potential both match the logged input potential. Maximum old↔current difference **{report['maximum_old_current_phi_abs_difference']:.12g}**; old↔source error **{report['maximum_source_phi_abs_error']:.12g}**. All254 values are **exactly equal after float32 encoding to X[17]**; maximum float64→float32 rounding {report['maximum_float64_phi_to_float32_X17_rounding']:.12g}.

The original video never saved the389 tensor directly: the candidate remains field-reconstructed, previously checked against original μ/history/σ, not bitwise-direct storage. These labels are actual CP201728 deterministic raw actions from a locally successful FR approach with later front continuation. This does not claim current CP211968 actor equality, closed-loop trajectory equivalence, full-task success, pure-policy FL placement, or current on-policy samples. Old value/reward targets are not admitted.

Only candidate P02 rows0–253 (source decisions3–256) are covered. P01/reset and other stages are excluded. A later finite AUX would still require explicit execution approval, latest-checkpoint binding, preservation checks and separate AUX credit. This check performs0 AUX/PPO updates and changes no data, old v1 helper, production file or checkpoint.

Current checkpoint SHA256 `{CURRENT_SHA}`; candidate manifest SHA256 `{report['candidate_manifest_sha256']}`. JSON contains both plan hashes, exact changed-file scope and all254 per-row phi/trigger checks. CPU helper exits after writing only this report.
'''
    (OUT / 'det_P02_current_semantics_admission.md').write_text(md, encoding='utf-8')
    require(sha(DATA / 'candidate_manifest.json') == report['candidate_manifest_sha256']
            and sha(DATA / 'candidate.npz') == report['candidate_npz_sha256'], 'candidate changed during admission read')
    print(json.dumps({key: report[key] for key in ('result', 'rows_checked', 'maximum_source_phi_abs_error',
        'maximum_old_current_phi_abs_difference', 'maximum_float64_phi_to_float32_X17_rounding',
        'all_float32_X17_exact', 'RR_retirement_active_inputs', 'capture_HOLD_to_AIR_eligible_inputs')}, indent=2))


if __name__ == '__main__':
    main()
