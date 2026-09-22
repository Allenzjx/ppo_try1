"""V2 pure-PPO pair: exact quantity-bound execution-profile metadata exception.
Output-only: never alters old review rules, runtime, checkpoints or simulation.
Usable only after production adoption, a persisted new checkpoint and real video.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location('quantity_checked_review', OUT/'review_video.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
require = r.require
def _persisted_infos(checkpoint):
    import torch
    # Called only after checked_review has verified the official local CP hash.
    return torch.load(checkpoint, map_location='cpu', weights_only=False)['infos']
def reject_auxiliary(value):
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            require(not (name == 'aux' or name.startswith('aux_') or 'auxiliary' in name),
                    'Pure PPO pair rejects nested/unknown auxiliary provenance; use a validated dedicated AUX adapter')
            reject_auxiliary(item)
    elif isinstance(value, (list, tuple)):
        for item in value: reject_auxiliary(item)
def verified_evaluation_configuration(a, z, factor, source, target):
    from wlr50_clean.ppo.semantic_migration import _version_bytes
    names = {'action_schema.json':'action_schema_path', 'execution_profile.yaml':'execution_profile',
        'observation_schema.json':'observation_schema_path', 'quality_score.yaml':'quality_score_path',
        'reward_config.yaml':'reward_config_path', 'stage_task_spec.yaml':'task_spec_path'}
    bindings = factor.get('configuration_bindings', {})
    require(set(bindings) == set(names), 'Quantity factor lacks exact six configuration bindings')
    expected = []
    for side, contract in (('source', source), ('target', target)):
        rows = {}
        for name, key in names.items():
            binding = bindings[name][side]
            rows[key] = {'path':str((ROOT/binding['path']).resolve()), 'sha256':binding['sha256'],
                'bytes':len(_version_bytes(ROOT, contract, binding['path'], prefer_worktree=True))}
        expected.append(rows)
    require(a.get('evaluation_configuration') == expected[0] and z.get('evaluation_configuration') == expected[1],
            'Evaluation metadata does not exactly match the officially verified source/target config bytes')
    require([key for key in expected[0] if expected[0][key] != expected[1][key]] == ['execution_profile']
        and {k:v for k,v in expected[0]['execution_profile'].items() if k != 'sha256'}
            == {k:v for k,v in expected[1]['execution_profile'].items() if k != 'sha256'},
        'Only the bound execution-profile hash may differ in evaluation metadata')
    return {'same_evaluation_configuration':False, 'same_control_configuration_via_verified_quantity_factor':True,
        'source_B_evaluation_configuration':expected[0], 'target_C_evaluation_configuration':expected[1],
        'evaluation_difference_scope':'only execution_profile.sha256 via the officially revalidated budget factor'}
def strict_quantity_capture(b, c):
    """No general hash waiver: one exact saved, officially revalidated factor."""
    require(b['receipt'].get('role') == 'B' and b['receipt'].get('mode') == 'N_plus_zero'
            and c['receipt'].get('role') == 'C'
            and c['receipt'].get('mode') == 'deterministic_conditional_mean',
            'Quantity pair requires B zero and deterministic C, never A or a diagnostic')
    a, z = b['manifest'], c['manifest']
    require(a.get('experiment_id') == z.get('experiment_id') == r.EXPERIMENT,
            'Quantity pair experiment mismatch')
    for key in ('camera',):
        require(a.get(key) is not None and a[key] == z.get(key), 'Pair '+key+' differs')
    require(a.get('seed') == z.get('seed') == 4001, 'Pair scene/reset seeds differ')
    entry = a.get('natural_reset_proof', {}).get('entry')
    require(entry is not None and entry == z.get('natural_reset_proof', {}).get('entry'),
            'Pair natural reset entry differs')
    identity = c['receipt']['checkpoint_identity']
    checkpoint = Path(identity['checkpoint']).resolve(strict=True)
    metadata = r.base.read_json(identity['manifest'])
    infos = _persisted_infos(checkpoint)
    reject_auxiliary(infos)
    require(infos == {k:v for k,v in metadata.items() if k not in
            ('checkpoint_path','checkpoint_sha256','save_load_round_trip')},
            'Persisted checkpoint infos differ from the verified manifest')
    extension = infos.get('training_quantity_budget_extension')
    require(isinstance(extension, dict) and extension.get('factor') is not None,
            'C has no persisted quantity-only continuation receipt')
    plan_path = Path(extension['plan_path']).resolve(strict=True)
    require(r.sha256(plan_path) == extension.get('plan_sha256'), 'Persisted quantity plan changed')
    plan = r.base.read_json(plan_path)
    sys.path.insert(0, str(ROOT/'src'))
    from wlr50_clean.ppo.semantic_migration import validate_migration_plan, checkpoint_metadata, digest
    source = Path(plan['source_checkpoint']).resolve(strict=True)
    target = z['runtime_contract']
    verified = validate_migration_plan(source, target, plan_path, project_root=ROOT)
    factor = verified.get('training_quantity_budget_factor')
    require(isinstance(factor, dict) and factor.get('schema') == 'wlr50_clean.training_quantity_budget_same372.v1'
            and extension == {'factor':factor, 'plan_path':verified['plan_path'],
                'plan_sha256':verified['plan_sha256'], 'source_checkpoint_sha256':verified['source_checkpoint_sha256'],
                'source_contract_sha256':verified['source_contract_sha256'],
                'target_contract_sha256':verified['target_contract_sha256']},
            'Saved receipt is not the exact officially revalidated quantity factor')
    require(all(value is None for key,value in verified.items()
                if key.endswith('_factor') and key != 'training_quantity_budget_factor'),
            'Quantity comparison cannot mix auxiliary/controller/other migration factors')
    require(factor.get('same_mdp_claimed') is True and factor.get('training_quantity_only') is True
            and all(factor.get(key) is False for key in ('kernel_changed','reward_changed',
                'task_acceptance_changed','nominal_control_changed','action_execution_changed','new_mdp')),
            'Quantity comparison cannot waive MDP/kernel/reward/task/control changes')
    source_metadata = checkpoint_metadata(source)
    source_contract = source_metadata['runtime_contract']
    require(set(metadata) - set(source_metadata) <= {'training_quantity_budget_extension'},
            'C adds unreviewed auxiliary or other checkpoint provenance beyond quantity continuation')
    require(source_contract == a.get('runtime_contract')
            and target == metadata.get('runtime_contract') == identity.get('runtime_contract')
            and digest(source_contract) == verified['source_contract_sha256']
            and digest(target) == verified['target_contract_sha256']
            and source_contract != target, 'Source B or target C does not exactly bind the quantity boundary')
    configuration = verified_evaluation_configuration(a, z, factor, source_contract, target)
    return {'same_control_via_verified_quantity_only_boundary':True, 'same_runtime_contract':False,
        'source_B_runtime_contract':source_contract, 'target_C_runtime_contract':target,
        'quantity_migration_plan':str(plan_path), 'quantity_migration_plan_sha256':verified['plan_sha256'],
        'saved_C_checkpoint_sha256':identity['checkpoint_sha256'], 'same_camera':a['camera'],
        **configuration, 'same_scene_seed':4001,
        'same_natural_reset_entry':entry, 'same_measured_initial_state_claimed':False,
        'single_pair_is_not_statistical_evidence':True, 'old_B_not_relabelled_as_new_runtime_B':True}
def build_pair(b_path, deterministic_path, destination):
    b, c = r.checked_review(b_path), r.checked_review(deterministic_path)
    common = strict_quantity_capture(b, c)
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUT) and destination != OUT, 'Use a new isolated pair directory')
    destination.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('quantity_encoder_only',
        ROOT/'outputs/ppo_fl_capture_quality_v1/paired_event_media.py')
    encoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(encoder)
    cp = c['receipt']['checkpoint_identity']['saved_global_policy_decisions']
    full = encoder._comparison_video(b,c,destination/f'N_vs_CP{cp}_deterministic_quantity_boundary.mp4',
        left_end=b['receipt']['frame_count'],right_end=c['receipt']['frame_count'],
        kind='full_attempt_same_elapsed_P01_verified_quantity_only_boundary')
    result = {'schema':'wlr50_clean.task_conditioned_quantity_only_control_pair.v2',
        'B_receipt':str(b['receipt_path']), 'deterministic_C_receipt':str(c['receipt_path']),
        'verified_quantity_only_comparison':common, 'full_episode':full,
        'B_physical_result':b['receipt']['physical_result'], 'C_physical_result':c['receipt']['physical_result'],
        'B_physical_duration_s':b['receipt']['physical_duration_s'],
        'C_physical_duration_s':c['receipt']['physical_duration_s'],
        'quality_improvement_claim':None, 'task_success_not_inferred_from_media':True,
        'shorter_side_freeze_is_not_new_physics':True}
    r.base.write_new_json(destination/'pair_receipt.json',result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b-receipt',type=Path,required=True)
    parser.add_argument('--deterministic-receipt',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build_pair(args.b_receipt,args.deterministic_receipt,args.destination),indent=2))
