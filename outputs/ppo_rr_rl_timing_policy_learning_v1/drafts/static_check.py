"""Stdlib-only dry patch/AST/signature check; never import the training runtime."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def preview_patch(text):
    lines = text.splitlines()
    assert lines[0] == '*** Begin Patch' and lines[-1] == '*** End Patch'
    result = {}; i = 1
    while i < len(lines)-1:
        header = lines[i]; i += 1
        if header.startswith('*** Add File: '):
            path = header.removeprefix('*** Add File: '); chunk = []
            assert not (ROOT/path).exists(), 'new production/test path already exists: '+path
            while i < len(lines) and not lines[i].startswith('*** '):
                assert lines[i].startswith('+')
                chunk.append(lines[i][1:]); i += 1
            result[path] = '\n'.join(chunk)+'\n'
        else:
            assert header.startswith('*** Update File: '), header
            path = header.removeprefix('*** Update File: ')
            source = result.get(path, (ROOT/path).read_text(encoding='utf-8'))
            while i < len(lines) and lines[i].startswith('@@'):
                i += 1; before = []; after = []
                while i < len(lines) and not lines[i].startswith(('@@','*** ')):
                    line = lines[i]; i += 1
                    assert line and line[0] in ' +-'
                    if line[0] in ' -': before.append(line[1:])
                    if line[0] in ' +': after.append(line[1:])
                old = '\n'.join(before)+'\n'; new = '\n'.join(after)+'\n'
                assert before and source.count(old) == 1, 'hunk is missing/ambiguous: '+path
                source = source.replace(old,new,1)
            result[path] = source
    return result


def constant(node, names):
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,ast.Name): return names[node.id]
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        return constant(node.left,names)+constant(node.right,names)
    if isinstance(node,ast.Dict):
        return {constant(k,names):constant(v,names) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == 'dict':
        assert not node.args
        return {k.arg:constant(k.value,names) for k in node.keywords}
    raise ValueError('not a simple literal binding')


def main():
    patch = (HERE/'same419_migration.patch').read_text(encoding='utf-8')
    previews = preview_patch(patch)
    for path,source in previews.items(): ast.parse(source,filename=path)
    module_path = 'src/wlr50_clean/ppo/semantic_rear_recapture_migration.py'
    module = (HERE/'semantic_rear_recapture_migration.py').read_text(encoding='utf-8')
    tests = (HERE/'test_semantic_rear_recapture_migration.py').read_text(encoding='utf-8')
    assert previews[module_path].strip() == module.strip()
    assert previews['tests/unit/test_semantic_rear_recapture_migration.py'].strip() == tests.strip()
    tree = ast.parse(module); names = {}
    wanted = {'SCHEMA','FACTOR_KEY','MIGRATION','SOURCE_HEAD','SOURCE_MODE','TARGET_MODE','CODE',
        'MEDIA_REVIEW_COMMIT','MEDIA_REVIEW'}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in wanted: names[key]=constant(node.value,names)
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    signatures={name:[arg.arg for arg in node.args.args+node.args.kwonlyargs] for name,node in functions.items()}
    for name in ('build_rear_recapture_migration','validate_rear_recapture_migration',
            'load_rear_recapture_migration','publish_rear_recapture_checkpoint','validate_rear_recapture_lineage'):
        assert name in functions
    publisher = ast.parse((HERE/'publish_rear_recapture.py').read_text(encoding='utf-8'))
    for node in ast.walk(publisher):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:
            signature=signatures[node.func.id]
            assert all(kw.arg in signature for kw in node.keywords)
            assert len(node.args) <= len(functions[node.func.id].args.args)
    assert 'rear_recapture_same419_factor' in previews['src/wlr50_clean/ppo/semantic_training.py']
    assert 'rear_recapture_same419_factor' in previews['src/wlr50_clean/ppo/semantic_cli.py']
    assert names['SCHEMA'] in previews['src/wlr50_clean/ppo/semantic_migration.py']
    assert 'validate_rear_recapture_lineage(metadata, contract, receipt)' in previews[
        'src/wlr50_clean/ppo/semantic_rear_policy_timing_migration.py']
    side = ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/checkpoints/history/checkpoint_step_000222080_manifest.json'
    assert hashlib.sha256(side.read_bytes()).hexdigest() == 'd2478db560db5bbd7f4e16452b5f58e250dd463d3b3b597c8e9b796a00124fcb'
    metadata = json.loads(side.read_text(encoding='utf-8'))
    assert metadata['checkpoint_sha256'] == '41fd0b7b71e1a7af6e8f61a906a96035172be6e54181ad640a8ff33f0ccac72c'
    assert metadata['runtime_contract']['source_git_commit'] == names['SOURCE_HEAD']
    assert [metadata[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')] == [222080,1700,34000]
    assert metadata['optimizer_learning_rate'] == 1e-5
    for path,pins in names['MEDIA_REVIEW'].items():
        assert metadata['runtime_contract']['files'][path] == pins['before']
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == pins['after']
    assert 'runner.alg.optimizer.load_state_dict' not in module
    assert 'zero_append' not in module
    print(json.dumps(dict(status='STATIC_PASS',patch_sha256=hashlib.sha256((HERE/'same419_migration.patch').read_bytes()).hexdigest(),
        previewed_paths=list(previews),source_counter_tuple=[222080,1700,34000],source_effective_lr=1e-5,
        source_runtime=names['SOURCE_HEAD'],reviewed_media_commit=names['MEDIA_REVIEW_COMMIT'],
        media_pins=names['MEDIA_REVIEW'],exported_signatures=signatures,
        production_written=False,training_runtime_imported=False,torch_imported=False,
        formal_plan_created=False,model_state_roundtrip_executed=False),indent=2))


if __name__ == '__main__': main()
