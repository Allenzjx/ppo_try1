"""UNAPPLIED candidate: pure metadata and mocked JSON-state tests, no tensors."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import finite_rr_mean_aux_candidate as helper

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'semantic_rr_capture_local.py'
PRODUCTION=ROOT/'src/wlr50_clean/ppo/semantic_rr_capture_local.py'
OUTPUT=HERE.parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def refresh(runtime):
    runtime['runtime_content_sha256']=digest(runtime['files'])


def runtime_pair():
    path=OUTPUT/'checkpoints/history/checkpoint_CP227840_local002560_lineage448_v2_g1e10d39c9a80_manifest.json'
    old=json.loads(path.read_text())['runtime_contract']
    new=copy.deepcopy(old);new['source_git_commit']='c'*40
    new['files']['src/wlr50_clean/ppo/semantic_rr_capture_local.py']='a'*64
    new['files']['src/wlr50_clean/ppo/semantic_rr_capture_local_aux.py']='b'*64
    refresh(new)
    return old,new


class StripLazyImports(ast.NodeTransformer):
    def visit_Import(self,node):
        return None
    def visit_ImportFrom(self,node):
        return None


def namespace():
    tree=ast.parse(SOURCE.read_text())
    names={'auxiliary_events','checkpoint_path','save','load',
        'validate_finite_aux_rebind_runtime','validate_same448_rebind_runtime','rebind_checkpoint'}
    constants={'SCHEMA','NAME','SAME448_REBIND_SOURCE_HEAD','SAME448_REBIND_REVISION',
        'SAME448_REBIND_FILES','FINITE_AUX_REBIND_SOURCE_HEAD','FINITE_AUX_ROUTE_FILE','FINITE_AUX_ADDED_FILE'}
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            selected.append(StripLazyImports().visit(copy.deepcopy(node)))
        elif isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in constants for x in node.targets):
            selected.append(node)
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    ns=dict(copy=copy,hashlib=hashlib,json=json,math=math,Path=Path,OUTPUT=OUTPUT)
    exec(compile(module,str(SOURCE),'exec'),ns)
    return ns


def json_model_fixture(ns,root):
    state=dict(actor_state_dict={'mean':[1.,2.]},critic_state_dict={'value':[3.]},
        optimizer_state_dict={'state':{'0':{'step':7}},'param_groups':[{'lr':1e-5}]})
    alg=SimpleNamespace(storage=SimpleNamespace(step=0),transition=SimpleNamespace(actions=None),
        optimizer=object(),learning_rate=1e-5)
    alg.actor=SimpleNamespace(assert_frozen_state=lambda *a:None,frozen_prior=object())
    alg.save=lambda:copy.deepcopy(state)
    def load(payload,*args):
        state.clear();state.update({k:copy.deepcopy(payload[k]) for k in
            ('actor_state_dict','critic_state_dict','optimizer_state_dict')})
    alg.load=load
    runner=SimpleNamespace(alg=alg,device='mock_cpu_not_a_tensor_device',local_configuration={'actor':'mock'},
        local_migration={'preserve':[1]},local_control_rebinds=[{'old':'receipt'}])
    def write(path,data,replace=False):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and not replace:raise FileExistsError(path)
        path.write_text(json.dumps(data))
    source_manifest=root/'prior_manifest.json';source_manifest.write_text(json.dumps({'actor_parameter_sha256':'prior'}))
    ns.update(OUTPUT=root,SOURCE_MANIFEST=source_manifest,
        validate_local_auxiliary_events=helper.validate_local_auxiliary_events,
        settings=lambda:dict(local_terminal='same task',prior_checkpoint_sha256='prior_sha'),
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        state_hash=digest,parameter_hash=lambda p:'prior',
        capture_training_rng_state=lambda seed:{'seed':seed,'state':'unchanged'},
        restore_training_rng_state=lambda *a,**k:None,write=write,
        torch=SimpleNamespace(save=lambda data,path:write(path,data),
            load=lambda path,**kw:json.loads(Path(path).read_text())))
    return runner


class FiniteAuxRouteTests(unittest.TestCase):
    def setUp(self):
        self.ns=namespace()
        self.old,self.new=runtime_pair()

    def test_declared_new_route_plus_one_helper_only(self):
        changes=self.ns['validate_same448_rebind_runtime'](self.old,self.new)
        self.assertEqual(set(changes),{self.ns['FINITE_AUX_ROUTE_FILE'],self.ns['FINITE_AUX_ADDED_FILE']})
        self.assertIsNone(changes[self.ns['FINITE_AUX_ADDED_FILE']]['before'])

    def test_legacy_5f_to_1e_still_uses_original_validator(self):
        path=OUTPUT/'checkpoints/history/checkpoint_CP227840_local002560_lineage448_v2_g5f8487b76f27_manifest.json'
        legacy=json.loads(path.read_text())['runtime_contract']
        changes=self.ns['validate_same448_rebind_runtime'](legacy,self.old)
        self.assertEqual(set(changes),self.ns['SAME448_REBIND_FILES'])

    def test_same_or_wrong_source_head_rejected(self):
        for old,new in ((self.old,self.old),({**self.old,'source_git_commit':'d'*40},self.new)):
            with self.assertRaises(ValueError):self.ns['validate_same448_rebind_runtime'](old,new)

    def test_protected_config_task_actor_physics_change_rejected(self):
        for path in ('configs/ppo_rr_capture_first_cp225280_v1/local_training.json',
            'src/wlr50_clean/ppo/semantic_rr_capture_local_actor.py',
            'src/wlr50_clean/ppo/semantic_rr_capture_local_task.py'):
            target=copy.deepcopy(self.new);target['files'][path]='f'*64;refresh(target)
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.ns['validate_same448_rebind_runtime'](self.old,target)
        target=copy.deepcopy(self.new);target['physics_hz']=60
        with self.assertRaises(ValueError):self.ns['validate_same448_rebind_runtime'](self.old,target)

    def test_missing_extra_removed_file_and_invalid_inventory_rejected(self):
        for kind in ('missing','extra','removed','digest'):
            target=copy.deepcopy(self.new)
            if kind=='missing':target['files'].pop(self.ns['FINITE_AUX_ADDED_FILE'])
            elif kind=='extra':target['files']['src/extra.py']='f'*64
            elif kind=='removed':target['files'].pop(next(iter(self.old['files'])))
            refresh(target)
            if kind=='digest':target['runtime_content_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.ns['validate_same448_rebind_runtime'](self.old,target)

    def test_local_contract_and_selected_config_remain_exact(self):
        for key in ('local_contract','selected_configuration'):
            target=copy.deepcopy(self.new);target[key]={}
            with self.assertRaises(ValueError):self.ns['validate_same448_rebind_runtime'](self.old,target)

    def test_checkpoint_name_aux_steps_and_new_head_avoid_collisions(self):
        fn=self.ns['checkpoint_path']
        normal=fn(3072,'a'*40)
        aux=fn(3072,'a'*40,8)
        self.assertNotEqual(normal,aux)
        self.assertIn('_aux000008_lineage448_v2_gaaaaaaaaaaaa',aux.name)
        self.assertNotEqual(fn(3072,'b'*40,8),aux)
        self.assertNotIn('_aux',normal.name)
        for invalid in (-1,True,1.5):
            with self.assertRaises(ValueError):fn(3072,'a'*40,invalid)

    def test_mock_save_load_preserves_whole_event_ledger_and_PPO_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=json_model_fixture(self.ns,Path(directory))
            event=dict(schema='wlr50_clean.finite_rr_mean_row_aux_event.v1',event_id='a'*64,
                actual_optimizer_steps=8,accepted_steps=7,PPO_credit=0,
                arbitrary_helper_receipt={'nested':[1,2,3]})
            runner.local_auxiliary_events=[event]
            counts=dict(local_policy_decisions=3072,local_ppo_updates=6,local_optimizer_steps=120,auxiliary_updates=8)
            prior={'sha256':'prior_sha'};before=copy.deepcopy(counts)
            pointer=self.ns['save'](runner,self.new,prior,counts,source_run='mock_only')
            metadata=json.loads(Path(pointer['manifest']).read_text())
            self.assertEqual(metadata['local_auxiliary_events'],[event])
            self.assertEqual(metadata['counts'],before)
            runner.local_auxiliary_events[0]['arbitrary_helper_receipt']['nested'].append(99)
            loaded_prior,loaded_counts=self.ns['load'](runner,pointer['checkpoint'],self.new)
            self.assertEqual(loaded_prior,prior);self.assertEqual(loaded_counts,before)
            self.assertEqual(runner.local_auxiliary_events,metadata['local_auxiliary_events'])
            self.assertEqual(runner.current_learning_iteration,6)

    def test_mock_legacy_absent_ledger_zero_count_supported(self):
        self.assertEqual(helper.validate_local_auxiliary_events(None,{'auxiliary_updates':0}),[])
        with self.assertRaises(ValueError):helper.validate_local_auxiliary_events(None,{'auxiliary_updates':1})

    def test_unaccounted_aux_steps_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner=json_model_fixture(self.ns,root)
            counts=dict(local_policy_decisions=3072,local_ppo_updates=6,auxiliary_updates=8)
            with self.assertRaisesRegex(ValueError,'disagree'):
                self.ns['save'](runner,self.new,{'sha256':'prior_sha'},counts,source_run='mock')
            self.assertFalse((root/'checkpoints').exists())

    def test_extra_ledger_field_tamper_in_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=json_model_fixture(self.ns,Path(directory))
            counts=dict(local_policy_decisions=3072,local_ppo_updates=6,auxiliary_updates=1)
            runner.local_auxiliary_events=[dict(schema='wlr50_clean.finite_rr_mean_row_aux_event.v1',
                event_id='b'*64,actual_optimizer_steps=1,accepted_steps=0,PPO_credit=0,status='rejected')]
            pointer=self.ns['save'](runner,self.new,{'sha256':'prior_sha'},counts,source_run='mock')
            path=Path(pointer['manifest']);metadata=json.loads(path.read_text())
            metadata['local_auxiliary_events'][0]['status']='forged_accepted'
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'embedded infos/AUX ledger'):
                self.ns['load'](runner,pointer['checkpoint'],self.new)

    def test_ordinary_load_runtime_never_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=json_model_fixture(self.ns,Path(directory))
            counts=dict(local_policy_decisions=3072,local_ppo_updates=6,local_optimizer_steps=120,auxiliary_updates=0)
            pointer=self.ns['save'](runner,self.old,{'sha256':'prior_sha'},counts,source_run='mock')
            with self.assertRaisesRegex(ValueError,'contract/hash mismatch'):
                self.ns['load'](runner,pointer['checkpoint'],self.new)

    def test_existing_cold_load_and_exact_state_checks_preserved(self):
        original=ast.parse(PRODUCTION.read_text());candidate=ast.parse(SOURCE.read_text())
        def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        rebind=ast.unparse(function(candidate,'rebind_checkpoint'))
        for required in ('load(runner, path, old_runtime)','parameter_ids','capture_training_rng_state',
            'metadata[\'state_hashes\']','runner.local_auxiliary_events','torch.nn.Identity'):
            self.assertIn(required,rebind)
        for forbidden in ('optim.Adam(','.step(','.backward(','initialize_prior('):self.assertNotIn(forbidden,rebind)
        self.assertEqual(ast.dump(function(original,'train')),ast.dump(function(candidate,'train')))
        self.assertEqual(ast.dump(function(original,'main')),ast.dump(function(candidate,'main')))

    def test_video_labels_include_aux_without_fake_eval_updates(self):
        tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        text=ast.unparse(fn)
        for key in ('local_auxiliary_events','local_auxiliary_optimizer_steps','AUX_updates_during_evaluation'):
            self.assertIn(key,text)
        self.assertIn("'AUX_updates_during_evaluation': 0",text)

    def test_no_actual_torch_pxr_or_sim_import(self):
        self.assertFalse(any(k in sys.modules for k in ('torch','pxr','isaaclab')))
        self.assertFalse(any(k.startswith('wlr50_clean') for k in sys.modules))


if __name__=='__main__':
    unittest.main(verbosity=2)
