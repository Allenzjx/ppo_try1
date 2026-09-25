"""Bounded metadata/early-mode checks; no Torch/PXR, no live artifacts."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('coordinate_video_candidate',HERE/'coordinate_video_validator_candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def apply_text_candidate(path):
    """Only memory strings, never applies candidate to production/output tools."""
    result={}
    for part in path.read_text().split('*** Update File: ')[1:]:
        name,body=part.split('\n',1);source=(ROOT/name).read_text()
        for hunk in body.split('@@\n')[1:]:
            lines=[line for line in hunk.splitlines() if not line.startswith('***')]
            before='\n'.join(line[1:] for line in lines if not line.startswith('+'))+'\n'
            after='\n'.join(line[1:] for line in lines if not line.startswith('-'))+'\n'
            if source.count(before)!=1:raise AssertionError(name+' ambiguous/missing patch context')
            source=source.replace(before,after,1)
        if name.endswith('.py'):ast.parse(source)
        result[name]=source
    return result


class CoordinateVideoContracts(unittest.TestCase):
    def fixture(self,directory):
        receipt=dict(schema='candidate.rr_mean_coordinate_migration.v1',status='PASS',
            flags={'conditional_mean_close':True,'conditional_std_exact':True,'full_RNG_exact':True},
            source_gain=list(c.IDENTITY),target_gain=list(c.RR10),**dict.fromkeys(c.ZERO_CREDIT,0))
        path=Path(directory)/'coordinate_receipt.json';path.write_text(json.dumps(receipt))
        runtime=dict(source_git_commit='a'*40,runtime_content_sha256='b'*64,
            local_contract={c.GAIN_KEY:list(c.RR10),'observation_dimension':448})
        event=dict(schema='wlr50_clean.rr_mean_coordinate_publication.v1',source_head=c.SOURCE_HEAD,
            destination_head=runtime['source_git_commit'],source_gain=list(c.IDENTITY),target_gain=list(c.RR10),
            source_checkpoint_sha256='c'*64,source_manifest_sha256='d'*64,source_runtime_sha256='e'*64,
            destination_runtime_sha256=runtime['runtime_content_sha256'],
            source_counts={'local_ppo_updates':7,'auxiliary_updates':64},
            destination_counts={'local_ppo_updates':7,'auxiliary_updates':64},
            receipt={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            **dict.fromkeys(c.ZERO_CREDIT,0))
        checkpoint=dict(schema='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2',runtime_contract=runtime,
            runner_config={'actor':{c.GAIN_KEY:list(c.RR10),'observation_layout':'role439_rr_capture_local_v2'}},
            counts={'local_ppo_updates':8,'auxiliary_updates':64},**{c.LEDGER_KEY:[event]})
        source=dict(schema='wlr50_clean.frozen_prior_rr_capture_video.v2',runtime_contract=copy.deepcopy(runtime),
            control_contributions={c.GAIN_KEY:list(c.RR10),c.LEDGER_KEY:copy.deepcopy([event])})
        return source,checkpoint,path

    def test_new_gain_and_request_pass_without_learning_claim(self):
        with tempfile.TemporaryDirectory() as d:
            source,checkpoint,_=self.fixture(d)
            result=c.validate_coordinate_video_binding(source,checkpoint)
            self.assertFalse(result['legacy_identity']);self.assertEqual(result['migration_count'],1)
            self.assertEqual(c.validate_coordinate_video_request({c.GAIN_KEY:list(c.RR10)},result),list(c.RR10))

    def test_current0ff_missing_fields_identity_and_unknown_missing_rejected(self):
        runtime={'source_git_commit':c.SOURCE_HEAD,'local_contract':{}}
        source=dict(schema='wlr50_clean.frozen_prior_rr_capture_video.v2',runtime_contract=runtime)
        checkpoint=dict(schema='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2',runtime_contract=runtime)
        binding=c.validate_coordinate_video_binding(source,checkpoint)
        self.assertEqual(c.validate_coordinate_video_request({},binding),list(c.IDENTITY))
        runtime['source_git_commit']='f'*40
        with self.assertRaisesRegex(RuntimeError,'omitted'):c.validate_coordinate_video_binding(source,checkpoint)

    def test_all_layer_missing_or_mixed_gains_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for layer in ('runtime','actor','control'):
                for replacement in (None,list(c.IDENTITY),[99]*12):
                    source,checkpoint,_=self.fixture(d)
                    mapping={'runtime':checkpoint['runtime_contract']['local_contract'],
                        'actor':checkpoint['runner_config']['actor'],
                        'control':source['control_contributions']}[layer]
                    if replacement is None:mapping.pop(c.GAIN_KEY)
                    else:mapping[c.GAIN_KEY]=replacement
                    if layer=='runtime':source['runtime_contract']=copy.deepcopy(checkpoint['runtime_contract'])
                    with self.subTest(layer=layer,replacement=replacement),self.assertRaises(RuntimeError):
                        c.validate_coordinate_video_binding(source,checkpoint)
            source,checkpoint,_=self.fixture(d);binding=c.validate_coordinate_video_binding(source,checkpoint)
            for request in ({},{c.GAIN_KEY:list(c.IDENTITY)},{c.GAIN_KEY:[99]*12},{c.GAIN_KEY:None}):
                with self.assertRaises(RuntimeError):c.validate_coordinate_video_request(request,binding)

    def test_receipt_runtime_credit_and_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for corruption in ('destination','credit','missing','bytes','flags'):
                source,checkpoint,path=self.fixture(d)
                event=checkpoint[c.LEDGER_KEY][0]
                if corruption=='destination':event['destination_head']='9'*40
                if corruption=='credit':event['new_PPO_updates']=1
                if corruption=='missing':checkpoint[c.LEDGER_KEY]=[]
                if corruption in ('bytes','flags'):
                    receipt=json.loads(path.read_text());receipt['flags']['conditional_mean_close']=False
                    path.write_text(json.dumps(receipt))
                    if corruption=='flags':event['receipt']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                source['control_contributions'][c.LEDGER_KEY]=copy.deepcopy(checkpoint[c.LEDGER_KEY])
                with self.subTest(corruption=corruption),self.assertRaises(RuntimeError):
                    c.validate_coordinate_video_binding(source,checkpoint)

    def test_mode_guard_is_before_creation_import_and_launcher(self):
        sources=apply_text_candidate(HERE/'route_config_aux_guard.apply_patch.txt')
        route=sources['src/wlr50_clean/ppo/semantic_rr_capture_local.py']
        main=next(node for node in ast.parse(route).body if isinstance(node,ast.FunctionDef) and node.name=='main')
        gate=next(node for node in main.body if isinstance(node,ast.If) and 'checked_gain' in ast.unparse(node.test))
        gate_text=ast.unparse(gate)
        self.assertLess(route.index("if checked_gain(runtime['local_contract'][GAIN_KEY])"),route.index('run.mkdir(parents=True,exist_ok=False)'))
        self.assertLess(gate.lineno,next(node for node in main.body if isinstance(node,ast.Try)).lineno)
        code=compile(ast.Module(body=[gate],type_ignores=[]),'<isolated mode gate>','exec')
        for mode in ('initialize','migrate','rebind','train','eval','diagnostic'):
            for checkpoint in (None,'explicit_cp.pt'):
                state={'checked_gain':c.checked_gain,'IDENTITY':c.IDENTITY,'GAIN_KEY':c.GAIN_KEY,
                    'runtime':{'local_contract':{c.GAIN_KEY:list(c.RR10)}},
                    'args':SimpleNamespace(mode=mode,checkpoint=checkpoint)}
                rejected=mode in ('initialize','migrate','rebind') or checkpoint is None
                if rejected:
                    with self.assertRaises(ValueError):exec(code,state)
                else:exec(code,state)
        self.assertIn('independent cold coordinate publisher',gate_text)
        self.assertIn("list(settings()['local_mean_coordinate_gain_full12'])",route)

    def test_exporter_patch_uses_validated_checkpoint_binding_and_no_torch(self):
        sources=apply_text_candidate(HERE/'exporter_coordinate_binding.apply_patch.txt')
        formal=sources['outputs/ppo_rr_capture_first_cp225280_v1/export_formal_from_sealed.py']
        export=sources['outputs/ppo_rr_capture_first_cp225280_v1/export_rr_capture_first_video.py']
        self.assertIn('COORDINATES.validate_coordinate_video_binding(source, checkpoint)',formal)
        self.assertIn('context["checkpoint"]["version_family"]["mean_coordinates"]',export)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules)


if __name__=='__main__':unittest.main()
