"""Isolated route/config/publisher checks. No Torch and no live-run reads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('coordinates',HERE/'rr_mean_coordinates_candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class IntegrationContracts(unittest.TestCase):
    def runtimes(self):
        files={p:'a'*64 for p in c.CHANGED_RUNTIME_FILES}
        config_path='configs/ppo_rr_capture_first_cp225280_v1/execution_profile.yaml'
        files[config_path]='b'*64
        old=dict(source_git_commit=c.SOURCE_HEAD,files=files,
            local_contract={'observation_dimension':448,'rear_task_assist':False,
                            'source_tracking_owner_revision':'pending_source_tracking_inheritance_v1'},
            selected_configuration={name:{'path':'configs/ppo_rr_capture_first_cp225280_v1/'+name,
                    'sha256':files['configs/ppo_rr_capture_first_cp225280_v1/'+name]}
                    for name in ('local_training.json','execution_profile.yaml')},
            unchanged_physics={'gravity':-9.81},unchanged_libraries={'RSL':'fixed'})
        new=copy.deepcopy(old);new['source_git_commit']='c'*40
        new['local_contract'][c.GAIN_KEY]=list(c.RR10)
        for key in c.CHANGED_RUNTIME_FILES:new['files'][key]='d'*64
        new['files'][c.ADDED_RUNTIME_FILE]='e'*64
        new['selected_configuration']['local_training.json']['sha256']='d'*64
        for runtime in (old,new):runtime['runtime_content_sha256']=hashlib.sha256(
            json.dumps(runtime['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return old,new

    def test_exact_source_and_protected_fields(self):
        old,new=self.runtimes()
        self.assertEqual(set(c.validate_coordinate_runtime_change(old,new)),set(c.CHANGED_RUNTIME_FILES)|{c.ADDED_RUNTIME_FILE})
        for mutation in ('old_head','physics','gain','extra_file','source_gain'):
            a,b=copy.deepcopy(old),copy.deepcopy(new)
            if mutation=='old_head':a['source_git_commit']='f'*40
            elif mutation=='physics':b['unchanged_physics']['gravity']=0
            elif mutation=='gain':b['local_contract'][c.GAIN_KEY][0]=10
            elif mutation=='source_gain':a['local_contract'][c.GAIN_KEY]=list(c.IDENTITY)
            else:b['files']['src/unrelated.py']='9'*64
            with self.assertRaises(ValueError):c.validate_coordinate_runtime_change(a,b)

    def test_old_missing_field_only0ff_and_new_triple_binding(self):
        old,new=self.runtimes();config={'actor':{'class_name':'saved'}}
        runner=SimpleNamespace(alg=SimpleNamespace(actor=SimpleNamespace(**{c.GAIN_KEY:c.IDENTITY})),
                              local_configuration=c.configuration_with_gain(config,c.IDENTITY))
        self.assertEqual(c.assert_coordinate_binding(runner,old,{'runner_config':config}),c.IDENTITY)
        with self.assertRaises(ValueError):c.assert_coordinate_binding(runner,new,{'runner_config':config})
        runner.alg.actor.local_mean_coordinate_gain_full12=c.RR10
        runner.local_configuration=c.configuration_with_gain(config,c.RR10)
        runner.local_mean_coordinate_migrations=[dict(schema='wlr50_clean.rr_mean_coordinate_publication.v1',
            source_head=c.SOURCE_HEAD,destination_head=new['source_git_commit'],source_gain=list(c.IDENTITY),
            target_gain=list(c.RR10),new_PPO_updates=0,new_AUX_steps=0)]
        self.assertEqual(c.assert_coordinate_binding(runner,new),c.RR10)
        runner.local_configuration['actor'][c.GAIN_KEY]=list(c.IDENTITY)
        with self.assertRaises(ValueError):c.assert_coordinate_binding(runner,new)

    def test_publisher_exact_future_source_counts_and_no_execution(self):
        namespace=runpy.run_path(str(HERE/'publish_rr_mean_coordinates_cold.py.txt'),run_name='_stdlib_only')
        check=namespace['validate_source_metadata']
        metadata=dict(schema='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2',
            runtime_contract={'source_git_commit':c.SOURCE_HEAD},rollout_empty=True,save_load_round_trip=True,
            front_FL_assist=True,rear_task_assists=False,counts=copy.deepcopy(namespace['EXPECTED_COUNTS']))
        check(metadata,c.SOURCE_HEAD)
        for key in namespace['EXPECTED_COUNTS']:
            bad=copy.deepcopy(metadata);bad['counts'][key]-=1
            with self.assertRaises(ValueError):check(bad,c.SOURCE_HEAD)
        self.assertNotIn('torch',__import__('sys').modules)

    def test_route_config_aux_patch_in_memory_only(self):
        draft=(HERE/'route_config_aux_guard.apply_patch.txt').read_text()
        for part in draft.split('*** Update File: ')[1:]:
            path,body=part.split('\n',1);source=(ROOT/path).read_text()
            for hunk in body.split('@@\n')[1:]:
                lines=[line for line in hunk.splitlines() if not line.startswith('***')]
                before='\n'.join(line[1:] for line in lines if not line.startswith('+'))+'\n'
                after='\n'.join(line[1:] for line in lines if not line.startswith('-'))+'\n'
                self.assertEqual(source.count(before),1,path+' unique context')
                source=source.replace(before,after,1)
            if path.endswith('.py'):ast.parse(source)
            else:self.assertEqual(json.loads(source)[c.GAIN_KEY],list(c.RR10))
        ast.parse((HERE/'publish_rr_mean_coordinates_cold.py.txt').read_text())


if __name__=='__main__':unittest.main()
