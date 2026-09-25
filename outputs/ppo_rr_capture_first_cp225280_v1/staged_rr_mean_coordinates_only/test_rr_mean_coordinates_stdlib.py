"""No Torch: scalar coordinates, serialization, guards and staged patch syntax."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('coordinates',HERE/'rr_mean_coordinates_candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CoordinateContract(unittest.TestCase):
    def test_only_named_RR_rows(self):
        self.assertEqual(c.checked_gain(list(c.RR10)),c.RR10)
        for index in (0,5,8,11):
            v=list(c.RR10);v[index]=10
            with self.assertRaises(ValueError):c.checked_gain(v)
        for v in ((1,)*11,[True]*12,[float('nan')]*12):
            with self.assertRaises(ValueError):c.checked_gain(v)

    def test_configuration_is_explicit_and_independent(self):
        config={'actor':{'hidden_dims':[128,128]},'algorithm':{'learning_rate':1e-5}}
        changed=c.configuration_with_gain(config,c.RR10)
        self.assertEqual(changed['actor'][c.GAIN_KEY],list(c.RR10))
        self.assertNotIn(c.GAIN_KEY,config['actor'])
        self.assertEqual(changed['algorithm'],config['algorithm'])

    def test_scalar_function_and_Adam_coordinate_mapping(self):
        w,b,h,rho,prior,history=2.75,-.33,.28,.9,.81,-.17
        old=(1-rho)*(prior+w*h+b)+rho*history
        new=(1-rho)*(prior+10*((w/10)*h+b/10))+rho*history
        self.assertAlmostEqual(old,new,places=14)
        # Fixed stored moments and unchanged LR/epsilon intentionally alter dynamics.
        m,v,alpha,eps=.12,.07,1e-5,1e-8
        old_step=-alpha*m/(v**.5+eps)
        effective_new_step=10*(-alpha*(10*m)/((100*v)**.5+eps))
        self.assertAlmostEqual(effective_new_step/old_step,10,places=5)

    def test_no_tensor_import_without_exit_acknowledgement(self):
        before=set(sys.modules)
        with self.assertRaisesRegex(ValueError,'Isaac exit'):
            c.migrate_rr_mean_coordinates(None,None,source_runtime_gain=c.IDENTITY,target_runtime_gain=c.RR10)
        self.assertNotIn('torch',set(sys.modules)-before)

    def test_patch_applies_in_memory_and_parses(self):
        source=(ROOT/'src/wlr50_clean/ppo/semantic_rr_capture_local_actor.py').read_text()
        draft=(HERE/'actor_rr_mean_coordinates.apply_patch.txt').read_text()
        for hunk in draft.split('@@\n')[1:]:
            lines=[line for line in hunk.splitlines() if not line.startswith('***')]
            before='\n'.join(line[1:] for line in lines if not line.startswith('+'))+'\n'
            after='\n'.join(line[1:] for line in lines if not line.startswith('-'))+'\n'
            self.assertEqual(source.count(before),1,'staged patch context must match uniquely')
            source=source.replace(before,after,1)
        ast.parse(source)
        self.assertIn('local_mean[..., 6:8] * self.local_mean_coordinate_gain_full12[6]',source)


if __name__=='__main__':unittest.main()
