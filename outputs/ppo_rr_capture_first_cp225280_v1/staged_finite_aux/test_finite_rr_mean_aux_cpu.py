"""DEFERRED CPU tensor unit tests, not real robot or real checkpoint evidence.

Run only after Isaac has exited, with --isaac-stopped. No files are published.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--isaac-stopped',action='store_true')
    args=parser.parse_args()
    if not args.isaac_stopped:
        parser.error('No Torch import: explicit --isaac-stopped required')
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    from wlr50_clean.ppo.semantic_training import state_hash
    spec=importlib.util.spec_from_file_location('finite_aux_candidate',HERE/'finite_rr_mean_aux_candidate.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

    class Actor(torch.nn.Module):
        """Small single-HISTORY stand-in; does not certify production physical success."""
        def __init__(self):
            super().__init__()
            self.observation_layout='role439_rr_capture_local_v2'
            self.mlp=torch.nn.Sequential(torch.nn.Linear(448,8),torch.nn.ELU(),
                torch.nn.Linear(8,24),torch.nn.Unflatten(-1,(2,12)))
            self.frozen_prior=torch.nn.Linear(439,12)
            self.frozen_prior.requires_grad_(False);self.frozen_prior.eval()
            self._last_forward_evidence=None
            self.anchor={k:v.detach().clone() for k,v in self.frozen_prior.state_dict().items()}

        def assert_frozen_state(self,optimizer):
            assert all(torch.equal(v,self.anchor[k]) for k,v in self.frozen_prior.state_dict().items())
            prior_ids={id(p) for p in self.frozen_prior.parameters()}
            assert all(not p.requires_grad and p.grad is None for p in self.frozen_prior.parameters())
            assert not any(id(p) in prior_ids for g in optimizer.param_groups for p in g['params'])

        def forward(self,obs,stochastic_output=False):
            assert not stochastic_output
            value=obs['policy'];head=self.mlp(value)
            with torch.no_grad():prior=self.frozen_prior(value[:,:439])
            mu=.1*(prior+head[:,0,:])+.9*value[:,:12]
            self._last_forward_evidence={'head_log_std':head[:,1,:].detach().clone()}
            return mu

    class TensorProtection(unittest.TestCase):
        def fixture(self):
            torch.manual_seed(51)
            actor=Actor();critic=torch.nn.Linear(448,1)
            adam=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad]+list(critic.parameters()),lr=1e-5)
            # Non-empty original PPO-style Adam moments must survive exactly.
            sum(p.square().sum() for g in adam.param_groups for p in g['params']).backward()
            adam.step();adam.zero_grad(set_to_none=True)
            alg=SimpleNamespace(actor=actor,critic=critic,optimizer=adam,learning_rate=1e-5,
                storage=SimpleNamespace(step=0),transition=SimpleNamespace(actions=None))
            alg.save=lambda:dict(actor_state_dict=actor.state_dict(),critic_state_dict=critic.state_dict(),
                                optimizer_state_dict=adam.state_dict())
            folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
            path=Path(folder.name)/'synthetic_bound_source.txt';path.write_text('synthetic tensor fixture, not robot data')
            binding=dict(path=str(path),sha256=helper.sha(path))
            runner=SimpleNamespace(alg=alg,checkpoint_load_provenance={'checkpoint_sha256':binding['sha256']})
            obs=torch.randn(41,448)*.1
            with torch.no_grad():raw=actor({'policy':obs},stochastic_output=False)
            selected=[]
            for i in range(11):
                sample=raw[i].clone();sample[6:8]+=torch.tensor([-.1,.1] if i<4 else [.1,.05])
                selected.append(dict(observation_full448=obs[i].tolist(),actual_issued_raw_full12=sample.tolist(),
                                     before_metrics={'current_top_contact':i>=4}))
            package=dict(bindings={key:dict(binding) for key in ['parent_checkpoint','parent_manifest','full_decisions']},
                parent_state_hashes={k:state_hash(v) for k,v in alg.save().items()},
                parent_counts={'auxiliary_updates':0,'local_policy_decisions':3072,'local_ppo_updates':6,'local_optimizer_steps':120},
                parent_learning_rate=1e-5,selected_rows=selected,reference_observations_full448=obs.tolist(),
                runtime_contract={'runtime_content_sha256':'b'*64,'source_git_commit':'c'*40},
                scope='SYNTHETIC tensor unit test only')
            package['package_sha256']=helper.canonical_sha(package)
            return runner,package

        def test_two_rows_change_protected_state_exact_and_grouped_metrics(self):
            runner,package=self.fixture()
            result=helper.fit_rr_mean_rows(runner,package,helper.recipe(steps=2),isaac_stopped=True)
            r=result['receipt']
            self.assertEqual(r['status'],'COMPLETE',r['reason'])
            self.assertEqual(r['AUX_optimizer_steps'],2)
            self.assertEqual(r['AUX_accepted_steps'],2)
            self.assertTrue(all(r['invariants'].values()))
            self.assertEqual(r['AIR_TOP_before_fit']['AIR_before']['rows'],4)
            self.assertEqual(r['AIR_TOP_before_fit']['TOP_before']['rows'],7)
            self.assertEqual(result['counts_after']['local_ppo_updates'],6)
            helper.validate_local_auxiliary_events([result['event']],result['counts_after'])

        def test_rejected_step_is_counted_but_actor_unchanged(self):
            runner,package=self.fixture()
            result=helper.fit_rr_mean_rows(runner,package,helper.recipe(steps=8,max_shift_sigma=1e-16),isaac_stopped=True)
            r=result['receipt']
            self.assertEqual(r['status'],'STOPPED_CONSTRAINT')
            self.assertEqual((r['AUX_optimizer_steps'],r['AUX_accepted_steps']),(1,0))
            self.assertEqual(r['state_hashes_before'],r['state_hashes_after'])
            self.assertFalse(r['actor_rows_committed'])
            self.assertEqual(result['counts_after']['auxiliary_updates'],1)
            self.assertIn('candidate_RR_rows',r['attempts'][0])

        def test_post_step_exception_keeps_credit_and_failure_evidence(self):
            runner,package=self.fixture();original=torch.func.functional_call;calls=0
            def fail_after_step(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('injected post-independent-step audit failure')
                return original(*args,**kwargs)
            with patch.object(torch.func,'functional_call',side_effect=fail_after_step):
                result=helper.fit_rr_mean_rows(runner,package,helper.recipe(steps=8),isaac_stopped=True)
            r=result['receipt']
            self.assertEqual(r['status'],'FAILED')
            self.assertEqual(r['AUX_optimizer_steps'],1)
            self.assertEqual(result['event']['actual_optimizer_steps'],1)
            self.assertEqual(r['state_hashes_before'],r['state_hashes_after'])
            self.assertIn('injected',r['attempts'][0]['error'])

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TensorProtection)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)


if __name__=='__main__':main()
