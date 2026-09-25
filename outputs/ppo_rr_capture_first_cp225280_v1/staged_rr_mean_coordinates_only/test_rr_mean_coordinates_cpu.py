"""Prepared CPU tensor tests only; requires explicit --isaac-stopped.

Uses synthetic actor inputs, not real robot data or physical success evidence.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--isaac-stopped',action='store_true');args=p.parse_args()
    if not args.isaac_stopped:p.error('No Torch import until explicit Isaac exit confirmation')
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('coordinates',HERE/'rr_mean_coordinates_candidate.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.observation_layout='role439_rr_capture_local_v2'
            self.local_mean_coordinate_gain_full12=c.IDENTITY
            self.mlp=torch.nn.Sequential(torch.nn.Linear(448,8),torch.nn.ELU(),torch.nn.Linear(8,24),
                                         torch.nn.Unflatten(-1,(2,12)))
            self.frozen_prior=torch.nn.Linear(439,24);self.frozen_prior.requires_grad_(False)
            self.frozen_prior.eval();self._last_forward_evidence=None
            self.anchor=copy.deepcopy(self.frozen_prior.state_dict())

        def assert_frozen_state(self,optimizer):
            assert all(torch.equal(v,self.anchor[k]) for k,v in self.frozen_prior.state_dict().items())
            prior_ids={id(p) for p in self.frozen_prior.parameters()}
            assert all(not p.requires_grad and p.grad is None for p in self.frozen_prior.parameters())
            assert not any(id(p) in prior_ids for g in optimizer.param_groups for p in g['params'])

        def forward(self,obs,stochastic_output=False):
            assert not stochastic_output
            x=obs['policy'];active=x[:,439].bool().unsqueeze(-1)
            with torch.no_grad():prior=self.frozen_prior(x[:,:439]).unflatten(-1,(2,12))
            head=self.mlp(x);mean=head[:,0,:]
            mean=torch.cat((mean[:,:6],mean[:,6:8]*self.local_mean_coordinate_gain_full12[6],mean[:,8:]),-1)
            applied=torch.where(active,mean,torch.zeros_like(mean))
            result=.1*(prior[:,0,:]+applied)+.9*x[:,:12]
            self._last_forward_evidence={'head_log_std':torch.where(active,head[:,1,:],prior[:,1,:]).detach().clone(),
                'applied_local_raw_mean_delta':applied.detach().clone()}
            return result

    class TensorCoordinates(unittest.TestCase):
        def fixture(self,amsgrad=False):
            torch.manual_seed(56);actor=Actor();critic=torch.nn.Linear(448,1)
            adam=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad]+list(critic.parameters()),
                lr=1.5e-5,amsgrad=amsgrad)
            sum(p.square().sum() for g in adam.param_groups for p in g['params']).backward()
            adam.step();adam.zero_grad(set_to_none=True)
            alg=SimpleNamespace(actor=actor,critic=critic,optimizer=adam,learning_rate=1.5e-5,
                storage=SimpleNamespace(step=0),transition=SimpleNamespace(actions=None))
            runner=SimpleNamespace(alg=alg,local_configuration={'actor':{}},
                local_auxiliary_events=[{'actual_optimizer_steps':64,'historical_evidence':'unchanged'}])
            obs=torch.randn(4,448)*.1;obs[:,439]=torch.tensor([0.,1.,1.,0.])
            return runner,obs

        def test_Adam_and_AMSGrad_function_preserving_coordinates(self):
            for amsgrad in (False,True):
                with self.subTest(amsgrad=amsgrad):
                    runner,obs=self.fixture(amsgrad);actor=runner.alg.actor
                    w=actor.mlp[2].weight;oldw=w.detach().clone()
                    moments=copy.deepcopy(runner.alg.optimizer.state[w]);ledger=copy.deepcopy(runner.local_auxiliary_events)
                    report=c.migrate_rr_mean_coordinates(runner,obs,source_runtime_gain=c.IDENTITY,
                        target_runtime_gain=c.RR10,isaac_stopped=True)
                    self.assertEqual(report['status'],'PASS',report)
                    self.assertTrue(all(report['flags'].values()))
                    self.assertTrue(torch.equal(w[6:8],oldw[6:8]/10))
                    state=runner.alg.optimizer.state[w]
                    self.assertTrue(torch.equal(state['exp_avg'][6:8],moments['exp_avg'][6:8]*10))
                    self.assertTrue(torch.equal(state['exp_avg_sq'][6:8],moments['exp_avg_sq'][6:8]*100))
                    self.assertTrue(torch.equal(state['step'],moments['step']))
                    self.assertEqual(ledger,runner.local_auxiliary_events)

        def test_bad_source_or_nonempty_rollout_rejected_before_mutation(self):
            runner,obs=self.fixture();original=runner.alg.actor.mlp[2].weight.detach().clone()
            runner.alg.storage.step=1
            with self.assertRaises(ValueError):c.migrate_rr_mean_coordinates(runner,obs,
                source_runtime_gain=c.IDENTITY,target_runtime_gain=c.RR10,isaac_stopped=True)
            self.assertTrue(torch.equal(original,runner.alg.actor.mlp[2].weight))
            runner.alg.storage.step=0
            runner.local_configuration['actor'][c.GAIN_KEY]=list(c.RR10)
            with self.assertRaises(ValueError):c.migrate_rr_mean_coordinates(runner,obs,
                source_runtime_gain=c.IDENTITY,target_runtime_gain=c.RR10,isaac_stopped=True)

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TensorCoordinates)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)


if __name__=='__main__':main()
