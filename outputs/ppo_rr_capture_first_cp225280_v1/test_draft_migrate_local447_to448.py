"""Prepared tensor tests, must NOT execute while Isaac is live.

Run with env_isaaclab Python and --isaac-stopped after explicit parent approval.
Uses synthetic models/Adam history to exercise structure, not robot ability.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

p=argparse.ArgumentParser()
p.add_argument("--isaac-stopped",action="store_true")
args,remaining=p.parse_known_args()
if not args.isaac_stopped:
    raise SystemExit("Refused: explicit --isaac-stopped required before Torch import")
import torch
torch.set_num_threads(1)
path=Path(__file__).with_name("draft_migrate_local447_to448.py")
spec=importlib.util.spec_from_file_location("migration_draft",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Model(torch.nn.Module):
    def __init__(self,n,actor=False):
        super().__init__()
        self.mlp=torch.nn.Sequential(torch.nn.Linear(n,7),torch.nn.ELU(),torch.nn.Linear(7,24 if actor else 1))
        if actor:
            self.frozen_prior=torch.nn.Linear(439,12)
            self.frozen_prior.requires_grad_(False)

    def forward(self,x):
        return self.mlp(x)


def setup_models():
    old_actor,old_critic=Model(447,True),Model(447)
    params=list(p for p in old_actor.parameters() if p.requires_grad)+list(old_critic.parameters())
    old_optimizer=torch.optim.Adam(params,lr=1e-5,amsgrad=True)
    x=torch.randn(3,447)
    loss=old_actor(x).square().sum()+old_critic(x).square().sum()
    loss.backward()
    old_optimizer.step()
    old_optimizer.zero_grad(set_to_none=True)
    new_actor,new_critic=Model(448,True),Model(448)
    new_optimizer=torch.optim.Adam(list(p for p in new_actor.parameters() if p.requires_grad)+list(new_critic.parameters()),lr=.03)
    return old_actor,new_actor,old_critic,new_critic,old_optimizer,new_optimizer,x


class MigrationTests(unittest.TestCase):
    def test_exact_old_columns_outputs_prior_and_Adam(self):
        a,b,c,d,o,n,x=setup_models()
        before={k:v.clone() for k,v in a.state_dict().items()}
        old_rng=torch.get_rng_state().clone()
        receipt=m.migrate_local447_to448(a,b,c,d,o,n,isaac_stopped=True)
        self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
        self.assertEqual(receipt["preserved_group_options"][0]["lr"],1e-5)
        self.assertEqual(receipt["preserved_group_options"][0]["amsgrad"],True)
        for k,v in before.items():
            self.assertTrue(torch.equal(a.state_dict()[k],v))
        self.assertTrue(torch.equal(a.mlp[0].weight,b.mlp[0].weight[:,:447]))
        self.assertEqual(torch.count_nonzero(b.mlp[0].weight[:,447]).item(),0)
        self.assertTrue(torch.equal(a.mlp[2].weight,b.mlp[2].weight))
        # Zero column guarantees either value of the new boolean initially
        # preserves the function, while making later learning possible.
        for bit in (0.,1.):
            xx=torch.cat((x,torch.full((len(x),1),bit)),dim=1)
            torch.testing.assert_close(a(x),b(xx),rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(c(x),d(xx),rtol=1e-6,atol=1e-6)
        for old,new in ((a.mlp[0].weight,b.mlp[0].weight),(c.mlp[0].weight,d.mlp[0].weight)):
            for key in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
                self.assertTrue(torch.equal(o.state[old][key],n.state[new][key][:,:447]))
                self.assertEqual(torch.count_nonzero(n.state[new][key][:,447]).item(),0)
            self.assertTrue(torch.equal(o.state[old]["step"],n.state[new]["step"]))

    def test_reject_wrong_real_optimizer_order(self):
        a,b,c,d,o,n,x=setup_models()
        n.param_groups[0]["params"].reverse()
        with self.assertRaisesRegex(ValueError,"order changed"):
            m.migrate_local447_to448(a,b,c,d,o,n,isaac_stopped=True)

    def test_reject_frozen_prior_in_optimizer(self):
        a,b,c,d,o,n,x=setup_models()
        n.param_groups[0]["params"].append(next(b.frozen_prior.parameters()))
        with self.assertRaisesRegex(ValueError,"frozen"):
            m.migrate_local447_to448(a,b,c,d,o,n,isaac_stopped=True)

    def test_reject_existing_new_optimizer_state(self):
        a,b,c,d,o,n,x=setup_models()
        n.state[next(b.parameters())]["step"]=torch.tensor(1.)
        with self.assertRaisesRegex(ValueError,"already contains"):
            m.migrate_local447_to448(a,b,c,d,o,n,isaac_stopped=True)

    def test_new_column_can_learn_with_preserved_Adam(self):
        a,b,c,d,o,n,x=setup_models()
        m.migrate_local447_to448(a,b,c,d,o,n,isaac_stopped=True)
        xx=torch.cat((x,torch.ones(len(x),1)),dim=1)
        loss=b(xx).square().sum()+d(xx).square().sum()
        loss.backward()
        n.step()
        self.assertGreater(torch.count_nonzero(b.mlp[0].weight[:,447]).item(),0)
        self.assertEqual(float(n.state[b.mlp[0].weight]["step"]),2.)
        self.assertTrue(torch.equal(a.frozen_prior.weight,b.frozen_prior.weight))


if __name__=="__main__":
    unittest.main(argv=[sys.argv[0],*remaining])
