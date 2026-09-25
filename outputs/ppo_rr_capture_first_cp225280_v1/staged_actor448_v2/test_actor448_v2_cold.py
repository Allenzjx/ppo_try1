"""CPU synthetic tests after the staged patch is integrated at a cold boundary.

This file refuses to import Torch without --isaac-stopped. No checkpoint or
simulator imports; these tests do not claim real robot success.
"""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

parser=argparse.ArgumentParser()
parser.add_argument("--isaac-stopped",action="store_true")
args,remaining=parser.parse_known_args()
if not args.isaac_stopped:
    raise SystemExit("Refused: --isaac-stopped required before Torch import")
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from wlr50_clean.ppo import semantic_rr_capture_local_actor as module
from wlr50_clean.ppo.semantic_rear_owner_actor import SemanticRearOwnerRecoveryHistoryMLPModel
from wlr50_clean.ppo.semantic_rear_owner_profile import REAR_OWNER_OBSERVATION_LAYOUT
torch.set_num_threads(1)


def obs(dimension=448,active=True,eligible=1.,batch=2):
    x=torch.zeros(batch,dimension)
    x[:,8]=1.
    x[:,20]=.02
    x[:,158:166]=1.
    x[:,195:207]=torch.linspace(-.4,.4,12)
    x[:,404]=x[:,410]=1.
    x[:,439]=float(active)
    if active:
        x[:,440:444]=torch.tensor([.01,.04,-.32,.55])
    if dimension==448:
        x[:,447]=eligible
    return TensorDict({"policy":x,"critic":x.clone()},batch_size=[batch])


def distribution():
    return dict(class_name="HeteroscedasticGaussianDistribution",std_type="log",init_std=.15)


def make(observation,legacy=False,bind=True):
    actor=module.SemanticRRCaptureLocalHistoryMLPModel(observation,{"actor":["policy"]},"actor",12,
        hidden_dims=(16,16),distribution_cfg=distribution(),
        observation_layout=module.LEGACY_OBSERVATION_LAYOUT if legacy else module.OBSERVATION_LAYOUT,
        legacy447_migration_only=legacy)
    if bind:
        prior=SemanticRearOwnerRecoveryHistoryMLPModel({"policy":observation["policy"][:,:439]},
            {"actor":["policy"]},"actor",12,distribution_cfg=distribution(),
            observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,exploration_std_temperature=.25)
        actor.load_frozen_prior_state(prior.state_dict())
    return actor


def last(actor):
    return [m for m in actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]


class Actor448Tests(unittest.TestCase):
    def setUp(self):
        self.rng=torch.get_rng_state().clone()
        torch.manual_seed(225280)

    def tearDown(self):
        torch.set_rng_state(self.rng)

    def test_defaults_are_explicit448_v2_and_prior_stays439(self):
        self.assertEqual(module.OBSERVATION_DIMENSION,448)
        a=make(obs())
        self.assertEqual(a.observation_dimension,448)
        self.assertFalse(a.legacy447_migration_only)
        self.assertEqual(a.frozen_prior.mlp[0].in_features,439)
        self.assertEqual(module.CAPTURE_FIELDS[-1],"current_attempt_capture_eligible")

    def test_447_requires_both_legacy_flag_and_layout(self):
        with self.assertRaises(ValueError):
            make(obs(447),legacy=False)
        with self.assertRaises(ValueError):
            make(obs(448),legacy=True)
        with self.assertRaisesRegex(ValueError,"explicit boolean"):
            module.SemanticRRCaptureLocalHistoryMLPModel(obs(),{"actor":["policy"]},"actor",12,
                distribution_cfg=distribution(),legacy447_migration_only="true")

    def test_legacy_det_only_no_distribution_or_rng_draw(self):
        x=obs(447)
        a=make(x,legacy=True)
        state=torch.get_rng_state().clone()
        with torch.inference_mode():
            a(x)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertIsNone(a.distribution._distribution)
        with self.assertRaisesRegex(ValueError,"rejects stochastic"):
            a(x,stochastic_output=True)
        with patch.object(a,"forward",side_effect=AssertionError("must reject before forward")):
            for stochastic in (False,True):
                with self.assertRaisesRegex(ValueError,"migration-only"):
                    module.audited_capture_local_policy_request(a,x,lambda:a(x),stochastic=stochastic)

    def test_learned447_mean_std_prior_migrate_without_reset(self):
        oldobs=obs(447)
        old=make(oldobs,legacy=True)
        with torch.no_grad():
            last(old).weight.uniform_(-.08,.08)
            last(old).bias[:12].copy_(torch.linspace(-.6,.6,12))
            last(old).bias[12:].copy_(torch.linspace(-2.,-1.,12))
        state=copy.deepcopy(old.state_dict())
        state["mlp.0.weight"]=torch.cat((state["mlp.0.weight"],torch.zeros(16,1)),dim=1)
        new=make(obs(),bind=False)
        new.load_state_dict(state,strict=True)
        for active in (False,True):
            xold=obs(447,active=active)
            with torch.inference_mode():
                expected=old(xold)
                oldstd=old._last_forward_evidence["head_log_std"].exp()
                for eligible in (0.,1.):
                    xnew=obs(active=active,eligible=eligible)
                    actual=new(xnew)
                    torch.testing.assert_close(actual,expected,rtol=3e-6,atol=3e-7)
                    torch.testing.assert_close(new._last_forward_evidence["head_log_std"].exp(),oldstd,rtol=3e-6,atol=3e-7)
        self.assertTrue(torch.equal(last(new).weight,last(old).weight))
        self.assertTrue(torch.equal(last(new).bias,last(old).bias))
        self.assertEqual(new.assert_frozen_state(),old.assert_frozen_state())

    def test_eligibility_is_observation_not_hidden_action_mask(self):
        a=make(obs())
        with torch.no_grad():
            last(a).bias[:12].fill_(.5)
        # AIR/GROUND recovery may be active but no longer capture-qualified.
        x=obs(eligible=0.)
        a(x)
        self.assertTrue(a._last_forward_evidence["applied_local_raw_mean_delta"].eq(.5).all())
        # Sensor TOP/bearing can be real even if current-attempt qualification
        # is false; actor must not manufacture qualification or fake success.
        x["policy"][:,444:446]=1.
        a(x)
        self.assertTrue(a._last_forward_evidence["active"].all())

    def test_eligibility_boolean_and_shape_checks(self):
        a=make(obs())
        with self.assertRaises(ValueError):
            a(obs(eligible=.5))
        with self.assertRaises(ValueError):
            module.validate_capture_local_latent(obs(447)["policy"])
        module.validate_capture_local_latent(obs(447)["policy"],legacy447_migration_only=True)

    def test_full12_single_HISTORY_and_real_Gaussian_audit(self):
        x=obs(batch=1)
        a=make(x)
        initial=a(x).detach()
        delta=torch.linspace(-.6,.6,12)
        with torch.no_grad():
            last(a).bias[:12].copy_(delta)
        with patch.object(module,"history_conditioned_head",wraps=module.history_conditioned_head) as history:
            value=a(x)
        self.assertEqual(history.call_count,1)
        torch.testing.assert_close(value-initial,.1*delta.unsqueeze(0),rtol=3e-6,atol=3e-7)
        raw,audit=module.audited_capture_local_policy_request(a,x,
            lambda:a(x,stochastic_output=True),stochastic=True)
        mean,std=a.output_distribution_params
        torch.testing.assert_close(a.get_output_log_prob(raw),torch.distributions.Normal(mean,std).log_prob(raw).sum(-1))
        self.assertEqual(audit["schema"],"wlr50_clean.actual_rr_capture_local_request.v2")
        self.assertEqual(audit["capture_context"]["current_attempt_capture_eligible"],1.)
        self.assertEqual(audit["history_kernel_applications"],1)
        self.assertEqual(audit["sampling_draws"],1)

    def test_prior_frozen_and_new_column_learns(self):
        x=obs()
        a=make(x)
        with torch.no_grad():
            last(a).weight.fill_(.01)
            a.mlp[0].weight[:,447].zero_()
        before=copy.deepcopy(a.frozen_prior.state_dict())
        optimizer=torch.optim.Adam(a.trainable_parameters(),lr=1e-5)
        loss=a(x).square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertGreater(a.mlp[0].weight[:,447].count_nonzero().item(),0)
        a.assert_frozen_state(optimizer)
        self.assertTrue(all(torch.equal(v,a.frozen_prior.state_dict()[k]) for k,v in before.items()))

    def test_save_reload448_does_not_reset_learned_head(self):
        x=obs()
        a=make(x)
        with torch.no_grad():
            last(a).bias[:12].fill_(.7)
        expected=a(x).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic448.pt"
            torch.save(a.state_dict(),path)
            b=make(x,bind=False)
            b.load_state_dict(torch.load(path,weights_only=True),strict=True)
        self.assertTrue(torch.equal(expected,b(x)))
        self.assertEqual(a.assert_frozen_state(),b.assert_frozen_state())


if __name__=="__main__":
    unittest.main(argv=[sys.argv[0],*remaining])
