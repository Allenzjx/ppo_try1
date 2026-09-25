"""Identity/order rejection tests only; no tensor/model imports."""
import importlib.util
from pathlib import Path
import sys
import unittest

path=Path(__file__).with_name("draft_migrate_local447_to448.py")
spec=importlib.util.spec_from_file_location("migration_layout",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Parameter:
    def __init__(self,trainable=True):
        self.requires_grad=trainable


class Model:
    def __init__(self,pairs):
        self.pairs=pairs

    def named_parameters(self):
        return iter(self.pairs)


class OptimizerFixture:
    def __init__(self,params,serialized_ids):
        self.param_groups=[dict(params=params)]
        self.ids=serialized_ids

    def state_dict(self):
        return dict(state={},param_groups=[dict(params=self.ids)])


class LayoutTests(unittest.TestCase):
    def test_import_safe(self):
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("pxr",sys.modules)

    def test_serialized_ids_come_from_optimizer_not_assumed_numbers(self):
        a,c=Parameter(),Parameter()
        frozen=Parameter(False)
        actor=Model([("mlp.0.weight",a),("frozen_prior.weight",frozen)])
        critic=Model([("mlp.0.weight",c)])
        groups,bindings,_=m.named_optimizer_layout(actor,critic,OptimizerFixture([a,c],[91,701]))
        self.assertEqual(groups,[["actor.mlp.0.weight","critic.mlp.0.weight"]])
        self.assertEqual(bindings["actor.mlp.0.weight"]["serialized_id"],91)
        self.assertEqual(bindings["critic.mlp.0.weight"]["serialized_id"],701)

    def test_reject_missing_trainable_parameter(self):
        a,c=Parameter(),Parameter()
        with self.assertRaisesRegex(ValueError,"exactly all"):
            m.named_optimizer_layout(Model([("weight",a)]),Model([("weight",c)]),OptimizerFixture([a],[50]))

    def test_reject_frozen_parameter(self):
        a=Parameter(False)
        with self.assertRaisesRegex(ValueError,"frozen"):
            m.named_optimizer_layout(Model([("frozen_prior.weight",a)]),Model([]),OptimizerFixture([a],[3]))

    def test_reject_shared_actor_critic_parameter(self):
        a=Parameter()
        with self.assertRaisesRegex(ValueError,"shared"):
            m.named_optimizer_layout(Model([("weight",a)]),Model([("weight",a)]),OptimizerFixture([a],[9]))


if __name__=="__main__":
    unittest.main()
