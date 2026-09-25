"""Isolated candidate checks only: no Torch, model, PXR, or physical success claim."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
TREE = ROOT / "tree/src/wlr50_clean/ppo"
spec = importlib.util.spec_from_file_location("collection_return_candidate", TREE/"semantic_return_profile.py")
returns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(returns)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def selected_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("missing exact functions")
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


COUNTERS = ("global_policy_decisions","ppo_updates","optimizer_steps")
ns = selected_functions(TREE/"semantic_front_retention439.py",
    ("require","_collection_counter_delta","_collection_frozen_metadata","validate_collection439_lineage"),
    dict(COUNTERS=COUNTERS,COLLECTION_KEY="collection_horizon439",
         COLLECTION_SCHEMA="wlr50_clean.collection_horizon439.v1",
         IDENTITY="front_retention439_runtime_identity",LEDGER="front_retention439_auxiliary",
         _hash=digest,Path=Path))


def config(length=128, marker=False):
    result = {"algorithm":{"gamma":.9985,"lam":.99},
        returns.RUNNER_PROFILE_KEY:returns.RETURN_PROFILE,"num_steps_per_env":length}
    if marker:
        result[returns.COLLECTION_PROFILE_KEY] = returns.COLLECTION_512
    return result


class ReturnTests(unittest.TestCase):
    def test_old128_exact(self):
        self.assertEqual(returns.runner_return_profile(config(),semantic_version="v3"),
                         returns.profile_parameters(returns.RETURN_PROFILE,semantic_version="v3"))

    def test512_math_unchanged(self):
        old = returns.runner_return_profile(config(),semantic_version="v3")
        new = returns.runner_return_profile(config(512,True),semantic_version="v3")
        self.assertTrue(returns.same_return_estimator(old,new))
        self.assertEqual(new["rollout_length"],512)
        self.assertEqual(returns.reward_return_profile({"return_profile":returns.RETURN_PROFILE,"gamma":.9985},
                                                      semantic_version="v3")["rollout_length"],128)

    def test_bad_markers_and_lengths(self):
        for length,marker in ((512,False),(128,True),(256,True),(True,False),(512.0,True)):
            with self.subTest(length=length,marker=marker), self.assertRaises(ValueError):
                returns.runner_return_profile(config(length,marker),semantic_version="v3")
        for marker in (None,"unknown",True):
            value=config()
            value[returns.COLLECTION_PROFILE_KEY]=marker
            with self.subTest(marker=marker),self.assertRaises(ValueError):
                returns.runner_return_profile(value,semantic_version="v3")

    def test_gamma_lambda_still_checked(self):
        for key in ("gamma","lam"):
            value=config(512,True)
            value["algorithm"][key]=.5
            with self.subTest(key=key),self.assertRaises(ValueError):
                returns.runner_return_profile(value,semantic_version="v3")
        self.assertFalse(returns.same_return_estimator({"version":"x","gamma":.9985},
                                                       {"version":"x","gamma":.9985}))

    def test_legacy512_rejected(self):
        value=config(512,True)
        value.pop(returns.RUNNER_PROFILE_KEY)
        value["algorithm"]={"gamma":.995,"lam":.95}
        with self.assertRaises(ValueError):
            returns.runner_return_profile(value,semantic_version="v3")

    def test_reward_no_input_mutation(self):
        values={"return_profile":returns.RETURN_PROFILE,"gamma":.9985}
        before=copy.deepcopy(values)
        returns.reward_return_profile(values,semantic_version="v3")
        self.assertEqual(values,before)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(global_policy_decisions=230000,ppo_updates=1760,optimizer_steps=35200,
            runner_config=config(),stage_requested_decisions={"full_episode":4000},
            runtime_contract={"source_git_commit":"frozen65a"},
            checkpoint_output_routing={"output_root":str((ROOT/"fixture_branch").resolve())},
            front_retention439_runtime_identity={"old128_receipt":"untouched"},
            front_retention439_auxiliary={"accepted_auxiliary_updates_total":32,"events":[{"immutable":True}]},
            rear_policy_timing_branch={"counter_origin":dict(global_policy_decisions=200000,ppo_updates=1500,optimizer_steps=30000)},
            rear_policy_timing_branch_counts=dict(global_policy_decisions=30000,ppo_updates=260,optimizer_steps=5200),
            actor_parameter_sha256="source_actor",optimizer_learning_rate=1e-5,
            last_update={"original128_receipt":True})
        self.receipt={"schema":ns["COLLECTION_SCHEMA"],"source_checkpoint":{"frozen":"binding"},
            "reason":"test", "target_runner_config":config(512,True),
            "frozen_historical_metadata_sha256":ns["_collection_frozen_metadata"](self.source)}
        self.target=copy.deepcopy(self.source)
        self.target.update(collection_horizon439=copy.deepcopy(self.receipt),
            runner_config=config(512,True),runtime_contract={"source_git_commit":"newcollection"})
        self.ancestor_calls=[]
        def historical(source,contract,root,**kwargs):
            self.assertEqual(source,self.source)
            self.assertEqual(source["runner_config"]["num_steps_per_env"],128)
            self.ancestor_calls.append("original128_full_chain")
        ns.update(_checkpoint_reference=lambda binding: copy.deepcopy(self.source),
            validate_front_retention439_lineage=historical,
            _collection_record=lambda *a:copy.deepcopy(self.receipt),
            _state=lambda m:{"test_protected":m.get("test_protected","same")})

    def validate(self):
        return ns["validate_collection439_lineage"](self.target,self.target["runtime_contract"],
            self.source["checkpoint_output_routing"]["output_root"],
            checkpoint_output_routing=self.source["checkpoint_output_routing"])

    def advance(self,decisions=512,updates=1,steps=20):
        self.target["global_policy_decisions"]+=decisions
        self.target["ppo_updates"]+=updates
        self.target["optimizer_steps"]+=steps
        origin=self.source["rear_policy_timing_branch"]["counter_origin"]
        self.target["rear_policy_timing_branch_counts"]={k:self.target[k]-origin[k] for k in COUNTERS}
        self.target["last_update"]={"global_policy_decisions":self.target["global_policy_decisions"],
            "ppo_update":self.target["ppo_updates"],"optimizer_steps":20,
            "actor_parameter_sha256_after":self.target["actor_parameter_sha256"],
            "optimizer_learning_rate":self.target["optimizer_learning_rate"]}

    def test_publication_zero_credit_and_old_chain(self):
        before=copy.deepcopy(self.target)
        self.validate()
        self.assertEqual(self.ancestor_calls,["original128_full_chain"])
        self.assertEqual(self.target,before)

    def test_actual512_one_update20adam(self):
        self.advance()
        self.validate()

    def test128_cannot_be_called_one_new_update(self):
        self.advance(decisions=128)
        with self.assertRaises(ValueError): self.validate()

    def test512_cannot_be_called_four_updates(self):
        self.advance(updates=4,steps=80)
        with self.assertRaises(ValueError): self.validate()

    def test_no_aux_or_historical_receipt_mutation(self):
        for key in ("front_retention439_auxiliary","front_retention439_runtime_identity"):
            with self.subTest(key=key):
                old=copy.deepcopy(self.target[key])
                self.target[key]={"tampered":True}
                with self.assertRaises(ValueError): self.validate()
                self.target[key]=old

    def test_no_new_segment_recursion(self):
        self.source["collection_horizon439"]={}
        with self.assertRaises(ValueError): self.validate()

    def test_zero_update_preserves_learned_state(self):
        self.target["actor_parameter_sha256"]="changed"
        with self.assertRaises(ValueError): self.validate()

    def test_branch_counts_actual_not_frozen(self):
        self.advance()
        self.target["rear_policy_timing_branch_counts"]=copy.deepcopy(self.source["rear_policy_timing_branch_counts"])
        with self.assertRaises(ValueError): self.validate()

    def test_reject_partial_last_update_claim(self):
        self.advance()
        self.target["last_update"]["global_policy_decisions"]-=1
        with self.assertRaises(ValueError): self.validate()

    def test_bools_not_real_credit(self):
        changed=copy.deepcopy(self.source)
        changed["ppo_updates"]=True
        with self.assertRaises(ValueError): ns["_collection_counter_delta"](changed,self.source)


class WiringTests(unittest.TestCase):
    def test_only_five_runtime_paths_parse_and_expected_entrypoints_select(self):
        self.assertEqual({p.name for p in TREE.glob("*.py")},{"semantic_return_profile.py","semantic_training.py",
            "semantic_front_retention439.py","semantic_cli.py","semantic_video_cli.py"})
        for p in TREE.glob("*.py"): ast.parse(p.read_text(encoding="utf-8"))
        cli=(TREE/"semantic_cli.py").read_text()
        self.assertEqual(cli.count("**_checkpoint_collection_options(args))"),2)
        self.assertIn("**_checkpoint_collection_options(args))",(TREE/"semantic_video_cli.py").read_text())

    def test_training_dynamic_collection_and_no_global_default_change(self):
        source=(TREE/"semantic_training.py").read_text()
        self.assertIn("ROLLOUT_LENGTH = 128",source)
        self.assertIn('batch = int(runner.cfg["num_steps_per_env"]) * env.num_envs',source)
        self.assertIn('for key in ("collection_horizon439", "front_retention439_runtime_identity"',source)
        self.assertIn('if not same_return_estimator(reward_horizon, horizon)',source)
        self.assertIn('tuple(storage.actions.shape) != (512, 1, 12)',source)

    def test_source128_validator_not_globally_relaxed(self):
        text=(TREE/"semantic_front_retention439.py").read_text()
        self.assertIn('== 128 * updates',text)
        self.assertIn('== 128 * parent_updates',text)
        self.assertIn('== 512 * updates',text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
