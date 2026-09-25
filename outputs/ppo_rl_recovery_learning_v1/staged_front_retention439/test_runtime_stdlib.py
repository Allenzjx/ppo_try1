"""Execute only AST-selected receipt helpers; no production/model imports."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def reference(value):
    assert set(value)=={"path","sha256"}
    path=Path(value["path"]).resolve(strict=True)
    if str(path)!=value["path"] or hashlib.sha256(path.read_bytes()).hexdigest()!=value["sha256"]:
        raise ValueError("changed")
    return path


def bound(path,value):
    path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")
    return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


tree=ast.parse((HERE/"semantic_front_retention439.py").read_text(encoding="utf-8"))
functions={"require","_empty_ledger","_data","_state","_invariance_evidence","_event"}
constants={"LEDGER","LEDGER_SCHEMA","FIT_SCHEMA","DATA_SCHEMA","PARAMETERS","COLUMNS","PPO_ZERO","PROBE_HASHES","PHASE_ROWS"}
selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in functions or
          isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets)]
namespace=dict(Path=Path,re=re,math=math,_hash=digest,_reference=reference,
    COUNTERS=("global_policy_decisions","ppo_updates","optimizer_steps"),
    _json_reference=lambda value:json.loads(reference(value).read_text(encoding="utf-8")))
exec(compile(ast.Module(body=selected,type_ignores=[]),"runtime_receipt_helpers_only","exec"),namespace)
spec=importlib.util.spec_from_file_location("front439_kernel_contract_only",HERE/"front_retention439.py")
kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.bindings={key:bound(self.path/(key+".json"),{"fixture":key}) for key in namespace["PROBE_HASHES"]}
        namespace["PROBE_HASHES"]={key:value["sha256"] for key,value in self.bindings.items()}
        self.data=dict(schema=namespace["DATA_SCHEMA"],observation_dimension=439,action_dimension=12,
            phase_columns=[1,4,5,8],decision_index_cutoff_inclusive=997,
            target_semantics="executed_student_conditional_raw_request",synthetic_or_intervention_rows_used=False,
            step_applied_action_validated_as="finite_physical_full12_not_raw_label",
            teacher_deployed=False,whole_failed_trajectory_used_as_success_label=False,targets_are_physical_success_labels=False,
            PPO_credit=0,new_AUX_credit=0,selected_rows_content_sha256="6"*64,
            train_row_ids=[2,347,646,844],holdout_row_ids=[10,355,660,854],**self.bindings)
        self.db=self.data_binding()
        self.source=dict(actor_parameter_sha256="1"*64,critic_parameter_sha256="2"*64,
            optimizer_state_sha256="3"*64,normalizer_state_sha256="4"*64,
            optimizer_learning_rate=1e-5,training_rng_state={"fixture":True},
            global_policy_decisions=234496,ppo_updates=1797,optimizer_steps=35940)
        self.report=dict(schema=namespace["FIT_SCHEMA"],accepted_auxiliary_updates=1,attempted_auxiliary_optimizer_steps=2,
            optimized_parameters=namespace["PARAMETERS"],optimized_scalar_count=1024,
            actor_parameter_sha256_before="1"*64,actor_parameter_sha256_after="5"*64,
            protected_state_before=namespace["_state"](self.source),protected_state_after=namespace["_state"](self.source),
            real_P10_P12_full_Gaussian_bitwise_unchanged=True,
            synthetic_P13_full_Gaussian_bitwise_unchanged=True,
            zero_selected_phase_columns_P10_P13_algebra_verified=True,nonselected_actor_state_unchanged=True,
            invariance_evidence=dict(schema=kernel.INVARIANCE_SCHEMA,
                actual_rows_by_phase={"P10":1,"P11":1,"P12":1,"P13":0},
                real_P13_validation_claimed=False,synthetic_P13_rows=3,
                synthetic_P13_origin="first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture",
                synthetic_rows_used_for_fit=False,selected_phase_columns=[1,4,5,8],
                real_observations_float32_sha256="a"*64,synthetic_P13_observations_float32_sha256="b"*64,
                same_future_trajectory_claimed=False),
            fresh_PPO_rollout_required=True,first_rejected_proposal_restored_and_stopped=True,
            stop_reason="first_rejected_proposal_restored_and_stopped",targets_were_executed_student_raw_requests=True,
            targets_were_success_or_teacher_labels=False,teacher_deployed=False,physical_success_claimed=False,
            same_future_trajectory_claimed=False,
            PPO_decisions_added=0,PPO_updates_added=0,PPO_optimizer_steps_added=0,data_receipt=self.db,
            budget=dict(max_attempts=2,learning_rate=.001,maximum_train_request_shift_full12=[1.]*12,
                maximum_validation_request_shift_full12=[1.]*12,maximum_per_state_full_gaussian_kl=.01,
                maximum_abs_log_sigma_change=.02))

    def data_binding(self):
        self.data["receipt_content_sha256"]=digest({k:v for k,v in self.data.items() if k!="receipt_content_sha256"})
        return bound(self.path/"data.json",self.data)

    def event(self):
        return namespace["_event"](self.source,{"fixture":"parent"},self.db,bound(self.path/"fit.json",self.report),1)

    def test_exact_event_and_kernel_agree(self):
        kernel.validate_fit_report(self.report,publishable=True)
        result=self.event()
        self.assertEqual(result["accepted_auxiliary_updates"],1)
        self.assertEqual(result["kind"],"executed_P02_P05_P06_P09_retention_sparse_phase_columns_not_PPO")
        self.assertEqual([result[k] for k in namespace["PPO_ZERO"]],[0,0,0])

    def test_normal_finite_stop_needs_no_rejection(self):
        for reason in ("finite_budget_exhausted","selected_student_raw_targets_already_match","zero_sparse_gradient_no_optimizer_step"):
            self.report.update(stop_reason=reason,first_rejected_proposal_restored_and_stopped=False)
            kernel.validate_fit_report(self.report,publishable=True)
            self.event()

    def test_wrong_stop_rejected(self):
        self.report["first_rejected_proposal_restored_and_stopped"]=False
        with self.assertRaises(ValueError):self.event()

    def test_data_semantics_mutations_rejected(self):
        for key,value in (("decision_index_cutoff_inclusive",998),("train_row_ids",[2,347,646,998]),
                ("holdout_row_ids",[2,355,660,854]),("train_row_ids",[2,3,347,646,844]),
                ("teacher_deployed",True),("PPO_credit",1)):
            original=copy.deepcopy(self.data);self.data[key]=value;self.db=self.data_binding()
            with self.subTest(key=key),self.assertRaises(ValueError):namespace["_data"](self.db)
            self.data=original

    def test_receipt_digest_rejected(self):
        self.data["receipt_content_sha256"]="0"*64
        with self.assertRaises(ValueError):namespace["_data"](bound(self.path/"data.json",self.data))

    def test_source_file_changes_rejected(self):
        Path(self.bindings["source_decisions"]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):self.event()

    def test_no_expanded_authority_or_credit(self):
        for key,value in (("accepted_auxiliary_updates",True),("PPO_updates_added",1),
                ("optimized_parameters",["actor.mlp.0.weight[:,0:9]"]),
                ("real_P10_P12_full_Gaussian_bitwise_unchanged",False),
                ("synthetic_P13_full_Gaussian_bitwise_unchanged",False),
                ("zero_selected_phase_columns_P10_P13_algebra_verified",False),
                ("physical_success_claimed",True),("same_future_trajectory_claimed",True)):
            original=copy.deepcopy(self.report);self.report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.event()
            self.report=original

    def test_real_and_synthetic_claims_cannot_be_mixed(self):
        for key,value in (("actual_rows_by_phase",{"P10":1,"P11":1,"P12":1,"P13":1}),
                ("real_P13_validation_claimed",True),("synthetic_rows_used_for_fit",True),
                ("selected_phase_columns",list(range(9))),("synthetic_P13_rows",0)):
            original=copy.deepcopy(self.report);self.report["invariance_evidence"][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):self.event()
                with self.assertRaises(ValueError):kernel.validate_fit_report(self.report,publishable=True)
            self.report=original
        self.report["same_input_P10_P13_full_Gaussian_bitwise_unchanged"]=True
        with self.assertRaises(ValueError):self.event()
        with self.assertRaises(ValueError):kernel.validate_fit_report(self.report,publishable=True)

    def test_no_numeric_runtime_imported(self):
        self.assertFalse(any(name=="torch" or name.startswith(("torch.","pxr","isaac")) for name in sys.modules))


if __name__=="__main__":unittest.main(verbosity=2)
