"""Portable CPU-only compatibility tests; no model forward or simulator."""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_probe_FL_hip_minus1_knee_plus1_candidate as parent
import direction_probe_preaction_candidate as shared
import direction_probe_receiving_FL_hip_minus1_knee_plus1_candidate as candidate
from test_direction_probe_preaction_candidate import fixture as parent_fixture


def fixture():
    """Explicit synthetic receiving request, not relabelled historical evidence."""
    data = parent_fixture()
    data['request'].update(policy_version=candidate.POLICY,
        schema='wlr50_clean.actual_receiving_wheel_sigma_policy_request.v1',
        receiving_continuation_active=False, RR_placed_history=False,
        receiving_sigma_gate_full12=[0.]*12, receiving_sigma_multiplier_full12=[1.]*12,
        receiving_state_semantics='historical_RR_placed_continuation_or_recovery_not_current_bearing')
    return data


def metadata_fixture():
    """Pure metadata shape fixture; not a real saved checkpoint or runtime proof."""
    cp = candidate.OUT/'checkpoints/history/synthetic_not_saved.pt'
    names = ('execution_profile.yaml','observation_schema.json','quality_score.yaml',
             'reward_config.yaml','action_schema.json','stage_task_spec.yaml')
    contract = dict(experiment_id=candidate.EXPERIMENT, source_git_commit='synthetic-only',
        selected_configuration={name:dict(path=f'configs/ppo_{candidate.EXPERIMENT}/{name}',sha256='0'*64)
                                for name in names})
    origin = dict(global_policy_decisions=185856,ppo_updates=1417,optimizer_steps=28340)
    counts = dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)
    metadata = dict(checkpoint_path=str(cp),save_load_round_trip=True,runtime_contract=copy.deepcopy(contract),
        policy_contract=candidate.receiving_wheel_policy_contract(observation_layout=candidate.LAYOUT),
        task_conditioned_hip_wheel_branch=dict(branch_id=candidate.EXPERIMENT,counter_origin=origin),
        task_conditioned_hip_wheel_branch_counts=counts,
        **{k:origin[k]+counts[k] for k in origin})
    return metadata, cp, contract


def function_ast(module, name):
    return next(n for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name==name)


class ReceivingCandidateTests(unittest.TestCase):
    def test_cpu_only_import_and_unchanged_shared_globals(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'-1')
        self.assertNotIn('torch',sys.modules)
        self.assertFalse(any(k=='isaaclab' or k.startswith('isaaclab.') for k in sys.modules))
        self.assertEqual(parent.POLICY,shared.POLICY)
        self.assertNotEqual(candidate.POLICY,shared.POLICY)
        self.assertIs(parent.validate_checkpoint_metadata,shared.validate_checkpoint_metadata)
        self.assertIsNot(candidate.validate_checkpoint_metadata,shared.validate_checkpoint_metadata)
        self.assertEqual((candidate.CONFIG,candidate.EXPERIMENT,candidate.LAYOUT),
                         (parent.CONFIG,parent.EXPERIMENT,parent.LAYOUT))

    def test_old_candidate_and_shared_bytes_preserved(self):
        self.assertEqual(hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest(),
            '86a890b2b73029d432dede04356edf70ac571bc98522138e35dbaf4f84119a17')
        self.assertEqual(hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
            'c7c1eef885df4cccda650e81175fb73a30b6a327e5a3abbd76adaa682597690d')

    def test_control_anchor_trigger_exact_source(self):
        for name in ('FiniteFLHipKnee','trigger_hip_knee','verified_ack_anchor'):
            self.assertEqual(inspect.getsource(getattr(candidate,name)),inspect.getsource(getattr(parent,name)))

    def test_two_channel_state_machine_matches_parent_ramp_release_and_follow(self):
        for capture_at in (None,15):
            a=parent.FiniteFLHipKnee('FL_hip_minus1_knee_plus1')
            b=candidate.FiniteFLHipKnee('FL_hip_minus1_knee_plus1')
            anchor=(2.,-8.)+(0.,)*10
            for p in (a,b):p.start(anchor)
            baseline=tuple(.001*i for i in range(12));caps=(18.,24.)+(32.,)*6+(.6,)*4
            for step in range(118):
                ev=dict(history=dict(placed=dict(FL=capture_at is not None and step>=capture_at)))
                ra,ia=a.apply(baseline,caps,'P05',ev);rb,ib=b.apply(baseline,caps,'P05',ev)
                self.assertEqual((ra,ia),(rb,ib))
                self.assertEqual(rb[2:],baseline[2:])
                self.assertEqual(b.anchor,anchor)
            self.assertTrue(b.complete)

    def test_capacity_guard_unchanged(self):
        for module in (parent,candidate):
            p=module.FiniteFLHipKnee('FL_hip_minus1_knee_plus1');p.start((-18.,23.9)+(0.,)*10)
            with self.assertRaises(ValueError):p.apply((0.,)*12,(18.,24.)+(32.,)*10,'P05',dict(history=dict(placed=dict(FL=False))))

    def test_receiving_metadata_passes_without_mutation(self):
        m,p,c=metadata_fixture();before=copy.deepcopy((m,c))
        candidate.validate_checkpoint_metadata(m,p,c)
        self.assertEqual((m,c),before)

    def test_old_profile_and_tampered_new_contract_rejected(self):
        for field in ('old','gate','actor'):
            m,p,c=metadata_fixture()
            if field=='old':m['policy_contract']['version']=parent.POLICY
            elif field=='gate':m['policy_contract']['receiving_continuation_gate']['P13_unchanged']=False
            else:m['policy_contract']['actor_class']='unreviewed.Actor'
            with self.assertRaises(ValueError):candidate.validate_checkpoint_metadata(m,p,c)

    def test_runtime_path_roundtrip_configs_and_counter_rejections(self):
        for field in ('runtime','path','roundtrip','configs','branch','count'):
            m,p,c=metadata_fixture()
            if field=='runtime':m['runtime_contract']['source_git_commit']='old-runtime'
            elif field=='path':m['checkpoint_path']=str(p.with_name('wrong.pt'))
            elif field=='roundtrip':m['save_load_round_trip']=False
            elif field=='configs':
                m['runtime_contract']['selected_configuration'].pop('action_schema.json')
                c=copy.deepcopy(m['runtime_contract'])
            elif field=='branch':m['task_conditioned_hip_wheel_branch']['branch_id']='other'
            else:m['global_policy_decisions']+=1
            with self.assertRaises(ValueError):candidate.validate_checkpoint_metadata(m,p,c)

    def test_actual_receiving_receipt_retains_history_gate_and_zero_credit(self):
        args=fixture();before=copy.deepcopy(args)
        r=candidate.pre_action_receipt(**args)
        self.assertEqual(args,before)
        self.assertEqual(r['original_policy_request_same_live_state'],args['request'])
        self.assertEqual(r['actual_live_history'],args['history'])
        self.assertEqual(r['manual_override_selector_full12'],(0,)*12)
        self.assertIsNone(r['manually_issued_action_policy_log_probability'])
        self.assertEqual([r[k] for k in ('new_PPO_decisions','new_PPO_updates','new_optimizer_steps',
            'new_auxiliary_updates','positive_auxiliary_labels_emitted')],[0]*5)
        digest=r.pop('pre_action_receipt_sha256')
        self.assertEqual(digest,hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())

    def test_old_request_rejected_and_manual_twochannel_action_explicit(self):
        with self.assertRaises(ValueError):candidate.pre_action_receipt(**parent_fixture())
        args=fixture();p=candidate.FiniteFLHipKnee('FL_hip_minus1_knee_plus1');p.start((2.,-8.)+(0.,)*10)
        for _ in range(12):raw,intervention=p.apply(args['baseline'],args['caps'],'P05',dict(history=dict(placed=dict(FL=False))))
        args.update(injected=raw,intervention=intervention)
        r=candidate.pre_action_receipt(**args)
        self.assertEqual(r['manual_override_selector_full12'],(1,1)+(0,)*10)
        self.assertEqual(r['manually_selected_raw_full12'][2:],args['baseline'][2:])
        self.assertIsNone(r['manually_issued_action_policy_log_probability'])
        self.assertEqual(r['original_policy_request_same_live_state']['selected_raw_full12'],args['baseline'])

    def test_receipt_AST_only_adds_explicit_no_likelihood_credit_fields(self):
        before,after=function_ast(parent,'pre_action_receipt'),function_ast(candidate,'pre_action_receipt')
        removed={}
        for node in ast.walk(after):
            if isinstance(node,ast.Call):
                for kw in list(node.keywords):
                    if kw.arg in ('manually_issued_action_policy_log_probability','manual_action_likelihood_semantics',
                                  'new_auxiliary_updates','positive_auxiliary_labels_emitted'):
                        removed[kw.arg]=ast.literal_eval(kw.value);node.keywords.remove(kw)
        self.assertEqual(len(removed),4)
        self.assertEqual(ast.dump(before),ast.dump(after))

    def test_main_AST_only_changes_diagnostic_metadata(self):
        before,after=function_ast(parent,'main'),function_ast(candidate,'main')
        removed=set()
        for node in ast.walk(after):
            if isinstance(node,ast.Constant) and node.value=='wlr50_clean.receiving_direction_probe_FL_hip_minus1_knee_plus1_candidate.v1':
                node.value='wlr50_clean.task_direction_probe_FL_hip_minus1_knee_plus1_candidate.v1'
            if isinstance(node,ast.Call):
                for kw in list(node.keywords):
                    if kw.arg in ('manual_action_policy_likelihood','implicit_checkpoint_migration_allowed',
                                  'diagnostic_policy_version','original_coupled_candidate_sha256'):
                        removed.add(kw.arg);node.keywords.remove(kw)
        self.assertEqual(len(removed),4)
        self.assertEqual(ast.dump(before),ast.dump(after))


if __name__=='__main__':
    unittest.main()
