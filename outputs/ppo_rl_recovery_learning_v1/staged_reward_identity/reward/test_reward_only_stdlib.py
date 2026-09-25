"""Execute staged pure helpers/full reward method via AST; no robot/Torch import.

Actual existing schema encode is executed too. This validates reward/input
immutability and arithmetic, not trained-network equivalence or physical gain.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OVERLAY=HERE/'v2/semantic_reward.py'
SOURCE=ROOT/'src/wlr50_clean/ppo/semantic_reward.py'


def extract():
    module=ModuleType('reward_only_isolated_ast')
    sys.modules[module.__name__]=module
    ns=module.__dict__
    ns.update(math=math,dataclass=dataclass,Path=Path,Any=Any,Mapping=Mapping,Sequence=Sequence,
        CONFIG_ROOT=ROOT/'configs/ppo_semantic_v2',yaml=SimpleNamespace(safe_load=json.loads),
        TASK_CONDITIONED_QUALITY_OBJECTIVE='task_conditioned_hip_wheel_quality_v1')
    names={'SemanticObservationError','SemanticNonFiniteObservation','finite','vector','SemanticObservationSchema'}
    observation=ast.parse((ROOT/'src/wlr50_clean/ppo/semantic_observation.py').read_text())
    nodes=[n for n in observation.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias('annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),'<actual-schema-encode-AST>','exec'),ns)
    tree=ast.parse(OVERLAY.read_text())
    constants={'FAMILIES','ROLE_TRANSFER_VERSION','CARRY_BODY_ALLOWANCE_MODE','TASK_FIRST_OBJECTIVE',
        'FRONT_QUALITY_OBJECTIVE','RR_RETENTION_REWARD_ONLY_MODE'}
    names={'_rr_retention_reward_binding','_rr_retention_reward_delta','_reward_only_potential_pair',
        '_square_cost','SemanticRewardCalculator'}
    nodes=[n for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
        or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets))]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias('annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(OVERLAY),'exec'),ns)
    return ns


N=extract()
B=dict(mode='rr_recapture_retention_reward_only_v1',task_spec_sha256='test-binding',
       gap_scale_m=.025,minimum_gap_m=-.015,minimum_top_samples=2,force_floor_n=.2,
       xy_scale_m=.25,preparation_share_of_geometry=.25,global_retention_share=.85/4*.2)


def ground():
    return dict(ground_contact=True,air=False,top_contact=False,top_surface_contact=False,
        obstacle_pair_active=False,within_top_xy=False,within_lateral_span=True,contact_surface='NONE',
        clearance_m=-.05,top_xy_outside_distance_m=.05,current_lift_valid=False,active_attempt=False,
        motion_continuation_allowed=True,support=True,bearing_verified=True,bearing_force_n=5.,
        consecutive_top_samples=0)


def air():
    return dict(ground(),ground_contact=False,air=True,within_top_xy=True,clearance_m=.02,
        top_xy_outside_distance_m=0.,current_lift_valid=True,active_attempt=True,support=False,bearing_force_n=0.)


def top():
    return dict(air(),air=False,top_contact=True,top_surface_contact=True,obstacle_pair_active=True,
        contact_surface='TOP',clearance_m=0.,support=True,bearing_force_n=5.,consecutive_top_samples=2)


def task(rr=None):
    return dict(task_progress_potential=.61625,substage='EXECUTION',stage_id='P12',physical_evaluator=dict(
        valid=True,termination_reason=None,physics_tick=7000,
        history=dict(placed=dict(RR=True,RL=False),active_lift=dict(RR=True),front_edge_crossed=dict(RR=True)),
        current_legs=dict(RR=ground() if rr is None else rr,RL=ground())))


def frame(t):
    data=json.loads((ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/observation_schema.json').read_text())
    schema=N['SemanticObservationSchema'](tuple(data['feature_groups']),tuple(data['fixed_chassis_to_body_wxyz']),
        data['maximum_task_duration_s'],data['clip'],Path('test-schema'))
    groups={r['name']:tuple(0. for _ in range(r['size'])) for r in data['feature_groups']}
    groups['stage_one_hot']=tuple(float(i==11) for i in range(13))
    groups['task_progress']=(.5,t['task_progress_potential'])
    metrics=dict(rpy=(0.,0.,0.),euler_roll_pitch_rate=(0.,0.),body_angular_acceleration=(0.,0.,0.),mass_kg=4.,
        wheels=[dict(contact=False,vertical_velocity=0.,force=0.,slip_speed=0.,chatter=0.) for _ in range(4)])
    return SimpleNamespace(task=t,groups=groups,metrics=metrics),schema


def calculator(binding):
    values=dict(objective_profile=None,attitude_scale_rad=.5,euler_rate_scale_rad_s=.5,
        angular_acceleration_scale_rad_s2=20.,transfer_attitude_weight=.2,transfer_contact_weight=.2,
        touchdown_speed_allowance_m_s=.05,touchdown_speed_scale_m_s=.5,
        slip_speed_allowance_m_s=.15,slip_speed_scale_m_s=.5,control_regularization_enabled=False,
        potential_weight=5.,success_reward=40.,failure_cost=40.,time_cost_per_s=.02,
        family_weights=dict(task_progress=1.,body_stability=.06,contact_motion_quality=0.,control_smoothness=0.,control_regularization=0.))
    obj=N['SemanticRewardCalculator'].__new__(N['SemanticRewardCalculator'])
    obj.config=SimpleNamespace(values=values,gamma=.9985)
    obj._rr_retention_binding=binding
    obj.reset()
    return obj


def samples(before,after,count=8):
    return [SimpleNamespace(previous=before,current=after,dt_s=1/120,
        nominal=(0.,)*12,previous_nominal=(0.,)*12,residual=(0.,)*12,previous_residual=(0.,)*12,
        actual_drive=(0.,)*12,previous_actual_drive=(0.,)*12,previous_previous_actual_drive=(0.,)*12,
        residual_caps=(1.,)*12) for _ in range(count)]


class RewardOnlyTests(unittest.TestCase):
    def delta(self,t): return N['_rr_retention_reward_delta'](t,B)

    def test_ground_low_continuous_no_support(self):
        d,a=self.delta(task())
        self.assertGreater(d,0);self.assertLessEqual(a['new_retention'],.125)
        self.assertEqual(a['old_retention'],0);self.assertEqual(a['contact_half'],0)
        closer=task(dict(ground(),top_xy_outside_distance_m=.01))
        self.assertGreater(self.delta(closer)[0],d)
        self.assertFalse(a['measured_TOP_bearing'])

    def test_legal_air_top_exact_old_arithmetic(self):
        for rr in (air(),dict(air(),current_lift_valid=False),top(),dict(top(),current_lift_valid=False)):
            d,a=self.delta(task(rr));self.assertEqual(d,0.);self.assertEqual(a['new_retention'],a['old_retention'])
        self.assertEqual(self.delta(task(top()))[1]['new_retention'],1.)

    def test_contact_does_not_follow_force_or_history(self):
        cases=[dict(top(),bearing_verified=False),dict(top(),ground_contact=True,current_lift_valid=False),
            dict(top(),contact_surface='FRONT_WALL',top_contact=False,top_surface_contact=False,current_lift_valid=False),
            dict(top(),bearing_force_n=.1),dict(air(),bearing_force_n=1000.)]
        for rr in cases:
            _,a=self.delta(task(rr));self.assertEqual(a['contact_half'],0.)
            self.assertFalse(a['measured_TOP_bearing'])

    def test_xy_gap_boundaries_current_qualified_continuous(self):
        for before,after in [(dict(air(),within_top_xy=False,top_xy_outside_distance_m=.005000001),
            dict(air(),within_top_xy=True,top_xy_outside_distance_m=.004999999)),
            (dict(air(),clearance_m=-.015000001),dict(air(),clearance_m=-.014999999))]:
            self.assertLess(abs(self.delta(task(before))[1]['new_retention']-
                self.delta(task(after))[1]['new_retention']),1e-7)

    def test_loss_and_no_higher_hover(self):
        captured=self.delta(task(top()))[1]['new_retention']
        hovering=self.delta(task(dict(air(),clearance_m=0.)))[1]['new_retention']
        self.assertEqual(captured-hovering,.5)
        self.assertGreater(hovering,self.delta(task(air()))[1]['new_retention'])
        self.assertGreater(hovering,self.delta(task())[1]['new_retention'])

    def test_scope_keeps_front_and_real_RL_continuation(self):
        for change in ('RR_unplaced','RL_placed','RL_swing'):
            t=task(); ev=t['physical_evaluator']
            if change=='RR_unplaced':ev['history']['placed']['RR']=False
            elif change=='RL_placed':ev['history']['placed']['RL']=True
            else:ev['current_legs']['RL']=air()
            d,a=self.delta(t);self.assertEqual(d,0.);self.assertEqual(a['scope'],'unchanged')

    def test_bad_scope_evidence_rejected(self):
        for key,value in [('top_xy_outside_distance_m',-1.),('clearance_m',float('nan'))]:
            with self.assertRaises(ValueError):self.delta(task(dict(ground(),**{key:value})))
        t=task();t['physical_evaluator']['history']['active_lift']['RR']=False
        with self.assertRaises(ValueError):self.delta(t)

    def test_actual_full_evaluator_no_input_mutation_439_encode(self):
        before,schema=frame(task());after,_=frame(task(dict(ground(),top_xy_outside_distance_m=.01)))
        original=deepcopy((before,after)); encoded_before=schema.encode(before.groups);encoded_after=schema.encode(after.groups)
        self.assertEqual(len(encoded_before),439)
        result=calculator(B).evaluate(before,after,samples(before,after),termination_reason=None,task_success=False)
        self.assertEqual((before,after),original)
        self.assertEqual(schema.encode(before.groups),encoded_before);self.assertEqual(schema.encode(after.groups),encoded_after)
        self.assertNotEqual(result['potential_after'],after.task['task_progress_potential'])
        self.assertEqual(result['rr_retention_reward_only']['shared_observation_potential_after'],after.task['task_progress_potential'])

    def test_no_double_count_or_per_physics_tick_repeat(self):
        before,_=frame(task());after,_=frame(task(dict(ground(),top_xy_outside_distance_m=.01)))
        for count in (1,8):
            old=calculator(None).evaluate(before,after,samples(before,after,count),termination_reason=None,task_success=False)
            new=calculator(B).evaluate(before,after,samples(before,after,count),termination_reason=None,task_success=False)
            expected=5*(.9985*self.delta(after.task)[0]-self.delta(before.task)[0])
            self.assertAlmostEqual(new['total']-old['total'],expected,places=13)
            self.assertAlmostEqual(new['potential_shaping'],5*(.9985*new['potential_after']-new['potential_before']),places=13)
            for family in N['FAMILIES'][1:]:self.assertEqual(new['families'][family],old['families'][family])

    def test_terminal_geometry_not_read_and_prior_delta_kept(self):
        before,_=frame(task());bad=SimpleNamespace(task={})
        p0,p1,a=N['_reward_only_potential_pair'](before,bad,'NAN_INF',B)
        self.assertEqual(p1,0.);self.assertGreater(p0,before.task['task_progress_potential'])
        self.assertIsNone(a['shared_observation_potential_after'])
        self.assertEqual(a['after']['scope'],'terminal_zero_geometry_not_read')
        # Existing numerical terminal physical-sample fallback still runs the
        # full evaluator using its last finite sample, not the malformed frame.
        result=calculator(B).evaluate(before,bad,samples(before,before,1),termination_reason='NAN_INF',task_success=False)
        self.assertEqual(result['terminal_event'],-40.);self.assertFalse(result['terminal_bootstrap_allowed'])

    def test_ordinary_phase_change_not_terminal_or_bonus(self):
        before,_=frame(task());after=deepcopy(before);after.task['stage_id']='P13'
        r=calculator(B).evaluate(before,after,samples(before,after),termination_reason=None,task_success=False)
        self.assertTrue(r['terminal_bootstrap_allowed']);self.assertEqual(r['terminal_event'],0.)
        self.assertGreater(r['potential_after'],0.)

    def test_no_static_or_same_state_cycle_bonus(self):
        values=[self.delta(task())[1]['new_retention'],self.delta(task(air()))[1]['new_retention'],1.,self.delta(task())[1]['new_retention']]
        gamma=.9985
        discounted=sum(gamma**i*(gamma*values[i+1]-values[i]) for i in range(3))
        self.assertLess(discounted,0.);self.assertAlmostEqual(discounted,(gamma**3-1)*values[0])
        self.assertLess(gamma*values[0]-values[0],0.)

    def test_old_mode_legacy_pair_does_not_read_RR(self):
        before=SimpleNamespace(task={'task_progress_potential':.3});after=SimpleNamespace(task={'task_progress_potential':.4})
        self.assertEqual(N['_reward_only_potential_pair'](before,after,None,None),(.3,.4,None))

    def test_bind_selected_path_schema_and_task_exactly_once(self):
        original_root=N['CONFIG_ROOT']
        spec=dict(schema='wlr50_clean.semantic_stage_task_spec.v2',potential_definition='global_physical_progress_v3',
            capture_retention_semantics='current_platform_region_after_placement',capture_approach_semantics='post_cross_FL_multiscale_positive_gap_plus_real_contact_v1',
            cooperative_preparation_mode='rr_capture_cooperative_preparation_v5',rear_owner_recovery_mode='issued_rear_transfer_owner_suspension_v1',
            p09_lift_semantics='functional_free_air_lift_v3',nominal={'rear_policy_timing':'rr_rl_edge_recovery_v4'},
            geometry={'top_gap_min_m':-.015,'top_gap_max_m':.025},history={'minimum_top_samples':2},support={'force_noise_floor_n':.2})
        with tempfile.TemporaryDirectory(prefix='rr_reward_stdlib_') as temporary:
            root=Path(temporary);N['CONFIG_ROOT']=root/'configs/ppo_semantic_v2'
            path=root/'configs/ppo_rr_rl_timing_policy_learning_v1/reward_config.yaml';path.parent.mkdir(parents=True)
            stage=path.with_name('stage_task_spec.yaml');stage.write_text(json.dumps(spec))
            config=SimpleNamespace(path=path,values={'schema':'wlr50_clean.semantic_reward.v2','objective_profile':'task_conditioned_hip_wheel_quality_v1',
                'cooperative_preparation':{'mode':'rr_capture_cooperative_preparation_v5'}})
            try:
                binding=N['_rr_retention_reward_binding'](config);self.assertEqual(binding['gap_scale_m'],.025)
                config.path=root/'configs/other_experiment/reward_config.yaml';self.assertIsNone(N['_rr_retention_reward_binding'](config))
                config.path=path;config.values['schema']='wrong'
                with self.assertRaises(ValueError):N['_rr_retention_reward_binding'](config)
                config.values['schema']='wlr50_clean.semantic_reward.v2';spec['geometry']['top_gap_max_m']=.03;stage.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):N['_rr_retention_reward_binding'](config)
                # Existing instance binding does not refresh from mutated files.
                self.assertEqual(N['_rr_retention_reward_delta'](task(),binding)[0],self.delta(task())[0])
                stage.unlink()
                with self.assertRaises(FileNotFoundError):N['_rr_retention_reward_binding'](config)
            finally:N['CONFIG_ROOT']=original_root

    def test_actual_completed384_matches_offline_candidate_and_preserves_tasks(self):
        expected=json.loads((HERE.parents[1]/'rr_recapture_potential_candidate_offline_v2.json').read_text())
        by_id={r['global_policy_decision']:r for r in expected['ground_rows']}
        path=Path(expected['source']['path']); matched=0
        with path.open() as stream:
            for _ in range(384):
                row=json.loads(next(stream));t=row['applied_audit']['semantic_task'];backup=deepcopy(t)
                if row['terminal']:continue
                d,a=self.delta(t)
                self.assertEqual(t,backup)
                if row['global_policy_decision'] in by_id:
                    r=by_id[row['global_policy_decision']];matched+=1
                    self.assertAlmostEqual(d,r['delta_phi'],places=13)
                    self.assertAlmostEqual(a['new_retention'],r['candidate_retention'],places=13)
        self.assertEqual(matched,75)

    def test_runtime_file_allowlist_and_no_model_import(self):
        identity=json.loads((HERE/'v2/reward_patch_identity.json').read_text())
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),identity['source_sha256'])
        self.assertEqual(hashlib.sha256(OVERLAY.read_bytes()).hexdigest(),identity['target_sha256'])
        self.assertFalse(any(name=='torch' or name=='pxr' or name.startswith('isaac') for name in sys.modules))
        original=ast.parse(SOURCE.read_text());candidate=ast.parse(OVERLAY.read_text())
        before={n.name:ast.dump(n) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        after={n.name:ast.dump(n) for n in candidate.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual([k for k in before if before[k]!=after[k]],['SemanticRewardCalculator'])

    def test_exact_staged_sequence_test_bodies_without_runtime_imports(self):
        class Approx:
            def __init__(self, value): self.value=value
            def __eq__(self, other):
                if isinstance(self.value,(tuple,list)):
                    return len(other)==len(self.value) and all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(other,self.value))
                return math.isclose(other,self.value,rel_tol=1e-10,abs_tol=1e-12)
        path=HERE/'tests/unit/test_semantic_rr_retention_reward_only.py'
        names={'leg','task','test_TOP_loss_recovery_TOP_sequence_has_one_retention_share',
            'test_outside_to_current_RL_swing_scope_exit_is_bounded_PBRS_not_event'}
        tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        for node in nodes:node.decorator_list=[]
        ns={'deepcopy':deepcopy,'SimpleNamespace':SimpleNamespace,'pytest':SimpleNamespace(approx=Approx),
            'reward':SimpleNamespace(_rr_retention_reward_delta=N['_rr_retention_reward_delta'],
                _reward_only_potential_pair=N['_reward_only_potential_pair'])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
        for middle in (ns['leg'](),ns['leg']('AIR',within_top_xy=False,top_xy_outside_distance_m=.07)):
            ns['test_TOP_loss_recovery_TOP_sequence_has_one_retention_share'](B,middle)
        ns['test_outside_to_current_RL_swing_scope_exit_is_bounded_PBRS_not_event'](B)


if __name__=='__main__':unittest.main(verbosity=2)
