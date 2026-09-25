"""Bounded stdlib-only tests. They prove transport semantics, not physical success."""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import rear_postcapture_feedback_intent_probe as probe


def evidence(tick=6000, phase='P10', valid=True):
    return dict(eligible=valid, tick=tick, phase=phase,
                blockers=[] if valid else ['RR_current_legal_TOP_bearing_absent'])


def committed(tick=6000, index=3, final=30., baseline=25.):
    return dict(episode_tick=tick, selected_index=index, final_deg=final,
                native_baseline_deg=baseline, safety_band_deg=[-58.,208.])


class FeedbackIntentTests(unittest.TestCase):
    def choose(self, p, tick=6000, *, raw=None, cap=112., filtered=5.,
               final=30., baseline=25., phase='P10', valid=True, snapshot=None):
        raw = tuple(.02*i for i in range(12)) if raw is None else raw
        return p.choose(raw, tick=tick, cap=cap, filtered_request=filtered,
            committed=committed(tick,p.index,final,baseline) if snapshot is None else snapshot,
            evidence=evidence(tick,phase,valid))

    def test_every_allowlisted_channel_both_directions_and_other11(self):
        raw = tuple(.02*i for i in range(12))
        for channel,index in probe.CHANNELS.items():
            for delta in (-4.,4.):
                p=probe.FeedbackIntent(channel=channel,offset_deg=delta,handoff_tick=6000)
                applied,changed,a=self.choose(p,raw=raw)
                self.assertEqual(changed,[index])
                self.assertAlmostEqual(112.*math.tanh(applied[index]),5.)
                self.assertEqual(a['desired_FINAL_intent_deg'],30.)
                self.assertTrue(a['first_step_retains_committed_filtered_REQUEST'])
                applied,changed,a=self.choose(p,6120,raw=raw,filtered=17.,final=36.,baseline=27.,phase='P12')
                self.assertEqual(a['fixed_entry_final_deg'],30.)
                self.assertEqual(a['fixed_goal_final_deg'],30.+delta)
                self.assertAlmostEqual(112.*math.tanh(applied[index]),30.+delta-27.)
                self.assertEqual([applied[i] for i in range(12) if i!=index],[raw[i] for i in range(12) if i!=index])

    def test_fixed_reference_not_accumulating_and_baseline_feedback(self):
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
        self.choose(p)
        for tick,base,actual in ((6120,25.,31.),(6128,27.,33.),(6232,24.,38.)):
            out,_,a=self.choose(p,tick,baseline=base,final=actual,filtered=-40.,phase='P12')
            self.assertEqual(a['desired_FINAL_intent_deg'],34.)
            self.assertEqual(a['fixed_entry_final_deg'],30.)
            self.assertAlmostEqual(112.*math.tanh(out[3]),34.-base)
            self.assertEqual(a['committed_FINAL_intent_error_deg'],34.-actual)

    def test_smooth_ramp_hold_release_and_no_history_reset(self):
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=-4.,handoff_tick=6000)
        raw=(0.,)*12
        self.choose(p,raw=raw)
        _,_,a=self.choose(p,6060,raw=raw,phase='P11')
        self.assertEqual(a['desired_FINAL_intent_deg'],28.)
        _,_,a=self.choose(p,6120,raw=raw,phase='P12')
        self.assertEqual(a['desired_FINAL_intent_deg'],26.)
        out,_,a=self.choose(p,6300,raw=raw,phase='P12')
        self.assertAlmostEqual(112.*math.tanh(out[3]),.5)
        self.assertEqual(a['desired_FINAL_intent_deg'],25.5)
        out,changed,a=self.choose(p,6360,raw=raw,phase='P12')
        self.assertEqual((out,changed),(raw,[]))
        self.assertEqual(p.state['release_reason'],'three_second_intent_window_complete')

    def test_real_support_loss_latched_even_if_recovered_before_next_decision(self):
        p=probe.FeedbackIntent(channel='RR_hip',offset_deg=-4.,handoff_tick=6000)
        self.choose(p)
        p.observe(6001,evidence(6001,'P11',False),cap=112.)
        raw=(.2,)*12
        out,changed,_=self.choose(p,6008,raw=raw,phase='P12')
        self.assertEqual((out,changed),(raw,[]))
        self.assertEqual(p.state['release_requested_tick'],6001)
        self.assertEqual(p.state['release_effective_tick'],6008)
        self.choose(p,6120,phase='P12')
        self.assertEqual(p.state['start_tick'],6000)

    def test_phase_continuity_but_cap_change_releases(self):
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
        self.choose(p)
        self.choose(p,6008,phase='P11')
        self.assertEqual(p.state['status'],'ACTIVE')
        self.choose(p,6016,phase='P12')
        self.assertEqual(p.state['status'],'ACTIVE')
        raw=(.1,)*12
        out,changed,_=self.choose(p,6024,raw=raw,cap=111.,phase='P12')
        self.assertEqual((out,changed),(raw,[]))
        self.assertEqual(p.state['release_reason'],'selected_cap_changed')

    def test_stale_baseline_rejected_at_entry_and_releases_active(self):
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
        raw=(.1,)*12
        out,changed,_=self.choose(p,raw=raw,snapshot=committed(5999))
        self.assertEqual((out,changed),(raw,[]))
        self.assertEqual(p.state['status'],'NOT_TRIGGERED')
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
        self.choose(p)
        out,changed,_=self.choose(p,6008,raw=raw,snapshot=committed(6000),phase='P11')
        self.assertEqual((out,changed),(raw,[]))
        self.assertEqual(p.state['release_reason'],'stale_or_invalid_committed_baseline')

    def test_entry_cap_or_physical_band_failure_is_no_intervention(self):
        for cap,final,base in ((8.,30.,25.),(112.,207.,205.)):
            p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
            raw=(.1,)*12
            out,changed,_=self.choose(p,raw=raw,cap=cap,final=final,baseline=base)
            self.assertEqual((out,changed),(raw,[]))
            self.assertEqual(p.state['status'],'NOT_TRIGGERED')

    def test_feedback_cap_clip_logged_not_hidden(self):
        p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
        self.choose(p)
        out,changed,a=self.choose(p,6120,baseline=-200.,phase='P12')
        self.assertTrue(a['request_cap_clipped'])
        self.assertTrue(math.isfinite(out[3]))
        self.assertLess(a['issued_REQUEST_before_native_filter_deg'],112.)
        self.assertFalse(a['exact_FINAL_intervention'])

    def test_only_first_real_p10_handoff_can_trigger(self):
        for tick,phase,valid in ((6008,'P10',True),(6000,'P11',True),(6000,'P10',False)):
            p=probe.FeedbackIntent(channel='FR_knee',offset_deg=4.,handoff_tick=6000)
            raw=(.1,)*12
            out,changed,_=self.choose(p,tick,raw=raw,phase=phase,valid=valid)
            self.assertEqual((out,changed),(raw,[]))
            self.assertEqual(p.state['status'],'NOT_TRIGGERED')

    def test_current_support_not_history_and_phase_or_safety_loss(self):
        rr=dict(top_contact=True,top_surface_contact=True,contact_surface='TOP',within_top_xy=True,
                ground_contact=False,air=False,support=True,bearing_verified=True,bearing_force_n=12.)
        task=dict(stage_id='P10',physical_evaluator=dict(valid=True,physics_tick=6000,current_legs={'RR':rr}))
        dep=dict(rr_top_contact=True,rr_current_bearing=True,support_transfer_permitted=True)
        self.assertTrue(probe.eligibility(task,phase='P10',tick=6000,dependency=dep)['eligible'])
        for kind in ('AIR','dependency','stale','terminal','phase'):
            t,d=deepcopy(task),deepcopy(dep)
            phase,tick='P10',6000
            if kind=='AIR':
                t['physical_evaluator']['current_legs']['RR'].update(air=True,bearing_force_n=0.)
                t['physical_evaluator']['history']={'placed':{'RR':True}}
            if kind=='dependency': d['support_transfer_permitted']=False
            if kind=='stale': tick=6001
            if kind=='terminal': t['termination_reason']='HARD_LIMIT'
            if kind=='phase': phase='P13'
            self.assertFalse(probe.eligibility(t,phase=phase,tick=tick,dependency=d)['eligible'])

    def test_snapshot_requires_current_real_ack(self):
        final=[0.]*12; baseline=[0.]*12
        q=dict(verified=True,actual_mapping_matches_dispatch=True,physics_tick=6179,
               native_drive_target_full12=baseline,
               tracking_reference_evidence=dict(bootstrap_physics_tick=180,dispatch_physics_tick=6179),
               policy_headroom_evidence=dict(baseline_native_plus_controller_full12=baseline,
                                            servo_safety_limits_deg=[[-58.,208.]]*8))
        ack=dict(physics_tick=6179,drive_target_full12=final,native_drive_target_full12=baseline)
        frame=SimpleNamespace(physics_tick=6000,info=dict(actuator_target_effect_audit=q,atomic_ack=ack,drive_target_full12=final))
        self.assertEqual(probe.committed_snapshot(frame,index=3)['episode_tick'],6000)
        ack['physics_tick']=6178
        with self.assertRaisesRegex(ValueError,'provenance'):
            probe.committed_snapshot(frame,index=3)
        q['physics_tick']=6178
        q['tracking_reference_evidence']['dispatch_physics_tick']=6178
        with self.assertRaisesRegex(ValueError,'provenance'):
            probe.committed_snapshot(frame,index=3)

    def test_identity_replacement_does_not_roundtrip_other_raw(self):
        raw=tuple(.1*i for i in range(12))
        for i in probe.CHANNELS.values():
            self.assertIs(probe.replace_request(raw,index=i,cap=112.,request=112.*math.tanh(raw[i])),raw)
        with self.assertRaises(ValueError):
            probe.replace_request(raw,index=0,cap=112.,request=0.)

    def test_control_never_replaces_or_holds_any_of_all12_raw(self):
        p=probe.NoOverrideControl(channel='FR_knee',handoff_tick=6000)
        with patch.object(probe,'replace_request',side_effect=AssertionError('control must not replace')):
            for tick,phase,valid,cap in ((6000,'P10',True,112.),(6008,'P11',True,112.),
                                        (6120,'P12',False,96.),(6360,'P12',True,112.)):
                raw=tuple((tick-6000)/1000.+i*.01 for i in range(12))
                out,changed,a=self.choose(p,tick,raw=raw,cap=cap,phase=phase,valid=valid,
                                         baseline=-20.,final=65.,filtered=-17.)
                self.assertIs(out,raw)
                self.assertEqual(changed,[])
                self.assertIsNone(a['desired_FINAL_intent_deg'])
                self.assertEqual(p.state['modified_decisions'],0)
        self.assertEqual(p.state['first_eligibility_loss_tick'],6120)
        self.assertEqual(p.state['entry_final_deg'],65.)
        p.finish(6360,'budget')
        self.assertEqual(p.state['status'],'CONTROL_SEALED')

    def test_control_and_offset_cli_are_mutually_exclusive(self):
        base=['--checkpoint','not-read.pt','--checkpoint-sha256','0'*64,
              '--checkpoint-manifest-sha256','0'*64,'--expected-head','0'*40,
              '--channel','FR_knee','--run-dir','not-created']
        args=probe.parser().parse_args(base+['--no-override-control'])
        self.assertTrue(args.no_override_control)
        self.assertIsNone(args.entry_final_offset_deg)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            probe.parser().parse_args(base+['--no-override-control','--entry-final-offset-deg','4'])
        with self.assertRaisesRegex(ValueError,'NOT EXECUTED'):
            probe.main(base+['--no-override-control'])

    def test_stdlib_import_ast_no_dispatch_or_history_overwrite(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pxr',sys.modules)
        tree=ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        methods={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(methods & {'step_physics','project_tick','write_data_to_sim','set_joint_position_target','backward'})
        self.assertTrue({'checkpoint_loader','_checkpoint_collection_options','CheckpointPolicyPrefixRslAdapter'} <= names)
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store):
                self.assertNotIn(n.attr,('_history','previous_projected_residual_full12','_last_atomic_ack'))
        argv=['--checkpoint','not-read.pt','--checkpoint-sha256','0'*64,
              '--checkpoint-manifest-sha256','0'*64,'--expected-head','0'*40,
              '--channel','FR_knee','--entry-final-offset-deg','4','--run-dir','not-created']
        with self.assertRaisesRegex(ValueError,'NOT EXECUTED'):
            probe.main(argv)
        self.assertTrue(all(v==0 for k,v in probe.ZERO.items() if k.startswith('new_')))

    def test_cleanup_all_resource_and_manifest_failures_preserve_primary(self):
        calls=[]
        def fail(name):
            def run():
                calls.append(name)
                raise OSError(name)
            return run
        primary=RuntimeError('original physics error')
        manifest={'lifecycle':'DIAGNOSTIC_ERROR'}
        receipt=Mock()
        def outer():
            error=None
            try:
                raise primary
            except BaseException as exc:
                error=exc
                raise
            finally:
                probe.independent_cleanup(manifest,
                    [('physical',fail('physical')),('app',fail('app')),('lock',fail('lock'))],
                    primary_error=error,manifest_writer=fail('manifest'),error_receipt_writer=receipt)
        with self.assertRaises(RuntimeError) as caught:
            outer()
        self.assertIs(caught.exception,primary)
        self.assertEqual(calls,['physical','app','lock','manifest'])
        self.assertEqual([x['step'] for x in manifest['cleanup_errors']],
                         ['physical','app','lock','final_manifest_write'])
        receipt.assert_called_once()
        self.assertEqual(len(receipt.call_args.args[0]['cleanup_errors']),4)

    def test_cleanup_without_primary_raises_first_only_after_remaining_actions(self):
        first=ValueError('first cleanup failure')
        physical=Mock(side_effect=first)
        app,lock,writer=Mock(),Mock(),Mock()
        manifest={'lifecycle':'DIAGNOSTIC_SEALED'}
        with self.assertRaises(ValueError) as caught:
            probe.independent_cleanup(manifest,[('physical',physical),('app',app),('lock',lock)],
                manifest_writer=writer,error_receipt_writer=Mock())
        self.assertIs(caught.exception,first)
        app.assert_called_once(); lock.assert_called_once(); writer.assert_called_once()
        self.assertEqual(manifest['lifecycle'],'DIAGNOSTIC_ERROR')

    def test_cleanup_manifest_and_receipt_failures_report_without_masking_primary(self):
        app,lock,reporter=Mock(),Mock(),Mock()
        primary=RuntimeError('original')
        manifest={'lifecycle':'DIAGNOSTIC_ERROR'}
        errors=probe.independent_cleanup(manifest,[('app',app),('lock',lock)],primary_error=primary,
            manifest_writer=Mock(side_effect=OSError('manifest disk error')),
            error_receipt_writer=Mock(side_effect=OSError('receipt disk error')),
            error_reporter=reporter)
        app.assert_called_once(); lock.assert_called_once(); reporter.assert_called_once()
        self.assertEqual([x['step'] for x in errors],['final_manifest_write','cleanup_error_receipt_write'])
        self.assertEqual(reporter.call_args.args[0]['primary_exception'],repr(primary))
        self.assertEqual(len(reporter.call_args.args[0]['cleanup_errors']),2)

    def test_cleanup_success_leaves_lifecycle_and_writes_once(self):
        app,lock,writer,receipt=Mock(),Mock(),Mock(),Mock()
        manifest={'lifecycle':'DIAGNOSTIC_SEALED'}
        self.assertEqual(probe.independent_cleanup(manifest,[('app',app),('lock',lock)],
            manifest_writer=writer,error_receipt_writer=receipt),[])
        app.assert_called_once(); lock.assert_called_once(); writer.assert_called_once()
        receipt.assert_not_called()
        self.assertEqual(manifest['lifecycle'],'DIAGNOSTIC_SEALED')


if __name__=='__main__':
    unittest.main()
