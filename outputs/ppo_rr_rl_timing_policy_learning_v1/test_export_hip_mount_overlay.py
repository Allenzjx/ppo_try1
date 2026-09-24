"""Synthetic output-helper tests; no model, encoder, or physical execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('mount_overlay_exporter',
    Path(__file__).with_name('export_policy_rear_no_assist_video.py'))
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def diagnostic(tick=8):
    return {'physics_tick': tick, 'frame_raw_physics_tick': tick,
        'same_rigid_body_hip_mount_geometry': {'value': {
            'schema': 'readonly.same_rigid_body_hip_mount_world_geometry.v1',
            'valid': True, 'source_verified': True, 'physics_tick': tick,
            'simulation_time_s': tick/120.,
            'parent_pose': {'observation_tick': tick},
            'source_resolution': {'source_verified': True, 'parent_body_name': 'base_link'},
            'left_minus_right_mean_z_m': .0123,
            'rear_minus_front_mean_z_m': -.0045, 'FR_world_z_m': .19}}}


def rows():
    result = []
    for index, phase in enumerate(('P01','P06','P07','P08','P09','P11')):
        result.append({'tick': 8*(index+1), 'phase': phase, 'rear_policy_timing': {
            'rr_carry_capture': phase == 'P09', 'rr_support_handoff': False,
            'rl_prep_transfer': phase == 'P11', 'rl_swing_capture': False}})
    return result


class MountOverlayTests(unittest.TestCase):
    def test_exact_verified_tick_units_signs(self):
        data = exporter.mount_frame_summary(diagnostic(), 8)
        self.assertTrue(data['valid'])
        text = exporter.hip_mount_panel_line({'hip_mount_geometry': data})
        self.assertIn('dL-R=12.30', text)
        self.assertIn('dRear-Front=-4.50', text)
        self.assertIn('FRz=190.00', text)
        self.assertIn('NOT CoM/bearing', text)
    def test_missing_measurement_NA_not_zero(self):
        text = exporter.hip_mount_panel_line({})
        self.assertEqual(text.count('N/A'), 3)
        self.assertNotIn('=0.00', text)
    def test_unverified_value_does_not_leak_numeric_display(self):
        item = diagnostic(); item['same_rigid_body_hip_mount_geometry']['value']['source_verified'] = False
        data = exporter.mount_frame_summary(item, 8)
        self.assertFalse(data['valid'])
        self.assertEqual(exporter.hip_mount_panel_line({'hip_mount_geometry': data}).count('N/A'), 3)
    def test_actual_field_missing_or_nonfinite_is_NA(self):
        for value in (None, float('nan'), float('inf'), True):
            item = diagnostic(); item['same_rigid_body_hip_mount_geometry']['value']['FR_world_z_m'] = value
            self.assertFalse(exporter.mount_frame_summary(item, 8)['valid'])
    def test_no_adjacent_tick_interpolation(self):
        self.assertFalse(exporter.mount_frame_summary(diagnostic(0), 8)['valid'])
    def test_raw_pose_clock_mismatch_NA(self):
        item = diagnostic(); item['frame_raw_physics_tick'] = 7
        self.assertFalse(exporter.mount_frame_summary(item, 8)['valid'])
    def test_moving_upper_link_not_accepted(self):
        item = diagnostic()
        item['same_rigid_body_hip_mount_geometry']['value']['source_resolution']['parent_body_name'] = 'front_right_upper'
        self.assertFalse(exporter.mount_frame_summary(item, 8)['valid'])
    def test_optional_height_artifact_missing_has_no_fabricated_rows(self):
        self.assertEqual(exporter.mount_evidence({'manifest': {}}), {})
    def test_declared_sealed_artifact_binding_and_duplicate_NA(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'height_diagnostics.jsonl'
            path.write_text(json.dumps(diagnostic())+'\n'+json.dumps(diagnostic())+'\n', encoding='utf-8')
            context = {'source': root, 'manifest': {'artifacts': {path.name: {
                'path': str(path), 'bytes': path.stat().st_size, 'sha256': exporter.sha256(path)}}}}
            result = exporter.mount_evidence(context)
            self.assertEqual(list(result), [8])
            self.assertFalse(result[8]['valid'])
            path.write_text('{}\n', encoding='utf-8')
            with self.assertRaises(RuntimeError): exporter.mount_evidence(context)
    def test_preparation_excerpt_includes_actual_P07_and_tail(self):
        data = rows(); plan = exporter.detail_plan(data, 223232, False, cooperative_preparation=True)
        self.assertEqual((plan['start'], plan['end']), (2, 6))
        self.assertIn('RR_to_RL_preparation_REAR_OFF_FL_ON_INCOMPLETE', plan['filename'])
        self.assertIn('REAR OFF FL ON', plan['title'])
        self.assertFalse(plan['contact_success_inferred_from_preparation'])
        self.assertEqual(data, rows())
    def test_preparation_does_not_claim_RL_reached_when_only_RR(self):
        plan = exporter.detail_plan(rows()[:-1], 223232, False, cooperative_preparation=True)
        self.assertIn('RL NOT REACHED', plan['title'])
    def test_predecessor_failure_not_faked_as_RR_segment(self):
        plan = exporter.detail_plan(rows()[:2], 223232, False, cooperative_preparation=True)
        self.assertEqual(plan['kind'], 'RR_NOT_REACHED_PREDECESSOR_FAILURE')
    def test_capture_abort_remains_not_task_terminal(self):
        plan = exporter.detail_plan(rows(), 223232, False, True, cooperative_preparation=True)
        self.assertEqual(plan['kind'], 'VIDEO_CAPTURE_ABORT_AVAILABLE_TAIL')
        self.assertIn('NOT TASK TERMINAL', plan['title'])
    def test_old_detail_default_preserved(self):
        plan = exporter.detail_plan(rows(), 223232, False)
        self.assertEqual(plan['start'], 4)
        self.assertEqual(plan['kind'], 'RR_ATTEMPT_TO_ACTUAL_TAIL')
    def test_overlay_can_hold_added_fifteenth_line(self):
        self.assertGreaterEqual(exporter.PANEL_HEIGHT, 3+15*14+14)


class ComWindowTests(unittest.TestCase):
    def observation(self):
        return {'episode_physics_tick': 120, 'center_of_mass': {'valid': True},
            'transfer_roles': {'RL': {'valid': True, 'diagonal_receiving_side': 'FR',
                'transfer_direction_context': {'reference_tick': 60, 'window_s': .5,
                    'com_toward_receiver_m': -.0001, 'fixed_direction_world': [1.,0.,0.],
                    'com_world_displacement_m': [-.0001,.002,0.]}}}}
    def test_fixed_direction_window_signed_projection_and_label(self):
        value = exporter.com_to_fr_window_summary(self.observation())
        self.assertTrue(value['valid'])
        text = exporter.com_to_fr_panel_line({'com_to_fr_local_window': value})
        self.assertIn('local 0.500s (max0.5s), fixed start-dir [60->120]', text)
        self.assertIn('-0.100mm', text)
        self.assertIn('NOT total transfer / FR bearing', text)
    def test_warmup_reports_actual_short_window_not_point_five(self):
        row = self.observation(); window = row['transfer_roles']['RL']['transfer_direction_context']
        window.update(reference_tick=112, window_s=8/120)
        value = exporter.com_to_fr_window_summary(row)
        self.assertTrue(value['valid'])
        self.assertIn('local 0.067s', exporter.com_to_fr_panel_line({'com_to_fr_local_window': value}))
    def test_missing_and_unverified_CoM_are_NA(self):
        self.assertFalse(exporter.com_to_fr_window_summary({})['valid'])
        row = self.observation(); row['center_of_mass']['valid'] = False
        self.assertFalse(exporter.com_to_fr_window_summary(row)['valid'])
        self.assertIn('N/Amm', exporter.com_to_fr_panel_line({}))
    def test_total_or_mismatched_window_rejected(self):
        for duration in (1., .25):
            row = self.observation()
            row['transfer_roles']['RL']['transfer_direction_context']['window_s'] = duration
            self.assertFalse(exporter.com_to_fr_window_summary(row)['valid'])
    def test_moving_direction_recomputation_not_substituted(self):
        row = self.observation()
        row['transfer_roles']['RL']['transfer_direction_context']['fixed_direction_world'] = [0.,1.,0.]
        self.assertFalse(exporter.com_to_fr_window_summary(row)['valid'])
    def test_mount_geometry_cannot_create_CoM_progress(self):
        row = {'hip_mount_geometry': exporter.mount_frame_summary(diagnostic(), 8)}
        self.assertFalse(exporter.com_to_fr_window_summary(row)['valid'])


if __name__ == '__main__':
    unittest.main()
