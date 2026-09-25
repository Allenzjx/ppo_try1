"""Small standard-library synthetic reader tests; zero robot/PPO credit."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("training_reader", Path(__file__).with_name("analyze_training_progress.py"))
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def local(tick, *, active=True, contact=False, gap=.05, q=None, hold=0.):
    return dict(active=active, activation_tick=10 if active else None,
                activation_time_s=10/120 if active else None, local_success=hold >= .5,
                hold_elapsed_s=hold, metrics=dict(tick=tick, time_s=tick/120, gap_m=gap,
                    current_top_contact=contact, current_top_bearing=contact, ground_contact=False,
                    actual_rr_hip_knee_deg=q, final_rr_hip_deg=-100., final_rr_knee_deg=100.))


def decision(tick=24, *, before=None, after=None):
    return dict(kind="activated_on_policy", PPO_credit=1, old_logp=[-1.25],
                observation=[0.] * 439 + [1.] + [0.] * 7,
                before_local=before or local(tick-8, q=[7., -58.]),
                policy_request=dict(schema='wlr50_clean.actual_rr_capture_local_request.v1', capture_active=True, selected_raw_log_probability=-1.25,
                                    selected_raw_full12=[.2]*12),
                step_info=dict(phase_id="P09", raw_policy_action_full12=[.2]*12,
                    rr_capture_local=after or local(tick, gap=.04, q=[6., -57.]),
                    actuator_target_effect_audit_summary=dict(physics_ticks=8, all_ticks_verified=True)))


class Tests(unittest.TestCase):
    def test_v2_uses_explicit_schema_and_start_eligibility(self):
        row = decision()
        row['observation'].append(1.)
        row['policy_request']['schema'] = 'wlr50_clean.actual_rr_capture_local_request.v2'
        row['before_local']['metrics']['current_attempt_capture_eligible'] = True
        row['step_info']['rr_capture_local']['metrics']['current_attempt_capture_eligible'] = False
        self.log('decisions.jsonl',[row])
        assert reader.analyze(self.run)['counts']['active_observation_old_likelihood_verified_samples']==1
        row['policy_request']['schema']='wlr50_clean.actual_rr_capture_local_request.v1'
        self.log('decisions.jsonl',[row])
        assert reader.analyze(self.run)['counts']['active_observation_old_likelihood_verified_samples']==0

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rr_training_reader_")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)

    def log(self, name, rows):
        path = self.run / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def test_missing_is_not_zero_and_empty_is_zero(self):
        result = reader.analyze(self.run, live=True)
        self.assertIsNone(result["counts"]["actual_on_policy_decisions"])
        self.assertIsNone(result["completed_updates_observed"])
        self.assertIsNone(result["rr_physical_evidence"])
        self.log("decisions.jsonl", [])
        self.log("updates.jsonl", [])
        result = reader.analyze(self.run, live=True)
        self.assertEqual(result["counts"]["actual_on_policy_decisions"], 0)
        self.assertEqual(result["completed_updates_observed"], 0)

    def test_open_seek_snapshot_never_stat_or_appended_tail(self):
        path = self.log("decisions.jsonl", [decision()])
        status = {}
        with patch.object(Path, "stat", side_effect=AssertionError("metadata forbidden")):
            rows = reader.snapshot_rows(path, True, status)
            self.assertEqual(next(rows)["PPO_credit"], 1)
            with path.open("ab") as stream:
                stream.write(b'{"partial":')
            self.assertEqual(list(rows), [])
        status = {}
        self.assertEqual(len(list(reader.snapshot_rows(path, True, status))), 1)
        self.assertEqual(status["ignored_incomplete_tail_bytes"], len(b'{"partial":'))
        with self.assertRaisesRegex(ValueError, "unfinished"):
            list(reader.snapshot_rows(path, False, {}))
        with path.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "malformed complete"):
            list(reader.snapshot_rows(path, True, {}))

    def test_prefix_activation_diagnostic_and_aux_not_ppo(self):
        prefix = dict(kind="frozen_prior_prefix", PPO_credit=0, phase="P09", local=local(16))
        diagnostic = decision(32)
        diagnostic["policy_request"]["independent_diagnostic"] = {"PPO_credit": 0}
        aux = dict(kind="auxiliary", PPO_credit=0)
        self.log("decisions.jsonl", [prefix, decision(), diagnostic, aux])
        result = reader.analyze(self.run, live=True)
        counts = result["counts"]
        self.assertEqual(counts["actual_on_policy_decisions"], 1)
        self.assertEqual(counts["prefix_decisions_credit0"], 1)
        self.assertEqual(counts["diagnostic_rows_not_PPO"], 1)
        self.assertEqual(counts["auxiliary_rows_not_PPO"], 1)
        self.assertEqual(counts["invalid_or_intervened_PPO_credit_claims"], 1)
        self.assertEqual(result["capture_opportunities_with_logged_activation"], 1)

    def test_measured_motion_contact_and_drop_not_final_target(self):
        first = decision(after=local(24, contact=True, gap=.005, q=[6., -57.], hold=.1))
        second = decision(32, before=first["step_info"]["rr_capture_local"],
                          after=local(32, contact=False, gap=.02, q=[6., -57.]))
        missing = decision(40, before=local(32), after=local(40))
        self.log("decisions.jsonl", [first, second, missing])
        result = reader.analyze(self.run)
        physical = result["rr_physical_evidence"]
        self.assertEqual(physical["actual_joint_motion_comparable_samples"], 2)
        self.assertEqual(physical["actual_hip_negative_and_knee_positive_samples"], 1)
        self.assertEqual(physical["gap_descent_samples"], 1)
        self.assertEqual(physical["bearing_drops_between_decision_endpoints"], 1)
        self.assertEqual(physical["bearing_gains_between_decision_endpoints"], 1)
        self.assertEqual(result["episodes"][0]["max_native_observer_hold_s"], .1)
        self.assertIsNone(result["episodes"][0]["local_success_endpoint"])

    def test_reset_creates_new_opportunity_same_activation_tick(self):
        self.log("decisions.jsonl", [decision(24), decision(32),
            dict(kind="frozen_prior_prefix", PPO_credit=0, local=local(8, active=False)), decision(24)])
        result = reader.analyze(self.run)
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(result["capture_opportunities_with_logged_activation"], 2)
        self.assertEqual(result["counts"]["actual_on_policy_decisions"], 3)

    def test_active_gap_minimum_excludes_ground_prefix(self):
        self.log("decisions.jsonl", [dict(kind="frozen_prior_prefix", PPO_credit=0,
            local=local(8, active=False, gap=-.05)), decision()])
        result = reader.analyze(self.run)
        self.assertEqual(result["episodes"][0]["min_active_logged_gap_m"], .04)

    def test_actual_entry_reporting_bins_and_ranges_are_not_commanded_targets(self):
        prefix = dict(kind="frozen_prior_prefix", PPO_credit=0, local=local(16, q=[7., -58.]))
        first = decision(after=local(24, q=[2., -53.]))
        second = decision(32, after=local(32, q=[6., -57.]))
        self.log("decisions.jsonl", [prefix, first, second])
        result = reader.analyze(self.run)
        episode = result["episodes"][0]
        self.assertEqual(episode["first_active_actual_rr_reference"]["tick"], 16)
        self.assertEqual(episode["active_actual_rr_angle_ranges_deg"], {"hip": [2., 7.], "knee": [-58., -53.]})
        evidence = result["rr_physical_evidence"]
        self.assertEqual(evidence["actual_entry_relative_5deg_bin_known_samples"], 2)
        self.assertEqual(evidence["actual_entry_relative_hip_le_minus5_knee_ge_plus5deg_samples"], 1)
        self.assertIsNone(episode["local_success_endpoint"])

    def test_per_update_json_evidence_not_historical_counter_or_tensor_proof(self):
        self.log("decisions.jsonl", [decision(24), decision(32), decision(40)])
        update = dict(counts={"local_policy_decisions": 100002, "local_ppo_updates": 196},
                      actual_phase_counts={"P09": 2}, optimizer_steps=20,
                      actor_parameters_changed=True, finite_nonzero_gradient_observed=True)
        self.log("updates.jsonl", [update])
        (self.run / "rollouts").mkdir()
        receipt = dict(source="actual_official_PPO_log_prob_call_before_each_optimizer_step",
                       extra_random_draws=0, extra_model_forwards=0,
                       minibatches=[dict(rollout_flat_indices=[[0], [1]], old_log_probability=[-1.25, -1.25])])
        path = self.run / "rollouts/likelihood_0196.json"
        path.write_text(json.dumps(receipt))
        result = reader.analyze(self.run)
        self.assertEqual(result["counts"]["actual_on_policy_decisions"], 3)
        self.assertEqual(result["pending_credited_samples_after_last_update"], 1)
        self.assertEqual(result["completed_updates_observed"], 1)
        row = result["updates"][0]
        self.assertTrue(row["phase_counts_match"])
        self.assertEqual(row["selected_vs_issued_raw_known_samples"], 2)
        self.assertEqual(row["selected_vs_issued_raw_mismatches"], 0)
        self.assertTrue(row["likelihood"]["old_logp_matches_decision_receipts"])
        self.assertFalse(row["likelihood"]["tensor_rollout_inspected"])
        receipt["minibatches"][0]["old_log_probability"][1] = 0.
        path.write_text(json.dumps(receipt))
        result = reader.analyze(self.run)
        self.assertFalse(result["updates"][0]["likelihood"]["old_logp_matches_decision_receipts"])

    def test_missing_metric_and_missing_likelihood_not_success(self):
        row = decision()
        del row["step_info"]["rr_capture_local"]["metrics"]["current_top_contact"]
        del row["old_logp"]
        self.log("decisions.jsonl", [row])
        self.log("updates.jsonl", [dict(counts={"local_ppo_updates": 1}, actual_phase_counts={"P09": 1})])
        result = reader.analyze(self.run)
        self.assertEqual(result["rr_physical_evidence"]["current_top_contact_missing_samples"], 1)
        self.assertEqual(result["counts"]["active_observation_or_old_likelihood_missing_or_mismatch"], 1)
        self.assertFalse(result["updates"][0]["likelihood"]["available"])
        self.assertIsNone(result["updates"][0]["likelihood"]["old_logp_matches_decision_receipts"])


if __name__ == "__main__":
    unittest.main()
