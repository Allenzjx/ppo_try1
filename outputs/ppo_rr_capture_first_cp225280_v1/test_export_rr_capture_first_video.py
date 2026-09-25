"""Small, CPU-only contract tests for the isolated video adapter."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "export_rr_capture_first_video.py"
SPEC = importlib.util.spec_from_file_location("_test_rr_capture_first_export", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def synthetic_decision(version_name="v2"):
    version=EXPORT.VERSIONS.VERSION_FAMILIES[version_name]
    local=dict(schema=version["task_schema"],active=True,local_success=False,
        hold_elapsed_s=0.,capture_hold_progress=0.,
        metrics=dict(tick=8,current_top_contact=False,current_top_bearing=False,ground_contact=False))
    request=dict(schema=version["request_schema"],policy_version=version["actor_policy"],
        sampling_draws=0,selected_raw_log_probability=None,
        selected_raw_full12=[0.]*12,selected_tanh_full12=[0.]*12,
        prior_same_observation_conditional_mean_full12=[0.]*12,
        applied_local_raw_mean_delta_full12=[0.]*12)
    if version_name=="v2":
        local["metrics"][EXPORT.VERSIONS.ELIGIBILITY_FIELD]=True
        local["obs9"]=[0.]*8+[1.]
        request["capture_context"]={EXPORT.VERSIONS.ELIGIBILITY_FIELD:1.}
    return dict(end_tick=8,request_phase="P09",environment_step_returned=True,policy_request=request,
        step_info=dict(physics_tick=8,phase_id="P09",end_phase_id="P09",
            actuator_target_effect_audit=dict(schema="wlr50_clean.actuator_target_effect_audit.v1",
                verified=True,source_phase_id="P09",policy_request_phase="P09",physics_tick=187),
            nominal_action_full12=[0.]*12,projected_residual_full12=[0.]*12,
            actual_drive_target_full12=[0.]*12,raw_policy_action_full12=[0.]*12,rr_capture_local=local))


def synthetic_checked_v2_version():
    """Use the real metadata validator, not a fabricated coordinate receipt."""
    version = EXPORT.VERSIONS.VERSION_FAMILIES["v2"]
    profile = {"schema": version["profile_schema"], "version": version["profile_policy"],
        "observation_dimension": 448,
        "local_features": EXPORT.VERSIONS.LOCAL_FIELDS_V1 + [EXPORT.VERSIONS.ELIGIBILITY_FIELD],
        "capture_source_dispatch": EXPORT.VERSIONS.SOURCE_DISPATCH_V2,
        "source_tracking_owner_revision": EXPORT.VERSIONS.SOURCE_TRACKING_INHERITANCE}
    runtime = {"source_git_commit": EXPORT.VERSIONS.COORDINATES.SOURCE_HEAD,
               "local_contract": profile}
    counts = {"local_policy_decisions": 3584, "local_ppo_updates": 7,
              "task_v2_policy_decisions": 1536, "task_v2_ppo_updates": 3}
    checkpoint = {"schema": version["checkpoint_schema"], "runtime_contract": runtime,
        "runner_config": {"actor": {"observation_layout": version["observation_layout"]}},
        "counts": counts}
    source = {"schema": version["source_schema"], "runtime_contract": copy.deepcopy(runtime),
        "local_task": synthetic_decision()["step_info"]["rr_capture_local"],
        "control_contributions": {"policy_version": version["profile_policy"],
            "observation_dimension": 448,
            "capture_source_dispatch": EXPORT.VERSIONS.SOURCE_DISPATCH_V2,
            "source_tracking_owner_revision": EXPORT.VERSIONS.SOURCE_TRACKING_INHERITANCE}}
    return EXPORT.VERSIONS.validate_version_bindings(source, checkpoint)


class ExportContractTests(unittest.TestCase):
    def test_formal_source_rejects_per_decision_diagnostic_override(self) -> None:
        item=synthetic_decision()
        item["decision"]=1
        item["policy_request"]["independent_diagnostic"]={"PPO_credit":0}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"decisions.jsonl"
            path.write_text(json.dumps(item)+"\n")
            version=synthetic_checked_v2_version()
            context=dict(decisions_path=path,checkpoint=dict(diagnostic=False,version_family=version),
                version_family=version)
            with self.assertRaisesRegex(RuntimeError,"contains a diagnostic"):
                EXPORT.capture_rows(context,[])

    def test_request_coordinate_gain_mismatch_rejected_before_media_join(self) -> None:
        item=synthetic_decision()
        item["decision"]=1
        item["policy_request"][EXPORT.VERSIONS.COORDINATES.GAIN_KEY]=list(EXPORT.VERSIONS.COORDINATES.RR10)
        version=synthetic_checked_v2_version()
        self.assertTrue(version["mean_coordinates"]["legacy_identity"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"decisions.jsonl"
            path.write_text(json.dumps(item)+"\n")
            context=dict(decisions_path=path,checkpoint=dict(diagnostic=False,version_family=version),
                         version_family=version)
            with self.assertRaisesRegex(RuntimeError,"request gain differs"):
                EXPORT.capture_rows(context,[])

    def test_sealed_old_v1_composite_identity_remains_accepted(self) -> None:
        source=EXPORT.ROOT/"runs"/EXPORT.EXPERIMENT/"diagnostic_hipminus25_kneeplus20_ecf205e"/"source"
        if not source.exists():
            self.skipTest("immutable old v1 source not present")
        manifest=json.loads((source/"source_manifest.json").read_text())
        proof=manifest["checkpoint_load_provenance"]
        counts=manifest["counts"]
        args=SimpleNamespace(checkpoint=proof["checkpoint"],checkpoint_sha256=proof["checkpoint_sha256"],
            checkpoint_manifest_sha256=proof["manifest_sha256"],
            expected_head=manifest["runtime_contract"]["source_git_commit"],
            expected_local_policy_decisions=counts["local_policy_decisions"],
            expected_local_ppo_updates=counts["local_ppo_updates"],
            expected_local_optimizer_steps=counts["local_optimizer_steps"])
        identity=EXPORT._checkpoint_identity(manifest,args)
        self.assertEqual(identity["version_family"]["version"],"v1")
        self.assertEqual(identity["version_family"]["observation_dimension"],447)
        self.assertIsNone(identity["source_tracking_owner_revision"])
        self.assertEqual(identity["source_tracking"]["hud_label"], "original v1")

    def test_v2_decision_requires_same_version_task_and_policy(self) -> None:
        expected=EXPORT.VERSIONS.version_family(EXPORT.VERSIONS.VERSION_FAMILIES["v2"]["source_schema"])
        EXPORT._decision_row(synthetic_decision(),expected)
        with self.assertRaisesRegex(RuntimeError,"not deterministic"):
            EXPORT._decision_row(synthetic_decision("v1"),expected)
        item=synthetic_decision()
        item["step_info"]["rr_capture_local"]["schema"]="wlr50_clean.rr_capture_local_task.v1"
        with self.assertRaisesRegex(RuntimeError,"observable RR-local"):
            EXPORT._decision_row(item,expected)
        item=synthetic_decision()
        item["policy_request"]["policy_version"]="frozen_cp225280_rr_capture_local_history_v1"
        with self.assertRaisesRegex(RuntimeError,"not deterministic"):
            EXPORT._decision_row(item,expected)

    def test_v2_request_and_endpoint_eligibility_can_change_within_decision(self) -> None:
        item=synthetic_decision()
        local=item["step_info"]["rr_capture_local"]
        local["metrics"][EXPORT.VERSIONS.ELIGIBILITY_FIELD]=False
        local["obs9"][8]=0.
        row=EXPORT._decision_row(item)
        self.assertEqual(row["request_capture_eligible"],1.)
        self.assertIs(row["endpoint_capture_eligible"],False)
        local["local_success"]=True
        with self.assertRaisesRegex(RuntimeError,"eligible current TOP"):
            EXPORT._decision_row(item)

    def test_v2_missing_or_inconsistent_observed_qualification_rejected(self) -> None:
        for location in ("request","metrics","obs9"):
            item=synthetic_decision()
            if location=="request":
                item["policy_request"]["capture_context"]={}
            elif location=="metrics":
                item["step_info"]["rr_capture_local"]["metrics"].pop(EXPORT.VERSIONS.ELIGIBILITY_FIELD)
            else:
                item["step_info"]["rr_capture_local"]["obs9"][8]=0.
            with self.subTest(location=location),self.assertRaises(RuntimeError):
                EXPORT._decision_row(item)

    def test_v2_outcome_cannot_use_old_placed_instead_of_current_qualification(self) -> None:
        local=synthetic_decision()["step_info"]["rr_capture_local"]
        local["local_success"]=True
        local["metrics"].update(current_top_contact=True,current_top_bearing=True,placed=True,crossed=True)
        manifest=dict(schema=EXPORT.VERSIONS.VERSION_FAMILIES["v2"]["source_schema"],
            local_RR_success=True,full_task_success=False,local_task=local,
            terminal_info=dict(local_task_success=True))
        self.assertTrue(EXPORT.outcome(dict(manifest=manifest))["success"])
        local["metrics"][EXPORT.VERSIONS.ELIGIBILITY_FIELD]=False
        with self.assertRaisesRegex(RuntimeError,"without current eligible"):
            EXPORT.outcome(dict(manifest=manifest))
        local["metrics"][EXPORT.VERSIONS.ELIGIBILITY_FIELD]=True
        local["metrics"]["ground_contact"]=True
        with self.assertRaisesRegex(RuntimeError,"without current eligible"):
            EXPORT.outcome(dict(manifest=manifest))

    def test_local_interval_receipt_is_accepted(self) -> None:
        ticks = 109
        EXPORT._time_receipt({
            "actual_ticks": ticks,
            "time_receipt": {
                "schema": "wlr50_clean.task_interval_video_window.v1",
                "experiment_id": EXPORT.EXPERIMENT,
                "endpoint_episode_tick": ticks,
                "frame_count": 14,
                "physical_duration_s": ticks / 120.0,
                "encoded_duration_s": 14 / 15.0,
                "extra_physics_ticks": 0,
                "extra_pre_frames": 0,
                "extra_post_frames": 0,
            },
        })

    def test_wrong_experiment_receipt_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrong or non-local"):
            EXPORT._time_receipt({
                "actual_ticks": 8,
                "time_receipt": {
                    "schema": "wlr50_clean.task_interval_video_window.v1",
                    "experiment_id": "fsm_reference_p09_stable_v2",
                    "endpoint_episode_tick": 8,
                    "frame_count": 1,
                    "physical_duration_s": 8 / 120.0,
                    "encoded_duration_s": 1 / 15.0,
                    "extra_physics_ticks": 0,
                    "extra_pre_frames": 0,
                    "extra_post_frames": 0,
                },
            })

    def test_detail_name_distinguishes_unreached_and_incomplete_capture(self) -> None:
        identity = {"display_step": 225280, "diagnostic": False,
                    "counts": {"local_policy_decisions": 0}}
        result = {"success": False, "full_task_success": False}
        unreached = EXPORT.detail_plan([
            {"local_active": False, "tick": 8},
            {"local_active": False, "tick": 16},
        ], identity, result)
        reached = EXPORT.detail_plan([
            {"local_active": False, "tick": 8},
            {"local_active": True, "tick": 16},
        ], identity, result)
        self.assertIn("not_reached_INCOMPLETE", unreached["filename"])
        self.assertIn("capture_detail_INCOMPLETE", reached["filename"])
        self.assertEqual(reached["start"], 1)
        diagnostic_identity = dict(identity, diagnostic=True)
        diagnostic = EXPORT.detail_plan([
            {"local_active": True, "tick": 8},
        ], diagnostic_identity, result)
        self.assertIn("_DIAGNOSTIC_", diagnostic["filename"])
        self.assertIn("PPO CREDIT 0", diagnostic["title"])

    def test_local_success_is_not_full_task_success(self) -> None:
        manifest = {
            "local_RR_success": True,
            "full_task_success": False,
            "terminal_info": {"semantic_task": {"termination_reason": "INCOMPLETE"}},
        }
        value = EXPORT.outcome({"manifest": manifest})
        self.assertEqual(value["result"], "RR_LOCAL_SUCCESS")
        self.assertTrue(value["success"])
        self.assertFalse(value["full_task_success"])

    def test_aux_detail_titles_disclose_training_not_realtime_assist(self) -> None:
        identity = {"display_step": 228864, "diagnostic": False,
                    "counts": {"local_policy_decisions": 3584, "auxiliary_updates": 64}}
        result = {"success": False, "full_task_success": False}
        font = EXPORT.MEDIA.ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
        for active in (False, True):
            plan = EXPORT.detail_plan([{"local_active": active, "tick": 8}], identity, result)
            self.assertIn(EXPORT.VERSIONS.AUX_TRAINING_LABEL, plan["title"])
            self.assertNotIn("pure", plan["title"].lower())
            self.assertLessEqual(font.getlength(plan["title"]), 1252)

    def test_actual_direct_source_endpoint_uses_native_tick_evidence(self) -> None:
        source = (EXPORT.ROOT / "runs" / EXPORT.EXPERIMENT /
            "diagnostic_hipminus25_kneeplus20_ecf205e" / "source")
        if not source.exists():
            self.skipTest("bounded real-source smoke is not present")
        decisions = []
        with (source / "video_policy_decisions.jsonl").open("rb") as stream:
            for _ in range(32):
                raw = next(stream)
                self.assertTrue(raw.endswith(b"\n"))
                decisions.append(json.loads(raw))
        endpoints = {value["end_tick"] for value in decisions}
        def rows_at(name: str) -> dict[int, dict]:
            result = {}
            with (source / name).open("rb") as stream:
                for raw in stream:
                    self.assertTrue(raw.endswith(b"\n"))
                    value = json.loads(raw)
                    tick = value["episode_physics_tick"]
                    if tick in endpoints:
                        result[tick] = value
                    if tick >= max(endpoints):
                        break
            self.assertEqual(set(result), endpoints)
            return result
        ticks = rows_at("capture_assist_ticks.jsonl")
        native = rows_at("native_tick_audit.jsonl")
        joined = []
        for decision in decisions:
            endpoint = decision["end_tick"]
            self.assertNotIn("raw_observation", decision["step_info"])
            self.assertNotIn("atomic_ack", decision["step_info"])
            joined.append(EXPORT._join_endpoint(EXPORT._decision_row(decision),
                ticks[endpoint], native[endpoint]))
        self.assertEqual(len(joined), 32)
        self.assertTrue(any(row["request_phase"] != row["phase"] for row in joined))
        self.assertEqual(joined[0]["evidence_alignment"],
            "episode endpoint -> same saved native dispatch/readback")
        self.assertNotEqual(joined[0]["selected_raw"], joined[0]["request_physical"])
        self.assertEqual(len(joined[0]["actual"]), 12)
        self.assertIsInstance(joined[0]["raw_physics_tick"], int)
        identity = {"counts": {"local_policy_decisions": 0, "local_ppo_updates": 0,
                                "local_optimizer_steps": 0, "prefix_decisions": 0},
                    "runtime_head": "0" * 40, "diagnostic": True}
        outcome = {"success": False, "full_task_success": False, "result": "INCOMPLETE"}
        self.assertEqual(len(EXPORT.panel_lines(joined[0], outcome, identity)), 15)
        v2identity=copy.deepcopy(identity)
        v2identity["version_family"]=EXPORT.VERSIONS.version_family(EXPORT.VERSIONS.VERSION_FAMILIES["v2"]["source_schema"])
        v2identity["counts"].update(local_policy_decisions=2048,local_ppo_updates=4,
            local_optimizer_steps=80,task_v2_policy_decisions=0,task_v2_ppo_updates=0)
        v2row=dict(joined[0],request_capture_eligible=1.,endpoint_capture_eligible=False)
        v2lines=EXPORT.panel_lines(v2row,outcome,v2identity)
        self.assertEqual(len(v2lines),15)
        self.assertIn("RR LOCAL v2",v2lines[1])
        self.assertIn("task-v2 +0 / 0 PPO",v2lines[3])
        self.assertIn("qualified request/end=1.0/False",v2lines[5])
        self.assertIn("TRACKING: legacy v2 late-derived", v2lines[1])
        revised_identity = dict(v2identity,
            source_tracking_owner_revision=EXPORT.VERSIONS.SOURCE_TRACKING_INHERITANCE)
        revised_lines = EXPORT.panel_lines(v2row, outcome, revised_identity)
        self.assertEqual(len(revised_lines), 15)
        self.assertIn("TRACKING: pending-source inherited", revised_lines[1])
        self.assertEqual(v2lines[3], revised_lines[3])
        self.assertEqual(v2identity["counts"], revised_identity["counts"])
        revised_diagnostic = dict(revised_identity, diagnostic=True)
        self.assertIn("DIAGNOSTIC PROBE / PPO CREDIT 0",
            EXPORT.panel_lines(v2row, outcome, revised_diagnostic)[0])
        aux_identity = copy.deepcopy(revised_identity)
        aux_identity["counts"]["auxiliary_updates"] = 64
        aux_lines = EXPORT.panel_lines(v2row, outcome, aux_identity)
        self.assertEqual(len(aux_lines), 15)
        self.assertIn(EXPORT.VERSIONS.AUX_TRAINING_LABEL, aux_lines[1])
        self.assertIn("AUX +64", aux_lines[3])
        self.assertIn("REAR ASSISTS: OFF", aux_lines[2])
        # Keep version/owner labels visible within the existing 1280px panel.
        font = EXPORT.MEDIA.ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
        for lines in (v2lines, revised_lines, aux_lines):
            self.assertLessEqual(font.getlength(lines[1]), 1260)
        original_panel = EXPORT.MEDIA.panel_lines
        EXPORT.MEDIA.panel_lines = EXPORT.panel_lines
        try:
            rgba = next(iter(EXPORT.MEDIA.rgba_panels([joined[0]], outcome, identity)))
            revised_rgba = next(iter(EXPORT.MEDIA.rgba_panels([v2row], outcome, revised_identity)))
        finally:
            EXPORT.MEDIA.panel_lines = original_panel
        self.assertEqual(len(rgba), 1280 * EXPORT.MEDIA.PANEL_HEIGHT * 4)
        self.assertEqual(len(revised_rgba), 1280 * EXPORT.MEDIA.PANEL_HEIGHT * 4)
        self.assertNotEqual(rgba, revised_rgba)

        # Raw bearing_verified is not the current-bearing conclusion.  AIR can
        # retain that sensor flag while the local task's conjunction is false.
        negative_tick = copy.deepcopy(ticks[decisions[0]["end_tick"]])
        negative_tick["current_legs"]["RR"]["bearing_verified"] = True
        negative = EXPORT._join_endpoint(EXPORT._decision_row(copy.deepcopy(decisions[0])),
            negative_tick, copy.deepcopy(native[decisions[0]["end_tick"]]))
        self.assertFalse(negative["rr_current_top_contact"])
        self.assertFalse(negative["rr_current_top_bearing"])
        self.assertTrue(negative["rr_sensor_bearing_flag"])

        positive_decision = copy.deepcopy(decisions[0])
        positive_metrics = positive_decision["step_info"]["rr_capture_local"]["metrics"]
        positive_metrics["current_top_contact"] = True
        positive_metrics["current_top_bearing"] = True
        positive_tick = copy.deepcopy(ticks[decisions[0]["end_tick"]])
        positive_rr = positive_tick["current_legs"]["RR"]
        positive_rr.update(top_surface_contact=True, ground_contact=False, air=False,
                           contact_surface="TOP", bearing_verified=True)
        positive = EXPORT._join_endpoint(EXPORT._decision_row(positive_decision),
            positive_tick, copy.deepcopy(native[decisions[0]["end_tick"]]))
        self.assertTrue(positive["rr_current_top_contact"])
        self.assertTrue(positive["rr_current_top_bearing"])
        self.assertTrue(positive["rr_sensor_bearing_flag"])

        # A phase may change before the eighth physics tick.  In that legal
        # handoff the final native source phase is the step audit's source
        # phase, while policy_request_phase remains the decision-start phase.
        handoff_decision = copy.deepcopy(decisions[1])
        handoff_tick = copy.deepcopy(ticks[handoff_decision["end_tick"]])
        handoff_native = copy.deepcopy(native[handoff_decision["end_tick"]])
        handoff_decision["request_phase"] = "P05"
        handoff_decision["step_info"]["phase_id"] = "P05"
        handoff_decision["step_info"]["end_phase_id"] = "P06"
        step_audit = handoff_decision["step_info"]["actuator_target_effect_audit"]
        step_audit["source_phase_id"] = "P06"
        step_audit["policy_request_phase"] = "P05"
        handoff_tick["phase"] = "P06"
        handoff_native["source_phase_id"] = "P06"
        handoff_native["native_audit"]["source_phase_id"] = "P06"
        handoff_native["native_audit"]["policy_request_phase"] = "P05"
        accepted = EXPORT._join_endpoint(EXPORT._decision_row(handoff_decision),
                                         handoff_tick, handoff_native)
        self.assertEqual((accepted["request_phase"], accepted["phase"]), ("P05", "P06"))
        wrong_source = copy.deepcopy(handoff_native)
        wrong_source["native_audit"]["source_phase_id"] = "P05"
        with self.assertRaisesRegex(RuntimeError, "clocks or phases"):
            EXPORT._join_endpoint(EXPORT._decision_row(handoff_decision),
                                  handoff_tick, wrong_source)
        wrong_request = copy.deepcopy(handoff_native)
        wrong_request["native_audit"]["policy_request_phase"] = "P06"
        with self.assertRaisesRegex(RuntimeError, "clocks or phases"):
            EXPORT._join_endpoint(EXPORT._decision_row(handoff_decision),
                                  handoff_tick, wrong_request)

    def test_diagnostic_raw_override_is_not_compared_across_units(self) -> None:
        request = [0.01] * 12
        decision = {
            "end_tick": 8, "request_phase": "P09",
            "environment_step_returned": True,
            "policy_request": {
                "schema": "wlr50_clean.actual_rr_capture_local_request.v1",
                "policy_version": "frozen_cp225280_rr_capture_local_history_v1",
                "sampling_draws": 0, "selected_raw_log_probability": None,
                "selected_raw_full12": [0.0] * 12,
                "selected_tanh_full12": [0.0] * 12,
                "prior_same_observation_conditional_mean_full12": [0.0] * 12,
                "applied_local_raw_mean_delta_full12": [0.0] * 12,
                "independent_diagnostic": {"PPO_credit": 0},
            },
            "step_info": {
                "physics_tick": 8, "phase_id": "P09", "end_phase_id": "P09",
                "actuator_target_effect_audit": {
                    "schema": "wlr50_clean.actuator_target_effect_audit.v1",
                    "verified": True, "source_phase_id": "P09",
                    "policy_request_phase": "P09", "physics_tick": 180,
                },
                "nominal_action_full12": [0.0] * 12,
                "projected_residual_full12": request,
                "actual_drive_target_full12": request,
                "raw_policy_action_full12": [0.5] * 12,
                "rr_capture_local": {
                    "schema": "wlr50_clean.rr_capture_local_task.v1",
                    "active": True, "local_success": False,
                    "hold_elapsed_s": 0.0, "capture_hold_progress": 0.0,
                    "metrics": {"tick": 8, "current_top_contact": False,
                                "current_top_bearing": False},
                },
            },
        }
        row = EXPORT._decision_row(copy.deepcopy(decision))
        self.assertNotEqual(row["issued_raw"], row["selected_raw"])
        self.assertNotEqual(row["issued_raw"], row["request_physical_step"])
        self.assertEqual(row["diagnostic_intervention"]["PPO_credit"], 0)


if __name__ == "__main__":
    unittest.main()
