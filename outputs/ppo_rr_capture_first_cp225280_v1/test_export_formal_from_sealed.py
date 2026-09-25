"""Pure-standard-library tests for the sealed formal-export wrapper."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_test_formal_export", HERE / "export_formal_from_sealed.py")
assert SPEC is not None and SPEC.loader is not None
FORMAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FORMAL
SPEC.loader.exec_module(FORMAL)


class FormalPlanTests(unittest.TestCase):
    def test_aux_training_ledger_exact_and_separate_from_realtime_assist(self) -> None:
        event = {"actual_optimizer_steps": 64, "accepted_steps": 64,
                 "source_checkpoint": {"path": "sealed.pt", "sha256": "a"*64},
                 "PPO_credit": 0, "extra_preserved": {"rows": [6, 7]}}
        checkpoint = {"counts": {"auxiliary_updates": 64}, "local_auxiliary_events": [event]}
        source = {"control_contributions": {"local_auxiliary_events": [copy.deepcopy(event)],
            "local_auxiliary_optimizer_steps": 64, "AUX_updates_during_evaluation": 0}}
        receipt = FORMAL.auxiliary_training_disclosure(source, checkpoint)
        self.assertEqual(receipt["label"], FORMAL.AUX_TRAINING_LABEL)
        self.assertEqual(receipt["actual_training_optimizer_steps"], 64)
        self.assertFalse(receipt["realtime_AUX_assist"])
        self.assertFalse(receipt["pure_PPO_training_claimed"])
        self.assertEqual(receipt["local_auxiliary_events"], [event])
        receipt["local_auxiliary_events"][0]["extra_preserved"]["rows"].append(8)
        self.assertEqual(event["extra_preserved"]["rows"], [6, 7])

    def test_aux_training_ledger_missing_mismatch_and_steps_rejected(self) -> None:
        base_checkpoint = {"counts": {"auxiliary_updates": 64},
                           "local_auxiliary_events": [{"actual_optimizer_steps": 64, "PPO_credit": 0}]}
        base_source = {"control_contributions": {
            "local_auxiliary_events": copy.deepcopy(base_checkpoint["local_auxiliary_events"]),
            "local_auxiliary_optimizer_steps": 64, "AUX_updates_during_evaluation": 0}}
        mutations = (
            lambda s,c:s["control_contributions"].pop("local_auxiliary_events"),
            lambda s,c:c.pop("local_auxiliary_events"),
            lambda s,c:s["control_contributions"]["local_auxiliary_events"][0].update(PPO_credit=False),
            lambda s,c:c["counts"].update(auxiliary_updates=63),
            lambda s,c:s["control_contributions"].update(local_auxiliary_optimizer_steps=63),
            lambda s,c:s["control_contributions"].update(AUX_updates_during_evaluation=1),
            lambda s,c:s["control_contributions"].pop("AUX_updates_during_evaluation"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                source, checkpoint = copy.deepcopy(base_source), copy.deepcopy(base_checkpoint)
                mutate(source, checkpoint)
                with self.assertRaisesRegex(RuntimeError, "AUX"):
                    FORMAL.auxiliary_training_disclosure(source, checkpoint)

    def test_old_aux_zero_without_optional_ledger_remains_supported(self) -> None:
        for counts in ({}, {"auxiliary_updates": 0}):
            receipt = FORMAL.auxiliary_training_disclosure({}, {"counts": counts})
            self.assertEqual(receipt["actual_training_optimizer_steps"], 0)
            self.assertEqual(receipt["local_auxiliary_events"], [])
            self.assertIsNone(receipt["label"])

    def fixture(self, root: Path, *, diagnostic: bool = False, version_name: str = "v1",
                tracking_revision: str | None = None) -> tuple[Path, Path]:
        version = FORMAL.VERSION_FAMILIES[version_name]
        experiment = root / "runs" / FORMAL.EXPERIMENT
        source = experiment / "video_eval" / "validation" / "sealed" / "source"
        source.mkdir(parents=True)
        output = root / "outputs" / FORMAL.EXPERIMENT
        checkpoint = output / "checkpoints" / "history" / "checkpoint_CP227328_local002048.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"immutable synthetic checkpoint")
        counts = {"local_policy_decisions": 2048, "local_ppo_updates": 4,
                  "local_optimizer_steps": 80, "prefix_decisions": 321,
                  "capture_opportunities": 4, "local_successes": 0,
                  "completed_local_episodes": 3, "auxiliary_updates": 0}
        if version_name == "v2":
            counts.update(task_v2_policy_decisions=0, task_v2_ppo_updates=0)
        profile = {"schema": version["profile_schema"], "version": version["profile_policy"],
            "observation_dimension": version["observation_dimension"],
            "local_features": FORMAL.LOCAL_FIELDS_V1 + ([FORMAL.ELIGIBILITY_FIELD] if version_name == "v2" else [])}
        if version_name == "v2":
            profile["capture_source_dispatch"] = FORMAL.SOURCE_DISPATCH_V2
        if tracking_revision is not None:
            profile["source_tracking_owner_revision"] = tracking_revision
        # Synthetic bytes, but an explicitly declared legacy coordinate family:
        # v1 original; v2 late-derived or tracking-inherited as this test selects.
        source_head = ("ecf205e80094693938057589fc91f7f31082ef89" if version_name == "v1" else
            "1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81" if tracking_revision is not None else
            "5f8487b76f27eb165a329a0b6f3096f54ebb4d48")
        runtime = {"source_git_commit": source_head, "experiment_id": FORMAL.EXPERIMENT,
                   "local_contract": profile}
        sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
        sidecar.write_text(json.dumps({"schema": version["checkpoint_schema"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": FORMAL.sha256(checkpoint),
            "save_load_round_trip": True, "runtime_contract": runtime,
            "runner_config": {"actor": {"observation_layout": version["observation_layout"]}},
            "counts": counts}), encoding="utf-8")
        manifest = {"schema": version["source_schema"], "runtime_contract": runtime,
            "mode": ("INDEPENDENT_DIRECTION_DIAGNOSTIC" if diagnostic else
                     "DETERMINISTIC_COMPOSITE_POLICY"),
            "diagnostic_intervention": diagnostic, "checkpoint_model_unchanged": True,
            "PPO_updates": 0, "counts": counts,
            "control_contributions": {"local_branch_counters": counts,
                "policy_version": version["profile_policy"],
                "observation_dimension": version["observation_dimension"],
                "diagnostic_only_RR_joint_override": diagnostic},
            "local_task": {"schema": version["task_schema"]},
            "checkpoint_load_provenance": {"checkpoint": str(checkpoint),
                "checkpoint_sha256": FORMAL.sha256(checkpoint), "manifest": str(sidecar),
                "manifest_sha256": FORMAL.sha256(sidecar),
                "strict_actual_composite_load_verified": True,
                "actor_critic_optimizer_hashes_verified": True}}
        if version_name == "v2":
            manifest["control_contributions"]["capture_source_dispatch"] = FORMAL.SOURCE_DISPATCH_V2
            manifest["local_task"].update(obs9=[0.]*8+[1.],
                metrics={FORMAL.ELIGIBILITY_FIELD: True})
        if tracking_revision is not None:
            manifest["control_contributions"]["source_tracking_owner_revision"] = tracking_revision
        (source / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (source.parent / "run_manifest.json").write_text(json.dumps({
            "lifecycle": "COMPLETE", "mode": "diagnostic" if diagnostic else "eval",
            "result": manifest}), encoding="utf-8")
        destination = output / "video_review" / "new_formal_review"
        return source, destination

    def test_derives_formal_hashes_counts_and_distinct_display_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = self.fixture(root)
            old_root, old_output, old_exporter = FORMAL.ROOT, FORMAL.OUTPUT, FORMAL.EXPORTER
            FORMAL.ROOT = root
            FORMAL.OUTPUT = root / "outputs" / FORMAL.EXPERIMENT
            FORMAL.EXPORTER = FORMAL.OUTPUT / "export_rr_capture_first_video.py"
            try:
                plan = FORMAL.build_plan(source, destination)
            finally:
                FORMAL.ROOT, FORMAL.OUTPUT, FORMAL.EXPORTER = old_root, old_output, old_exporter
            self.assertEqual(plan["display_id"], "CP227328_local002048")
            self.assertEqual(plan["counts"]["local_ppo_updates"], 4)
            self.assertFalse(plan["diagnostic"])
            self.assertNotIn("--allow-diagnostic", plan["command"])
            self.assertIn("not the historical soft-KL checkpoint",
                          plan["numeric_step_collision_note"])

    def test_rejects_diagnostic_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = self.fixture(root, diagnostic=True)
            old_root, old_output = FORMAL.ROOT, FORMAL.OUTPUT
            FORMAL.ROOT = root
            FORMAL.OUTPUT = root / "outputs" / FORMAL.EXPERIMENT
            try:
                with self.assertRaisesRegex(RuntimeError, "rejects diagnostic"):
                    FORMAL.build_plan(source, destination)
            finally:
                FORMAL.ROOT, FORMAL.OUTPUT = old_root, old_output

    def test_v2_has_explicit_family_and_zero_new_task_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = self.fixture(root, version_name="v2")
            old_root, old_output = FORMAL.ROOT, FORMAL.OUTPUT
            FORMAL.ROOT, FORMAL.OUTPUT = root, root/"outputs"/FORMAL.EXPERIMENT
            try:
                plan = FORMAL.build_plan(source, destination)
            finally:
                FORMAL.ROOT, FORMAL.OUTPUT = old_root, old_output
            self.assertEqual(plan["version_family"]["version"], "v2")
            self.assertEqual(plan["version_family"]["observation_dimension"], 448)
            self.assertEqual(plan["counts"]["local_policy_decisions"], 2048)
            self.assertEqual(plan["counts"]["task_v2_policy_decisions"], 0)
            self.assertIsNone(plan["source_tracking_owner_revision"])
            self.assertEqual(plan["source_tracking"]["hud_label"], "legacy v2 late-derived")

    def test_tracking_inheritance_plan_preserves_counts_and_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = self.fixture(root, version_name="v2",
                tracking_revision=FORMAL.SOURCE_TRACKING_INHERITANCE)
            old_root, old_output = FORMAL.ROOT, FORMAL.OUTPUT
            FORMAL.ROOT, FORMAL.OUTPUT = root, root/"outputs"/FORMAL.EXPERIMENT
            try:
                plan = FORMAL.build_plan(source, destination)
            finally:
                FORMAL.ROOT, FORMAL.OUTPUT = old_root, old_output
            self.assertEqual(plan["version_family"]["version"], "v2")
            self.assertEqual(plan["source_tracking_owner_revision"], FORMAL.SOURCE_TRACKING_INHERITANCE)
            self.assertEqual(plan["source_tracking"]["hud_label"], "pending-source inherited")
            self.assertEqual(plan["counts"]["local_policy_decisions"], 2048)
            self.assertEqual(plan["counts"]["local_ppo_updates"], 4)
            self.assertEqual(plan["counts"]["task_v2_policy_decisions"], 0)
            self.assertEqual(plan["display_id"], "CP227328_local002048")
            self.assertFalse(plan["diagnostic"])
            self.assertIn("no tracking-revision-local counts inferred", plan["source_tracking"]["training_count_scope"])

    def test_tracking_revision_unknown_mixed_or_v1_rejected(self) -> None:
        for version_name in ("v1", "v2"):
            with tempfile.TemporaryDirectory() as directory:
                source, _ = self.fixture(Path(directory), version_name=version_name)
                manifest = FORMAL.read_json(source/"source_manifest.json")
                metadata = FORMAL.read_json(Path(manifest["checkpoint_load_provenance"]["manifest"]))
                for source_revision, checkpoint_revision in (
                    (FORMAL.SOURCE_TRACKING_INHERITANCE, None),
                    (None, FORMAL.SOURCE_TRACKING_INHERITANCE),
                    ("unverified_tracking", "unverified_tracking"),
                    *(([(FORMAL.SOURCE_TRACKING_INHERITANCE, FORMAL.SOURCE_TRACKING_INHERITANCE)])
                      if version_name == "v1" else []),
                ):
                    with self.subTest(version=version_name, source=source_revision, checkpoint=checkpoint_revision):
                        s, c = copy.deepcopy(manifest), copy.deepcopy(metadata)
                        s["control_contributions"]["source_tracking_owner_revision"] = source_revision
                        c["runtime_contract"]["local_contract"]["source_tracking_owner_revision"] = checkpoint_revision
                        with self.assertRaisesRegex(RuntimeError, "tracking owner revision"):
                            FORMAL.validate_version_bindings(s, c)

    def test_mixed_family_legacy_mode_and_wrong_source_rule_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,_=self.fixture(root,version_name="v2")
            manifest=FORMAL.read_json(source/"source_manifest.json")
            metadata=FORMAL.read_json(Path(manifest["checkpoint_load_provenance"]["manifest"]))
            FORMAL.validate_version_bindings(manifest,metadata)
            mutations=(
                ("checkpoint v1", lambda s,c:c.update(schema=FORMAL.CHECKPOINT_SCHEMA)),
                ("actor v1", lambda s,c:c["runner_config"]["actor"].update(observation_layout="role439_rr_capture_local_v1")),
                ("legacy", lambda s,c:c["runner_config"]["actor"].update(legacy447_migration_only=True)),
                ("profile v1", lambda s,c:c["runtime_contract"]["local_contract"].update(version=FORMAL.VERSION_FAMILIES["v1"]["profile_policy"])),
                ("old obs", lambda s,c:s["control_contributions"].update(observation_dimension=447)),
                ("source rule", lambda s,c:s["control_contributions"].update(capture_source_dispatch="freeze_all_legs")),
                ("task v1", lambda s,c:s["local_task"].update(schema="wlr50_clean.rr_capture_local_task.v1")),
                ("eligibility mismatch", lambda s,c:s["local_task"]["obs9"].__setitem__(8,0.)),
                ("missing new counts", lambda s,c:c["counts"].pop("task_v2_ppo_updates")),
            )
            for label,mutate in mutations:
                with self.subTest(label=label):
                    s,c=copy.deepcopy(manifest),copy.deepcopy(metadata)
                    mutate(s,c)
                    with self.assertRaises(RuntimeError):
                        FORMAL.validate_version_bindings(s,c)

    def test_checkpoint_tamper_still_fails_before_version_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,destination=self.fixture(root,version_name="v2")
            manifest=FORMAL.read_json(source/"source_manifest.json")
            checkpoint=Path(manifest["checkpoint_load_provenance"]["checkpoint"])
            checkpoint.write_bytes(b"not the sealed checkpoint")
            old_root,old_output=FORMAL.ROOT,FORMAL.OUTPUT
            FORMAL.ROOT,FORMAL.OUTPUT=root,root/"outputs"/FORMAL.EXPERIMENT
            try:
                with self.assertRaisesRegex(RuntimeError,"differs from load provenance"):
                    FORMAL.build_plan(source,destination)
            finally:
                FORMAL.ROOT,FORMAL.OUTPUT=old_root,old_output

    def test_unknown_version_and_no_tensor_import(self) -> None:
        with self.assertRaisesRegex(RuntimeError,"unknown"):
            FORMAL.version_family("wlr50_clean.frozen_prior_rr_capture_video.v3")
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("pxr",sys.modules)


if __name__ == "__main__":
    unittest.main()
