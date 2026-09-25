"""Pure-stdlib checks for the isolated CP225280 front-preservation adapter."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("export_front_preserved_video.py")
SPEC = importlib.util.spec_from_file_location("export_front_preserved_video", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


DATASET = (PATH.parent / "staged_cp225280_front_branch" /
           "CP225280_front_replay_dataset.json").resolve()
DATASET_SHA = "d78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38"
SOURCE_MANIFEST = (MODULE.ROOT / "outputs" /
    "ppo_rr_rl_timing_policy_learning_v1" / "branches" /
    "ancestor220544_recapture_v2" / "checkpoints" / "history" /
    "checkpoint_rear_owner_CP225280_gf6d1d2df8d87_manifest.json")


def source_args() -> argparse.Namespace:
    publication_path = PATH.with_name(
        "rear_owner_CP225280_gf6d1d2df8d87_publication.json")
    return argparse.Namespace(
        rear_owner_publication=publication_path,
        rear_owner_publication_sha256=MODULE.CORE.OWNER_PUBLICATION_SHA,
        rear_owner_plan_sha256=MODULE.CORE.OWNER_PLAN_SHA)


class FrontPreservedAdapterTest(unittest.TestCase):
    def test_actual_f6d_source_identity_and_legacy_aux_are_preserved(self):
        source = MODULE.CORE.BASE.read_json(SOURCE_MANIFEST)
        receipt = {"source_checkpoint": {
                "checkpoint": source["checkpoint_path"],
                "checkpoint_sha256": MODULE.SOURCE_SHA,
                "manifest": str(SOURCE_MANIFEST.resolve()),
                "manifest_sha256": MODULE.SOURCE_MANIFEST_SHA},
            "source_manifest": source,
            "source_manifest_content_sha256": MODULE.CORE.BASE.json_digest(source)}
        identity = MODULE._source_identity(receipt, source_args())
        self.assertEqual(identity["runtime_head"], MODULE.SOURCE_HEAD)
        self.assertEqual(identity["lifetime_counters"], MODULE.ORIGIN)
        self.assertEqual(identity["inherited_auxiliary"], {
            "legacy_limited_AUX": {"accepted": 7, "attempted": 8},
            "mixed_front_and_RR_AUX": {"accepted": 103, "attempted": 104}})

    def test_counter_and_replay_accounting_are_exact(self):
        counters = {"global_policy_decisions": 226304,
                    "ppo_updates": 1727, "optimizer_steps": 34540}
        delta = MODULE._counter_delta(counters)
        self.assertEqual(delta, {"global_policy_decisions": 1024,
                                 "ppo_updates": 2, "optimizer_steps": 40})
        self.assertEqual(MODULE._expected_replay_counts(2), {
            "ppo_updates_with_replay": 2, "optimizer_minibatches": 40,
            "replay_row_exposures": 1280, "on_policy_samples_added": 0,
            "separate_auxiliary_optimizer_steps": 0})
        for bad in (
            {"global_policy_decisions": 225792,
             "ppo_updates": 1727, "optimizer_steps": 34540},
            {"global_policy_decisions": 226304,
             "ppo_updates": 1727, "optimizer_steps": 34520},
        ):
            with self.assertRaisesRegex(RuntimeError, r"\+512/\+1/\+20"):
                MODULE._counter_delta(bad)

    def test_runtime_reconstruction_allows_only_reviewed_paths(self):
        source = MODULE.CORE.BASE.read_json(SOURCE_MANIFEST)
        old = source["runtime_contract"]
        target = copy.deepcopy(old)
        target["source_git_commit"] = "1" * 40
        target["files"][MODULE.CODE + "semantic_front_preservation.py"] = "2" * 64
        target["files"][MODULE.CODE + "semantic_front_replay.py"] = "3" * 64
        target["runtime_content_sha256"] = MODULE.CORE.BASE.json_digest(target["files"])
        changed = {
            path: {"before": old["files"].get(path), "after": target["files"][path]}
            for path in MODULE.REQUIRED_NEW_PATHS}
        receipt = {"source_manifest": source, "changed_file_hashes": changed}
        self.assertEqual(MODULE._runtime_source(target, receipt), old)
        bad = copy.deepcopy(target)
        bad["files"]["src/wlr50_clean/semantic_task.py"] = "4" * 64
        bad["runtime_content_sha256"] = MODULE.CORE.BASE.json_digest(bad["files"])
        bad_receipt = copy.deepcopy(receipt)
        bad_receipt["changed_file_hashes"]["src/wlr50_clean/semantic_task.py"] = {
            "before": old["files"].get("src/wlr50_clean/semantic_task.py"),
            "after": "4" * 64}
        with self.assertRaisesRegex(RuntimeError, "reviewed narrow set"):
            MODULE._runtime_source(bad, bad_receipt)

    def test_actual_bounded_dataset_binding(self):
        args = argparse.Namespace(front_replay_dataset=DATASET,
                                  front_replay_dataset_sha256=DATASET_SHA)
        spec = {"schema": MODULE.REPLAY_SCHEMA,
                "version": MODULE.REPLAY_VERSION,
                "dataset_path": str(DATASET), "dataset_sha256": DATASET_SHA,
                "coefficient": 1.0, "minibatch_size": 32}
        binding = MODULE._validate_dataset(spec, args)
        self.assertEqual(binding["training_rows"], 100)
        self.assertEqual(binding["heldout_rows"], 99)
        wrong = copy.deepcopy(spec)
        wrong["coefficient"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "coefficient1/batch32"):
            MODULE._validate_dataset(wrong, args)

    def test_replay_report_requires_actual_same_adam_accounting(self):
        spec = {"schema": MODULE.REPLAY_SCHEMA, "coefficient": 1.0,
                "minibatch_size": 32, "dataset_path": str(DATASET),
                "dataset_sha256": DATASET_SHA}
        batch = {"gradient_consumed_once": True, "on_policy_samples_added": 0,
                 "separate_optimizer_steps": 0}
        metadata = {"last_update": {"front_replay_regularization": {
            "schema": MODULE.REPLAY_SCHEMA, "spec": spec,
            "minibatches": [copy.deepcopy(batch) for _ in range(20)],
            "actual_replay_row_exposures": 640, "on_policy_samples_added": 0,
            "separate_auxiliary_optimizer_steps": 0,
            "realtime_teacher_deployed": False}}}
        MODULE._validate_last_replay(metadata, spec, 1)
        metadata["last_update"]["front_replay_regularization"][
            "separate_auxiliary_optimizer_steps"] = 1
        with self.assertRaisesRegex(RuntimeError, "actual20/640"):
            MODULE._validate_last_replay(metadata, spec, 1)

    def test_visible_labels_keep_replay_and_aux_credit_separate(self):
        identity = {"rear_owner_recovery_revision": {},
            "front_preservation439_branch_identity": {
                "revision_branch_counts": {"global_policy_decisions": 512,
                    "ppo_updates": 1, "optimizer_steps": 20}},
            "front_replay_counts": MODULE._expected_replay_counts(1),
            "legacy_inherited_source_auxiliary": {
                "legacy_limited_AUX": {"accepted": 7, "attempted": 8},
                "mixed_front_and_RR_AUX": {"accepted": 103, "attempted": 104}}}
        first, second = MODULE.CORE.owner_status_lines(identity)
        self.assertIn("RR/RL TASK ASSIST OFF", first)
        self.assertIn("CP225280/f6d restored", second)
        self.assertIn("not PPO samples/AUX steps", second)
        labels = MODULE.CORE.owner_visible_labels(identity)
        self.assertFalse(labels["front_replay_rows_are_on_policy_samples"])
        self.assertFalse(labels["front_replay_adds_separate_optimizer_steps"])
        self.assertFalse(labels["front_replay_is_AUX"])
        self.assertFalse(labels["front_retention439_AUX32_inherited"])
        self.assertEqual(labels["legacy_inherited_source_auxiliary"],
                         identity["legacy_inherited_source_auxiliary"])


if __name__ == "__main__":
    unittest.main()
