"""Focused base-Python checks for the strict owner439 media adapter."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("export_rl_recovery_video.py")
SPEC = importlib.util.spec_from_file_location("export_rl_recovery_video", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def publication_path() -> Path:
    return PATH.with_name("rear_owner_CP225280_gf6d1d2df8d87_publication.json")


def retention_publication_path() -> Path:
    return PATH.with_name("rr_retention_CP226048_g72e63592bdf4_publication.json")


def front_identity_publication_path() -> Path:
    return PATH.with_name(
        "front_retention439_CP229120_g65a9255be6d9_identity_publication.json")


def front_aux_publication_path() -> Path:
    return PATH.with_name(
        "front_retention439_CP229120_g65a9255be6d9_aux_publication.json")


def collection_publication_path() -> Path:
    return PATH.with_name(
        "collection512_CP229632_g59e868f3e223_publication.json")


def args() -> argparse.Namespace:
    publication = MODULE.BASE.read_json(publication_path())
    checkpoint = Path(publication["checkpoint"])
    return argparse.Namespace(
        checkpoint=checkpoint,
        checkpoint_sha256=publication["checkpoint_sha256"],
        checkpoint_manifest_sha256=publication["manifest_sha256"],
        expected_global_policy_decisions=225280,
        expected_ppo_updates=1725,
        expected_optimizer_steps=34500,
        expected_head=MODULE.OWNER_TARGET_HEAD,
        checkpoint_runtime_head=None,
        rear_owner_publication=publication_path(),
        rear_owner_publication_sha256=MODULE.OWNER_PUBLICATION_SHA,
        rear_owner_plan_sha256=MODULE.OWNER_PLAN_SHA,
        rr_retention_publication=None,
        rr_retention_publication_sha256=None,
        rr_retention_plan_sha256=None,
        front_retention439_publication=None,
        front_retention439_publication_sha256=None,
        front_retention439_plan_sha256=None,
        collection512_publication=None,
        collection512_publication_sha256=None,
    )


def retention_args() -> argparse.Namespace:
    value = args()
    publication = MODULE.BASE.read_json(retention_publication_path())
    value.checkpoint = Path(publication["checkpoint"])
    value.checkpoint_sha256 = publication["checkpoint_sha256"]
    value.checkpoint_manifest_sha256 = publication["manifest_sha256"]
    value.expected_global_policy_decisions = publication["global_policy_decisions"]
    value.expected_ppo_updates = publication["ppo_updates"]
    value.expected_optimizer_steps = publication["optimizer_steps"]
    value.expected_head = MODULE.RETENTION_TARGET_HEAD
    value.rr_retention_publication = retention_publication_path()
    value.rr_retention_publication_sha256 = MODULE.RETENTION_PUBLICATION_SHA
    value.rr_retention_plan_sha256 = MODULE.RETENTION_PLAN_SHA
    return value


def front_args() -> argparse.Namespace:
    value = retention_args()
    publication = MODULE.BASE.read_json(front_aux_publication_path())
    value.checkpoint = Path(publication["checkpoint"])
    value.checkpoint_sha256 = publication["checkpoint_sha256"]
    value.checkpoint_manifest_sha256 = publication["manifest_sha256"]
    value.expected_global_policy_decisions = publication["global_policy_decisions"]
    value.expected_ppo_updates = publication["ppo_updates"]
    value.expected_optimizer_steps = publication["optimizer_steps"]
    value.expected_head = MODULE.FRONT_RETENTION_TARGET_HEAD
    value.front_retention439_publication = front_identity_publication_path()
    value.front_retention439_publication_sha256 = \
        MODULE.FRONT_RETENTION_PUBLICATION_SHA
    value.front_retention439_plan_sha256 = MODULE.FRONT_RETENTION_PLAN_SHA
    return value


def collection_args() -> argparse.Namespace:
    value = front_args()
    publication = MODULE.BASE.read_json(collection_publication_path())
    value.checkpoint = Path(publication["checkpoint"])
    value.checkpoint_sha256 = publication["checkpoint_sha256"]
    value.checkpoint_manifest_sha256 = publication["manifest_sha256"]
    value.expected_global_policy_decisions = publication["global_policy_decisions"]
    value.expected_ppo_updates = publication["ppo_updates"]
    value.expected_optimizer_steps = publication["optimizer_steps"]
    value.expected_head = MODULE.COLLECTION_TARGET_HEAD
    value.collection512_publication = collection_publication_path()
    value.collection512_publication_sha256 = MODULE.BASE.sha256(
        collection_publication_path())
    return value


def manifest_for(value: argparse.Namespace) -> dict:
    checkpoint = Path(value.checkpoint)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    metadata = MODULE.BASE.read_json(sidecar)
    return {
        "policy_sampling_mode": "deterministic_conditional_mean",
        "runtime_contract": metadata["runtime_contract"],
        "checkpoint_load_provenance": {
            "checkpoint_loaded_and_verified": True,
            "stochastic_policy": False,
            "policy_seed": None,
            "saved_global_policy_decisions": value.expected_global_policy_decisions,
            "checkpoint_runtime_compatibility": None,
            "parameter_hashes": {
                "actor_parameter_sha256": metadata["actor_parameter_sha256"]},
            "source": {"checkpoint": str(checkpoint.resolve()),
                       "checkpoint_sha256": value.checkpoint_sha256,
                       "manifest": str(sidecar.resolve()),
                       "manifest_sha256": value.checkpoint_manifest_sha256},
        },
    }


class OwnerIdentityTest(unittest.TestCase):
    def test_actual_published_zero_credit_owner439_identity(self):
        value = args()
        identity = MODULE.checkpoint_identity(manifest_for(value), value)
        self.assertEqual(identity["policy_version"], MODULE.OWNER_POLICY)
        self.assertEqual(identity["observation_dimension"], 439)
        self.assertEqual(identity["rear_owner_recovery_revision"][
            "revision_branch_counts"], {
                "global_policy_decisions": 0, "ppo_updates": 0,
                "optimizer_steps": 0})
        self.assertEqual(identity["rear_policy_branch_counts"], {
            "global_policy_decisions": 4736, "ppo_updates": 37,
            "optimizer_steps": 740})
        self.assertNotIn("rr_retention_reward_revision", identity)

    def test_actual_published_zero_credit_reward_only72e_identity(self):
        value = retention_args()
        identity = MODULE.checkpoint_identity(manifest_for(value), value)
        revision = identity["rr_retention_reward_revision"]
        self.assertEqual(identity["runtime_head"], MODULE.RETENTION_TARGET_HEAD)
        self.assertEqual(revision["reward_revision"],
                         MODULE.RETENTION_REWARD_REVISION)
        self.assertTrue(revision["reward_only_boundary"])
        self.assertFalse(revision["policy_control_distribution_changed_at_boundary"])
        self.assertEqual(revision["revision_branch_counts"], {
            "global_policy_decisions": 0, "ppo_updates": 0,
            "optimizer_steps": 0})
        first, second = MODULE.owner_status_lines(identity)
        self.assertIn("RR/RL TASK ASSIST OFF", first)
        self.assertIn("ISSUED-OWNER SUSPENSION/PROJECTION ON", first)
        self.assertIn("edge-v4", second)
        self.assertIn("reward-only72e boundary", second)
        self.assertNotIn(MODULE.BASE.NOMINAL_TIMING, first + second)
        labels = MODULE.owner_visible_labels(identity)
        self.assertEqual(labels["RR_RL_task_assists"], "OFF")
        self.assertEqual(labels["front_FL_assist"], "ON")
        self.assertEqual(labels["issued_owner_suspension_projection"], "ON")
        self.assertEqual(labels["timing_revision"], MODULE.OWNER_TIMING_REVISION)
        self.assertEqual(labels["reward_revision"],
                         MODULE.RETENTION_REWARD_REVISION)
        self.assertFalse(labels["source_manifest_nominal_timing_used_as_authority"])

    def test_actual_front_retention439_aux_identity_is_separate_from_ppo(self):
        value = front_args()
        identity = MODULE.checkpoint_identity(manifest_for(value), value)
        revision = identity["front_retention439_runtime_revision"]
        auxiliary = identity["front_retention439_auxiliary"]
        self.assertEqual(identity["runtime_head"],
                         MODULE.FRONT_RETENTION_TARGET_HEAD)
        self.assertTrue(revision["accounting_runtime_identity_only"])
        self.assertEqual(revision["revision_branch_counts"], {
            "global_policy_decisions": 0, "ppo_updates": 0,
            "optimizer_steps": 0})
        self.assertEqual(auxiliary["event_count"], 1)
        self.assertEqual(auxiliary["accepted_auxiliary_updates"], 32)
        self.assertEqual(auxiliary["attempted_auxiliary_optimizer_steps"], 32)
        self.assertEqual(auxiliary["PPO_credit"], 0)
        self.assertFalse(auxiliary["physical_success_claimed"])
        first, second = MODULE.owner_status_lines(identity)
        self.assertIn("RR/RL TASK ASSIST OFF", first)
        self.assertIn("front-ret AUX 32/32 (not PPO)", second)
        labels = MODULE.owner_visible_labels(identity)
        self.assertFalse(labels["front_retention439_AUX_is_PPO"])
        self.assertEqual(labels["front_retention439_auxiliary_ledger"], auxiliary)

    def test_actual_zero_credit_collection512_identity_keeps_old_counts_separate(self):
        value = collection_args()
        identity = MODULE.checkpoint_identity(manifest_for(value), value)
        collection = identity["collection_horizon439_revision"]
        self.assertEqual(identity["runtime_head"], MODULE.COLLECTION_TARGET_HEAD)
        self.assertEqual(identity["lifetime_counters"], {
            "global_policy_decisions": 229632, "ppo_updates": 1759,
            "optimizer_steps": 35180})
        self.assertEqual(collection["revision_branch_counts"], {
            "global_policy_decisions": 0, "ppo_updates": 0,
            "optimizer_steps": 0})
        self.assertEqual(identity["front_retention439_runtime_revision"][
            "revision_branch_counts"], {
                "global_policy_decisions": 512, "ppo_updates": 4,
                "optimizer_steps": 80})
        self.assertEqual(identity["front_retention439_auxiliary"][
            "accepted_auxiliary_updates"], 32)
        first, second = MODULE.owner_status_lines(identity)
        self.assertIn("RR/RL TASK ASSIST OFF", first)
        self.assertIn("65a +512d/4P/80A", second)
        self.assertIn("collection512 +0d/0P/0A", second)
        labels = MODULE.owner_visible_labels(identity)
        self.assertEqual(labels["collection_steps_per_update"], 512)
        self.assertFalse(labels["historical_128_counters_reinterpreted"])

    def test_collection512_runtime_counter_and_publication_tamper_fail_closed(self):
        value = collection_args()
        metadata = MODULE.BASE.read_json(
            Path(value.checkpoint).with_name(
                Path(value.checkpoint).stem + "_manifest.json"))
        receipt = metadata[MODULE.COLLECTION_KEY]
        previous = MODULE.reconstruct_collection439_source_runtime(
            metadata["runtime_contract"], receipt)
        self.assertEqual(MODULE.BASE.json_digest(previous),
                         MODULE.FRONT_RETENTION_TARGET_CONTRACT_SHA)
        changed = copy.deepcopy(receipt)
        changed["changed_file_hashes"].pop(
            next(iter(changed["changed_file_hashes"])))
        with self.assertRaisesRegex(RuntimeError, "six-path delta"):
            MODULE.reconstruct_collection439_source_runtime(
                metadata["runtime_contract"], changed)
        origin = receipt["counter_origin"]
        self.assertEqual(MODULE._collection_counter_updates({
            "global_policy_decisions": origin["global_policy_decisions"] + 512,
            "ppo_updates": origin["ppo_updates"] + 1,
            "optimizer_steps": origin["optimizer_steps"] + 20}, origin), 1)
        for bad in ({"global_policy_decisions": origin["global_policy_decisions"] + 128,
                     "ppo_updates": origin["ppo_updates"] + 1,
                     "optimizer_steps": origin["optimizer_steps"] + 20},
                    {"global_policy_decisions": origin["global_policy_decisions"] + 512,
                     "ppo_updates": origin["ppo_updates"] + 4,
                     "optimizer_steps": origin["optimizer_steps"] + 80}):
            with self.assertRaisesRegex(RuntimeError, r"\+512/\+1/\+20"):
                MODULE._collection_counter_updates(bad, origin)
        wrong = copy.copy(value)
        wrong.collection512_publication_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "publication bytes"):
            MODULE.checkpoint_identity(manifest_for(wrong), wrong)

    def test_front_retention439_reconstruction_and_ledger_tamper_fail(self):
        value = front_args()
        sidecar = Path(value.checkpoint).with_name(
            Path(value.checkpoint).stem + "_manifest.json")
        metadata = MODULE.BASE.read_json(sidecar)
        receipt = metadata[MODULE.FRONT_RETENTION_IDENTITY]
        previous = MODULE.reconstruct_front_retention439_source_runtime(
            metadata["runtime_contract"], receipt)
        self.assertEqual(MODULE.BASE.json_digest(previous),
                         MODULE.RETENTION_TARGET_CONTRACT_SHA)
        changed = copy.deepcopy(receipt)
        changed["changed_file_hashes"].pop(
            next(iter(changed["changed_file_hashes"])))
        with self.assertRaisesRegex(RuntimeError, "three-path delta"):
            MODULE.reconstruct_front_retention439_source_runtime(
                metadata["runtime_contract"], changed)
        bad = copy.deepcopy(metadata)
        bad[MODULE.FRONT_RETENTION_LEDGER][
            "accepted_auxiliary_updates_total"] += 1
        with self.assertRaisesRegex(RuntimeError, "totals or append-only"):
            MODULE._validate_front_retention_ledger(
                bad, receipt, metadata["runtime_contract"], {
                    key: metadata[key] for key in MODULE.BASE.COUNTERS})
        missing = copy.copy(value)
        missing.front_retention439_plan_sha256 = None
        with self.assertRaisesRegex(RuntimeError, "requires the front-retention439"):
            MODULE.checkpoint_identity(manifest_for(missing), missing)

    def test_reward_runtime_reconstruction_and_explicit_arguments_fail_closed(self):
        value = retention_args()
        metadata = MODULE.BASE.read_json(
            Path(value.checkpoint).with_name(Path(value.checkpoint).stem + "_manifest.json"))
        receipt = metadata[MODULE.RETENTION_MIGRATION]
        previous = MODULE.reconstruct_retention_source_runtime(
            metadata["runtime_contract"], receipt)
        self.assertEqual(MODULE.BASE.json_digest(previous),
                         MODULE.OWNER_TARGET_CONTRACT_SHA)
        changed = copy.deepcopy(receipt)
        changed["changed_file_hashes"].pop(next(iter(changed["changed_file_hashes"])))
        with self.assertRaisesRegex(RuntimeError, "six-file delta"):
            MODULE.reconstruct_retention_source_runtime(
                metadata["runtime_contract"], changed)
        missing = copy.copy(value)
        missing.rr_retention_plan_sha256 = None
        with self.assertRaisesRegex(RuntimeError, "requires its publication"):
            MODULE.checkpoint_identity(manifest_for(missing), missing)

    def test_runtime_reconstruction_is_exact_and_tamper_fails(self):
        value = args()
        metadata = MODULE.BASE.read_json(
            Path(value.checkpoint).with_name(Path(value.checkpoint).stem + "_manifest.json"))
        receipt = metadata[MODULE.OWNER_MIGRATION]
        previous = MODULE.reconstruct_owner_source_runtime(
            metadata["runtime_contract"], receipt)
        self.assertEqual(MODULE.BASE.json_digest(previous),
                         receipt["source_contract_sha256"])
        changed = copy.deepcopy(receipt)
        first = next(iter(changed["changed_file_hashes"]))
        changed["changed_file_hashes"][first]["after"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "changed-file binding"):
            MODULE.reconstruct_owner_source_runtime(metadata["runtime_contract"], changed)

    def test_wrong_publication_hash_and_arbitrary_439_rejected(self):
        value = args()
        bad_hash = copy.copy(value)
        bad_hash.rear_owner_publication_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "publication receipt"):
            MODULE.checkpoint_identity(manifest_for(bad_hash), bad_hash)
        bad_head = copy.copy(value)
        bad_head.expected_head = MODULE.OWNER_SOURCE_HEAD
        with self.assertRaisesRegex(RuntimeError, "exact f6d/72e"):
            MODULE.checkpoint_identity(manifest_for(bad_head), bad_head)

    def test_f6d_path_rejects_reward_only_arguments(self):
        value = args()
        value.rr_retention_publication = retention_publication_path()
        with self.assertRaisesRegex(RuntimeError, "must not carry reward-only72e"):
            MODULE.checkpoint_identity(manifest_for(value), value)

    def test_pre65a_path_rejects_front_retention_arguments(self):
        value = retention_args()
        value.front_retention439_publication = front_identity_publication_path()
        with self.assertRaisesRegex(RuntimeError, "must not carry front-retention439"):
            MODULE.checkpoint_identity(manifest_for(value), value)

    def test_precollection_path_rejects_collection_arguments(self):
        value = front_args()
        value.collection512_publication = collection_publication_path()
        with self.assertRaisesRegex(RuntimeError, "must not carry collection512"):
            MODULE.checkpoint_identity(manifest_for(value), value)


class NamingTest(unittest.TestCase):
    @staticmethod
    def row(phase: str) -> dict:
        return {"phase": phase,
                "rear_policy_timing": {"rr_carry_capture": False,
                                       "rr_support_handoff": False,
                                       "rl_prep_transfer": False,
                                       "rl_swing_capture": False}}

    def test_predecessor_failure_filename_does_not_claim_rear_window(self):
        plan = MODULE.detail_plan([self.row("P02")] * 20, 225280, False)
        self.assertEqual(plan["filename"],
            "CP225280_DET_RR_NOT_REACHED_PREDECESSOR_FAILURE_detail_INCOMPLETE.mp4")
        self.assertEqual(plan["kind"], "RR_NOT_REACHED_PREDECESSOR_FAILURE")
        self.assertEqual(plan["requested_RR_detail_unavailable_reason"],
                          "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW")

    def test_rear_window_without_placement_is_capture_incomplete(self):
        rows = [self.row("P09")]
        rows[0]["rr_placed"] = False
        plan = MODULE.detail_plan(rows, 225792, False)
        self.assertTrue(plan["RR_window_reached"])
        self.assertFalse(plan["RR_placed_history_observed"])
        self.assertEqual(plan["filename"],
                         "CP225792_DET_RR_capture_detail_INCOMPLETE.mp4")
        self.assertEqual(plan["kind"], "FL_TO_FR_RL_RECOVERY_ACTUAL_WINDOW")
        self.assertIsNone(plan["requested_RR_detail_unavailable_reason"])
        self.assertIn("RR CAPTURE INCOMPLETE", plan["title"])
        self.assertIn("RL NOT REACHED", plan["title"])

    def test_rr_placed_without_rl_still_uses_capture_incomplete_name(self):
        rows = [self.row("P09")]
        rows[0]["rr_placed"] = True
        plan = MODULE.detail_plan(rows, 226304, False)
        self.assertEqual(plan["filename"],
                         "CP226304_DET_RR_capture_detail_INCOMPLETE.mp4")
        self.assertIn("RR CAPTURE OBSERVED", plan["title"])
        self.assertIn("RL NOT REACHED", plan["title"])

    def test_actual_rl_window_alone_enables_rr_to_rl_filename(self):
        rows = [self.row("P09"), self.row("P12")]
        rows[0]["rr_placed"] = True
        rows[1]["rear_policy_timing"]["rl_prep_transfer"] = True
        incomplete = MODULE.detail_plan(rows, 227328, False)
        self.assertEqual(incomplete["filename"],
                         "CP227328_DET_RR_to_RL_detail_INCOMPLETE.mp4")
        self.assertEqual(incomplete["kind"],
                         "FL_TO_FR_RL_RECOVERY_ACTUAL_WINDOW")
        self.assertIn("RL WINDOW OBSERVED", incomplete["title"])
        complete = MODULE.detail_plan(rows, 227328, True)
        self.assertEqual(complete["filename"],
                         "CP227328_DET_RR_to_RL_detail.mp4")


if __name__ == "__main__":
    unittest.main()
