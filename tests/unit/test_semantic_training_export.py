"""Standard-library exporter tests. Fixtures are NOT optimizer or physical evidence."""
import csv
import importlib.util
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("semantic_export", ROOT / "tools/export_semantic_training.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    history = root / "outputs/ppo_semantic_v2/checkpoints/history"
    history.mkdir(parents=True)
    files = {name: "f" * 64 for name in (
        "configs/ppo_semantic_v2/reward_config.yaml", "src/wlr50_clean/ppo/semantic_reward.py",
        "configs/ppo_semantic_v2/stage_task_spec.yaml", "src/wlr50_clean/ppo/semantic_supervisor.py")}
    contract = {"files": files, "runtime_content_sha256": module.digest(files),
                "source_git_commit": "a" * 40, "frozen_A_files": {"frozen": "b" * 64}, "physics_hz": 120}
    def checkpoint(name, end, actor, source_run=None, update=None):
        path = history / name
        path.write_bytes(("fixture " + name).encode())
        data = {"schema": "wlr50_clean.semantic_checkpoint.v1", "checkpoint_path": str(path),
            "checkpoint_sha256": module.sha(path), "save_load_round_trip": True,
            "global_policy_decisions": end, "ppo_updates": end // 128, "optimizer_steps": end // 128 * 20,
            "actor_parameter_sha256": actor, "optimizer_state_sha256": "d" * 64,
            "runtime_contract": contract, "runner_config": {"num_steps_per_env": 128}, "seed": 1001,
            "stage_requested_decisions": {"smoke": end, "phase_suffix": 0, "full_episode": 0}}
        if source_run is not None:
            data.update(source_run=str(source_run), last_update=update)
        write(path.with_name(path.stem + "_manifest.json"), data)
        return path
    initial = checkpoint("checkpoint_initial_semantic.pt", 0, "0" * 64)
    previous = initial
    for number, (start, end, tail) in enumerate(((1, 128, 0), (129, 256, 7), (257, 384, 0)), 1):
        run = root / f"runs/ppo_semantic_v2/train/run{number}"
        run.mkdir(parents=True)
        launched = None if number == 1 else str(previous)
        write(run / "run_manifest.started.json", {"command": "train", "runtime_contract": contract,
            "arguments": {"checkpoint": launched, "run_dir": str(run), "seed": 1001}})
        args = ["--run-dir", str(run), "--expected-head", contract["source_git_commit"]]
        if launched:
            args += ["--checkpoint", str(history.parent / "checkpoint_last.pt")]
        write(run.with_name(run.name + "_launcher") / "arguments.json", args)
        update = {"ppo_update": number, "global_policy_decisions": end, "optimizer_steps": 20,
            "actor_parameter_sha256_before": str(number - 1) * 64, "actor_parameter_sha256_after": str(number) * 64,
            "actor_parameters_changed": True, "finite_nonzero_gradient_observed": True,
            "gradient_norm_min": 1., "gradient_norm_max": 2., "kl_mean": .1,
            "clip_fraction": .1, "entropy": .1, "value_loss": .1}
        (run / "optimizer_updates.jsonl").write_text(json.dumps(update) + "\n")
        (run / "rollouts").mkdir()
        (run / "rollouts" / f"rollout_{number:06d}.pt").write_bytes(b"fixture only")
        with (run / "residual_and_projection_audit.jsonl").open("w") as stream:
            for index in range(start, end + tail + 1):
                info = {"phase_id": "P01", "end_phase_id": "P01", "raw_policy_action_full12": [0.] * 12,
                    "nominal_action_full12": [0.] * 12, "projected_residual_full12": [0.] * 12,
                    "applied_action_full12": [0.] * 12, "physics_ticks": 8, "sim_time_s": index / 15,
                    "termination_reason": None, "task_success": False,
                    "reward_breakdown": {"total": 1., "families": dict.fromkeys(module.FAMILIES, .2)}}
                stream.write(json.dumps({"global_policy_decision": index, "raw_policy_action_full12": [0.] * 12,
                    "old_log_probability": .1, "old_value": .1, "reward": 1., "terminal": False,
                    "applied_audit": info}) + "\n")
        previous = checkpoint(f"checkpoint_step_{end:09d}.pt", end, str(number) * 64, run, update)
        if tail:
            write(run / "external_stop.json", {"lifecycle": "EXTERNALLY_STOPPED_AFTER_VERIFIED_CHECKPOINT",
                "partial_rollout_will_not_be_reused": True, "checkpoint": str(previous),
                "checkpoint_sha256": module.sha(previous)})
        else:
            write(run / "run_manifest.json", {"lifecycle": "SUCCEEDED"})
    return initial, previous


class ExportTests(unittest.TestCase):
    def test_old_launch_resolves_parent_and_unoptimized_overlap_is_excluded(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            initial, latest = fixture(root)
            output = root / "outputs/ppo_semantic_v2/metrics/training_exports/new"
            result = module.export(root, latest, initial, output)
            self.assertEqual(result["optimized_policy_decisions"], 384)
            self.assertEqual(result["cumulative_optimizer_steps"], 60)
            self.assertEqual(result["ancestry_ranges"][1]["excluded_rows_after_selected_checkpoint"], 7)
            with (output / "residual_and_projection_audit.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["global_policy_decision"]) for row in rows], list(range(1, 385)))
            self.assertTrue(all(row["all_tick_native_audit"] == "unavailable" for row in rows))
            with (output / "phase_coverage.csv").open(newline="") as stream:
                phases = list(csv.DictReader(stream))
            self.assertEqual(len(phases), 13)
            self.assertEqual(phases[-1]["optimized_policy_decisions"], "0")
            with self.assertRaises(FileExistsError):
                module.export(root, latest, initial, output)

    def test_checkpoint_corruption_and_duplicate_optimized_rows_fail(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            initial, latest = fixture(root)
            latest.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                module.Exporter(root, latest, initial)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            initial, latest = fixture(root)
            path = root / "runs/ppo_semantic_v2/train/run2/residual_and_projection_audit.jsonl"
            text = path.read_text()
            path.write_text(text.splitlines()[0] + "\n" + text)
            output = root / "outputs/ppo_semantic_v2/metrics/training_exports/invalid"
            with self.assertRaises(ValueError):
                module.export(root, latest, initial, output)
            self.assertTrue((output / "export_failed.json").exists())
            self.assertFalse((output / "training_manifest_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
