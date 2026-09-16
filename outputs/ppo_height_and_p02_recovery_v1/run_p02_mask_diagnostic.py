"""Independent short frozen-policy mask diagnostics; never formal PPO success.

Reuses the existing semantic_video_cli runtime, reset, checkpoint/migration loader,
physical evaluator, recorder and finally-close path. Only output-side wrappers
select a temporary raw-channel intervention. No production modules are edited.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.ppo import semantic_video_cli as cli
from wlr50_clean.infrastructure.command_batch import FULL12_ORDER

MASKS = {"wheels4": (8, 9, 10, 11), "fl_knee": (1,)}
LEGACY = PROJECT / "outputs/ppo_rr_video_diagnosis_v1/run_rr_channels_off_diagnostic.py"
WRAPPER = {"path": str(Path(__file__).resolve()),
           "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class DiagnosticBoundedWindow(RuntimeError):
    pass


def dispatch_raw(proposed, indices):
    values = tuple(float(x) for x in proposed)
    cli.require(len(values) == 12 and all(math.isfinite(x) for x in values),
                "diagnostic actor output must be finite Full12")
    return tuple(0. if i in indices else x for i, x in enumerate(values))


def checked_history(core, observation, last_dispatched):
    actual = tuple(core._history["previous_raw_full12"])
    cli.require(actual == last_dispatched, "HISTORY differs from actual dispatched diagnostic raw")
    cli.require(len(observation) == 372, "diagnostic requires unchanged HISTORY372")
    feature = tuple(float(x) for x in observation[195:207])
    clipped = tuple(max(-20., min(20., x)) for x in actual)
    cli.require(feature == clipped, "actor observation HISTORY differs from clipped actual raw")
    return actual, feature


def main(argv=None):
    options = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    options.add_argument("--diagnostic-mask", required=True, choices=tuple(MASKS))
    options.add_argument("--diagnostic-max-decisions", type=int, default=240)
    diagnostic, official_argv = options.parse_known_args(argv)
    args = cli.parser().parse_args(official_argv)
    cli.require(cli.validate_video_args(args) == "C", "diagnostic requires official frozen C loader")
    cli.require(args.semantic_version == "v3" and args.from_phase == "P01" and args.num_envs == 1,
                "diagnostic requires existing natural P01 v3 path")
    limit = diagnostic.diagnostic_max_decisions
    cli.require(1 <= limit <= 300, "diagnostic decision limit must be1..300; task horizon remains200s")
    indices = MASKS[diagnostic.diagnostic_mask]
    cli.require(tuple(FULL12_ORDER[i] for i in MASKS["wheels4"]) ==
                ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
                and FULL12_ORDER[1] == "front_left_knee", "diagnostic channel mapping changed")
    intervention = {"schema": "wlr50_clean.p02_mask_diagnostic.v1", "wrapper": WRAPPER,
        "mode": "DIAGNOSTIC_" + diagnostic.diagnostic_mask.upper() + "_RAW_OFF_FROM_FIRST_DECISION",
        "mask_selection": diagnostic.diagnostic_mask, "masked_raw_indices": list(indices),
        "masked_names": [FULL12_ORDER[i] for i in indices], "formal_ppo_success": False,
        "formal_P02_evidence": False, "diagnostic_only": True, "on_policy_training_samples": 0,
        "optimizer_updates": 0, "maximum_diagnostic_decisions": limit,
        "maximum_diagnostic_physics_ticks": 8 * limit, "task_horizon_unchanged_s": 200,
        "other_channels": "same frozen actor on current measured observation including actual dispatched raw HISTORY",
        "nominal_mapper_projection_physics_evaluator": "unchanged current production functions",
        "expected_head": args.expected_head, "checkpoint": str(args.checkpoint),
        "resume_migration": None if args.resume_migration is None else str(args.resume_migration)}
    spec = importlib.util.spec_from_file_location("prior_output_only_rr_probe", LEGACY)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)  # Reuse its proven read-only getter function, not its RR-mask main.
    original_build, original_loader = cli.build_video_core, cli.checkpoint_loader
    original_capture, original_write = cli.capture_semantic_video, cli.write_json
    state = {"core": None, "stream": None, "check_model": None, "last_dispatched": (0.,) * 12,
             "issued": 0, "history_checks": 0, "bounded_stop": False, "owned_source": None,
             "model_unchanged": None}

    def write(path, payload, **kwargs):
        if kwargs.get("replace"):
            cli.require(Path(path).resolve() == state["owned_source"],
                        "replacement restricted to this new capture source")
        if Path(path).name in ("run_manifest.started.json", "run_manifest.json",
                              "semantic_video_source_manifest.json"):
            payload = {**payload, "diagnostic_intervention": intervention}
        return original_write(path, payload, **kwargs)

    def build(*positional, **kwargs):
        state["core"] = original_build(*positional, **kwargs)
        return state["core"]

    def loader(request, contract):
        official_load = original_loader(request, contract)
        def load(observation):
            actor_action, proof, unchanged = official_load(observation)
            core = state["core"]
            state["check_model"] = unchanged
            probe_path = request.run_dir / "source/live_preaction_drive_getters.json"
            probe = prior.live_probe(core)
            probe["intervention"] = intervention
            probe["reused_probe_source"] = str(LEGACY)
            original_write(probe_path, probe)
            state["stream"] = (request.run_dir / "source/diagnostic_policy_decisions.jsonl").open("x", encoding="utf-8")
            def action(current, decision):
                actual, feature = checked_history(core, current, state["last_dispatched"])
                state["history_checks"] += 1
                if decision >= limit:
                    state["bounded_stop"] = True
                    raise DiagnosticBoundedWindow("DIAGNOSTIC_BOUNDED_WINDOW")
                proposed = tuple(actor_action(current, decision))
                effective = dispatch_raw(proposed, indices)
                record = {"decision": decision + 1, "start_tick": core.backend._episode_tick,
                    "phase": core.frame.state_id, "policy_proposed_raw_full12": proposed,
                    "diagnostic_dispatched_raw_full12": effective,
                    "previous_actual_dispatched_raw_full12": actual,
                    "actual_policy_HISTORY_feature_full12": feature,
                    "masked_indices": list(indices), "diagnostic_only": True, "training_credit": False}
                state["stream"].write(json.dumps(record, allow_nan=False) + "\n")
                state["stream"].flush()
                state["last_dispatched"], state["issued"] = effective, decision + 1
                return effective
            return action, {**proof, "diagnostic_intervention": intervention,
                            "live_probe": str(probe_path)}, unchanged
        return load

    def capture(*positional, **kwargs):
        owned = (Path(kwargs["output_directory"]) / "semantic_video_source_manifest.json").resolve()
        cli.require(owned == (Path(args.run_dir) / "source/semantic_video_source_manifest.json").resolve()
                    and not owned.exists(), "annotation requires a fresh uniquely owned source")
        state["owned_source"] = owned
        try:
            result = original_capture(*positional, **kwargs)
        finally:
            if state["stream"] is not None:
                state["stream"].close()
            if state["check_model"] is not None:
                state["check_model"]()
                state["model_unchanged"] = True
        cli.require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == WRAPPER["sha256"],
                    "diagnostic wrapper changed during run")
        cli.require(state["issued"] <= limit and (result.get("episode_physics_ticks") or 0) <= 8 * limit,
                    "diagnostic resource window exceeded")
        result.update(diagnostic_only=True, success_candidate=False, improved_claim=False,
            diagnostic_intervention=intervention, diagnostic_bounded_stop=state["bounded_stop"],
            diagnostic_stop_reason="DIAGNOSTIC_BOUNDED_WINDOW" if state["bounded_stop"] else "EARLIER_REAL_ENDPOINT_OR_ERROR",
            masked_raw_history_checks=state["history_checks"], frozen_model_unchanged_after_capture=state["model_unchanged"],
            diagnostic_policy_decisions=state["issued"], on_policy_training_samples=0,
            formal_ppo_success=False, formal_P02_evidence=False,
            diagnostic_action_log=str(Path(kwargs["output_directory"]) / "diagnostic_policy_decisions.jsonl"),
            live_preaction_drive_probe=str(Path(kwargs["output_directory"]) / "live_preaction_drive_getters.json"))
        write(owned, result, replace=True)
        return result

    cli.build_video_core, cli.checkpoint_loader = build, loader
    cli.capture_semantic_video, cli.write_json = capture, write
    try:
        return cli.main(official_argv)  # Official --expected-head/checkpoint/resume-migration untouched.
    finally:
        cli.build_video_core, cli.checkpoint_loader = original_build, original_loader
        cli.capture_semantic_video, cli.write_json = original_capture, original_write


if __name__ == "__main__":
    raise SystemExit(main())
