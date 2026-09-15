"""Output-only, <=225-decision RR raw-channel intervention; never a PPO eval.

Pass the ordinary semantic_video_cli arguments, including --max-decisions 3000.
The unchanged task horizon stays200s; this explicitly labeled diagnostic stops
before decision226, with no extra physics, state writes, optimizer, or prefix.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.ppo import semantic_video_cli as cli
from wlr50_clean.infrastructure.command_batch import FULL12_ORDER

MAX_DECISIONS = 225
WRAPPER = {"path": str(Path(__file__).resolve()),
           "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
INTERVENTION = {"schema": "wlr50_clean.rr_channels_off_diagnostic.v1", "wrapper": WRAPPER,
    "mode": "DIAGNOSTIC_RR_RAW_CHANNELS_OFF_FROM_FIRST_DECISION", "formal_C0": False,
    "masked_raw_indices": [6, 7], "masked_names": list(FULL12_ORDER[6:8]),
    "other_channels": "same frozen actor, current measured observation including actual masked action history",
    "maximum_diagnostic_decisions": MAX_DECISIONS, "maximum_diagnostic_physics_ticks": 1800,
    "diagnostic_stop_reason": "DIAGNOSTIC_BOUNDED_WINDOW", "task_horizon_unchanged_s": 200,
    "optimizer_updates": 0, "on_policy_training_samples": 0, "formal_ppo_success": False,
    "nominal_feedback_reset_projection_physics_and_evaluator": "unchanged production functions"}


class DiagnosticBoundedWindow(RuntimeError):
    pass


def object_file(value):
    if value is None:
        return {"value": None, "reason": "object unavailable"}
    module = sys.modules.get(type(value).__module__)
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}",
            "__file__": None if module is None else getattr(module, "__file__", None)}


def live_probe(core):
    """Only installed root_physx_view getters; not configured defaults or estimates."""
    backend, adapter = core.backend, core.backend._adapter
    robot, view = adapter.robot, adapter.robot.root_physx_view
    clock = lambda: {"episode_tick": backend._episode_tick, "decision_count": core.decision_count,
        "reset_count": backend._reset_count, "authoritative_frame_tick": core.frame.physics_tick,
        "controller_next_tick": backend._controller.physics_tick}
    before = clock()
    cli.require(before["episode_tick"] == before["decision_count"] == before["authoritative_frame_tick"] == 0
                and before["controller_next_tick"] == 1,
                "probe is permitted only at freshly reset P01 before first action")
    getters = {"stiffness": ("get_dof_stiffnesses", "native PhysX drive gain"),
        "damping": ("get_dof_dampings", "native PhysX drive gain"),
        "armature": ("get_dof_armatures", "native PhysX armature"),
        "friction_properties": ("get_dof_friction_properties", "Isaac5.x static,dynamic,viscous columns"),
        "position_limits": ("get_dof_limits", "radians for these revolute DOFs"),
        "velocity_limits": ("get_dof_max_velocities", "rad/s for these revolute DOFs"),
        "maximum_forces": ("get_dof_max_forces", "Nm for these revolute DOFs; configured effective limit, not measured torque")}
    fields = {}
    for name, (getter, units) in getters.items():
        try:
            values = getattr(view, getter)()
            values = values.detach().cpu().tolist() if hasattr(values, "detach") else values.tolist()
            nonfinite = []
            def finite(value, index=()):
                if isinstance(value, list):
                    return [finite(item, index + (i,)) for i, item in enumerate(value)]
                if isinstance(value, float) and not math.isfinite(value):
                    nonfinite.append({"index": list(index), "reported": str(value)})
                    return None
                return value
            values = finite(values)
            fields[name] = {"value": values, "getter": getter, "units_or_native_semantics": units,
                            "provenance": "actual root_physx_view getter", "nonfinite_entries": nonfinite,
                            "reason": "nonfinite entries replaced with null" if nonfinite else None}
        except Exception as exc:
            fields[name] = {"value": None, "getter": getter, "units_or_native_semantics": units,
                            "reason": f"{type(exc).__name__}: {exc}"}
    controller = backend._controller
    nominal = getattr(controller, "nominal_provider", None)
    objects = {"core": core, "backend": backend, "robot_adapter": adapter, "articulation": robot,
        "root_physx_view": view, "controller": controller, "nominal_provider": nominal,
        "motion_executor": getattr(nominal, "_source_motion", None),
        "mapper": getattr(adapter, "servo_target_mapper", None), "sensor_reader": backend._reader}
    cli.require(clock() == before, "read-only probe changed clocks")
    return {"schema": "wlr50_clean.live_preaction_drive_getters.v1", "intervention": INTERVENTION,
        "python": sys.executable, "joint_names": list(robot.joint_names), "full12_order": list(FULL12_ORDER),
        "joint_map": cli.jsonable(asdict(adapter.joint_map)), "prim_path": str(robot.cfg.prim_path),
        "fields": fields, "actual_objects": {key: object_file(obj) for key, obj in objects.items()},
        "clock_before": before, "clock_after": clock(), "measurement_timing": "after ordinary reset/settle and before first policy action",
        "probe_setters_resets_steps_updates": 0}


def main(argv=None):
    cli.require(FULL12_ORDER[6:8] == ("rear_right_hip", "rear_right_knee"), "RR index mapping changed")
    args = cli.parser().parse_args(argv)
    cli.require(cli.validate_video_args(args) == "C", "intervention requires official frozen C checkpoint loader")
    cli.require(args.semantic_version == "v3" and args.experiment_id == "fsm_reference_p09_stable_v2",
                "only frozen natural task-window recipe is supported")
    original_build, original_loader = cli.build_video_core, cli.checkpoint_loader
    original_capture, original_write = cli.capture_semantic_video, cli.write_json
    state = {"core": None, "stream": None, "check_model": None, "last_masked": (0.,) * 12,
             "issued": 0, "history_checks": 0, "bounded_stop": False, "model_unchanged": None, "owned_source": None}

    def write(path, payload, **options):
        if options.get("replace"):
            cli.require(Path(path).resolve() == state["owned_source"], "replacement restricted to this capture's newly owned source")
        if Path(path).name in ("run_manifest.started.json", "run_manifest.json", "semantic_video_source_manifest.json"):
            payload = {**payload, "diagnostic_intervention": INTERVENTION}
        return original_write(path, payload, **options)

    def build(*positional, **keywords):
        state["core"] = original_build(*positional, **keywords)
        return state["core"]

    def loader(request, contract):
        official_load = original_loader(request, contract)
        def load(observation):
            actor_action, proof, unchanged = official_load(observation)
            core = state["core"]
            state["check_model"] = unchanged
            probe_path = request.run_dir / "source/live_preaction_drive_getters.json"
            original_write(probe_path, live_probe(core))
            state["stream"] = (request.run_dir / "source/rr_channels_off_decisions.jsonl").open("x", encoding="utf-8")
            def action(current, decision):
                actual_history = tuple(core._history["previous_raw_full12"])
                cli.require(actual_history == state["last_masked"], "HISTORY differs from actually dispatched masked raw")
                state["history_checks"] += 1
                if decision >= MAX_DECISIONS:
                    state["bounded_stop"] = True
                    raise DiagnosticBoundedWindow("DIAGNOSTIC_BOUNDED_WINDOW")
                raw = tuple(actor_action(current, decision))
                masked = raw[:6] + (0., 0.) + raw[8:]
                record = {"decision": decision + 1, "start_tick": core.backend._episode_tick,
                    "phase": core.frame.state_id, "actor_raw_full12": raw, "dispatched_raw_full12": masked,
                    "previous_raw_history_full12": actual_history, "diagnostic_only": True, "training_credit": False}
                state["stream"].write(json.dumps(record, allow_nan=False) + "\n")
                state["stream"].flush()
                state["last_masked"], state["issued"] = masked, decision + 1
                return masked
            return action, {**proof, "diagnostic_intervention": INTERVENTION, "live_probe": str(probe_path)}, unchanged
        return load

    def capture(*positional, **keywords):
        owned = (Path(keywords["output_directory"]) / "semantic_video_source_manifest.json").resolve()
        cli.require(owned == (Path(args.run_dir) / "source/semantic_video_source_manifest.json").resolve()
                    and not owned.exists(), "diagnostic annotation requires this fresh run's not-yet-created source")
        state["owned_source"] = owned
        try:
            result = original_capture(*positional, **keywords)
        finally:
            if state["stream"] is not None:
                state["stream"].close()
            if state["check_model"] is not None:
                state["check_model"]()
                state["model_unchanged"] = True
        cli.require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == WRAPPER["sha256"], "wrapper changed during run")
        cli.require(state["issued"] <= MAX_DECISIONS and (result.get("episode_physics_ticks") or 0) <= 1800,
                    "diagnostic window exceeded")
        result.update(diagnostic_only=True, success_candidate=False, improved_claim=False,
            diagnostic_intervention=INTERVENTION, diagnostic_bounded_stop=state["bounded_stop"],
            diagnostic_stop_reason="DIAGNOSTIC_BOUNDED_WINDOW" if state["bounded_stop"] else "EARLIER_REAL_ENDPOINT_OR_ERROR",
            masked_raw_history_checks=state["history_checks"], frozen_model_unchanged_after_capture=state["model_unchanged"],
            diagnostic_policy_decisions=state["issued"], on_policy_training_samples=0, formal_ppo_success=False,
            diagnostic_action_log=str(Path(keywords["output_directory"]) / "rr_channels_off_decisions.jsonl"),
            live_preaction_drive_probe=str(Path(keywords["output_directory"]) / "live_preaction_drive_getters.json"))
        write(owned, result, replace=True)  # Annotate only the source just created by original_capture.
        return result

    cli.build_video_core, cli.checkpoint_loader = build, loader
    cli.capture_semantic_video, cli.write_json = capture, write
    try:
        return cli.main(argv)
    finally:
        cli.build_video_core, cli.checkpoint_loader = original_build, original_loader
        cli.capture_semantic_video, cli.write_json = original_capture, original_write


if __name__ == "__main__":
    raise SystemExit(main())
