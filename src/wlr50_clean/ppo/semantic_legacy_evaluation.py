"""Fresh, read-only A control and common physical evidence for A/B/C."""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .isaac_fsm_backend import _member
from .semantic_metrics import SemanticMetricsAccumulator
from .semantic_supervisor import TaskEvaluator
from .semantic_training import verified_native_effect, write_json

ZERO12 = (0.0,) * 12
WRITE_COUNTERS = ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
                  "in_episode_force_or_impulse_writes", "in_episode_gravity_writes")


def physical_json(value: Any) -> Any:
    """Preserve invalid raw floats as explicit tokens, never fabricated zeros."""
    if is_dataclass(value):
        return physical_json(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): physical_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [physical_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_raw_float": str(value)}
    if value is None or isinstance(value, (str, bool, float, int)):
        return value
    raise TypeError(f"unsupported physical evidence {type(value).__name__}")


def measured_observation(raw: Any) -> dict[str, Any]:
    # Complete evaluator inputs, not the legacy compact stream which omitted
    # base position, IMU, joint velocities and exact pair active flags.
    fields = ("schema", "physics_tick", "simulation_time_s", "physics_dt_s",
              "all_finite", "base", "imu", "obstacle", "joints", "wheels",
              "contacts", "bodies", "center_of_mass", "support", "body_collision",
              "actual_full12", "commanded_full12", "data_quality")
    return {name: physical_json(_member(raw, name)) for name in fields}


def _line(stream, row):
    stream.write(json.dumps(physical_json(row), allow_nan=False, separators=(",", ":")) + "\n")


class PhysicalEvaluationRecorder:
    """Same independent evaluator; no controller success label is substituted."""
    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.evaluator = TaskEvaluator()
        self.metrics = SemanticMetricsAccumulator()
        self._streams = {name: (self.run_dir / name).open("x", encoding="utf-8") for name in (
            "physical_observations.jsonl", "stage_transition_evidence.jsonl", "native_tick_audit.jsonl")}
        self._started = False
        self._ticks = 0
        self._first_success_time = None
        self._last_frame = None
        self._history = None

    def start(self, initial_frame):
        if self._started:
            raise RuntimeError("one recorder owns one continuous physical episode")
        raw = initial_frame.info["raw_observation"]
        ev = self.evaluator.observe(raw)
        self._history = ev["history"]
        self._last_frame = initial_frame
        self._started = True
        _line(self._streams["physical_observations.jsonl"], measured_observation(raw))

    def observe(self, before, after, projection):
        if not self._started or before.physics_tick != self._last_frame.physics_tick:
            raise RuntimeError("physical recorder requires contiguous start/ticks")
        for key in WRITE_COUNTERS:
            if type(after.info.get(key)) is not int or after.info[key] != 0:
                raise RuntimeError(f"missing or forbidden in-episode state write evidence: {key}")
        native = after.info.get("actuator_target_effect_audit")
        if not isinstance(native, Mapping):
            raise RuntimeError("physical evaluation requires same-tick native audit")
        raw_action = tuple(native.get("raw_policy_action_full12", ()))
        verified_native_effect(after.info, raw_action)
        if native.get("source_phase_id") != before.state_id:
            raise RuntimeError("physical evaluation audit source phase mismatch")
        ev = self.evaluator.observe(after.info["raw_observation"])
        # No standing tail is allowed to dilute RMS after real completion.
        if self._first_success_time is None:
            self.metrics.observe(before, after, projection)
            if ev["success"]:
                self._first_success_time = after.sim_time_s
        _line(self._streams["physical_observations.jsonl"], measured_observation(after.info["raw_observation"]))
        _line(self._streams["native_tick_audit.jsonl"], {
            "episode_physics_tick": after.physics_tick, "source_phase_id": before.state_id,
            "nominal_full12": before.nominal_action_full12,
            "projected_residual_full12": projection.safe_projected_residual_full12,
            "native_audit": native, **{k: after.info[k] for k in WRITE_COUNTERS}})
        if before.state_id != after.state_id or ev["history"] != self._history:
            _line(self._streams["stage_transition_evidence.jsonl"], {
                "physics_tick": after.physics_tick, "sim_time_s": after.sim_time_s,
                "from_stage": before.state_id, "to_stage": after.state_id,
                "physical_history": ev["history"], "physical_goal_features": ev["goal_features"],
                "semantic_transition_evidence": after.info.get("semantic_task", {}).get("transition_evidence", []),
                "legacy_phase_label_alone_is_not_task_completion": True})
        self._history = ev["history"]
        self._last_frame = after
        self._ticks += 1
        if self._ticks % 64 == 0:
            for stream in self._streams.values():
                stream.flush()

    def summary(self):
        if not self._started:
            raise RuntimeError("cannot summarize an unstarted physical episode")
        quality = self.metrics.summary()
        path = self.run_dir / "phase_metrics.csv"
        rows = [{"phase": phase, **row} for phase, row in quality["phases"].items()]
        columns = ["phase"] + sorted(set().union(*(row.keys() for row in rows)) - {"phase"})
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        with (self.run_dir / "physics_quality_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            if self.metrics.rows:
                writer = csv.DictWriter(stream, fieldnames=list(self.metrics.rows[0]))
                writer.writeheader(); writer.writerows(self.metrics.rows)
        ev = self.evaluator.snapshot
        return {"task_success": bool(ev["success"]), "physical_task_evaluation": ev,
                "physical_task_duration_s": self._first_success_time if ev["success"] else self._last_frame.sim_time_s,
                "observed_physics_ticks": self._ticks, "quality_metrics": quality,
                "evaluation_artifacts": {name: str(self.run_dir / name) for name in (
                    *self._streams, "phase_metrics.csv", "physics_quality_metrics.csv")}}

    def close(self):
        for stream in self._streams.values():
            stream.close()


def _evaluation_legacy(app, args, contract):
    from .isaac_fsm_backend import IsaacFSMBackend
    from .residual_direct_env import ResidualEpisodeEnv
    backend = IsaacFSMBackend(app, audit_actuator_target_effect=True)
    core = ResidualEpisodeEnv(backend, collect_trace=False)
    core.reset(seed=args.seed)
    recorder = PhysicalEvaluationRecorder(args.run_dir)
    recorder.start(core.frame)
    core.tick_callback = recorder.observe
    last_info = {}
    count = 0
    try:
        with (args.run_dir / "residual_and_projection_audit.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.max_decisions):
                backend.set_actuator_target_audit_request(phase_id=core.frame.state_id,
                    raw_policy_action_full12=ZERO12, phase_mask_full12=core.phase_actions.mask_for(core.frame.state_id))
                step = core.step(ZERO12)
                last_info = step.info
                count += 1
                _line(stream, {"decision": index + 1, "phase_id": core.frame.state_id,
                    "sim_time_s": core.frame.sim_time_s, "raw_policy_action_full12": ZERO12,
                    "legacy_termination_reason": last_info.get("termination_reason"),
                    "actuator_target_effect_audit": last_info.get("actuator_target_effect_audit")})
                # Independent physical completion/failure ends the evaluation;
                # the original controller and all its guards remain untouched.
                if core.done or recorder.evaluator.snapshot["success"] or recorder.evaluator.snapshot["termination_reason"]:
                    break
        physical = recorder.summary()
        reason = physical["physical_task_evaluation"]["termination_reason"]
        if physical["task_success"]:
            reason = "SUCCESS"
        result = {"schema": "wlr50_clean.semantic_evaluation.v1", "mode": "legacy_fsm_eval",
            "seed": args.seed, "checkpoint": None, "deterministic_policy": False,
            "deterministic_legacy_controller": True, "from_phase": "P01", "policy_decisions": count,
            "duration_s": physical["physical_task_duration_s"],
            "legacy_controller_end_time_s": core.frame.sim_time_s,
            "legacy_controller_result": core.frame.info.get("controller_task_result"),
            "legacy_controller_reason": last_info.get("termination_reason"),
            "termination_reason": reason or ("LEGACY_ENDED_WITHOUT_PHYSICAL_TASK_COMPLETION" if core.done else "EVALUATION_WINDOW_END"),
            "window_ended_before_task_terminal": not physical["task_success"] and reason is None and not core.done,
            "runtime_contract": contract, "evaluation_seed_interpretation": "deterministic_repetition_without_randomization",
            "original_controller_and_physics_unmodified": True, **physical}
        write_json(args.run_dir / "evaluation_manifest.json", result)
        return result
    finally:
        recorder.close()
