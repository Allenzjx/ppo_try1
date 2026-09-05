"""Actual-time quality measurements shared by legacy A and semantic B/C.

No controller success label or reward is used to manufacture quality. Collection
stops at the caller's real episode end and excludes video padding. Phase metrics
are unavailable when unsampled, rather than assigned an artificially good zero.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .isaac_fsm_backend import _member
from .semantic_observation import _rpy, _quaternion
from .observation_schema_v2 import _quat_rotate_inverse

PHASES = tuple(f"P{i:02}" for i in range(1, 14))
DEFAULT_SCORE_PATH = Path(__file__).resolve().parents[3] / "configs/ppo_semantic_v2/quality_score.yaml"


def _norm(v):
    return math.sqrt(sum(float(x)**2 for x in v))


class SemanticMetricsAccumulator:
    def __init__(self, score_path: Path | str = DEFAULT_SCORE_PATH):
        self.config = yaml.safe_load(Path(score_path).read_text(encoding="utf-8"))
        self.reset()

    def reset(self):
        self.rows: list[dict[str, Any]] = []
        self._previous_omega = None
        self._previous_applied_rate = None
        self._last_time = None

    def observe(self, before: Any, after: Any, projection: Any) -> None:
        raw0, raw1 = before.info["raw_observation"], after.info["raw_observation"]
        if not bool(_member(raw1, "all_finite", False)):
            return  # Nonfinite terminal classified separately, never fabricated.
        dt = float(after.sim_time_s) - float(before.sim_time_s)
        if not math.isclose(dt, 1 / 120, abs_tol=1e-10):
            raise ValueError("quality sample must represent one real physics tick")
        if self._last_time is not None and not math.isclose(before.sim_time_s, self._last_time, abs_tol=1e-10):
            raise ValueError("quality samples must be continuous within an episode")
        base0, base1 = _member(raw0, "base"), _member(raw1, "base")
        q0, q1 = (_quaternion(_member(base, "orientation_wxyz")) for base in (base0, base1))
        rpy0, rpy1 = _rpy(q0), _rpy(q1)
        rates = tuple(math.atan2(math.sin(a-b), math.cos(a-b))/dt for a,b in zip(rpy1[:2], rpy0[:2]))
        omega0 = _quat_rotate_inverse(q0, _member(base0, "angular_velocity_w_rad_s"))
        omega1 = _quat_rotate_inverse(q1, _member(base1, "angular_velocity_w_rad_s"))
        acceleration = tuple((a-b)/dt for a,b in zip(omega1, omega0))
        ack0, ack1 = before.info["atomic_ack"], after.info["atomic_ack"]
        actual0, actual1 = ack0["drive_target_full12"], ack1["drive_target_full12"]
        action_rate = tuple((a-b)/dt for a,b in zip(actual1, actual0))
        action_acceleration = ((0.,)*12 if self._previous_applied_rate is None else
                               tuple((a-b)/dt for a,b in zip(action_rate, self._previous_applied_rate)))
        nominal_rate = tuple((a-b)/dt for a,b in zip(after.nominal_action_full12, before.nominal_action_full12))
        residual = tuple(projection.safe_projected_residual_full12)
        task = before.info.get("semantic_task", {})
        row = {
            "sim_time_s": after.sim_time_s, "dt_s": dt, "phase": before.state_id,
            "substage": task.get("substage", "LEGACY_UNSEGMENTED"),
            "roll_rad": rpy1[0], "pitch_rad": rpy1[1],
            "roll_rate_rad_s": rates[0], "pitch_rate_rad_s": rates[1],
            "body_x_omega_rad_s": omega1[0], "body_y_omega_rad_s": omega1[1],
            "angular_acceleration_rad_s2": _norm(acceleration),
            "applied_servo_rate_deg_s": _norm(action_rate[:8])/math.sqrt(8),
            "applied_wheel_rate_rad_s2": _norm(action_rate[8:])/2,
            "applied_servo_acceleration_deg_s2": _norm(action_acceleration[:8])/math.sqrt(8),
            "applied_wheel_acceleration_rad_s3": _norm(action_acceleration[8:])/2,
            "nominal_servo_rate_deg_s": _norm(nominal_rate[:8])/math.sqrt(8),
            "nominal_wheel_rate_rad_s2": _norm(nominal_rate[8:])/2,
            "residual_servo_deg": _norm(residual[:8])/math.sqrt(8),
            "residual_wheel_rad_s": _norm(residual[8:])/2,
        }
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float,int))):
            raise ValueError("quality data contains nonfinite values")
        self.rows.append(row)
        self._previous_applied_rate = action_rate
        self._last_time = after.sim_time_s

    @staticmethod
    def _aggregate(rows):
        if not rows:
            return {"sampled": False, "physics_ticks": 0, "duration_s": 0.0}
        time = np.array([r["dt_s"] for r in rows]); total = float(time.sum())
        result = {"sampled": True, "physics_ticks": len(rows), "duration_s": total}
        fields = [k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ("sim_time_s", "dt_s")]
        for field in fields:
            values = np.array([r[field] for r in rows], dtype=float)
            stem, unit = field.rsplit("_",1) if field.endswith("_deg") else (field, "")
            # Insert statistic before physical units to keep fixed-score names readable.
            suffix = next((u for u in ("rad_s3","rad_s2","rad_s","deg_s2","deg_s","rad","deg") if field.endswith("_"+u)), "")
            stem = field[:-(len(suffix)+1)] if suffix else field
            label = "_"+suffix if suffix else ""
            result[stem+"_rms"+label] = float(np.sqrt(np.sum(values*values*time)/total))
            result[stem+"_p95"+label] = float(np.percentile(np.abs(values),95))
            result[stem+"_peak"+label] = float(np.max(np.abs(values)))
        return result

    def summary(self) -> dict[str, Any]:
        by_phase = {phase: self._aggregate([r for r in self.rows if r["phase"]==phase]) for phase in PHASES}
        by_substage = {phase: {sub: self._aggregate([r for r in self.rows if r["phase"]==phase and r["substage"]==sub])
                              for sub in ("TRANSFER", "CAPTURE", "EXECUTION", "LEGACY_UNSEGMENTED")}
                       for phase in PHASES}
        score = 0.0
        available = all(row["sampled"] for row in by_phase.values())
        for phase, row in by_phase.items():
            row["quality_score"] = None
            if row["sampled"]:
                row["quality_score"] = sum(row[name]*cfg["weight"]/cfg["scale"] for name,cfg in self.config["components"].items())
                score += row["quality_score"]*self.config["phase_weights"][phase]
        return {"schema":"wlr50_clean.semantic_quality_metrics.v1", "global": self._aggregate(self.rows),
                "phases":by_phase, "transfer_capture_split":by_substage,
                "fixed_quality_score":score if available else None,
                "all_phases_sampled":available, "comparison_window":"actual_physics_ticks_excluding_video_padding",
                "score_is_not_success":True, "score_config":str(DEFAULT_SCORE_PATH)}
