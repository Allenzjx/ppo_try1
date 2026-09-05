"""Actual-time quality measurements shared by legacy A and semantic B/C.

No controller success label or reward is used to manufacture quality. Collection
stops at the caller's real episode end and excludes video padding. Phase metrics
are unavailable when unsampled, rather than assigned an artificially good zero.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .isaac_fsm_backend import _member
from .semantic_observation import _rpy, _quaternion
from .observation_schema_v2 import _quat_rotate_inverse
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.sensing.contact_classifier import SENSED_BODIES

PHASES = tuple(f"P{i:02}" for i in range(1, 14))
DEFAULT_SCORE_PATH = Path(__file__).resolve().parents[3] / "configs/ppo_semantic_v2/quality_score.yaml"
WHEEL_RADIUS_ESTIMATE_M = 0.04998999834060672
LEGS = ("FL", "FR", "RL", "RR")
DIAGNOSTIC_NOTES = {
    "motion": "Measured base-link velocity; whole-robot linear momentum is measured CoM velocity times measured total mass, not base-link velocity times mass.",
    "contact": "Exact verified ground/obstacle pairs. Events compare consecutive samples once, not overlapping history windows. Ground-to-obstacle transfer alone is not a new wheel touchdown.",
    "force_impulse": "Integral of *_force_n is N*s; it includes support and intentional propulsion, not only impact, and is not a universal badness cost. Signed world-force components retain direction.",
    "touchdown": "Pre-contact downward wheel-body velocity at a measured AIR-to-contact edge; not a resolved contact-point impact velocity.",
    "rebound": "Upward wheel-body speed at contact loss within 0.25 s of a measured touchdown; a diagnostic proxy that can also represent intentional lift.",
    "slip": "Estimate: nominal wheel radius times angular speed minus wheel-body forward velocity in chassis coordinates. Not ground-truth tangential contact slip; orientation, tyre deformation and wall contact can invalidate the rolling approximation. Reported only for verified contacting wheels.",
    "absence": "Unavailable fields are null with valid/missing tick counts; verified zero contact/event/force is a real zero. No absent phase or missing sensor is scored as perfect.",
    "aggregation": "Means, RMS and integrals use actual physics dt; all ticks are 120 Hz. Sum of *_event_count gives events, sum of *_change_* gives the signed measured change across observed intervals.",
}


def _norm(v):
    return math.sqrt(sum(float(x)**2 for x in v))


def _finite_number(value):
    if isinstance(value,bool):
        return None
    try:
        result = float(value)
    except (TypeError,ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3:
        return None
    values = tuple(_finite_number(x) for x in value)
    return None if any(x is None for x in values) else values


def _exact_pair(raw, body_name, pair_name):
    pair = _member(_member(_member(raw,"contacts",{}),body_name),pair_name)
    force = _finite_number(_member(pair,"normal_force_n"))
    if (_member(pair,"pair_verified") is not True or type(_member(pair,"active")) is not bool
            or _member(pair,"sensor_body") != body_name or force is None or force<0):
        return None
    return pair


def _motion_contact_diagnostics(raw0, raw1, q1, *, dt, now, last_touchdown):
    values = {}
    def vector_fields(stem, unit, value):
        for index,axis in enumerate("xyz"):
            values[f"{stem}_{axis}_{unit}"] = None if value is None else value[index]
    base0,base1 = _member(raw0,"base"),_member(raw1,"base")
    velocity0,velocity1 = (_finite_vector(_member(base,"linear_velocity_w_m_s")) for base in (base0,base1))
    vector_fields("body_linear_velocity_world","m_s",velocity1)
    vector_fields("body_linear_velocity_chassis","m_s",None if velocity1 is None else _quat_rotate_inverse(q1,velocity1))
    vector_fields("body_linear_velocity_change_world","m_s",None if velocity0 is None or velocity1 is None else tuple(b-a for a,b in zip(velocity0,velocity1)))
    values["body_linear_speed_m_s"] = None if velocity1 is None else _norm(velocity1)
    com0,com1 = _member(raw0,"center_of_mass"),_member(raw1,"center_of_mass")
    def com_state(com):
        velocity = _finite_vector(_member(com,"velocity_w_m_s"))
        mass = _finite_number(_member(com,"total_mass_kg"))
        if _member(com,"valid") is not True or velocity is None or mass is None or mass<=0:
            return None,None
        return velocity,tuple(mass*x for x in velocity)
    cv0,p0 = com_state(com0)
    cv1,p1 = com_state(com1)
    vector_fields("com_velocity_world","m_s",cv1)
    vector_fields("com_velocity_change_world","m_s",None if cv0 is None or cv1 is None else tuple(b-a for a,b in zip(cv0,cv1)))
    vector_fields("com_linear_momentum_world","kg_m_s",p1)
    vector_fields("com_linear_momentum_change_world","kg_m_s",None if p0 is None or p1 is None else tuple(b-a for a,b in zip(p0,p1)))
    values["com_speed_m_s"] = None if cv1 is None else _norm(cv1)
    values["com_linear_momentum_magnitude_kg_m_s"] = None if p1 is None else _norm(p1)
    bank = [_exact_pair(raw1,body,pair) for body in SENSED_BODIES for pair in ("ground","obstacle")]
    previous_bank = [_exact_pair(raw0,body,pair) for body in SENSED_BODIES for pair in ("ground","obstacle")]
    values["all_body_exact_active_pair_count"] = sum(_member(p,"active") for p in bank) if all(p is not None for p in bank) else None
    values["all_body_exact_normal_force_n"] = sum(_member(p,"normal_force_n") for p in bank) if all(p is not None for p in bank) else None
    values["all_body_exact_pair_toggle_event_count"] = sum(_member(a,"active")!=_member(b,"active") for a,b in zip(previous_bank,bank)) if all(p is not None for p in previous_bank+bank) else None
    for leg,name in zip(LEGS,WHEEL_ORDER,strict=True):
        wheel0,wheel1 = (_member(_member(raw,"wheels",{}),name) for raw in (raw0,raw1))
        body = _member(wheel1,"body_name")
        previous_body = _member(wheel0,"body_name")
        pairs0 = [_exact_pair(raw0,previous_body,p) for p in ("ground","obstacle")]
        pairs1 = [_exact_pair(raw1,body,p) for p in ("ground","obstacle")]
        current_valid = all(p is not None for p in pairs1)
        prior_valid = previous_body==body and all(p is not None for p in pairs0)
        active = any(_member(p,"active") for p in pairs1) if current_valid else None
        previous_active = any(_member(p,"active") for p in pairs0) if prior_valid else None
        touchdown = bool(active and not previous_active) if current_valid and prior_valid else None
        lost = bool(previous_active and not active) if current_valid and prior_valid else None
        if touchdown:
            last_touchdown[leg] = now
        if not current_valid or not prior_valid:
            last_touchdown[leg] = None
        v0,v1 = (_finite_vector(_member(_member(_member(raw,"bodies",{}),body_name),"linear_velocity_w_m_s"))
                 for raw,body_name in ((raw0,previous_body),(raw1,body)))
        omega = _finite_number(_member(wheel1,"velocity_rad_s"))
        prefix = f"{leg}_"
        values[prefix+"exact_active_pair_count"] = sum(_member(p,"active") for p in pairs1) if current_valid else None
        values[prefix+"contact_normal_force_n"] = sum(_member(p,"normal_force_n") for p in pairs1) if current_valid else None
        for pair_name,pair in zip(("ground","obstacle"),pairs1,strict=True):
            values[prefix+pair_name+"_normal_force_n"] = None if pair is None else _member(pair,"normal_force_n")
        force_vectors = [_finite_vector(_member(pair,"force_w_n")) for pair in pairs1]
        force_sum = tuple(sum(v[index] for v in force_vectors) for index in range(3)) if current_valid and all(v is not None for v in force_vectors) else None
        vector_fields(prefix+"contact_force_world","n",force_sum)
        values[prefix+"exact_pair_toggle_event_count"] = sum(_member(a,"active")!=_member(b,"active") for a,b in zip(pairs0,pairs1)) if current_valid and prior_valid else None
        values[prefix+"wheel_contact_toggle_event_count"] = int(active!=previous_active) if current_valid and prior_valid else None
        values[prefix+"touchdown_event_count"] = None if touchdown is None else int(touchdown)
        values[prefix+"contact_loss_event_count"] = None if lost is None else int(lost)
        values[prefix+"wheel_vertical_velocity_m_s"] = None if v1 is None else v1[2]
        values[prefix+"touchdown_descent_speed_m_s"] = max(0.,-v0[2]) if touchdown and v0 is not None else None
        values[prefix+"touchdown_normal_force_n"] = values[prefix+"contact_normal_force_n"] if touchdown else None
        recent = last_touchdown[leg] is not None and 0<=now-last_touchdown[leg]<=.25
        rebound = bool(lost and recent and v1 is not None and v1[2]>0) if lost is not None and v1 is not None else None
        values[prefix+"post_touchdown_rebound_event_count"] = None if rebound is None else int(rebound)
        values[prefix+"post_touchdown_rebound_upward_speed_m_s"] = v1[2] if rebound else None
        slip_valid = (active is True and v1 is not None and omega is not None
                      and _member(wheel1,"geometry_verified") is True)
        slip = WHEEL_RADIUS_ESTIMATE_M*omega-_quat_rotate_inverse(q1,v1)[0] if slip_valid else None
        values[prefix+"contact_rolling_slip_estimate_signed_m_s"] = slip
        values[prefix+"contact_rolling_slip_estimate_abs_m_s"] = None if slip is None else abs(slip)
    counts = [values[f"{leg}_exact_active_pair_count"] for leg in LEGS]
    values["wheel_contact_count"] = sum(count>0 for count in counts) if all(count is not None for count in counts) else None
    values["wheel_exact_active_pair_count"] = sum(counts) if all(count is not None for count in counts) else None
    return values


def _aggregate_diagnostics(rows):
    keys = sorted({key for row in rows for key in row.get("motion_contact_diagnostics",{})})
    result = {}
    for key in keys:
        samples = [(row["motion_contact_diagnostics"].get(key),row["dt_s"]) for row in rows]
        valid = [(value,dt) for value,dt in samples if value is not None]
        count = len(valid)
        stats = {"available":bool(count),"valid_ticks":count,"missing_or_inapplicable_ticks":len(rows)-count,
                 "valid_duration_s":sum(dt for _,dt in valid),"mean":None,"rms":None,"p95_abs":None,
                 "peak_abs":None,"first":None,"last":None,"integral":None}
        if valid:
            numbers = np.array([x for x,_ in valid],dtype=float)
            times = np.array([dt for _,dt in valid],dtype=float)
            stats.update(mean=float(np.dot(numbers,times)/times.sum()),
                         rms=float(np.sqrt(np.dot(numbers*numbers,times)/times.sum())),
                         p95_abs=float(np.percentile(np.abs(numbers),95)),peak_abs=float(np.abs(numbers).max()),
                         first=float(numbers[0]),last=float(numbers[-1]),integral=float(np.dot(numbers,times)))
            if key.endswith("_event_count") or "_change_" in key:
                stats["sum"] = float(numbers.sum())
        elif key.endswith("_event_count") or "_change_" in key:
            stats["sum"] = None
        result[key] = stats
    return result


class SemanticMetricsAccumulator:
    def __init__(self, score_path: Path | str = DEFAULT_SCORE_PATH):
        self.config = yaml.safe_load(Path(score_path).read_text(encoding="utf-8"))
        self.reset()

    def reset(self):
        self.rows: list[dict[str, Any]] = []
        self._previous_omega = None
        self._previous_applied_rate = None
        self._last_time = None
        self._last_touchdown = dict.fromkeys(LEGS)
        self._nonfinite_terminal_ticks_skipped = 0

    def observe(self, before: Any, after: Any, projection: Any) -> None:
        raw0, raw1 = before.info["raw_observation"], after.info["raw_observation"]
        if not bool(_member(raw1, "all_finite", False)):
            self._nonfinite_terminal_ticks_skipped += 1
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
        row["motion_contact_diagnostics"] = _motion_contact_diagnostics(
            raw0,raw1,q1,dt=dt,now=float(after.sim_time_s),last_touchdown=self._last_touchdown)
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float,int))):
            raise ValueError("quality data contains nonfinite values")
        self.rows.append(row)
        self._previous_applied_rate = action_rate
        self._last_time = after.sim_time_s

    @staticmethod
    def _aggregate(rows):
        if not rows:
            return {"sampled": False, "physics_ticks": 0, "duration_s": 0.0,"motion_contact_diagnostics":{}}
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
        result["motion_contact_diagnostics"] = _aggregate_diagnostics(rows)
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
                "score_is_not_success":True, "score_config":str(DEFAULT_SCORE_PATH),
                "diagnostic_schema":"wlr50_clean.semantic_motion_contact_metrics.v1",
                "diagnostic_notes":dict(DIAGNOSTIC_NOTES),
                "nonfinite_terminal_ticks_skipped":self._nonfinite_terminal_ticks_skipped}
