"""NOTEXECUTED: pure target planner only, no backend/Isaac/Torch imports or runner.

Requires a FUTURE audited same-tick post-mapper diagnostic hook. This file
cannot inject actions today and is not a production controller/capture assist.
"""
from dataclasses import dataclass
import math

RR_HIP, RR_KNEE = 6, 7
SELECTED = (RR_HIP, RR_KNEE)


def quintic(value):
    x = min(1., max(0., value))
    return x * x * x * (10. + x * (-15. + 6. * x))


@dataclass(frozen=True)
class ProbeConfig:
    hip_candidate_deg: float = -20.
    ramp_s: float = 3.
    maximum_intervention_s: float = 4.
    no_response_check_s: float = 1.
    minimum_actual_negative_response_deg: float = .25
    minimum_commanded_negative_change_deg: float = 1.

    def __post_init__(self):
        if not (0 < self.ramp_s <= self.maximum_intervention_s <= 4.
                and 0 < self.no_response_check_s < self.maximum_intervention_s
                and self.minimum_actual_negative_response_deg > 0.
                and self.minimum_commanded_negative_change_deg > 0.):
            raise ValueError("finite one-shot diagnostic configuration required")


def qualified_carry(evaluation, phase, minimum_other_supports, force_noise_floor_n):
    """Diagnostic trigger's CURRENT AIR-safe-drop region, never task success.

    Mirrors the AIR branch of the existing supervisor's geometric drop test,
    not the separate late FL/RL group's current-RR-bearing permission. The
    caller supplies the existing support specification, never a new threshold.
    """
    if (phase != "P09" or evaluation.get("valid") is not True
            or evaluation.get("termination_reason") is not None
            or evaluation.get("physical_evidence_status") not in
                ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")):
        return False
    if not math.isfinite(force_noise_floor_n) or force_noise_floor_n < 0.:
        raise ValueError("existing finite support noise floor required")
    legs = evaluation.get("current_legs", {})
    rr = legs.get("RR", {})
    def measured_bearing(row):
        force = row.get("bearing_force_n")
        return bool(row.get("support") is True and row.get("bearing_verified") is True
            and row.get("air") is False
            and (row.get("ground_contact") is True or row.get("top_surface_contact") is True)
            and isinstance(force, (int, float)) and math.isfinite(force)
            and force >= force_noise_floor_n)
    others = sum(measured_bearing(row) for leg, row in legs.items() if leg != "RR")
    distance, clearance = rr.get("front_distance_m"), rr.get("clearance_m")
    safe_drop_geometry = bool(rr.get("within_top_xy") is True
        and isinstance(distance, (int, float)) and math.isfinite(distance) and distance >= 0.
        and isinstance(clearance, (int, float)) and math.isfinite(clearance) and clearance > 0.)
    return bool(rr.get("current_lift_valid") is True
        and rr.get("motion_continuation_allowed") is True
        and rr.get("body_control_evidence") is True
        and rr.get("air") is True and rr.get("contact_mode") == "AIR"
        and rr.get("ground_contact") is False and not rr.get("obstacle_pair_active")
        and rr.get("within_lateral_span") is True and safe_drop_geometry
        and others >= minimum_other_supports)


class OneShotRRDirection:
    """Plans RR final targets; refuses rather than silently violating a hold.

    Context values must come from the unique real mapper pass, current physical
    evaluator and adjacent committed ACK. No second mapper or predicted N.
    """
    def __init__(self, config=ProbeConfig()):
        self.config = config
        self.state = "WAIT"
        self.anchor = None
        self.last_tick = None
        self.last_plan = None
        self.release_reason = None

    def _release(self, reason):
        self.state, self.release_reason = "RELEASED", reason
        return None, {"state": self.state, "release_reason": reason,
                      "continue_same_episode_with_original_policy": True,
                      "new_PPO_decisions": 0, "new_PPO_updates": 0}

    def _wait(self, reason):
        # No anchor/time/one-shot is consumed until the entire candidate is
        # locally expressible. Later ticks use their THEN-current real ACK.
        return None, {"state": "WAIT", "eligible": True, "reason": reason,
                      "one_shot_consumed": False,
                      "new_PPO_decisions": 0, "new_PPO_updates": 0}

    @staticmethod
    def _expressible(ctx, i, desired, *, check_step):
        mapped = ctx["same_tick_mapped_nominal_full12"][i] + ctx["controller_bias_full12"][i]
        value = desired - mapped
        lo, hi = ctx["headroom_residual_intervals_servo_deg"][i]
        hard_lo, hard_hi = ctx["servo_hard_limits_deg"][i]
        valid = (abs(value) < ctx["residual_caps_full12"][i]
                 and lo <= value <= hi and hard_lo <= desired <= hard_hi)
        if check_step:
            valid = (valid and abs(value - ctx["previous_effective_residual_full12"][i])
                <= ctx["residual_rate_deg_s"] * ctx["physics_dt_s"] + 1e-9
                and abs(desired - ctx["previous_final_full12"][i])
                <= ctx["final_slew_deg_per_tick"] + 1e-9)
        return valid, value

    def plan(self, ctx):
        tick = ctx["dispatch_tick"]
        if tick == self.last_tick:
            return self.last_plan
        if (ctx["same_tick_mapped_tick"] != tick or ctx["previous_ack_tick"] != tick - 1
                or self.last_tick is not None and tick <= self.last_tick):
            raise ValueError("same-tick unique mapper and adjacent actual ACK are mandatory")
        self.last_tick = tick
        if self.state == "RELEASED":
            self.last_plan = self._release(self.release_reason)
            return self.last_plan
        eligible = qualified_carry(ctx["evaluation"], ctx["phase"], ctx["minimum_other_supports"],
                                   ctx["force_noise_floor_n"])
        if ctx.get("physical_abort") or not eligible:
            self.last_plan = (self._release("physical_permission_or_contact_lost")
                if self.state == "ACTIVE" else (None, {"state": "WAIT", "eligible": False}))
            return self.last_plan
        vectors = ("previous_final_full12", "actual_full12", "same_tick_mapped_nominal_full12",
                   "controller_bias_full12", "policy_projected_residual_full12",
                   "previous_effective_residual_full12", "residual_caps_full12", "policy_raw_full12")
        for key in vectors:
            if len(ctx[key]) != 12 or not all(math.isfinite(x) for x in ctx[key]):
                raise ValueError("finite Full12 required for " + key)
        if not 0 < ctx["physics_dt_s"] <= 1 / 15:
            raise ValueError("positive actual bounded physics interval required")
        if self.state == "WAIT":
            before = ctx["previous_final_full12"]
            target = self.config.hip_candidate_deg
            lo, hi = ctx["servo_hard_limits_deg"][RR_HIP]
            if not lo <= target <= hi or target >= before[RR_HIP]:
                self.last_plan = self._wait("negative_absolute_candidate_not_locally_applicable")
                return self.last_plan
            # Admit the full ABSOLUTE endpoint and held knee under this tick's
            # actual mapped N/controller/caps/headroom, not merely the first
            # infinitesimal ramp step. This is a conservative diagnostic window,
            # not a new production pose gate or a promise of future reachability.
            endpoint = {RR_HIP: target, RR_KNEE: before[RR_KNEE]}
            if any(not self._expressible(ctx, i, endpoint[i], check_step=False)[0]
                   for i in SELECTED):
                self.last_plan = self._wait("candidate_endpoint_or_knee_hold_not_currently_expressible")
                return self.last_plan
            if any(not self._expressible(ctx, i, before[i], check_step=True)[0]
                   for i in SELECTED):
                self.last_plan = self._wait("initial_anchor_compensation_not_currently_expressible")
                return self.last_plan
            self.anchor = dict(start_time_s=ctx["sim_time_s"],
                previous_final_hip_deg=before[RR_HIP], knee_hold_deg=before[RR_KNEE],
                actual_hip_deg=ctx["actual_full12"][RR_HIP],
                owner_ids=tuple(ctx["rr_owner_ids"]), previous_ack_tick=ctx["previous_ack_tick"])
            self.state = "ACTIVE"
        age = ctx["sim_time_s"] - self.anchor["start_time_s"]
        if age < 0:
            raise ValueError("probe actual physics clock moved backwards")
        if tuple(ctx["rr_owner_ids"]) != self.anchor["owner_ids"]:
            self.last_plan = self._release("RR_source_owner_changed")
            return self.last_plan
        if age >= self.config.maximum_intervention_s:
            self.last_plan = self._release("finite_4s_intervention_complete")
            return self.last_plan
        hip = self.anchor["previous_final_hip_deg"] + quintic(age / self.config.ramp_s) * (
            self.config.hip_candidate_deg - self.anchor["previous_final_hip_deg"])
        desired = {RR_HIP: hip, RR_KNEE: self.anchor["knee_hold_deg"]}
        changed = self.anchor["previous_final_hip_deg"] - hip
        response = self.anchor["actual_hip_deg"] - ctx["actual_full12"][RR_HIP]
        if (age >= self.config.no_response_check_s
                and changed >= self.config.minimum_commanded_negative_change_deg
                and response < self.config.minimum_actual_negative_response_deg):
            self.last_plan = self._release("commanded_negative_hip_without_measured_negative_response")
            return self.last_plan
        injected = list(ctx["policy_projected_residual_full12"])
        required = {}
        for i in SELECTED:
            # m+c is the same-tick mapped nominal INCLUDING bounded controller
            # correction. A previous-dispatch m is not acceptable here.
            expressible, value = self._expressible(ctx, i, desired[i], check_step=True)
            if not expressible:
                self.last_plan = self._release("required_compensation_saturates_cap_headroom_rate_or_final_slew")
                return self.last_plan
            required[i] = value
            injected[i] = value
        receipt = {"state": "ACTIVE", "kind": "finite_post_mapper_direction_diagnostic_not_PPO",
            "age_s": age, "anchor": dict(self.anchor), "desired_final_servo_deg": desired,
            "required_selected_residual_deg": required,
            "original_policy_raw_full12": tuple(ctx["policy_raw_full12"]),
            "original_policy_log_probability": ctx.get("policy_log_probability"),
            "original_filtered_policy_residual_full12": tuple(ctx["policy_projected_residual_full12"]),
            "diagnostic_injected_residual_full12": tuple(injected),
            "diagnostic_injected_raw": None, "injected_raw_semantics": "not_a_new_Gaussian_sample",
            "same_tick_mapped_tick": tick, "changed_channels": list(SELECTED),
            "remaining_ten_channels_exact_original_request": all(
                injected[i] == ctx["policy_projected_residual_full12"][i] for i in range(12) if i not in SELECTED),
            "post_write_ACK_verification_still_required": True,
            "new_PPO_decisions": 0, "new_PPO_updates": 0}
        self.last_plan = tuple(injected), receipt
        return self.last_plan
