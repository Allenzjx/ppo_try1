"""Measured post-RR front-pair preparation and RL continuation task.

The accepted RR task is delegated unchanged. This wrapper first latches after
an already-observed, qualified real TOP sample; consumers can therefore act
only on a subsequent control update. It writes no targets or physical state.
Preparation is a disclosed control candidate elsewhere, never a claim that
these task predicates or the learned policy generated its commands.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

from .semantic_rr_capture_continuation_task import RRCaptureContinuationTask
from .semantic_rr_capture_local_task import _finite, _vector12

SCHEMA = "wlr50_clean.post_rr_front_prep_task.v2"
STAGES = ("POST_RR_FRONT_PREP", "FL_RECOIL_TRANSFER_TO_FR", "RL_SWING_CAPTURE", "FINISH")
PREP_INDICES = (1, 3, 8, 9)
POST_RR_FIELDS = (
    "post_rr_active", "post_rr_preparing", "post_rr_transferring", "post_rr_swinging",
    "post_rr_finishing", "post_rr_prep_age_norm", "post_rr_prep_actual_progress",
    "post_rr_prep_tracking", "post_rr_prepared", "post_rr_support_usable",
    "post_rr_fixed_fr_direction_x", "post_rr_fixed_fr_direction_y",
    "post_rr_fixed_fr_com_progress", "post_rr_prep_exhausted",
    "entry_request_ratio_FL_knee", "entry_request_ratio_FR_knee",
    "entry_request_ratio_FL_wheel", "entry_request_ratio_FR_wheel",
    "entry_final_norm_FL_knee", "entry_final_norm_FR_knee",
    "entry_final_norm_FL_wheel", "entry_final_norm_FR_wheel",
    "finish_settle_pending", "finish_endpoint_missed", "finish_clean_elapsed_norm",
)
POST_RR_OBSERVATION_DIM = len(POST_RR_FIELDS)


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _vec(value, size=3):
    if isinstance(value, (list, tuple)) and len(value) == size and all(map(_finite, value)):
        return tuple(map(float, value))
    return None


def _clip(value):
    return min(1., max(0., float(value))) if _finite(value) else 0.


def _info(source):
    source = source.frame if hasattr(source, "frame") else source
    return source if isinstance(source, Mapping) else source.info


@dataclass(frozen=True)
class PostRRFrontPrepTaskConfig:
    candidate_knee_deg: float = 30.
    candidate_wheel_rad_s: float = 1.
    knee_rate_deg_s: float = 30.
    wheel_rate_rad_s2: float = 1.8
    prep_diagnostic_window_s: float = 8.
    minimum_actual_fraction: float = .8
    tracking_scale_deg: float = 20.
    minimum_tracking_fraction: float = .25
    meaningful_wheel_response_rad_s: float = .05
    com_progress_scale_m: float = .06

    def __post_init__(self):
        if not all(_finite(x) and x > 0. for x in (
                self.prep_diagnostic_window_s, self.tracking_scale_deg,
                self.meaningful_wheel_response_rad_s, self.com_progress_scale_m,
                self.knee_rate_deg_s, self.wheel_rate_rad_s2)):
            raise ValueError("post-RR task scales must be positive finite")
        if (not _finite(self.candidate_knee_deg) or not 20. <= self.candidate_knee_deg <= 40.
                or not _finite(self.candidate_wheel_rad_s) or not 0. < self.candidate_wheel_rad_s <= 1.5
                or self.knee_rate_deg_s > 60. or self.wheel_rate_rad_s2 > 1.8
                or not .5 <= self.minimum_actual_fraction < 1.
                or not 0. < self.minimum_tracking_fraction < 1.):
            raise ValueError("invalid bounded front preparation candidate")


class PostRRFrontPrepTask:
    """Read-only wrapper with unchanged accepted obs16 and a visible 25 tail.

    ``active`` retains the old RR branch gate; ``post_rr_active`` is the new
    learner gate. Do not give the new optimizer prefix credit using ``active``.
    All entry offsets come from the committed native ACK, not a counterfactual
    nominal reconstruction. A saturated inverse is rejected, never clamped.
    """
    def __init__(self, config=None, *, accepted_task=None, continuation_config=None,
                 phase_caps_full12=None):
        self.post_config = PostRRFrontPrepTaskConfig() if config is None else config
        if not isinstance(self.post_config, PostRRFrontPrepTaskConfig):
            raise ValueError("validated post-RR task config required")
        if accepted_task is not None and continuation_config is not None:
            raise ValueError("provide accepted task or continuation config, not both")
        self.accepted = accepted_task or RRCaptureContinuationTask(continuation_config)
        if not isinstance(self.accepted, RRCaptureContinuationTask):
            raise ValueError("accepted RR continuation task required")
        self.phase_caps_full12 = deepcopy(phase_caps_full12)
        if not isinstance(self.phase_caps_full12, Mapping) or not self.phase_caps_full12:
            raise ValueError("actual phase_caps_full12 are required for entry coordinates")
        for caps in self.phase_caps_full12.values():
            if any(x <= 0. for x in _vector12(caps)):
                raise ValueError("strictly positive action capacities required")
        self.reset()

    def __getattr__(self, name):
        return getattr(self.accepted, name)

    def reset(self):
        self.accepted.reset()
        self.post_rr_active = self.post_rr_prepared = self.post_rr_prep_exhausted = False
        self.post_rr_stage = None
        self.post_rr_touch_tick = self.post_rr_touch_time_s = None
        self.entry_final = self.entry_residual = self.entry_caps = self.entry_raw_reference = None
        self.entry_provenance = self.entry_evidence = None
        self.evidence = None
        self.prep_progress = self.prep_tracking = self.prep_locked_potential = 0.
        self.prep_stall_reason = None
        self.transfer_reference = None
        self.transfer_progress_m = None
        self.transfer_locked_potential = None
        self.rl_events = dict(rl_lift_seen=False, rl_cross_seen=False, rl_touch_seen=False)
        self._last_observed_tick = None
        self._last_read = None

    def obs9(self):
        return self.accepted.obs9()

    def obs16(self):
        return self.accepted.obs16()

    def _read(self, source):
        info = _info(source)
        ev = info["semantic_task"]["physical_evaluator"]
        raw = info.get("raw_observation")
        actual = _vec(_get(raw, "actual_full12"), 12)
        final = _vector12(info.get("drive_target_full12", info.get("actual_drive_target_full12")))
        role = (ev.get("transfer_roles") or {}).get("RL") or {}
        audit = info.get("actuator_target_effect_audit") or {}
        workspace = role.get("receiver_workspace_state") or {}
        role_valid = bool(role.get("valid") is True and role.get("diagonal_receiving_side") == "FR")
        com = _get(raw, "center_of_mass")
        com_position = _vec(_get(com, "position_w_m")) if _get(com, "valid") is True else None
        wheels = _get(raw, "wheels", {})
        fr_center = _vec(_get(_get(wheels, "front_right_ankle"), "center_w_m"))
        rl = (ev.get("current_legs") or {}).get("RL") or {}
        # Edge/leg contact alone is legal and is not a blocked-space verdict.
        rl_blocked = rl.get("unrecoverable_obstruction") is True
        return dict(actual=actual, final=final, com=com_position, fr_center=fr_center,
            finish_settle_pending=ev.get("finish_settle_pending") is True,
            finish_endpoint_missed=ev.get("finish_endpoint_missed") is True,
            finish_clean_elapsed_norm=_clip(ev.get("finish_clean_elapsed_norm", 0.)),
            receiver_role_valid=role_valid,
            receiver_workspace=_clip(role.get("workspace_progress")) if role_valid else None,
            receiver_geometry=_vec(workspace.get("wheel_relative_body_m")) if role_valid else None,
            receiver_contraction_m=workspace.get("radial_contraction_m") if role_valid else None,
            rl_available=bool(rl), rl_blocked=rl_blocked,
            rl_obstacle_contact=rl.get("obstacle_pair_active") is True,
            rl_gap_m=rl.get("clearance_m") if _finite(rl.get("clearance_m")) else None,
            ack=deepcopy(info.get("atomic_ack")),
            request_phase=audit.get("policy_request_phase"), source_phase=audit.get("source_phase_id"),
            audit_command_tick=audit.get("physics_tick"),
            stage_id=info["semantic_task"].get("stage_id"))

    def _activate(self, read):
        ack = read["ack"]
        if not isinstance(ack, Mapping):
            raise ValueError("post-RR entry requires the committed atomic ACK")
        final = _vector12(ack.get("drive_target_full12"))
        if final != read["final"]:
            raise ValueError("post-RR entry FINAL differs from atomic ACK")
        if type(ack.get("physics_tick")) is not int or ack["physics_tick"] < 0:
            raise ValueError("post-RR entry requires a real ACK command tick")
        if read["audit_command_tick"] is not None and read["audit_command_tick"] != ack["physics_tick"]:
            raise ValueError("post-RR entry ACK and native audit clocks disagree")
        requested = _vector12(ack.get("independent_policy_residual_requested_full12"))
        phase = read["request_phase"] or read["source_phase"] or read["stage_id"]
        if phase not in self.phase_caps_full12:
            raise ValueError("post-RR entry request phase has no bound capacities")
        caps = _vector12(self.phase_caps_full12[phase])
        references = [0.] * 12
        for i in PREP_INDICES:
            ratio = requested[i] / caps[i]
            if not -1. < ratio < 1.:
                raise ValueError("post-RR entry residual inverse is saturated or outside capacity")
            references[i] = math.atanh(ratio)
        if read["actual"] is None:
            raise ValueError("post-RR entry requires actual joint/wheel measurements")
        self.post_rr_active = True
        self.post_rr_stage = STAGES[0]
        self.post_rr_touch_tick = self.metrics["tick"]
        self.post_rr_touch_time_s = self.metrics["time_s"]
        self.entry_final, self.entry_residual, self.entry_caps = final, requested, caps
        self.entry_raw_reference = tuple(references)
        self.entry_evidence = deepcopy(read)
        self.entry_provenance = dict(phase=phase, request_phase=read["request_phase"],
            source_phase=read["source_phase"], observation_tick=self.metrics["tick"],
            ack_command_tick=ack.get("physics_tick"),
            request_source="atomic_ack.independent_policy_residual_requested_full12",
            final_source="atomic_ack.drive_target_full12", capacities_source="bound_execution_profile",
            raw_reference_formula="atanh(committed_requested_residual / bound_phase_capacity)",
            inverse_clamping=False, simulator_state_writes=0)

    def _set_transfer_reference(self):
        read = self.evidence
        if self.transfer_reference is not None or not (read and read["com"] and read["fr_center"]):
            return
        delta = tuple(read["fr_center"][i] - read["com"][i] for i in range(2))
        length = math.hypot(*delta)
        if length <= 1.e-9:
            return
        self.transfer_reference = dict(tick=self.metrics["tick"], time_s=self.metrics["time_s"],
            com_world_m=list(read["com"]), receiver_world_m=list(read["fr_center"]),
            direction_world=[delta[0]/length, delta[1]/length, 0.], receiver="FR")

    def _update_prep(self, previous):
        read, cfg = self.evidence, self.post_config
        actual = read["actual"]
        reasons = []
        if actual is None:
            self.prep_progress = self.prep_tracking = 0.
            reasons.append("actual_joint_or_wheel_measurement_unavailable")
        else:
            entry = self.entry_evidence["actual"]
            fractions = [1.-abs(cfg.candidate_knee_deg-actual[i]) /
                         max(15., abs(cfg.candidate_knee_deg-entry[i])) for i in (1, 3)]
            self.prep_progress = min(map(_clip, fractions))
            self.prep_tracking = _clip(1.-max(abs(actual[i]-read["final"][i]) for i in (1, 3))
                                       / cfg.tracking_scale_deg)
            if self.prep_progress < cfg.minimum_actual_fraction:
                reasons.append("front_knee_actual_travel_not_yet_effective")
            if self.prep_tracking < cfg.minimum_tracking_fraction:
                reasons.append("front_knee_final_to_actual_tracking_lag")
            if any(actual[i] < cfg.meaningful_wheel_response_rad_s or read["final"][i] <= 0.
                   for i in (8, 9)):
                reasons.append("front_wheel_forward_request_or_measured_rotation_missing")
        geometry = bool(read["receiver_role_valid"] and read["receiver_geometry"] is not None)
        if not geometry:
            reasons.append("FR_receiver_geometry_unavailable")
        elif (read["receiver_workspace"] + .02 < (self.entry_evidence["receiver_workspace"] or 0.)
              and not (_finite(read["receiver_contraction_m"]) and read["receiver_contraction_m"] > 0.)):
            reasons.append("FR_receiver_geometry_deteriorated")
        rr = self.metrics
        recovering = bool(previous and previous.get("gap_m") is not None and rr["gap_m"] is not None
            and rr["current_attempt_capture_eligible"] and rr["within_top_xy"]
            and rr["within_lateral_span"] and rr["gap_m"] < previous["gap_m"]-1.e-5)
        rr_usable = self.accepted.rr_support_continuation_valid or rr["current_top_bearing"]
        if not (rr_usable or recovering):
            reasons.append("RR_outside_contact_grace_without_current_gap_recovery")
        if not read["rl_available"] or read["rl_blocked"]:
            reasons.append("RL_space_unavailable_or_explicitly_unrecoverable")
        self.prep_stall_reason = reasons[0] if reasons else None
        if not reasons:
            self.post_rr_prepared = True
            self.prep_locked_potential = .2*self.prep_progress
            self.post_rr_stage = STAGES[1]
            self._set_transfer_reference()
        age = self.metrics["time_s"]-self.post_rr_touch_time_s
        self.post_rr_prep_exhausted |= bool(not self.post_rr_prepared and age >= cfg.prep_diagnostic_window_s)

    def observe(self, source):
        read = self._read(source)
        tick = _info(source)["semantic_task"]["physical_evaluator"]["physics_tick"]
        if tick == self._last_observed_tick:
            if read != self._last_read:
                raise ValueError("conflicting post-RR evidence at same observation tick")
            self.accepted.observe(source)
            return self.snapshot()
        previous = deepcopy(self.metrics)
        self.accepted.observe(source)
        self._last_observed_tick, self._last_read = tick, deepcopy(read)
        self.evidence = read
        if (not self.post_rr_active and self.accepted.rr_touch_seen
                and self.metrics["current_top_contact"] and self.metrics["current_attempt_capture_eligible"]):
            self._activate(read)
        if not self.post_rr_active:
            return self.snapshot()
        if self.post_rr_stage == STAGES[0]:
            self._update_prep(previous)
        old = self.accepted.snapshot()
        rl = old.get("rl_metrics") or {}
        if self.metrics["live"] and rl.get("current_qualified"):
            self.rl_events["rl_lift_seen"] |= rl.get("current_air_lift_valid") is True
            self.rl_events["rl_cross_seen"] |= bool(rl.get("crossed") and rl.get("within_top_xy"))
            self.rl_events["rl_touch_seen"] |= bool(rl.get("crossed") and rl.get("current_top_contact")
                and rl.get("consecutive_top_samples", 0) >= self.accepted.config.minimum_top_samples)
        if self.post_rr_stage != STAGES[0]:
            self._set_transfer_reference()
        if self.transfer_reference and read["com"]:
            self.transfer_progress_m = sum((read["com"][i]-self.transfer_reference["com_world_m"][i])
                * self.transfer_reference["direction_world"][i] for i in range(3))
        if self.rl_events["rl_lift_seen"] and self.post_rr_stage in STAGES[:2]:
            if self.transfer_locked_potential is None:
                self.transfer_locked_potential = .2*_clip(
                    (self.transfer_progress_m or 0.)/self.post_config.com_progress_scale_m)
            self.post_rr_stage = STAGES[2]
        if self.rl_events["rl_touch_seen"]:
            self.post_rr_stage = STAGES[3]
        return self.snapshot()

    def potential_components(self):
        result = dict(front_preparation=0., RR_current_support=0., transfer_to_FR=0.,
                      RL_current_lift=0., RL_cross=0., RL_current_contact=0., full_finish=0.)
        if not self.post_rr_active or not self.metrics or not self.metrics["live"]:
            return result
        old = self.accepted.snapshot()
        rl = old.get("rl_metrics") or {}
        result["front_preparation"] = (self.prep_locked_potential if self.post_rr_prepared
                                       else .2*self.prep_progress)
        result["RR_current_support"] = .1*float(self.metrics["current_top_bearing"])
        if self.transfer_locked_potential is not None:
            result["transfer_to_FR"] = self.transfer_locked_potential
        elif self.transfer_reference is not None:
            result["transfer_to_FR"] = .2*_clip((self.transfer_progress_m or 0.)/self.post_config.com_progress_scale_m)
        if rl.get("current_qualified"):
            result["RL_current_lift"] = .2
            result["RL_cross"] = .1*float(bool(rl.get("crossed") and rl.get("within_top_xy")))
            result["RL_current_contact"] = .1*float(bool(rl.get("crossed") and rl.get("current_top_contact")))
        result["full_finish"] = .1*float(old["full_task_success"])
        return result

    def post_rr_observation(self):
        if not self.post_rr_active:
            return (0.,)*POST_RR_OBSERVATION_DIM
        direction = self.transfer_reference["direction_world"] if self.transfer_reference else (0., 0., 0.)
        values = [1., *[float(self.post_rr_stage == s) for s in STAGES],
            _clip((self.metrics["time_s"]-self.post_rr_touch_time_s)/self.post_config.prep_diagnostic_window_s),
            self.prep_progress, self.prep_tracking, float(self.post_rr_prepared),
            float(self.accepted.rr_support_continuation_valid), direction[0], direction[1],
            (self.transfer_locked_potential/.2 if self.transfer_locked_potential is not None
             else max(-1., min(1., (self.transfer_progress_m or 0.)/self.post_config.com_progress_scale_m))),
            float(self.post_rr_prep_exhausted)]
        values += [self.entry_residual[i]/self.entry_caps[i] for i in PREP_INDICES]
        values += [self.entry_final[i]/(180. if i < 8 else 2.0943951023931953) for i in PREP_INDICES]
        # These states change permissible finish timing, not actuator targets.
        # Keep them explicit for both actor and critic; never hide a retry clock
        # behind a saturated old phase-progress value. The existing22 stay put.
        values += [float(self.evidence["finish_settle_pending"]),
                   float(self.evidence["finish_endpoint_missed"]),
                   self.evidence["finish_clean_elapsed_norm"]]
        return tuple(values)

    def snapshot(self):
        old = self.accepted.snapshot()
        parts = self.potential_components()
        old.update(schema=SCHEMA, accepted_task_schema=old["schema"],
            post_rr_active=self.post_rr_active, post_rr_stage=self.post_rr_stage,
            post_rr_touch_tick=self.post_rr_touch_tick, post_rr_touch_time_s=self.post_rr_touch_time_s,
            post_rr_preparing=self.post_rr_stage == STAGES[0], post_rr_prepared=self.post_rr_prepared,
            post_rr_prep_exhausted=self.post_rr_prep_exhausted,
            post_rr_prep_stall_reason=self.prep_stall_reason,
            post_rr_prep_actual_progress=self.prep_progress, post_rr_prep_tracking=self.prep_tracking,
            post_rr_candidate_knee_deg=self.post_config.candidate_knee_deg,
            post_rr_candidate_wheel_rad_s=self.post_config.candidate_wheel_rad_s,
            post_rr_knee_rate_deg_s=self.post_config.knee_rate_deg_s,
            post_rr_wheel_rate_rad_s2=self.post_config.wheel_rate_rad_s2,
            post_rr_entry_final_full12=self.entry_final, post_rr_entry_residual_full12=self.entry_residual,
            post_rr_entry_actual_full12=(self.entry_evidence["actual"] if self.entry_evidence else None),
            post_rr_entry_capacities_full12=self.entry_caps,
            post_rr_entry_raw_reference_full12=self.entry_raw_reference,
            post_rr_entry_provenance=deepcopy(self.entry_provenance),
            post_rr_fixed_fr_reference=deepcopy(self.transfer_reference),
            post_rr_fixed_fr_com_progress_m=self.transfer_progress_m,
            post_rr_retired_transfer_potential=self.transfer_locked_potential,
            post_rr_transfer_observation_semantics="signed_live_progress_then_retired_potential_fraction_after_RL_lift",
            post_rr_measured_front=deepcopy({k: v for k, v in (self.evidence or {}).items() if k != "ack"}),
            post_rr_obs=self.post_rr_observation(), post_rr_potential=sum(parts.values()),
            finish_settle_pending=bool(self.evidence and self.evidence["finish_settle_pending"]),
            finish_endpoint_missed=bool(self.evidence and self.evidence["finish_endpoint_missed"]),
            finish_clean_elapsed_norm=(self.evidence["finish_clean_elapsed_norm"] if self.evidence else 0.),
            post_rr_potential_components=parts, post_rr_events=dict(self.rl_events),
            post_rr_prep_timeout_is_terminal=False, physical_state_or_target_writes=0)
        return old

    def read_local_state(self):
        return self.snapshot()

    def reward(self, before, *, termination_reason=None):
        if not isinstance(before, Mapping) or before.get("schema") != SCHEMA:
            raise ValueError("post-RR reward requires its decision-start snapshot")
        # Existing global/safety terminal validation remains authoritative.
        legacy_before = dict(before, schema=before["accepted_task_schema"])
        legacy_reward = self.accepted.reward(legacy_before, termination_reason=termination_reason)
        after = self.snapshot()
        credited = before.get("post_rr_active") is True
        previous = before.get("post_rr_potential")
        if not _finite(previous) or not 0. <= previous <= 1.+1.e-12:
            raise ValueError("invalid post-RR decision-start potential")
        current = 0. if legacy_reward["terminated"] else after["post_rr_potential"]
        cfg = self.accepted.config
        weights = dict(rl_lift_seen=cfg.rl_lift_reward, rl_cross_seen=cfg.rl_cross_reward,
                       rl_touch_seen=cfg.rl_touch_reward)
        events = {key: weight if self.rl_events[key] and not before["post_rr_events"][key] else 0.
                  for key, weight in weights.items()}
        shaping = cfg.potential_weight*(cfg.gamma*current-previous)
        terminal = legacy_reward["terminal_event"]
        cost = cfg.time_cost_per_s*legacy_reward["elapsed_physics_s"]
        return dict(legacy_reward, schema=SCHEMA,
            reward=(shaping+sum(events.values())+terminal-cost) if credited else 0.,
            on_policy_rr_sample=False, on_policy_continuation_sample=credited,
            on_policy_post_rr_sample=credited, prefix_excluded=not credited,
            potential_before=previous, potential_after=current,
            potential_shaping=shaping if credited else 0.,
            milestone_events=events if credited else dict.fromkeys(events, 0.),
            terminal_event=terminal if credited else 0., time_cost=cost if credited else 0.,
            current_potential_components=after["post_rr_potential_components"],
            accepted_prefix_reward_diagnostic=legacy_reward["reward"],
            internal_state_change_is_terminal=False)
