"""Measured diagonal transfer context; preferences are not contact templates.

The sliding physical window is episode-local, shared across macro phases, and
never resets on P07/P08/P09 handoff. No force, pose, or simulator state writes.
Workspace is explicitly a joint/geometry proxy, not a dynamics feasibility proof.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER, servo_limits_deg

MODE = "diagonal_transfer_roles_v1"
ALL_STAGE_ACCEPTANCE_VERSION = "all_stage_v1"
LEGS = ("FL", "FR", "RL", "RR")
DIAGONAL = {"FR": "RL", "FL": "RR", "RR": "FL", "RL": "FR"}
ROLE_OBSERVATION_LAYOUT = "diagonal_transfer_state_v1"
ROLE_OBSERVATION_GROUP = "transfer_role_context_full48"
ROLE_OBSERVATION_BASE_DIM = 324
ROLE_OBSERVATION_DIM = 372
ROLE_OBSERVATION_FIELDS = (
    "valid", "workspace_progress", "preparation_progress", "transfer_progress",
    "motion_fraction", "preparation_ready", "transfer_ready",
    "fixed_direction_world_x", "fixed_direction_world_y",
    "short_support_continuity_fraction", "continued_response_fraction",
    "window_evidence_fraction",
)


def get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def vec(value, n=3):
    if value is None:
        return None
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == n and all(math.isfinite(x) for x in result) else None


def clip(x):
    return max(0., min(1., x))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def body_vector(q, v):
    # Inverse unit-quaternion rotation, with no convention change to sensors.
    w, x, y, z = q
    u = (-x, -y, -z)
    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    return tuple(2*dot(u, v)*u[i] + (w*w-dot(u, u))*v[i] + 2*w*cross[i] for i in range(3))


def validate_config(spec):
    cfg = spec.get("transfer_roles")
    if cfg is None:
        return
    if spec.get("physical_acceptance_version") not in (None, ALL_STAGE_ACCEPTANCE_VERSION):
        raise ValueError("unsupported physical acceptance version for transfer roles")
    if cfg.get("mode") != MODE or set(cfg.get("mapping", {})) != set(LEGS):
        raise ValueError("invalid diagonal transfer role config")
    for leg, row in cfg["mapping"].items():
        if row["diagonal_receiving_side"] != DIAGONAL[leg]:
            raise ValueError("receiving side must be diagonal, not a fixed contact requirement")
        if not set(row["preferred_bridge_contacts"]) <= set(LEGS)-{leg}:
            raise ValueError("invalid bridge preference")
    for key in ("window_s", "minimum_evidence_s", "motion_scale_m", "velocity_scale_m_s", "load_change_scale"):
        if not isinstance(cfg.get(key), (int, float)) or not 0 < cfg[key] <= 1:
            raise ValueError(f"invalid transfer scale {key}")
    if cfg["minimum_evidence_s"] > cfg["window_s"]:
        raise ValueError("transfer evidence window too short")


class TransferRoleTracker:
    def __init__(self, spec):
        validate_config(spec)
        self.spec = spec
        self.cfg = spec["transfer_roles"]
        self.samples = deque()
        self.response_since = dict.fromkeys(LEGS, None)
        self.transfer_response_since = dict.fromkeys(LEGS, None)
        self._separate_transfer_response = (
            spec.get("physical_acceptance_version") == ALL_STAGE_ACCEPTANCE_VERSION)

    def observe(self, raw, evaluation):
        current, hist = evaluation["current_legs"], evaluation["history"]
        com, base = get(raw, "center_of_mass"), get(raw, "base")
        c, v = vec(get(com, "position_w_m")), vec(get(com, "velocity_w_m_s"))
        b, q = vec(get(base, "position_w_m")), vec(get(base, "orientation_wxyz"), 4)
        wheels, joints = get(raw, "wheels", {}), get(raw, "joints", {})
        centers = {leg: vec(get(wheels[WHEEL_ORDER[i]], "center_w_m")) for i, leg in enumerate(LEGS)}
        valid = (get(com, "valid") is True and c is not None and v is not None
                 and b is not None and q is not None and all(x is not None for x in centers.values()))
        invalid_load = self._separate_transfer_response and any(
            current[leg].get("load_fraction_valid") is False for leg in LEGS)
        valid = valid and not invalid_load
        if not valid:
            # Detailed diagnostics unavailable is NOT invented zero CoM/support.
            self.samples.clear()
            self.response_since = dict.fromkeys(LEGS, None)
            self.transfer_response_since = dict.fromkeys(LEGS, None)
            return {leg: {"valid": False, "preparation_progress": 0., "transfer_progress": 0.,
                "preparation_ready": False, "transfer_ready": False, "motion_fraction": 0.,
                "reason": ("normalized bearing load is unavailable" if invalid_load else
                    "missing verified mass-weighted CoM or wheel/body geometry")} for leg in LEGS}
        now = evaluation["simulation_time_s"]
        supports = tuple(leg for leg in LEGS if current[leg]["support"]
                         and (current[leg]["ground_contact"] or current[leg]["top_contact"]))
        row = dict(t=now, tick=evaluation["physics_tick"], c=c, v=v, b=b, q=q, centers=centers,
                   loads={leg: current[leg]["load_fraction"] for leg in LEGS}, supports=supports,
                   joint_positions=tuple(float(get(joints[n], "position_deg")) for n in SERVO_ORDER),
                   joint_commands=tuple(float(get(joints[n], "command_deg", get(joints[n], "position_deg"))) for n in SERVO_ORDER),
                   wheel_commands=tuple(float(get(wheels[n], "command_rad_s")) for n in WHEEL_ORDER),
                   omega=vec(get(base, "angular_velocity_w_rad_s")))
        self.samples.append(row)
        while len(self.samples) > 1 and now-self.samples[0]["t"] > self.cfg["window_s"]+1e-10:
            self.samples.popleft()
        first = self.samples[0]
        duration = now-first["t"]
        time_credit = clip(duration/self.cfg["minimum_evidence_s"])
        command_motion = sum(sum(abs(x-y) for x, y in zip(a["joint_commands"], b_["joint_commands"]))
                             for a, b_ in zip(self.samples, list(self.samples)[1:]))
        actual_motion = sum(sum(abs(x-y) for x, y in zip(a["joint_positions"], b_["joint_positions"]))
                            for a, b_ in zip(self.samples, list(self.samples)[1:]))
        demand = (command_motion >= self.spec["history"]["minimum_command_motion_deg"]
                  or max(abs(x-y) for x, y in zip(row["joint_commands"], row["joint_positions"])) >= .5
                  or max(map(abs, row["wheel_commands"])) >= .02)
        actuated_response = demand and (actual_motion >= .1 or norm(sub(c, first["c"])) >= .001)
        result = {}
        for leg in LEGS:
            mapping = self.cfg["mapping"][leg]
            receiver = mapping["diagonal_receiving_side"]
            r = current[receiver]
            # Freeze direction at the beginning of THIS physical window.
            direction = (*sub(first["centers"][receiver], first["c"])[:2], 0.)
            length = norm(direction)
            direction = tuple(x/length for x in direction) if length > 1e-9 else (0., 0., 0.)
            delta_c, delta_receiver = sub(c, first["c"]), sub(centers[receiver], first["centers"][receiver])
            com_toward, velocity_toward = dot(delta_c, direction), dot(v, direction)
            load_drop = first["loads"][leg]-row["loads"][leg]
            support_fraction = sum(sum(x != leg for x in s["supports"]) >= 2 for s in self.samples)/len(self.samples)
            short_samples = [s for s in self.samples if now-s["t"] <= self.cfg["minimum_evidence_s"]+1e-10]
            short_support_fraction = sum(sum(x != leg for x in s["supports"]) >= 2 for s in short_samples)/len(short_samples)
            # Two contacts are supporting observations, not a static stability proof.
            continuation = time_credit*short_support_fraction if sum(x != leg for x in supports) >= 2 else 0.
            relative = body_vector(q, sub(centers[receiver], b))
            old_relative = body_vector(first["q"], sub(first["centers"][receiver], first["b"]))
            contraction = norm(old_relative)-norm(relative)
            margins = {}
            i = LEGS.index(receiver)
            for name in SERVO_ORDER[2*i:2*i+2]:
                actual = float(get(joints[name], "position_deg"))
                lo, hi = servo_limits_deg(name)
                margins[name] = {"negative_deg": actual-lo, "positive_deg": hi-actual}
            margin_proxy = min(clip(min(x.values())/10.) for x in margins.values())
            directional = max(clip(com_toward/self.cfg["motion_scale_m"]),
                              time_credit*clip(velocity_toward/self.cfg["velocity_scale_m_s"]))
            redistribution = clip(load_drop/self.cfg["load_change_scale"])
            space_response = clip(contraction/self.cfg["motion_scale_m"])
            initial = bool(current[leg].get("initial_clearance") and current[leg]["air"])
            actuated = bool(actuated_response or current[leg].get("whole_body_actuation_evidence"))
            # Preference-direction evidence is one route; measured redistribution
            # or actual whole-body clearance can establish an alternative route.
            measured_transfer = max(directional, redistribution, float(initial))
            response = bool(actuated and max(measured_transfer, space_response) > 0. and continuation > 0.)
            if not response:
                self.response_since[leg] = None
            elif self.response_since[leg] is None:
                self.response_since[leg] = now
            response_duration = 0. if self.response_since[leg] is None else now-self.response_since[leg]
            preparation_duration = response_duration
            preparation_continuation = continuation*clip(preparation_duration/self.cfg["minimum_evidence_s"])
            if self._separate_transfer_response:
                # Receiver-space preparation is useful, but its elapsed time
                # cannot mature a newly observed load/directional/lift response.
                # These clocks belong to physical evidence, never phase labels.
                transfer_response = bool(actuated and measured_transfer > 0. and continuation > 0.)
                if not transfer_response:
                    self.transfer_response_since[leg] = None
                elif self.transfer_response_since[leg] is None:
                    self.transfer_response_since[leg] = now
                response_duration = (0. if self.transfer_response_since[leg] is None
                                     else now-self.transfer_response_since[leg])
            # Legacy mode retains the original shared preparation/transfer age.
            continuation *= clip(response_duration/self.cfg["minimum_evidence_s"])
            preparation = preparation_continuation*max(directional, redistribution, space_response,
                                            float(initial))
            unload = clip((1.-current[leg]["load_fraction"])/(1.-self.spec["support"]["unloaded_leg_maximum_load_fraction"]))
            progress = continuation*measured_transfer*unload
            ready = bool(actuated and continuation >= 1.-1e-9 and measured_transfer >= 1.
                         and (current[leg]["load_fraction"] <= self.spec["support"]["unloaded_leg_maximum_load_fraction"] or initial))
            crossed = hist["front_edge_crossed"][leg]
            capture = (clip(current[leg]["consecutive_top_samples"]/self.spec["history"]["minimum_top_samples"])
                       if crossed else 0.)
            # Do not reward every receiver dropout as intentional reopening.
            reopening_evidence = bool(hist["placed"][receiver] and r["air"] and actuated
                                      and preparation_continuation > .5 and max(directional, space_response) > 0.)
            degraded = [x for x in LEGS if hist["placed"][x] and x not in supports
                        and not (x == receiver and reopening_evidence)]
            result[leg] = {"valid": True, "target_swing_leg": leg,
                "diagonal_receiving_side": receiver,
                "preferred_bridge_contacts": list(mapping["preferred_bridge_contacts"]),
                "observed_support_contacts": list(supports),
                "preferred_bridge_observed": [x for x in mapping["preferred_bridge_contacts"] if x in supports],
                "historical_placement_without_current_support": degraded,
                "receiver_workspace_state": {"validity": "measured_geometry_and_joint_margin_proxy_not_FK_feasibility",
                    "joint_range_margin_deg": margins, "joint_margin_proxy": margin_proxy,
                    "wheel_relative_body_m": relative, "wheel_to_top_m": r["clearance_m"],
                    "radial_contraction_m": contraction, "body_to_receiver_space_m": norm(relative),
                    "receiver_current_support": receiver in supports, "receiver_air": r["air"],
                    "reopening_context": bool(hist["placed"][receiver]),
                    "reopening_measured_response": reopening_evidence,
                    "exact_cartesian_feasibility": None},
                "transfer_direction_context": {"reference_tick": first["tick"], "window_s": duration,
                    "minimum_evidence_s": self.cfg["minimum_evidence_s"],
                    "fixed_direction_world": direction, "com_world_displacement_m": delta_c,
                    "receiver_world_displacement_m": delta_receiver,
                    "com_toward_receiver_m": com_toward, "com_velocity_toward_receiver_m_s": velocity_toward,
                    "body_world_displacement_m": sub(b, first["b"]),
                    "mass_weighted_com_position_w_m": c, "mass_weighted_com_velocity_w_m_s": v,
                    "load_fraction_change": load_drop, "support_continuity_fraction": support_fraction,
                    "short_support_continuity_fraction": short_support_fraction,
                    "continued_response_duration_s": response_duration,
                    "body_angular_velocity_w_rad_s": row["omega"],
                    "body_angular_velocity_change_w_rad_s": sub(row["omega"], first["omega"]) if row["omega"] and first["omega"] else None,
                    "whole_body_command_motion_deg": command_motion,
                    "whole_body_actual_motion_deg": actual_motion, "actuated_response": actuated,
                    "two_contact_static_stability_proven": False,
                    "exact_angular_momentum": None, "ground_reaction_impulse": None,
                    "target_contact_reaction_possible": leg in supports},
                "pending_capture": bool(hist["active_lift"][leg] and not hist["placed"][leg]
                    and not current[leg]["ground_contact"] and current[leg]["within_lateral_span"]),
                "preparation_progress": clip(preparation),
                "workspace_progress": .5*margin_proxy+.5*space_response,
                "preparation_ready": bool(actuated and preparation >= 1.-1e-9),
                "transfer_progress": clip(progress), "transfer_ready": ready,
                "motion_fraction": clip(max(preparation, progress)*float(actuated)*(1.-capture))
                    if not hist["placed"][leg] else 0.,
                "crossing_and_placement_evidence": {key: hist[key][leg] for key in
                    ("active_lift", "front_edge_crossed", "placed")}}
            if self._separate_transfer_response:
                result[leg]["transfer_direction_context"].update(
                    preparation_response_duration_s=preparation_duration,
                    response_maturity_semantics="separate_preparation_and_transfer")
        return result
