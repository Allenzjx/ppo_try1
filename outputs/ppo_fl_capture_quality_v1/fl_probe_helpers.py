"""Output-only, finite FL controllability intervention; no production mutation."""
from __future__ import annotations

import math


def quintic(u):
    u = max(0., min(1., float(u)))
    return u*u*u*(10.+u*(-15.+6.*u))


def trigger_ready(*, phase, endpoint, evaluator, tick):
    if phase != "P05" or not endpoint or evaluator.get("physics_tick") != tick:
        return False
    if evaluator.get("valid") is not True or evaluator.get("termination_reason") is not None:
        return False
    current = evaluator.get("current_legs", {}).get("FL", {})
    history = evaluator.get("history", {})
    return (history.get("front_edge_crossed", {}).get("FL") is True
            and history.get("placed", {}).get("FL") is False
            and current.get("air") is True
            and current.get("obstacle_pair_active") is False
            and current.get("within_top_xy") is True
            and current.get("ground_contact") is False
            and current.get("support") is False)


class FiniteFLHold:
    """Override only nominated FL channels around one frozen residual anchor.

    Adding a physical offset to each current HISTORY conditional mean would
    feed the intervention back at rho=.9 and accumulate it. This intentionally
    fixes the nominated channel's requested physical residual during the hold.
    Other channels use the live actor on actual, unmodified observations.
    """
    def __init__(self, offsets_deg, *, ramp_decisions=8, hold_decisions=32,
                 release_decisions=8, follow_decisions=12):
        if len(offsets_deg) != 2 or not all(math.isfinite(x) and abs(x) <= 2 for x in offsets_deg):
            raise ValueError("FL diagnostic offsets must be finite and within +/-2 degrees")
        if ramp_decisions < 1 or hold_decisions < 30 or release_decisions < 1 or follow_decisions < 1:
            raise ValueError("finite probe requires a ramp and at least two seconds hold")
        self.offsets = tuple(offsets_deg)
        self.ramp, self.hold, self.release, self.follow = ramp_decisions, hold_decisions, release_decisions, follow_decisions
        self.anchor = None
        self.start_tick = None
        self.index = 0
        self.release_at = None
        self.release_start = None
        self.last_requested = None

    def start(self, *, tick, projected_residual):
        if self.anchor is not None or tick % 8 != 0:
            raise ValueError("probe starts once, on an ordinary policy-decision boundary")
        if len(projected_residual) != 12 or not all(math.isfinite(x) for x in projected_residual):
            raise ValueError("finite actual residual anchor required")
        self.anchor = tuple(projected_residual[:2])
        self.last_requested = self.anchor
        self.start_tick = tick

    def action(self, baseline_raw, *, caps, capture_or_leave=False):
        if self.anchor is None:
            raise ValueError("probe not started")
        if len(baseline_raw) != 12 or len(caps) != 12:
            raise ValueError("full12 baseline and caps required")
        if self.release_at is None and (capture_or_leave or self.index >= self.ramp+self.hold):
            self.release_at, self.release_start = self.index, self.last_requested
        requested = [caps[i]*math.tanh(baseline_raw[i]) for i in range(2)]
        state, amount = "ramp", quintic((self.index+1)/self.ramp)
        if self.release_at is None:
            if self.index >= self.ramp:
                state, amount = "hold", 1.
            for i in range(2):
                if self.offsets[i]:
                    requested[i] = self.anchor[i]+amount*self.offsets[i]
        else:
            age = self.index-self.release_at
            state, amount = ("release", quintic((age+1)/self.release)) if age < self.release else ("follow", 1.)
            for i in range(2):
                if self.offsets[i]:
                    requested[i] = (1.-amount)*self.release_start[i]+amount*requested[i]
        raw = list(baseline_raw)
        for i in range(2):
            if self.offsets[i]:
                fraction = requested[i]/caps[i]
                if not math.isfinite(fraction) or abs(fraction) >= 1:
                    raise ValueError("diagnostic request exceeds current physical residual cap")
                raw[i] = math.atanh(fraction)
        receipt = dict(kind="external_FL_controllability_hold_not_PPO", index=self.index,
                       state=state, anchor_deg=self.anchor, planned_offset_deg=self.offsets,
                       requested_FL_residual_deg=requested, interpolation=amount,
                       baseline_raw=list(baseline_raw), injected_raw=raw,
                       no_accumulation_on_conditional_mean=True)
        self.last_requested = tuple(requested)
        self.index += 1
        return tuple(raw), receipt

    @property
    def complete(self):
        return self.release_at is not None and self.index >= self.release_at+self.release+self.follow


def live_fl_jacobian(backend):
    """Read current FL link-center Jacobian; fixed-base estimate, not contact model."""
    import torch
    adapter, robot = backend._adapter, backend._adapter.robot
    names, joints = tuple(robot.body_names), tuple(robot.joint_names)
    body = names.index("front_left_wheel")
    ids = tuple(adapter.joint_map.servo_ids[:2])
    if tuple(joints[i] for i in ids) != ("front_left_hip", "front_left_knee"):
        raise ValueError("FL actuator order mismatch")
    fixed = robot.is_fixed_base
    if type(fixed) is not bool:
        raise ValueError("missing fixed-base flag")
    columns = [i+(0 if fixed else 6) for i in ids]
    data = robot.data
    jac = robot.root_physx_view.get_jacobians()[0, body-int(fixed)]
    offset = data.body_link_pos_w[0, body]-data.body_com_pos_w[0, body]
    linear = jac[:3]+torch.cross(jac[3:].T, offset.expand(jac.shape[1], 3), dim=1).T
    velocity = data.joint_vel[0] if fixed else torch.cat((data.root_com_vel_w[0], data.joint_vel[0]))
    error = float((linear@velocity-data.body_link_vel_w[0, body, :3]).abs().max().item())
    if not torch.isfinite(linear).all() or error > 1e-3:
        raise ValueError("same-state Jacobian/velocity consistency failed")
    return dict(physics_tick=backend._episode_tick, body="front_left_wheel", canonical_joint_ids=ids,
        world_xyz_mm_per_canonical_deg=(linear[:, columns]*math.pi/180*1000).cpu().tolist(),
        velocity_identity_max_error_m_s=error, physical_q_rad=data.joint_pos[0, list(ids)].cpu().tolist(),
        body_position_w_m=data.body_link_pos_w[0, body].cpu().tolist(),
        fixed_base_estimate_only=True, contact_or_body_response_guaranteed=False,
        derivative_scope="wheel_link_center_not_pose_dependent_lowest_collider_vertex")
