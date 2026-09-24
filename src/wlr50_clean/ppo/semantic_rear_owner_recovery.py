"""Cancel an inapplicable issued rear-transfer owner, not a rear teacher.

Only P09-late FL/RL and P12 dependent RL servos are affected. A frozen entry target and request
reference make policy changes bidirectional without chasing the old distant N.
No wheel owner, stop, mapper, physics or task/contact evidence is changed.
All persistent state is exported to the actor/critic; genuine reset alone clears it.
"""
from __future__ import annotations

import copy
import math

from .semantic_headroom import project_semantic_servo_headroom

MODE = "issued_rear_transfer_owner_suspension_v1"
INDICES = (0, 1, 4, 5)


def _values(values, size, label):
    if (not isinstance(values, (tuple, list)) or len(values) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("invalid owner recovery " + label)
    return tuple(float(v) for v in values)


def empty_state():
    return dict(anchor_final_deg=[0.] * 4, anchor_request_deg=[0.] * 4,
                active=[False] * 4, winning_late_owner=[False] * 4,
                rl_edge_recovery_permitted=False)


def validate_state(state):
    _values(state["anchor_final_deg"], 4, "target anchors")
    _values(state["anchor_request_deg"], 4, "request anchors")
    for key in ("active", "winning_late_owner"):
        if len(state[key]) != 4 or any(type(v) is not bool for v in state[key]):
            raise ValueError("owner bits must be four booleans")
    if type(state["rl_edge_recovery_permitted"]) is not bool:
        raise ValueError("edge permission must be boolean")
    for j in range(4):
        if not state["active"][j] and (state["anchor_final_deg"][j] != 0.
                                     or state["anchor_request_deg"][j] != 0.):
            raise ValueError("inactive owner anchors must be zero")


def transition(state, context, previous_final, previous_request):
    """Pure state transition; previous values are adjacent committed ACK inputs."""
    validate_state(state)
    previous_final = _values(previous_final, 12, "previous FINAL")
    previous_request = _values(previous_request, 12, "previous request")
    owners = context["winning_late_owner"]
    if len(owners) != 4 or any(type(v) is not bool for v in owners):
        raise ValueError("missing exact winning owner receipt")
    for key in ("support_transfer_permitted", "rl_current_swing", "rl_edge_recovery_permitted", "live"):
        if type(context[key]) is not bool:
            raise ValueError("invalid current physical owner permission")
    result = copy.deepcopy(state)
    result["winning_late_owner"] = list(owners)
    result["rl_edge_recovery_permitted"] = context["rl_edge_recovery_permitted"]
    # EDGE is not permission to resume strong FL/RL source goals. It retains
    # independent policy recovery while stale source ownership stays suspended.
    for j, i in enumerate(INDICES):
        # An airborne RL may continue its own swing, but does not authorize
        # FL to restart a strong transfer that depended on RR bearing.
        released = context["support_transfer_permitted"] or (i in (4, 5) and context["rl_current_swing"])
        active = bool(owners[j] and context["live"] and not released)
        if active and not state["active"][j]:
            result["anchor_final_deg"][j] = previous_final[i]
            result["anchor_request_deg"][j] = previous_request[i]
        if not active:
            result["anchor_final_deg"][j] = result["anchor_request_deg"][j] = 0.
        result["active"][j] = active
    validate_state(result)
    return result


def apply_state(candidate, request, state, capacities):
    """Replayable projection before the unchanged final hard clamp/slew."""
    candidate = list(_values(candidate, 12, "candidate"))
    request = _values(request, 12, "request")
    capacities = _values(capacities, 12, "capacity")
    validate_state(state)
    anchor, relative = [0.] * 12, [0.] * 12
    for j, i in enumerate(INDICES):
        if capacities[i] <= 0.:
            raise ValueError("owner recovery must preserve positive policy capacity")
        if state["active"][j]:
            anchor[i] = state["anchor_final_deg"][j]
            relative[i] = max(-capacities[i], min(capacities[i],
                request[i] - state["anchor_request_deg"][j]))
    margin = project_semantic_servo_headroom(anchor, [0.] * 12, relative)
    effective = margin["effective_combined_post_mapper_bias_full12"]
    for j, i in enumerate(INDICES):
        if state["active"][j]:
            candidate[i] = anchor[i] + effective[i]
    return tuple(candidate)


class RearOwnerRecovery:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = empty_state()

    def snapshot(self):
        return copy.deepcopy(self.state)

    def advance(self, *, context, previous_ack, write_count, previous_tick,
                request, candidate, capacities):
        if (not isinstance(previous_ack, dict)
                or previous_ack.get("write_count") != write_count
                or previous_ack.get("physics_tick") != previous_tick
                or context.get("dispatch_physics_tick") != previous_tick + 1):
            raise ValueError("owner suspension requires adjacent committed ACK")
        previous_final = previous_ack["drive_target_full12"]
        previous_request = previous_ack.get("independent_policy_residual_requested_full12", [0.] * 12)
        before = self.snapshot()
        self.state = transition(before, context, previous_final, previous_request)
        after = self.snapshot()
        output = apply_state(candidate, request, after, capacities)
        return dict(mode=MODE, state_before=before, state_after=after,
                    context=copy.deepcopy(context), previous_final_full12=list(previous_final),
                    previous_requested_full12=list(previous_request), requested_full12=list(request),
                    capacities_full12=list(capacities), candidate_before_full12=list(candidate),
                    candidate_after_full12=list(output),
                    owner_indices=[i for j, i in enumerate(INDICES) if after["active"][j]],
                    semantics="entry_FINAL_plus_bounded_request_change_no_old_N_pull",
                    task_assist=False, source_stops_modified=False, physical_evidence_modified=False)
