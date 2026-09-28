"""Explicit P06 ordinary-forward permission and scalar probability checks."""
from dataclasses import dataclass
import math

FL = 8
B_FIELDS = (
    'b_p06_fl_forward_active', 'b_p06_atomic_stop_horizon_norm',
    'b_p06_current_rolling_gain', 'b_fl_current_bearing',
)


@dataclass(frozen=True)
class WindowFacts:
    tick: int
    phase: str
    front_left_placed: bool
    front_right_placed: bool
    owner_fl_nominal_rad_s: float  # Effective winning owner, NOT changed_channels.
    owner_explicit_stop: bool
    safety_stop: bool
    accepted_recoil_or_reverse_pulse: bool
    winning_owner_is_p06: bool = False
    p06_rolling_gain: float = 0.
    pending_fl_capture: bool = False
    p06_endpoint_issued: bool = False
    native_ticks_to_atomic_stop: int | None = None


def select_window(facts):
    if not isinstance(facts, WindowFacts) or type(facts.tick) is not int or facts.tick < 0:
        raise ValueError('same-tick explicit owner facts required')
    if not all(math.isfinite(v) for v in (facts.owner_fl_nominal_rad_s, facts.p06_rolling_gain)):
        raise ValueError('finite canonical owner speed required')
    if facts.safety_stop or facts.owner_explicit_stop or facts.accepted_recoil_or_reverse_pulse:
        return 'OFF'
    if (facts.phase == 'P06' and facts.front_left_placed and facts.front_right_placed
            and facts.winning_owner_is_p06 and facts.owner_fl_nominal_rad_s == .3
            and facts.p06_rolling_gain == 1. and not facts.pending_fl_capture
            and not facts.p06_endpoint_issued and type(facts.native_ticks_to_atomic_stop) is int
            and facts.native_ticks_to_atomic_stop > 8):
        return 'P06_FORWARD'
    # Post-RR preparation excluded. Its negative entry is a finite physical
    # ramp, not a reason to replace its 1.8 rad/s2 rate or authored recoil.
    return 'OFF'


def scalar_logp(value, mean, std, active=True):
    if not math.isfinite(std) or std <= 0.:
        raise ValueError('scalar std must be strictly positive')
    return -.5*((value-mean)/std)**2-math.log(std)-.5*math.log(2.*math.pi) if active else 0.


def scalar_entropy(std, active=True):
    if not math.isfinite(std) or std <= 0.:
        raise ValueError('scalar std must be strictly positive')
    return .5*math.log(2.*math.pi*math.e)+math.log(std) if active else 0.


def scalar_kl(old_mean, old_std, new_mean, new_std, *, old_active=True, new_active=True):
    if old_active != new_active:
        raise ValueError('same stored observation must have the same window permission')
    if min(old_std,new_std) <= 0. or not all(math.isfinite(v) for v in (old_mean,old_std,new_mean,new_std)):
        raise ValueError('finite scalar Gaussians with positive std required')
    if not old_active:
        return 0.
    return math.log(new_std/old_std)+(old_std**2+(old_mean-new_mean)**2)/(2*new_std**2)-.5
