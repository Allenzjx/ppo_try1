"""OUTPUTS-ONLY REVIEW DRAFT. Not imported by production or any running Isaac.

The candidate earns finite opportunity from new measured distance progress, not
wheel rotation, actor output, a phase label, or repeated traversal of old space.
No action/nominal/reward is generated here. Public values require schema migration.
"""
from dataclasses import dataclass
import math

MODE = "p02_measured_progress_credit_v1"


@dataclass
class FrontProgressCredit:
    """Tiny candidate state; reset only at episode/P02 entry or exit.

    Candidate constants are derived from current P02's existing stall contract:
    0.01 phase progress * 3 equal predicates * 0.25 m approach-shaping scale.
    This is 7.5 mm credit, spent at 7.5 mm / 6 s. These are reviewable design
    parameters, NOT a physical-success threshold or a modified P03 boundary.
    """

    capacity_m: float = 0.0075
    drain_m_s: float = 0.00125
    best_remaining_m: float | None = None
    credit_m: float = 0.0

    def observe(self, *, remaining_m: float, dt_s: float,
                eligible: bool, p02: bool, episode_age_s: float,
                physical_abort: bool = False) -> dict:
        if any(not math.isfinite(v) for v in
               (remaining_m, dt_s, episode_age_s, self.capacity_m, self.drain_m_s)):
            raise ValueError("finite measured state and finite positive credit configuration required")
        if (remaining_m < 0 or dt_s <= 0 or dt_s > 1 / 15 + 1e-9
                or self.capacity_m <= 0 or self.drain_m_s <= 0):
            raise ValueError("distance, physics interval or credit contract invalid")
        if not p02:
            self.best_remaining_m = None
            self.credit_m = 0.0
            return {"eligible": False, "continuation_allowed": False,
                    "best_remaining_m": 0.0, "credit_m": 0.0,
                    "new_forward_progress_m": 0.0, "observation": (0.0, 0.0, 0.0)}
        first = self.best_remaining_m is None
        previous_best = remaining_m if first else self.best_remaining_m
        measured_new_progress = max(0.0, previous_best - remaining_m)
        # Track the geometric best even while ineligible, so later eligibility
        # cannot sell an old, unsafe or ground-contact traversal a second time.
        self.best_remaining_m = min(previous_best, remaining_m)
        valid_now = bool(eligible and not physical_abort and episode_age_s < 200.0)
        earned = measured_new_progress if valid_now else 0.0
        self.credit_m = min(self.capacity_m,
                            max(0.0, self.credit_m + earned - self.drain_m_s * dt_s))
        allowed = bool(valid_now and self.credit_m > 0.0)
        return {"eligible": valid_now, "continuation_allowed": allowed,
                "best_remaining_m": self.best_remaining_m, "credit_m": self.credit_m,
                "new_forward_progress_m": measured_new_progress, "earned_progress_m": earned,
                "credit_without_new_progress_s": self.credit_m / self.drain_m_s,
                # Proposed appended3, not production419 or a hidden latch.
                "observation": (self.best_remaining_m, self.credit_m / self.capacity_m,
                                float(valid_now))}
