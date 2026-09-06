"""Five disjoint reward families for the finite-horizon physical task.

Potential uses augmented physical task/history state, not a phase-local origin.
All costs are bounded and integrated in actual simulated seconds. No reference
trajectory, support-leg template, or incentive to produce nonzero actions exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .semantic_observation import CONFIG_ROOT, SemanticObservationFrame, finite, vector

FAMILIES = ("task_progress", "body_stability", "contact_motion_quality",
            "control_smoothness", "control_regularization")
DEFAULT_REWARD_CONFIG = CONFIG_ROOT / "reward_config.yaml"


@dataclass(frozen=True)
class SemanticRewardConfig:
    values: Mapping[str, Any]
    path: Path

    @property
    def gamma(self) -> float:
        return float(self.values["gamma"])

    @property
    def failure_avoidance_bound(self) -> float:
        v = self.values
        costs = sum(v["family_weights"][name] for name in FAMILIES[1:])
        costs += v["time_cost_per_s"] * v["family_weights"]["task_progress"]
        # Infinite discounted future is an upper bound on the finite task.
        return costs / v["decision_hz"] / (1-self.gamma) + v["potential_weight"]


def load_semantic_reward_config(path: Path | str = DEFAULT_REWARD_CONFIG) -> SemanticRewardConfig:
    selected = Path(path).resolve()
    v = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if v.get("schema") != "wlr50_clean.semantic_reward.v2" or tuple(v["family_weights"]) != FAMILIES:
        raise ValueError("semantic reward requires exactly five ordered families")
    if (v["physics_hz"], v["decision_hz"], v["maximum_task_duration_s"]) != (120,15,200):
        raise ValueError("semantic task timing must be 120/15 Hz and 200 seconds")
    if not 0 < finite(v["gamma"],"gamma") < 1 or v["potential_terminal_value"] != 0:
        raise ValueError("potential requires gamma in (0,1) and zero terminal value")
    if v["time_limit_semantics"] != "finite_horizon_task_termination_no_bootstrap":
        raise ValueError("task deadline is terminal, not a rollout cutoff")
    for key in ("reference_imitation_reward", "phase_label_transition_bonus", "nonzero_residual_bonus"):
        if v.get(key) is not False:
            raise ValueError(f"{key} is forbidden")
    for key,value in v.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool) and finite(value,key) < 0:
            raise ValueError(f"negative reward configuration {key}")
    if any(finite(value,key) < 0 for key,value in v["family_weights"].items()):
        raise ValueError("negative family weight")
    for key in ("attitude_scale_rad", "euler_rate_scale_rad_s", "angular_acceleration_scale_rad_s2",
                "touchdown_speed_scale_m_s", "slip_speed_scale_m_s"):
        if finite(v[key],key) <= 0:
            raise ValueError(f"{key} must be positive")
    if type(v["control_regularization_enabled"]) is not bool:
        raise ValueError("regularization switch must be boolean")
    config = SemanticRewardConfig(v,selected)
    if v["failure_cost"]*v["family_weights"]["task_progress"] <= config.failure_avoidance_bound:
        raise ValueError("failure event must exceed avoidable future costs and potential")
    return config


def _square_cost(values: Sequence[float], scales: Sequence[float]) -> float:
    return min(1.0, sum(min(1.0,(x/s)**2) for x,s in zip(values,scales,strict=True))/max(1,len(values)))


@dataclass(frozen=True)
class SemanticRewardSample:
    previous: SemanticObservationFrame
    current: SemanticObservationFrame
    dt_s: float
    nominal: tuple[float, ...]
    previous_nominal: tuple[float, ...]
    residual: tuple[float, ...]
    previous_residual: tuple[float, ...]
    actual_drive: tuple[float, ...]
    previous_actual_drive: tuple[float, ...]
    previous_previous_actual_drive: tuple[float, ...]
    residual_caps: tuple[float, ...]


class SemanticRewardCalculator:
    def __init__(self, config: SemanticRewardConfig | None = None):
        self.config = config or load_semantic_reward_config()
        self.reset()

    def reset(self) -> None:
        """Episode reset only; a task phase change never clears landing history."""
        self._clock_s = 0.
        self._touchdowns: dict[int,tuple[float,tuple[float,...]]] = {}
        self._command_excursion = dict.fromkeys(range(4),0.)

    def evaluate(self, previous: SemanticObservationFrame, current: SemanticObservationFrame,
                 samples: Sequence[SemanticRewardSample], *, termination_reason: str | None,
                 task_success: bool) -> dict[str, Any]:
        if not samples:
            raise ValueError("reward requires an actually executed physical tick")
        if task_success != (termination_reason == "SUCCESS"):
            raise ValueError("success event and termination reason disagree")
        v = self.config.values
        costs = {name:0.0 for name in FAMILIES[1:]}
        diagnostics: dict[str,float] = {}
        total_dt = 0.0
        for sample in samples:
            dt = finite(sample.dt_s,"reward dt")
            if not 0 < dt <= 1/120+1e-9:
                raise ValueError("reward samples must be actual 120 Hz ticks")
            total_dt += dt
            self._clock_s += dt
            metrics = sample.current.metrics
            transfer = sample.current.task["substage"] == "TRANSFER"
            transfer_fraction = (max(0.,min(1.,float(sample.current.task.get("physical_transfer_fraction",float(transfer)))))
                                 if v.get("physical_transfer_weighting") else float(transfer))
            attitude = _square_cost(metrics["rpy"][:2], (v["attitude_scale_rad"],)*2)
            rates = _square_cost(metrics["euler_roll_pitch_rate"], (v["euler_rate_scale_rad_s"],)*2)
            acceleration = _square_cost(metrics["body_angular_acceleration"], (v["angular_acceleration_scale_rad_s2"],)*3)
            attitude *= 1.-(1.-v["transfer_attitude_weight"])*transfer_fraction
            body = (attitude+rates+acceleration)/3
            contact_terms = []
            for index,(before,after) in enumerate(zip(sample.previous.metrics["wheels"],metrics["wheels"],strict=True)):
                touchdown = after["contact"] and not before["contact"]
                descent = max(0.0,-before["vertical_velocity"]-v["touchdown_speed_allowance_m_s"])
                impact = min(1.0,(descent/v["touchdown_speed_scale_m_s"])**2) if touchdown else 0.0
                load = min(1.0,after["force"]/(metrics["mass_kg"]*9.81)) if touchdown else 0.0
                # A contact loss/rebound is not forbidden: penalize only excess
                # upward speed at a just-lost contact, independent of leg identity.
                rebound = min(1.0,(max(0.0,after["vertical_velocity"]-.15)/.5)**2) if before["contact"] and not after["contact"] else 0.0
                slip = min(1.0,(max(0.0,after["slip_speed"]-v["slip_speed_allowance_m_s"])/v["slip_speed_scale_m_s"])**2) if after["contact"] else 0.0
                if v.get("contact_loss_semantics") == "recent_touchdown_without_active_command_change":
                    command_change=sum(abs(a-b) for a,b in zip(sample.actual_drive[:8],sample.previous_actual_drive[:8]))
                    command_change+=sum(abs(a-b)*60./1.8 for a,b in zip(sample.actual_drive[8:],sample.previous_actual_drive[8:]))
                    self._command_excursion[index]+=command_change
                    if touchdown:
                        self._touchdowns[index]=(self._clock_s,sample.actual_drive)
                        self._command_excursion[index]=0.
                    last=self._touchdowns.get(index)
                    passive_recent_landing=(last is not None and self._clock_s-last[0]<=v["rebound_window_s"]
                        and self._command_excursion[index]<=v["rebound_command_motion_deg"])
                    # Departure alone proves neither intention nor rebound. A
                    # recent touchdown followed by upward reversal under an
                    # unchanged whole-body command is the conservative cost.
                    # Actuated/ambiguous departures stay diagnostic, not taxed.
                    if not passive_recent_landing: rebound=0.
                    contact_terms.append((impact*(.5+.5*load)+rebound+slip)/3)
                    diagnostics["confirmed_post_touchdown_rebound"] = diagnostics.get("confirmed_post_touchdown_rebound",0.)+rebound*dt
                    diagnostics["contact_chatter_diagnostic"] = diagnostics.get("contact_chatter_diagnostic",0.)+after["chatter"]*dt
                    diagnostics["touchdown_events"] = diagnostics.get("touchdown_events",0.)+float(touchdown)
                else:
                    contact_terms.append((impact*(.5+.5*load)+rebound+after["chatter"]+slip)/4)
            contact = sum(contact_terms)/4 * (1.-(1.-v["transfer_contact_weight"])*transfer_fraction)
            rates12 = (60.0,)*8+(1.8,)*4
            changes = {
                "nominal_first_difference":tuple((a-b)/dt for a,b in zip(sample.nominal,sample.previous_nominal,strict=True)),
                "residual_first_difference":tuple((a-b)/dt for a,b in zip(sample.residual,sample.previous_residual,strict=True)),
                "actual_drive_first_difference":tuple((a-b)/dt for a,b in zip(sample.actual_drive,sample.previous_actual_drive,strict=True)),
                "actual_drive_second_difference":tuple((a-2*b+c)/(dt*dt) for a,b,c in zip(sample.actual_drive,sample.previous_actual_drive,sample.previous_previous_actual_drive,strict=True)),
            }
            smooth_parts = {key:_square_cost(values,tuple(x*120 for x in rates12) if key.endswith("second_difference") else rates12)
                            for key,values in changes.items()}
            smooth = ((smooth_parts["actual_drive_first_difference"]+smooth_parts["actual_drive_second_difference"])/2
                      if v.get("smoothness_components")=="applied_only" else sum(smooth_parts.values())/4)
            regularization = _square_cost(sample.residual,sample.residual_caps) if v["control_regularization_enabled"] else 0.0
            for name,cost in zip(FAMILIES[1:],(body,contact,smooth,regularization),strict=True):
                costs[name] += min(1.0,max(0.0,cost))*dt
            for name,cost in {**smooth_parts,"gravity_attitude":attitude,"euler_rate":rates,
                              "angular_acceleration":acceleration,"contact_quality":contact}.items():
                diagnostics[name] = diagnostics.get(name,0.0)+cost*dt
        phi_before = finite(previous.task["task_progress_potential"],"potential before")
        phi_after = 0.0 if termination_reason else finite(current.task["task_progress_potential"],"potential after")
        potential = v["potential_weight"]*(self.config.gamma*phi_after-phi_before)
        event = v["success_reward"] if task_success else (-v["failure_cost"] if termination_reason else 0.0)
        unweighted = {"task_progress":potential+event-v["time_cost_per_s"]*total_dt,
                      **{name:-costs[name] for name in FAMILIES[1:]}}
        families = {name:unweighted[name]*v["family_weights"][name] for name in FAMILIES}
        return {"total":sum(families.values()),"families":families,"unweighted_families":unweighted,
                "potential_before":phi_before,"potential_after":phi_after,"potential_shaping":potential,
                "terminal_event":event,"elapsed_physics_s":total_dt,"cost_components":diagnostics,
                "discount_convention":"one_gamma_per_policy_decision; short_N1_terminal_interval_has_zero_next_potential_and_no_bootstrap",
                "terminal_bootstrap_allowed":False if termination_reason else True}
