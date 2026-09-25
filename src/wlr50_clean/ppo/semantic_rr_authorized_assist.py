"""Declared RR-only fallback through the existing raw-to-target actuator path.

No simulator writes, import side effects, policy/noise calls or nominal-history
mutation. The caller retains the selected Gaussian sample/logp separately from
this deterministic execution mapping. OFF is literally the identity mapping.
"""
from __future__ import annotations

import math


class AuthorizedRRAssist:
    def __init__(self, *, enabled=False):
        if type(enabled) is not bool:
            raise ValueError('RR assistance needs an explicit Boolean configuration')
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.active = False
        self.mode = 0
        self.started = None
        self.age = 0.
        self.entry = [0., 0.]
        self.goal = [0., 0.]
        self.last_time = None

    def obs7(self):
        return (float(self.active), self.mode/4., min(1., self.age/200.),
                self.entry[0]/180., self.entry[1]/180.,
                self.goal[0]/180., self.goal[1]/180.)

    def request(self, core, policy_raw):
        selected = list(policy_raw)
        if len(selected) != 12 or not all(math.isfinite(x) for x in selected):
            raise ValueError('RR assist requires the original finite raw12 request')
        state = core.task.snapshot()
        metrics = state.get('metrics') or {}
        now = float(core.frame.sim_time_s)
        final = core.frame.info['drive_target_full12']
        if not self.enabled or not state.get('active'):
            return selected, dict(enabled=self.enabled, active=False,
                policy_raw_full12=selected, applied_raw_full12=selected,
                overridden_channels=[], PPO_sample_is_original_raw=True,
                forced_RR_actions_are_not_student_discovery=True)
        if self.started is None:
            # This is a delta from the actual capture-entry FINAL, not an
            # increment repeatedly added to an already changed target.
            self.started = now
            self.last_time = now
            self.entry = [float(final[6]), float(final[7])]
            self.goal = list(self.entry)
        dt = max(0., min(8./120., now-self.last_time))
        self.last_time = now
        self.age = now-self.started
        self.active = True
        legal = bool(metrics.get('live') and metrics.get('current_attempt_capture_eligible')
                     and metrics.get('within_top_xy') and not metrics.get('ground_contact'))
        contact = bool(metrics.get('current_top_contact'))
        if self.mode == 4:
            pass  # The gradual hand-back below remains authoritative.
        elif contact:
            # Keep the actually reached target, not the old AIR target. No
            # claim that an AIR wheel bears weight in the later grace state.
            self.mode = 2
            self.goal = [float(final[6]), float(final[7])]
        elif not legal:
            self.mode = 3
            self.goal = [float(final[6]), float(final[7])]
        elif self.mode in (0, 1):
            self.mode = 1
            fraction = min(1., self.age/1.5)
            self.goal = [self.entry[0]-25.*fraction, self.entry[1]+20.*fraction]
            if fraction == 1.:
                self.mode = 2
        else:
            self.mode = 2
            # A small bounded candidate beyond the recorded +20deg diagnostic,
            # not a measured success label or an unbounded per-step increment.
            if metrics.get('gap_m', 0.) > .002:
                self.goal[1] = min(self.entry[1]+28., self.goal[1]+1.5*dt)
        ack = core.frame.info['atomic_ack']
        baseline = ack['policy_headroom_evidence']['baseline_native_plus_controller_full12']
        caps = core.backend.execution_profile['residual']['phase_caps_full12'][core.frame.state_id]
        issued = list(selected)
        for offset, index in enumerate((6, 7)):
            if not math.isfinite(caps[index]) or caps[index] <= 0:
                raise ValueError('RR assist needs actual positive RR residual capacity')
        # After real RL touchdown and entry to final settling, return RR
        # authority gradually. This is not a fixed home pose or a new touch
        # claim; the student's current target remains the destination.
        if self.mode == 4 or (core.frame.state_id == 'P13'
                             and state.get('rl_touch_seen') and contact):
            self.mode = 4
            for offset, index in enumerate((6, 7)):
                destination = baseline[index] + caps[index]*math.tanh(selected[index])
                self.goal[offset] += max(-4.*dt, min(4.*dt, destination-self.goal[offset]))
        for offset, index in enumerate((6, 7)):
            ratio = max(-.985, min(.985, (self.goal[offset]-baseline[index])/caps[index]))
            issued[index] = math.atanh(ratio)
        return issued, dict(schema='wlr50_clean.authorized_rr_target_assist.v1',
            enabled=True, active=True, mode=self.mode, entry_FINAL_rr_deg=list(self.entry),
            goal_absolute_rr_deg=list(self.goal), baseline_previous_ACK_full12=list(baseline),
            not_same_tick_counterfactual=True, policy_raw_full12=selected,
            applied_raw_full12=issued, overridden_channels=[6, 7],
            same_train_eval_rule=True, PPO_sample_is_original_raw=True,
            likelihood='original_full12_Gaussian_before_declared_deterministic_execution_mapping',
            forced_RR_actions_are_not_student_discovery=True,
            current_contact=contact, current_bearing=bool(metrics.get('current_top_bearing')),
            no_state_or_force_writes=True, nominal_history_mutation=False)
