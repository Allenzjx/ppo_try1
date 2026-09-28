"""P06 FL task context and local reward around the unchanged A physics core.

Preserves controller closures, physical history, original sampling hold and
true task outcomes. It does not modify nominal, actuator targets or contacts.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from .semantic_fl_forward_window import B_FIELDS, WindowFacts, select_window

SCHEMA = 'wlr50_clean.p06_fl_forward_window.v1'
WHEELS = ('front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel')
ANKLES = tuple(x.replace('_wheel', '_ankle') for x in WHEELS)
SIGNS = (-1., 1., -1., 1.)
DT = 1./120.


def get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def vec(x, length):
    return tuple(float(a) for a in x) if isinstance(x, (tuple, list)) and len(x) == length and all(map(finite, x)) else None


@dataclass(frozen=True)
class FlWindowConfig:
    p06_stop_source_tick: int
    normal_time_scale: float = 1.
    forward_progress_reward_per_m: float = 5.
    negative_final_cost_per_s: float = .10
    loaded_counterdrive_cost_per_s: float = .10
    force_noise_floor_n: float = .2
    progress_epsilon_m: float = 1.e-7
    source_receipt: dict | None = None

    def __post_init__(self):
        if type(self.p06_stop_source_tick) is not int or self.p06_stop_source_tick <= 8:
            raise ValueError('finite authored P06 source stop required')
        for name in ('normal_time_scale', 'forward_progress_reward_per_m', 'negative_final_cost_per_s',
                     'loaded_counterdrive_cost_per_s', 'force_noise_floor_n', 'progress_epsilon_m'):
            if not finite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('positive finite B configuration required: '+name)


def config_from_sources(contract_path, fsm_spec_path):
    """Small fixed source files only; no rollout, simulator or model imports."""
    import yaml
    contract_path, fsm_spec_path = Path(contract_path), Path(fsm_spec_path)
    raw, fsm_raw = contract_path.read_bytes(), fsm_spec_path.read_bytes()
    contract, fsm = json.loads(raw), yaml.safe_load(fsm_raw)
    phase = next(p for p in contract['phases'] if p['state_id'] == 'P06')
    state = next(s for s in fsm['states'] if s['state_id'] == 'P06')
    scale = float(state['normal_time_scale'])
    points = phase['waypoints']
    if (contract['rear_leg_order'] != 'RR_FIRST' or contract['physics_hz'] != 120.
            or len(points) != 3 or points[0]['kind'] != 'phase_entry'
            or points[1]['time_s'] != 0. or points[1]['full12'][8:] != [.3]*4
            or points[2]['full12'][8:] != [0.]*4
            or set(points[1]['atomic_channels']) != set(ANKLES)
            or set(points[2]['atomic_channels']) != set(ANKLES)):
        raise ValueError('B requires the exact finite four-wheel P06 launch/stop structure')
    stop = round(float(points[2]['time_s'])*scale*120.)
    return FlWindowConfig(stop, scale, source_receipt=dict(
        contract=str(contract_path.resolve()), contract_sha256=hashlib.sha256(raw).hexdigest(),
        fsm_spec=str(fsm_spec_path.resolve()), fsm_spec_sha256=hashlib.sha256(fsm_raw).hexdigest(),
        stop_authored_s=points[2]['time_s'], normal_time_scale=scale,
        stop_source_tick=stop, clock='MotionExecutor native source index, NOT episode age',
        owner_proof='P06 newest layer owns all four wheels by verified atomic launch'))


def physical(frame, cfg):
    raw = get(frame, 'info', {}).get('raw_observation')
    base = get(raw, 'base')
    pos = vec(get(base, 'position_w_m'), 3)
    actual = vec(get(raw, 'actual_full12'), 12)
    contacts = get(raw, 'contacts', {})
    bearing, forces = [], []
    for name in WHEELS:
        pairs = get(contacts, name, {})
        load = sum(max(0., get(p, 'normal_force_n')) for surface in ('ground', 'obstacle')
                   if (p := get(pairs, surface)) is not None
                   and get(p, 'active') is True and get(p, 'pair_verified') is True
                   and finite(get(p, 'normal_force_n')))
        forces.append(load)
        bearing.append(load > cfg.force_noise_floor_n)
    return dict(body_position_w_m=pos, actual_full12=actual,
                current_bearing=bearing, verified_normal_force_n=forces)


def read_window(frame, cfg):
    """Current SOURCE intention for the next dispatch, not previous ACK targets."""
    info = get(frame, 'info', {})
    task = info.get('semantic_task', {})
    ev = task.get('physical_evaluator', {})
    provider = task.get('nominal_provider_diagnostics', {})
    capture = provider.get('capture_continuation', {})
    rolling = provider.get('p06_rolling_retirement', {})
    tail = provider.get('p06_wheel_tail', {})
    nominal = vec(get(frame, 'nominal_action_full12'), 12)
    nominal_fl = nominal[8] if nominal else 0.
    tick = get(frame, 'physics_tick', -1)
    phase = get(frame, 'state_id', '')
    cursor = capture.get('P06_source_tick')
    remaining = cfg.p06_stop_source_tick-cursor if type(cursor) is int else None
    gain = capture.get('P06_wheel_gain')
    gain = float(gain) if finite(gain) else 0.
    # No generic channel-owner field exists in the accepted provider. This is
    # a declared DERIVATION from current latest P06 + its atomic launch; held
    # channels are not mistaken for zero changed_channels.
    proof = bool(phase == 'P06' and ev.get('valid') is True and ev.get('physics_tick') == tick
        and capture.get('source_observation_tick') == tick
        and capture.get('P06_layer_present') is True and capture.get('P06_source_advanced') is True
        and capture.get('P06_wheel_contribution_enabled') is True
        and tail.get('layer_present') is True and tail.get('status') == 'finite_source_before_endpoint'
        and rolling.get('layer_present') is True and rolling.get('wheel_gain') == 1.
        and nominal is not None and nominal[8:] == (.3,)*4)
    placed = ev.get('history', {}).get('placed', {})
    safety = get(frame, 'safety_projection')
    stopped = bool(get(safety, 'force_wheels_zero', False) or get(safety, 'residual_enabled', True) is False
                   or task.get('termination_reason') or ev.get('termination_reason'))
    facts = WindowFacts(tick, phase, placed.get('FL') is True, placed.get('FR') is True,
        nominal_fl, nominal_fl == 0., stopped, nominal_fl < 0., proof, gain,
        capture.get('fl_capture_pending') is not False,
        tail.get('source_endpoint_issued') is not False, remaining)
    mode = select_window(facts)
    measured = physical(frame, cfg)
    active = mode == 'P06_FORWARD'
    horizon = max(0., min(1., (remaining or 0)/cfg.p06_stop_source_tick))
    return dict(schema=SCHEMA, active=active, mode=mode, observation_tick=tick, phase=phase,
        p06_source_tick=cursor, native_ticks_to_atomic_stop=remaining, rolling_gain=gain,
        derived_winning_owner_is_p06=proof, owner_semantics='derived_latest_finite_P06_atomic_wheel_owner',
        source_nominal_full12=nominal, current_FL_bearing=measured['current_bearing'][0],
        current_FL_verified_normal_force_n=measured['verified_normal_force_n'][0],
        pending_fl_capture=capture.get('fl_capture_pending'), source_endpoint_issued=tail.get('source_endpoint_issued'),
        observation=(float(active), horizon, max(0., min(1., gain)), float(measured['current_bearing'][0])),
        current_FINAL_FL=get(info, 'drive_target_full12', [None]*12)[8],
        current_actual_FL=measured['actual_full12'][8] if measured['actual_full12'] else None,
        permission_never_depends_on_FINAL_sign_or_current_contact=True)


def transition_evidence(before, after, cfg, *, request_active):
    """Use actual completed native interval: previous SOURCE + current ACK/raw."""
    source = read_window(before, cfg)
    a, b = physical(before, cfg), physical(after, cfg)
    info = get(after, 'info', {})
    ack, audit = info.get('atomic_ack', {}), info.get('actuator_target_effect_audit', {})
    final = vec(ack.get('drive_target_full12'), 12)
    direct = vec(info.get('drive_target_full12'), 12)
    native = vec(audit.get('actual_native_targets', {}).get('wheel_velocity_rad_s'), 4)
    verified = bool(final is not None and direct == final and native is not None
        and all(audit.get(k) is True for k in ('verified', 'actual_mapping_matches_dispatch', 'setter_dispatch_targets_equal'))
        and type(ack.get('physics_tick')) is int and audit.get('physics_tick') == ack['physics_tick']
        and audit.get('source_phase_id') == get(before, 'state_id')
        and all(abs(native[i]*SIGNS[i]-final[8+i]) <= 2.e-6 for i in range(4)))
    adjacent = get(after, 'physics_tick') == get(before, 'physics_tick')+1
    valid = bool(adjacent and verified and a['body_position_w_m'] and b['body_position_w_m'] and b['actual_full12'])
    eligible = bool(request_active and source['active'] and valid)
    dx = b['body_position_w_m'][0]-a['body_position_w_m'][0] if valid else 0.
    # The sealed baseline was around -0.7 rad/s. Clipping at -0.3 would
    # erase useful small-action distinctions throughout that actual window.
    negative_final = max(0., -final[8]) if eligible else 0.
    final_conflict = negative_final/(.3+negative_final)
    actual = b['actual_full12']
    bearing = b['current_bearing']
    counterdrive = bool(eligible and bearing[0] and actual[8] < 0.
        and any(bearing[i] and actual[8+i] > .05 for i in range(1, 4))
        and dx <= cfg.progress_epsilon_m)
    negative_actual = -actual[8] if counterdrive else 0.
    loaded_conflict = negative_actual/(.3+negative_actual)
    progress_reward = cfg.forward_progress_reward_per_m*dx if eligible else 0.
    command_cost = cfg.negative_final_cost_per_s*DT*final_conflict
    loaded_cost = cfg.loaded_counterdrive_cost_per_s*DT*loaded_conflict
    return dict(physics_tick=get(after, 'physics_tick'), command_tick=ack.get('physics_tick'),
        request_active=bool(request_active), native_source_window_active=source['active'], eligible=eligible,
        verified_adjacent_ACK=verified and adjacent, source=source, body_dx_world_m=dx,
        final_FL=final[8] if final else None, actual_FL=actual[8] if actual else None,
        current_FL_bearing=bearing[0], current_bearing_full4=bearing,
        progress_reward=progress_reward, negative_FINAL_command_cost=command_cost,
        nonprogress_loaded_counterdrive_cost=loaded_cost,
        local_reward=progress_reward-command_cost-loaded_cost)


class FlWindowCore:
    """One existing physics core; task/control closures are never replaced."""
    def __init__(self, inner, config):
        self.inner, self.config = inner, config
        self._external_observer = inner.tick_observer
        inner.tick_observer = self._observe
        self._request_active = False
        self._collect = False
        self._rows = []
        self._seen_active_decision = False
        self.fl_window = None
        self.observation = None

    def __getattr__(self, key):
        return getattr(self.inner, key)

    @property
    def tick_observer(self):
        return self._external_observer

    @tick_observer.setter
    def tick_observer(self, value):
        self._external_observer = value

    def _observe(self, before, after, projection):
        if self._collect:
            self._rows.append(transition_evidence(before, after, self.config, request_active=self._request_active))
        if self._external_observer is not None:
            self._external_observer(before, after, projection)

    def _encode(self, original):
        self.fl_window = read_window(self.inner.frame, self.config)
        if len(original) != 531:
            raise ValueError('B must preserve exact A531 observation prefix')
        self.observation = tuple(original)+self.fl_window['observation']
        return self.observation

    def reset(self, seed=1001):
        self._collect = self._request_active = self._seen_active_decision = False
        self._rows = []
        return self._encode(self.inner.reset(seed=seed))

    def step(self, raw):
        start = deepcopy(self.fl_window)
        if start is None:
            raise RuntimeError('reset before B decision')
        self._request_active = start['active']
        finish_active_at_request = self.inner.task.finish_active
        self._rows, self._collect = [], True
        try:
            step = self.inner.step(raw)  # Same raw12/one dispatch/one HISTORY update.
        finally:
            self._collect = False
        original_reward = step.reward
        local = sum(r['local_reward'] for r in self._rows)
        self._seen_active_decision |= self._request_active
        # A discarded this prefix because it trained only after legal RL TOP.
        # B has an earlier causal action and continuous GAE: preserve the
        # ALREADY EXISTING real-terminal cost instead of erasing P09 failures.
        # Do not call task.reward twice or invent another failure criterion.
        inherited_terminal = 0.
        reason = step.info.get('termination_reason')
        if (self._seen_active_decision and not finish_active_at_request
                and step.terminated and not step.truncated and reason
                and reason != 'SUCCESS' and not step.info.get('full_task_success')
                and not step.info.get('local_reward', {}).get('terminal_event')):
            failure_cost = self.inner.task.config.failure_cost
            if not finite(failure_cost) or failure_cost <= 0:
                raise ValueError('original accepted task failure_cost required')
            inherited_terminal = -float(failure_cost)
        observation = self._encode(step.observation)
        info = dict(step.info)
        info['fl_forward_window'] = dict(schema=SCHEMA, request=start, current=deepcopy(self.fl_window),
            active=self._request_active, actor_sample=self._request_active,
            critic_only_sample=self._seen_active_decision and not self._request_active,
            prefix_excluded=not self._seen_active_decision,
            accepted_finish_reward=original_reward, local_reward=local,
            inherited_pre_finish_terminal_event=inherited_terminal,
            inherited_terminal_source='original task.config.failure_cost; actual terminal only',
            total_reward=original_reward+local+inherited_terminal, eligible_native_ticks=sum(r['eligible'] for r in self._rows),
            native_ticks=self._rows, ordinary_phase_change_is_terminal=False,
            frozen_A531_prefix_preserved=True, post_window_HISTORY_and_physical_influence_preserved=True)
        return replace(step, observation=observation, reward=original_reward+local+inherited_terminal, info=info)


def build_core(app, config=None):
    # Deferred until the parent explicitly installs/runs B. Merely importing
    # this prototype never imports Isaac, torch or the active training route.
    from wlr50_clean.ppo import semantic_finish_advance as finish
    if config is None:
        config = config_from_sources(finish.ROOT/'configs/recording_motion_contract.json', finish.ROOT/'configs/fsm_states.yaml')
    if isinstance(config, dict):
        config = FlWindowConfig(**config)
    return FlWindowCore(finish.build_core(app), config)
