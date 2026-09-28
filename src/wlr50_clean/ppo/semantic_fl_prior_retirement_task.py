"""Explicit observed one-second prior-retirement clock; no action writes."""
from dataclasses import replace
import math

from wlr50_clean.ppo.semantic_fl_forward_task import FlWindowCore, read_window

RETIREMENT_FIELD = 'b_fl_old_prior_retirement_lambda'
OBSERVATION_DIMENSION = 536
LAMBDA_INDEX = 535
RAMP_DURATION_S = 1.
SCHEMA = 'wlr50_clean.p06_fl_prior_retirement_context.v1'


class RetirementClock:
    """Clock over actual observations, not counted decisions or private actor state."""
    def __init__(self, ramp_duration_s=RAMP_DURATION_S):
        if type(ramp_duration_s) not in (int, float) or not math.isfinite(ramp_duration_s) or ramp_duration_s <= 0.:
            raise ValueError('positive declared retirement ramp duration required')
        self.ramp_duration_s = float(ramp_duration_s)
        self.reset()

    def reset(self):
        self.tick = None
        self.active = False
        self.value = 0.
        self.entry_tick = None

    def observe(self, tick, active):
        if type(tick) is not int or tick < 0 or type(active) is not bool:
            raise ValueError('explicit native tick and window permission required')
        if self.tick is not None and (tick < self.tick or tick == self.tick and active != self.active):
            raise ValueError('same-tick permission cannot change; observation time cannot reverse')
        if tick == self.tick:
            return self.snapshot()
        if not active:
            self.value, self.entry_tick = 0., None
        elif not self.active:
            self.value, self.entry_tick = 0., tick
        else:
            # Absolute measured age avoids 120 additions accumulating enough
            # rounding to make an exactly one-second ramp miss lambda=1.
            self.value = min(1., (tick-self.entry_tick)/(120.*self.ramp_duration_s))
        self.tick, self.active = tick, active
        return self.snapshot()

    def snapshot(self):
        return dict(schema=SCHEMA, active=self.active, observation_tick=self.tick,
            retirement_lambda=self.value, entry_tick=self.entry_tick,
            ramp_duration_s=self.ramp_duration_s, ramp_is_declared_control_change=True, physical_validation_is_run_specific=True,
            owner_indices=[], additional_projection=False, HISTORY_reset=False,
            source_clock_or_stop_modified=False)


class PriorRetirementCore(FlWindowCore):
    """Same FinishCore/task/observer; one appended observable scalar only."""
    def __init__(self, inner, config, *, ramp_duration_s=RAMP_DURATION_S):
        self.retirement = RetirementClock(ramp_duration_s)
        self._retirement_rows = []
        super().__init__(inner, config)

    def _observe(self, before, after, projection):
        current = read_window(after, self.config)
        self.retirement.observe(after.physics_tick, current['active'])
        if self._collect:
            self._retirement_rows.append(self.retirement.snapshot())
        super()._observe(before, after, projection)

    def _encode(self, original):
        prefix = super()._encode(original)
        if len(prefix) != LAMBDA_INDEX:
            raise ValueError('unchanged complete B535 prefix required')
        self.retirement.observe(self.inner.frame.physics_tick, self.fl_window['active'])
        self.observation = tuple(prefix)+(self.retirement.value,)
        return self.observation

    def reset(self, seed=1001):
        self.retirement.reset()
        return super().reset(seed=seed)

    def step(self, raw):
        request = self.retirement.snapshot()
        self._retirement_rows = []
        step = super().step(raw)
        info = dict(step.info)
        info['fl_prior_retirement'] = dict(schema=SCHEMA, request=request,
            current=self.retirement.snapshot(), original_eight_native_tick_raw_hold=True,
            native_observation_contexts=self._retirement_rows,
            no_native_instantaneous_raw_cancellation_claim=True)
        return replace(step, info=info)


def build_core(app, config=None, ramp_duration_s=RAMP_DURATION_S):
    """Construct exactly one accepted physics core; the CPU tests never call it."""
    from wlr50_clean.ppo import semantic_finish_advance as finish
    from wlr50_clean.ppo.semantic_fl_forward_task import config_from_sources, FlWindowConfig
    if config is None:
        config = config_from_sources(finish.ROOT/'configs/recording_motion_contract.json', finish.ROOT/'configs/fsm_states.yaml')
    if isinstance(config, dict):
        config = FlWindowConfig(**config)
    return PriorRetirementCore(finish.build_core(app), config, ramp_duration_s=ramp_duration_s)
