import math
from direction_probe import FiniteDirection, trigger

def test_single_anchored_delta_not_recursive():
    probe = FiniteDirection('FR_RL_minus3')
    anchor = [0.]*12
    anchor[4] = 1.2
    probe.start(anchor)
    ev = {'history': {'placed': {'FR': False, 'FL': False, 'RR': False}}}
    base = [0.1]*12
    for _ in range(60):
        raw, receipt = probe.apply(base, [24.]*12, 'P02', ev)
        base = list(raw)
    assert abs(24*math.tanh(raw[4])-(-1.8)) < 1e-12
    assert all(raw[i] == .1 for i in range(12) if i != 4)
    assert receipt['delta_deg'] == {4: -3.}

def test_cap_transition_preserves_degrees():
    probe = FiniteDirection('FL_minus3')
    probe.start([1.]*12)
    ev = {'history': {'placed': {'FR': True, 'FL': False, 'RR': False}}}
    for _ in range(20):
        raw, _ = probe.apply([.2]*12, [18.]*12, 'P05', ev)
    before = 18*math.tanh(raw[0])
    raw, _ = probe.apply([.2]*12, [32.]*12, 'P06', ev)
    assert abs(before-32*math.tanh(raw[0])) < 1e-12

def test_FR_trigger_requires_real_carry():
    ev = {'valid': True, 'current_legs': {'FR': {'air': True, 'clearance_m': .020,
           'consecutive_air_samples': 3}}, 'history': {'active_lift': {'FR': True}}}
    assert trigger('FR_RL_minus3', 'P02', ev, False)
    ev['current_legs']['FR']['clearance_m'] = .010
    assert not trigger('FR_RL_minus3', 'P02', ev, False)

def test_all_zero_control_has_no_intervention():
    probe = FiniteDirection('control')
    base = tuple(i/10 for i in range(12))
    result, receipt = probe.apply(base, [24.]*12, 'P02', {})
    assert result == base and receipt is None

def test_finite_release_retains_nonselected_actions():
    probe = FiniteDirection('FL_minus3')
    probe.start([0.]*12)
    ev = {'history': {'placed': {'FR': True, 'FL': True, 'RR': False}}}
    for _ in range(66):
        raw, _ = probe.apply([.123]*12, [24.]*12, 'P06', ev)
    assert probe.complete
    assert all(abs(x-.123) < 1e-12 for x in raw)
