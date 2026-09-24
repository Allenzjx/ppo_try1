"""UNEXECUTED outputs-only source counterexample; not a guard/success test.

Future invocation needs repository src plus tests/unit on PYTHONPATH. These
are synthetic scheduler fixtures, not rollout state edits or real contacts.
"""
from test_semantic_source_partial_order_v1 import contract, layer
from test_semantic_rear_policy_timing_control import provider, issue, position_source
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER


def test_current_late_drop_load_retains_remote_source_targets_not_executed_hold(contract):
    p = provider(contract)
    issue(p, contract, 0)
    source = layer(p, 'P09')
    start = source['motion']._scaled_source_tick(p._p09_late_source[1])
    position_source(source, start)  # CPU scheduler positioning only.
    issued, _ = issue(p, contract, 1, rr_bearing=True, gap=-.001)
    count = source['motion'].source_atomic_emitted
    assert issued[8:] == (-1.07, 0., 0., 0.)
    remote = tuple(issued[i] for i in (0, 1, 4, 5))
    dropped, item = issue(p, contract, 2, rr_bearing=False, gap=.035)
    assert source['ticks'] == start + 2  # Existing start-only gate, not continuous.
    assert tuple(dropped[i] for i in (0, 1, 4, 5)) == remote
    assert dropped[8] == -1.07 and source['motion'].source_atomic_emitted == count
    assert p.rear_policy_timing(item)['rr_carry_capture']
    # This pass exposes an unresolved target-persistence counterexample.
    # It does NOT assert final actuator motion stopped on lost bearing.


def test_source_stop_is_consumed_once_after_loss_and_must_not_be_delayed_by_joint_pause(contract):
    p = provider(contract)
    issue(p, contract, 0)
    source = layer(p, 'P09')
    start = source['motion']._scaled_source_tick(p._p09_late_source[1])
    position_source(source, start)
    issued, _ = issue(p, contract, 1, rr_bearing=True, gap=-.001)
    stop = next(w for w in contract.phase('P09').waypoints
        if w.time_s > p._p09_late_source[1] and set(w.atomic_channels) == set(WHEEL_ORDER)
        and w.full12[8:] == (0.,)*4)
    position_source(source, source['motion']._scaled_source_tick(stop.time_s))
    stopped, _ = issue(p, contract, 2, rr_bearing=False, gap=.035)
    assert stopped[8:] == (0.,)*4
    assert any(set(g.channels) == set(WHEEL_ORDER) for g in source['sample'].atomic_groups)
    assert tuple(stopped[i] for i in (0, 1, 4, 5)) == tuple(issued[i] for i in (0, 1, 4, 5))
    issue(p, contract, 3, rr_bearing=False, gap=.035)
    assert not source['sample'].atomic_groups
