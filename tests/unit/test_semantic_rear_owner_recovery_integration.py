"""CPU tensor seams, not an Isaac run or evidence of physical task success."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import plan
from test_semantic_rr_capture_context import measured, top_rr, SUPPORT_SPEC
from test_semantic_source_partial_order_v1 import contract, layer
from test_semantic_rear_policy_timing_control import (
    provider, issue, position_source,
)
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.actuator_target_effect import (
    ActuatorTargetEffectError, actuator_target_audit_request,
    build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.semantic_rear_owner_recovery import RearOwnerRecovery, INDICES
from wlr50_clean.ppo.semantic_rear_policy_timing import EDGE_RECOVERY_MODE

ZERO = (0.,) * 12
CAPS = (24.,) * 8 + (1.,) * 4


class Chain:
    """Real backend composition, real adapter/mapper, independent native audit."""
    def __init__(self):
        self.adapter = _adapter()
        self.owner = RearOwnerRecovery()
        self.owners = [False] * 4
        self.task, _ = measured()
        self.task['stage_id'] = 'P12'
        self.backend = SemanticIsaacBackend.__new__(SemanticIsaacBackend)
        b = self.backend
        b._rear_owner_recovery = None
        b._reset_prime_tick_count = 0
        b._rear_policy_timing_mode = EDGE_RECOVERY_MODE
        b._rr_support_spec = SUPPORT_SPEC
        b._policy_headroom_mode = HEADROOM_MODE
        b._rr_carry_pre_dispatch_context = lambda _tick: None
        b.execution_profile = {'residual': {'phase_caps_full12': {'P12': CAPS}}}
        b._controller = SimpleNamespace(task_snapshot=self.task,
            nominal_provider=SimpleNamespace(rear_late_owner_bits=lambda: list(self.owners)))
        self.tick = 0
        self.dispatch()  # A real committed ACK, not fabricated preceding history.
        b._rear_owner_recovery = self.owner

    def dispatch(self, request=ZERO, nominal=ZERO, *, tick=None, audit=True):
        b, a = self.backend, self.adapter
        actuation = plan(request, nominal=nominal)
        b._semantic_actuation_plan = actuation
        previous = tuple(a._final_drive_servo_deg[name] for name in SERVO_ORDER)
        self.tick = self.tick + 1 if tick is None else tick
        count, events = a.write_count, len(a.robot.events)
        ack = b._atomic_apply(a, nominal, physics_tick=self.tick,
            tracking_servo_names=(),
            drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12)
        assert a.write_count == count + 1
        assert a.robot.events[events:] == ['position.setter', 'velocity.setter', 'dispatch']
        assert ack['articulation_writes_this_call'] == 1
        assert ack['independent_policy_residual_requested_full12'] == list(request)
        assert tuple(a._final_drive_servo_deg[name] for name in SERVO_ORDER) == tuple(ack['drive_target_full12'][:8])
        for before, after in zip(previous, ack['drive_target_full12']):
            assert abs(after-before) <= a.servo_target_mapper.maximum_delta_deg + 1e-12
        inputs = dict(adapter=a, actuation=actuation, raw_ack=ack,
            previous_final_drive_servo_deg=previous, source_phase_id='P12',
            policy_request=actuator_target_audit_request('P12', (.1,) * 12, (1,) * 12),
            policy_headroom_mode=HEADROOM_MODE)
        if audit:
            checked = build_actuator_target_effect_audit(**inputs)
            assert checked['verified'] and checked['setter_dispatch_targets_equal']
            assert checked['actual_mapping_matches_dispatch']
            assert a.write_count == count + 1  # Readback never dispatches twice.
        return ack, inputs


def with_channels(values):
    result = [0.] * 12
    for i, value in values.items():
        result[i] = value
    return tuple(result)


def test_issued_remote_targets_suspend_at_previous_final_without_request_creep():
    c = Chain()
    request = with_channels({0: 2., 1: -2., 4: 3., 5: -3.})
    remote = with_channels({0: 40., 1: -40., 4: 40., 5: -40.})
    before, _ = c.dispatch(request, remote)
    c.owners[:] = [True] * 4
    anchor = [before['drive_target_full12'][i] for i in INDICES]
    for _ in range(5):
        ack, _ = c.dispatch(request, remote)
        receipt = ack['rear_owner_recovery_evidence']
        assert receipt['state_after']['active'] == [True] * 4
        assert receipt['state_after']['anchor_final_deg'] == anchor
        assert [ack['drive_target_full12'][i] for i in INDICES] == anchor
        assert receipt['state_after']['anchor_request_deg'] == [request[i] for i in INDICES]
        assert not receipt['task_assist'] and not receipt['source_stops_modified']
    assert any(abs(ack['native_drive_target_full12'][i]-anchor[j]) > 1.
               for j, i in enumerate(INDICES))  # Mapper continues, its old N no longer pulls.


@pytest.mark.parametrize('channel', INDICES)
@pytest.mark.parametrize('direction', [-1., 1.])
def test_policy_request_changes_remain_bidirectional_under_suspended_owner(channel, direction):
    c = Chain()
    c.owners[:] = [True] * 4
    before, _ = c.dispatch()
    anchor = deepcopy(c.owner.snapshot())
    request = with_channels({channel: direction * .5})
    ack, _ = c.dispatch(request, nominal=(30.,) * 8 + ZERO[8:])
    assert ack['drive_target_full12'][channel] == pytest.approx(before['drive_target_full12'][channel] + direction*.5)
    assert c.owner.snapshot() == anchor
    assert ack['independent_policy_residual_requested_full12'][channel] == direction*.5


def test_explicit_wheel_stop_remains_zero_while_all_four_servo_owners_suspend():
    c = Chain()
    c.owners[:] = [True] * 4
    ack, _ = c.dispatch(nominal=ZERO[:8]+(-.3, .3, -.2, .2))
    assert ack['drive_target_full12'][8:] == [-.3, .3, -.2, .2]
    ack, _ = c.dispatch(nominal=ZERO)
    assert ack['drive_target_full12'][8:] == [0.] * 4
    assert ack['rear_owner_recovery_evidence']['owner_indices'] == list(INDICES)


@pytest.mark.parametrize('release', ['new_owner', 'rr_current_bearing', 'rl_air', 'rl_edge'])
def test_release_is_per_channel_and_still_final_slew_bounded(release):
    c = Chain()
    c.owners[:] = [True] * 4
    before, _ = c.dispatch()
    if release == 'new_owner':
        c.owners[:] = [False] * 4
    elif release == 'rr_current_bearing':
        top_rr(c.task)
    else:
        ev = c.task['physical_evaluator']
        ev['current_legs']['RL'].update(current_lift_valid=True,
            motion_continuation_allowed=True, active_attempt=True,
            current_lift_qualified_tick=ev['physics_tick'], ground_contact=False,
            air=release == 'rl_air', obstacle_pair_active=release == 'rl_edge',
            contact_reaction=release == 'rl_edge', contact_reaction_force_n=1.,
            contact_mode='FRONT_WALL' if release == 'rl_edge' else 'AIR',
            contact_surface='FRONT_WALL' if release == 'rl_edge' else 'NONE',
            top_contact=False, top_surface_contact=False, support=False, bearing_force_n=0.)
    ack, _ = c.dispatch(nominal=(30.,)*8+ZERO[8:])
    active = ack['rear_owner_recovery_evidence']['state_after']['active']
    expected = [True]*4 if release == 'rl_edge' else [True, True, False, False] if release == 'rl_air' else [False]*4
    assert active == expected
    for j, i in enumerate(INDICES):
        if active[j]:
            assert ack['drive_target_full12'][i] == before['drive_target_full12'][i]
        else:
            assert ack['drive_target_full12'][i] > before['drive_target_full12'][i]


def test_nonadjacent_ack_rejected_before_actuator_write():
    c = Chain()
    c.owners[:] = [True] * 4
    count = c.adapter.write_count
    with pytest.raises(ValueError, match='adjacent committed ACK'):
        c.dispatch(tick=c.tick+2)
    assert c.adapter.write_count == count


@pytest.mark.parametrize('field', ['state_before', 'previous_requested_full12', 'context', 'previous_final_full12'])
def test_native_audit_rejects_ack_only_owner_provenance_changes(field):
    c = Chain()
    c.owners[:] = [True] * 4
    ack, inputs = c.dispatch()
    receipt = ack['rear_owner_recovery_evidence']
    if field == 'state_before':
        receipt[field]['winning_late_owner'][0] = True
    elif field == 'context':
        receipt[field]['rl_edge_recovery_permitted'] = not receipt[field]['rl_edge_recovery_permitted']
    else:
        receipt[field][0] += 1.
    with pytest.raises(ActuatorTargetEffectError, match='independent pre-dispatch'):
        build_actuator_target_effect_audit(**inputs)


def test_p12_issued_rl_owner_pauses_without_stalling_stop_or_catching_up(contract):
    p = provider(contract, 'P12')
    issue(p, contract, 0, 'P12', rr_bearing=True)
    source = layer(p, 'P12')
    phase = contract.phase('P12')
    pulse = next(w for w in phase.waypoints if any(w.full12[8:]) and w.time_s > 0.)
    position_source(source, source['motion']._scaled_source_tick(pulse.time_s))
    command, _ = issue(p, contract, 1, 'P12', rr_bearing=True)
    assert any(command[8:])
    joint_ticks = source['rl_ticks']
    stop = next(w for w in phase.waypoints if w.time_s > pulse.time_s
        and set(w.atomic_channels) == set(WHEEL_ORDER) and not any(w.full12[8:]))
    position_source(source, source['motion']._scaled_source_tick(stop.time_s))
    stopped, _ = issue(p, contract, 2, 'P12')
    assert stopped[8:] == (0.,)*4
    assert source['rl_ticks'] == joint_ticks and source['rl_dependency_wait']
    assert source['rl_touched']
    assert p.rear_late_owner_bits()[2:] == [i in source['rl_touched'] for i in (4, 5)]
    assert sum(set(g.channels) == set(WHEEL_ORDER) for g in source['sample'].atomic_groups) == 1
    issue(p, contract, 3, 'P12')
    assert not source['sample'].atomic_groups
    assert source['rl_ticks'] == joint_ticks
    issue(p, contract, 4, 'P12', rr_bearing=True)
    assert source['rl_ticks'] == joint_ticks+1
    assert source['rl_sample'].tick_index == joint_ticks


def test_p09_started_late_owner_is_identified_after_drop_load_without_replaying_group(contract):
    p = provider(contract)
    issue(p, contract, 0)
    source = layer(p, 'P09')
    pending = source['motion']._scaled_source_tick(p._p09_late_source[1])
    position_source(source, pending)
    started, _ = issue(p, contract, 1, rr_bearing=True)
    issued = source['motion'].source_atomic_emitted
    first_owner = p.rear_late_owner_bits()
    assert any(first_owner)
    c = Chain()
    top_rr(c.task)
    c.owners[:] = first_owner
    before, _ = c.dispatch(nominal=started)
    loss, _ = measured()
    c.task['physical_evaluator'] = loss['physical_evaluator']
    continued, _ = issue(p, contract, 2)
    c.owners[:] = p.rear_late_owner_bits()
    after, _ = c.dispatch(nominal=continued)
    assert source['ticks'] == pending+2  # No fictional source pause or replay.
    assert source['motion'].source_atomic_emitted == issued
    for j, i in enumerate(INDICES):
        if c.owners[j]:
            assert after['drive_target_full12'][i] == before['drive_target_full12'][i]
    assert after['drive_target_full12'][8:] == list(continued[8:])
