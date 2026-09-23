"""Output-only CPU controller replay. Zero new physics/PPO; no contact invented."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wlr50_clean.ppo import semantic_rr_capture_assist as a

DATA = HERE / 'video_v5_c53119a_capture_follow_review/v5_capture_follow_fixed_COM.json'
DATA_SHA = '386b2bfa5f00f51b9fb5a5d1f5f86b8a4c8fe3ad6c760a9471c8dfa2e644463d'
DT = 1. / 120.


def tail_two(path):
    """Read only a bounded tail of the already sealed native stream."""
    with path.open('rb') as stream:
        stream.seek(0, 2); end = stream.tell(); start = end; data = b''
        while data.count(b'\n') < 3 and start:
            amount = min(start, 65536); start -= amount
            stream.seek(start); data = stream.read(amount) + data
            if len(data) > 2_000_000:
                raise RuntimeError('unexpected native-row size')
    assert data.endswith(b'\n')
    return [json.loads(line) for line in data.splitlines()[-2:]]


def upgrade_metadata_only(old):
    assert old['feedback_revision'] == 'progress_reserve_captured_incremental_v4'
    new = deepcopy(old)
    new.update(feedback_revision=a.RR_CAPTURE_FEEDBACK_REVISION,
               window_reference_semantics=a.RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS,
               capture_search_semantics=a.RR_CAPTURE_SEARCH_SEMANTICS)
    assert all(new[k] == old[k] for k in a.RR_CAPTURE_ASSIST_FEATURE_NAMES)
    a.validate_rr_capture_assist_snapshot(new)
    return new


def advance(snapshot, context, previous):
    obj = a.RRHipOnlyCaptureAssist.from_snapshot(snapshot)
    receipt = obj.advance(context=context, previous_final_full12=previous, physics_dt_s=DT)
    return receipt


def small(receipt):
    b, s = receipt['state_before'], receipt['state_after']
    return dict(before_mode=b['mode_name'], after_mode=s['mode_name'], reason=s['reason'],
                travel_before=b['travel_used_deg'], travel_after=s['travel_used_deg'],
                exposure_after=s['descent_elapsed_s'], knee_before=b['knee_hold_deg'],
                knee_after=s['knee_hold_deg'], knee_delta=s['knee_hold_deg']-b['knee_hold_deg'],
                contact_seen=s['contact_seen'], context=receipt['context'])


def main():
    blob = DATA.read_bytes(); assert hashlib.sha256(blob).hexdigest() == DATA_SHA
    evidence = json.loads(blob); terminal = evidence['milestones'][-1]
    p, n = terminal['physical_post_step'], terminal['dispatch_pre_step']
    source = Path(evidence['source'])
    prev, last = tail_two(source / 'native_tick_audit.jsonl')
    assert (prev['episode_physics_tick'], last['episode_physics_tick']) == (14594, 14595)
    old_receipt = last['native_audit']['rr_capture_assist_evidence']
    assert old_receipt == n['rr_assist']
    audit = last['native_audit']
    assert audit['rr_carry_wheel_context_and_previous_FINAL_independently_verified'] is True
    previous_final = (audit['previous_final_drive_servo_deg']
                      + audit['previous_final_drive_wheel_rad_s'])
    assert previous_final == prev['native_audit']['rr_carry_wheel_evidence']['output_full12']
    actual_final = audit['rr_carry_wheel_evidence']['output_full12']
    assert actual_final == p['commanded_full12']
    try:
        a.validate_rr_capture_assist_snapshot(old_receipt['state_after'])
    except ValueError:
        legacy_rejected = True
    else:
        raise AssertionError('legacy metadata silently accepted')

    # The next context reuses the ACTUAL terminal post-sensors, not a predicted
    # contact. The following dispatch is offline, not a new physical sample.
    next_context = deepcopy(old_receipt['context'])
    rr = terminal['current_capture_post_step']['current_legs']['RR']
    next_context.update(dispatch_physics_tick=old_receipt['state_after']['last_dispatch_physics_tick']+1,
                        source_observation_tick=p['post_tick'], gap_m=p['RR_gap_m'],
                        hip_actual_deg=p['actual_full12_canonical_deg_rad_s'][6],
                        knee_actual_deg=p['actual_full12_canonical_deg_rad_s'][7])
    for key in ('air', 'ground_contact', 'top_surface_contact', 'obstacle_pair_active'):
        assert next_context[key] == rr[key]
    assert rr['within_top_xy'] and rr['current_lift_valid'] and rr['front_edge_crossed']
    assert rr['bearing_force_n'] == 0. and next_context['current_top_bearing'] is False
    literal_terminal = advance(upgrade_metadata_only(old_receipt['state_after']), next_context, actual_final)
    assert literal_terminal['state_after']['mode_name'] == 'BLOCKED'
    assert literal_terminal['state_after']['travel_used_deg'] == old_receipt['state_after']['travel_used_deg']

    # Different, explicit control counterfactual: at the last v5 dispatch the
    # actual input still had mode6 earned credit. v6 does NOT exhaust at 52 here.
    boundary = advance(upgrade_metadata_only(old_receipt['state_before']), old_receipt['context'], previous_final)
    assert boundary['state_after']['mode_name'] == 'DESCEND_PROGRESS'
    assert all(boundary['state_after'][k] == old_receipt['state_after'][k]
               for k in ('hip_target_deg', 'knee_hold_deg', 'travel_used_deg', 'descent_elapsed_s'))
    continued = advance(boundary['state_after'], next_context, actual_final)
    assert continued['state_after']['mode_name'] == 'DESCEND_PROGRESS'
    assert 0. < continued['state_after']['knee_hold_deg']-boundary['state_after']['knee_hold_deg'] <= DT + 1e-12
    assert continued['state_after']['contact_seen'] == 0.

    negatives = {}
    for name, change in (('invalid_physical', {'physical_valid': False}),
                         ('support_lost', {'other_support_count': 1}),
                         ('XY_lost', {'within_top_xy': False}),
                         ('rebound_above_1mm', {'gap_m': .0011}),
                         ('tracking_lost', {'knee_actual_deg': 10.})):
        ctx = deepcopy(next_context); ctx.update(change)
        r = advance(boundary['state_after'], ctx, actual_final)
        assert r['state_after']['mode_name'] == 'BLOCKED'
        assert r['state_after']['travel_used_deg'] == boundary['state_after']['travel_used_deg']
        negatives[name] = small(r)

    # Pure controller counterexample, not future physics: repeat this actual
    # last sensor input for 121 calls; only public controller clocks advance.
    obj = a.RRHipOnlyCaptureAssist.from_snapshot(boundary['state_after'])
    initial_knee = obj.state['knee_hold_deg']
    for _ in range(121):
        ctx = deepcopy(next_context); ctx['dispatch_physics_tick'] = obj.last_tick+1
        out = obj.advance(context=ctx, previous_final_full12=actual_final, physics_dt_s=DT)
    assert abs(obj.state['travel_used_deg']-53.) < 1e-9
    assert obj.state['knee_hold_deg']-initial_knee <= 1. + 1e-9
    assert obj.snapshot()['mode_name'] == 'BLOCKED'
    assert obj.state['contact_seen'] == 0.
    result = dict(schema='readonly.v6_contact_onset_actual_inputs.v1', source=str(source),
                  input_sha256=DATA_SHA, production_module_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
                  new_physics_ticks=0, new_PPO_updates=0, legacy_metadata_rejected=legacy_rejected,
                  metadata_upgrade='Only three explicit metadata strings replaced; all 14 numeric states preserved.',
                  literal_v5_terminal_next_step=small(literal_terminal),
                  continuous_v6_at_actual_last_v5_input=small(boundary),
                  continuous_v6_next_step_using_actual_terminal_sensors=small(continued),
                  counterfactual_negative_contexts_not_physical_samples=negatives,
                  repeated_frozen_sensor_controller_bound=dict(calls=121, physical_samples_added=0,
                      knee_delta_deg=obj.state['knee_hold_deg']-initial_knee, state=obj.snapshot()),
                  caveat='No contact was created; existing unit contact fixtures separately test stop behavior. '
                         'Literal already-BLOCKED v5 terminal cannot regain credit merely by metadata migration; '
                         'fresh v6 execution preserves mode6 across the old 52-degree boundary instead.')
    output = HERE / 'v6_contact_onset_actual_replay.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), terminal_mode=literal_terminal['state_after']['mode_name'],
                          boundary_mode=boundary['state_after']['mode_name'],
                          next_delta=small(continued)['knee_delta'], maximum_extra=obj.state['knee_hold_deg']-initial_knee)))


if __name__ == '__main__':
    main()
