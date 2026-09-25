"""OFFLINE candidate only. Never imported by production or an Isaac run.

Preserves the existing .2 RR retention share, geometric/contact halves, scale,
current support predicate and completion semantics. Alpha is a proposed soft
allocation, NOT a contact threshold or fitted learned parameter.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from inspect_completed384_live import RUN, OUT, bounded_prefix

ALPHA = .25
GAP_SCALE = .025
XY_SCALE = .25
LOWER_GAP = -.015
GLOBAL_RR_RETENTION_WEIGHT = .85*.2/4
GAMMA, POTENTIAL_WEIGHT = .9985, 5.


def clamp(value): return max(0., min(1., value))


def measured_top(r):
    return bool(r.get('ground_contact') is False and r.get('air') is False
        and r.get('top_contact') is True and r.get('top_surface_contact') is True
        and r.get('obstacle_pair_active') is True and r.get('within_top_xy') is True
        and r.get('within_lateral_span') is True and r.get('contact_surface') == 'TOP'
        and r['clearance_m'] >= LOWER_GAP)


def measured_bearing(r):
    return bool(measured_top(r) and r.get('support') is True
        and r.get('bearing_verified') is True and r.get('bearing_force_n',-1.) >= .2)


def parts(r, *, live=True):
    if not live:
        return dict(old=0., candidate=0., geometry=0., preparation=0., established_geometry=0.,
                    contact=0., mode='invalid_or_terminal', current_bearing=False)
    outside, gap = r['top_xy_outside_distance_m'], r['clearance_m']
    assert math.isfinite(outside) and outside >= 0 and math.isfinite(gap)
    # Existing bounded geometric kernel. There is no target angle, velocity,
    # reference copying, force maximization or high-clearance reward.
    geometry = clamp(1.-outside/XY_SCALE)*GAP_SCALE/(GAP_SCALE+abs(gap))
    top, bearing = measured_top(r), measured_bearing(r)
    contact = clamp(r.get('consecutive_top_samples',0)/2.) if bearing else 0.
    legal_old = bool(r.get('within_top_xy') is True and r.get('within_lateral_span') is True
        and r.get('ground_contact') is False
        and (top or (r.get('air') is True and r.get('obstacle_pair_active') is False))
        and gap >= LOWER_GAP)
    old = .5*geometry+.5*contact if legal_old else 0.
    established = bool(legal_old or (r.get('current_lift_valid') is True
        and r.get('active_attempt') is True and r.get('motion_continuation_allowed') is True
        and r.get('ground_contact') is False))
    # Unqualified/ground geometry consumes only 1/4 of the OLD geometric half.
    # Current established motion unlocks its remainder outside the old region.
    # Keep the original legal-AIR/TOP kernel exactly, including a temporary
    # current-qualification drop: no new penalty/gate on that accepted path.
    # Do not unlock based on historical placed/current phase or source clocks.
    preparation = .5*ALPHA*geometry
    established_geometry = .5*(1.-ALPHA)*float(established)*geometry
    current_contact = .5*contact
    candidate = preparation+established_geometry+current_contact
    mode=('TOP_BEARING' if bearing else 'TOP_NOT_BEARING' if top else
          'GROUND_PREPARATION_ONLY' if r.get('ground_contact') else
          'QUALIFIED_AIR' if r.get('current_lift_valid') and r.get('air') else
          'LEGAL_AIR_GEOMETRY_NOT_QUALIFICATION' if legal_old and r.get('air') else
          'QUALIFIED_EDGE_NO_BEARING' if established else 'UNQUALIFIED_GEOMETRY_ONLY')
    return dict(old=old,candidate=candidate,geometry=geometry,preparation=preparation,
        established_geometry=established_geometry,contact=current_contact,mode=mode,current_bearing=bearing)


def tests():
    base=dict(ground_contact=True,air=False,top_contact=False,top_surface_contact=False,
        obstacle_pair_active=False,within_top_xy=False,within_lateral_span=True,contact_surface='NONE',
        clearance_m=-.05,top_xy_outside_distance_m=.05,current_lift_valid=False,active_attempt=False,
        motion_continuation_allowed=True,support=True,bearing_verified=True,bearing_force_n=5.,
        consecutive_top_samples=0)
    ground=parts(base)
    closer=dict(base,top_xy_outside_distance_m=.01)
    assert 0 < ground['candidate'] < parts(closer)['candidate'] <= .125
    assert ground['contact']==0 and not ground['current_bearing']
    air=dict(base,ground_contact=False,air=True,bearing_force_n=0.,support=False,
        clearance_m=.02,top_xy_outside_distance_m=0.,within_top_xy=True)
    unqualified=parts(air)
    assert unqualified['candidate'] == unqualified['old'] <= .5 and unqualified['contact']==0
    qualified=dict(air,current_lift_valid=True,active_attempt=True)
    assert parts(qualified)['candidate']==parts(qualified)['old'] <= .5
    assert parts(dict(qualified,clearance_m=.005))['candidate']>parts(qualified)['candidate']
    outside=dict(qualified,within_top_xy=False,top_xy_outside_distance_m=.005000001)
    inside=dict(outside,within_top_xy=True,top_xy_outside_distance_m=.004999999)
    assert abs(parts(outside)['candidate']-parts(inside)['candidate'])<1e-8
    below=dict(qualified,clearance_m=LOWER_GAP-1e-9)
    above=dict(qualified,clearance_m=LOWER_GAP+1e-9)
    assert abs(parts(below)['candidate']-parts(above)['candidate'])<1e-7
    top=dict(qualified,air=False,top_contact=True,top_surface_contact=True,obstacle_pair_active=True,
        contact_surface='TOP',clearance_m=0.,bearing_force_n=5.,support=True,consecutive_top_samples=2)
    assert parts(top)['candidate']==1.
    assert parts(dict(top,bearing_verified=False))['candidate']==.5
    assert parts(dict(top,ground_contact=True,current_lift_valid=False))['candidate']<=.125
    assert parts(dict(top,current_lift_valid=False))['candidate']==1., 'actual TOP bearing is not revoked by transient other-support qualification'
    wall=dict(base,ground_contact=False,obstacle_pair_active=True,bearing_force_n=100.,
        contact_surface='FRONT_WALL',within_top_xy=True,clearance_m=-.001)
    assert parts(wall)['candidate'] <= .125 and parts(wall)['contact']==0
    assert parts(top,live=False)['candidate']==0
    assert parts(top)['candidate']-parts(dict(qualified,clearance_m=0.))['candidate']==.5
    static_phi=GLOBAL_RR_RETENTION_WEIGHT*ground['candidate']
    assert POTENTIAL_WEIGHT*(GAMMA*static_phi-static_phi)<0
    sequence=[ground['candidate'],parts(qualified)['candidate'],1.,ground['candidate']]
    discounted=sum(GAMMA**i*(GAMMA*sequence[i+1]-sequence[i]) for i in range(len(sequence)-1))
    assert math.isclose(discounted,(GAMMA**3-1)*sequence[0],abs_tol=1e-12) and discounted<0
    return dict(status='PASS',cases=17,scope='pure scalar counterexamples, not physical validation or PPO learning',
        ground=ground,qualified_air=parts(qualified),top=parts(top),unqualified_air=unqualified,
        discounted_same_state_loop=discounted,alpha=ALPHA)


def summarize(rows):
    def interval(key):
        xs=[r[key] for r in rows if r[key] is not None]
        return [min(xs),max(xs)] if xs else None
    return dict(count=len(rows),old_retention_range=interval('old_retention'),
        candidate_retention_range=interval('candidate_retention'),
        old_global_phi_range=interval('old_global_phi'),candidate_global_phi_range=interval('candidate_global_phi'),
        delta_phi_range=interval('delta_phi'),aligned_local_delta_shaping_range=interval('delta_shaping_reward'),
        aligned_local_delta_shaping_sum=sum(r['delta_shaping_reward'] for r in rows if r['delta_shaping_reward'] is not None))


def main():
    target=OUT/'rr_recapture_potential_candidate_offline_v2.json'
    assert not target.exists(), 'retain previous candidate report'
    data,source=bounded_prefix(RUN/'residual_and_projection_audit.jsonl',384)
    assert data[-1]['global_policy_decision']==225664
    rows=[]; previous=None; episode=1
    for row in data:
        a=row['applied_audit']; ev=a['semantic_task']['physical_evaluator']; rr=ev['current_legs']['RR']; rl=ev['current_legs']['RL']
        if previous is not None and a['physics_tick']<previous['tick']: episode+=1
        active=(ev['history']['placed']['RR'] and not ev['history']['placed']['RL']
            and not (rl.get('current_lift_valid') and rl.get('motion_continuation_allowed') and rl.get('air') and not rl.get('ground_contact')))
        p=parts(rr,live=ev['valid'] and ev['termination_reason'] is None)
        phi=a['semantic_task']['task_progress_potential']
        delta=GLOBAL_RR_RETENTION_WEIGHT*(p['candidate']-p['old']) if active else 0.
        candidate_phi=phi+delta
        assert 0 <= candidate_phi <= 1.
        start=a['physics_tick']-a['physics_ticks']
        aligned=previous is not None and previous['tick']==start and not previous['terminal']
        if aligned:
            assert math.isclose(previous['old_global_phi'],a['reward_breakdown']['potential_before'],abs_tol=1e-9)
            delta_reward=POTENTIAL_WEIGHT*(GAMMA*(0. if row['terminal'] else delta)-previous['delta_phi'])
        else: delta_reward=None
        record=dict(global_policy_decision=row['global_policy_decision'],episode=episode,tick=a['physics_tick'],
            sim_time_s=a['sim_time_s'],phase=a['phase_id'],terminal=row['terminal'],candidate_scope=active,
            mode=p['mode'],RR_ground=rr['ground_contact'],RR_gap_m=rr['clearance_m'],
            RR_outside_xy_m=rr['top_xy_outside_distance_m'],RR_current_qualification=rr['current_lift_valid'],
            RL_current_qualification=rl['current_lift_valid'],old_retention=p['old'],candidate_retention=p['candidate'],
            old_global_phi=phi,candidate_global_phi=candidate_phi,delta_phi=delta,
            delta_shaping_reward=delta_reward,original_reward=row['reward'],
            offline_hypothetical_reward=(row['reward']+delta_reward if delta_reward is not None else None),
            parts=p)
        rows.append(record);previous=record
    eligible=[r for r in rows if r['candidate_scope']]
    ground=[r for r in eligible if r['episode']==3 and r['RR_ground'] and not r['RL_current_qualification']]
    selected_ticks={5992,6000,6008,6016,6056,6064,6080,6088,6120,6128,6320,6328,7000}
    result=dict(schema='offline.unapplied_RR_recapture_potential_candidate.v2',source=source,
        supersedes='rr_recapture_potential_candidate_offline.json: v1 reduced preexisting legal-AIR geometry when qualification temporarily dropped; v2 preserves every original legal-AIR/TOP value.',
        optimized_prefix=dict(decisions=384,updates=[1726,1727,1728],last_global=225664),
        applied_to_training=False,physics_interventions=0,new_PPO_updates=0,new_auxiliary_updates=0,
        formula='R=.5*g*(alpha+(1-alpha)*(old_legal_AIR_or_TOP OR current_qualified_non_ground))+.5*actual_TOP_contact_fraction; g=clip(1-outside_xy/.25)*.025/(.025+abs(gap)); alpha=.25',
        scope='Only RR already history-placed but lacking a legitimate ongoing RL swing/RL placement; all other old paths unchanged.',
        total_retention_global_weight=GLOBAL_RR_RETENTION_WEIGHT,
        semantics=['Ground or unqualified geometry outside the old legal-AIR/TOP region <=.125 is preparation only, not lift/cross/TOP/support.',
            'Existing legal-AIR geometric share remains <=.5 even if current qualification momentarily drops; it is not new active-lift evidence and acceptance stays unchanged.',
            'Current-qualified AIR/EDGE geometric progress <=.5, never bearing; actual TOP contact half requires existing exact sensor-bearing predicate.',
            'Actual TOP remains eligible despite momentary current_lift_valid=False due other-support evidence, matching existing contact handling.',
            'Same physical-mode geometry continuous across old XY/top-gap zero cuts; qualification/contact evidence gates remain discrete and are not claimed globally continuous.',
            'Alpha quarter is an unvalidated allocation hypothesis within the old half-share, not physical evidence or an increased reward weight.',
            'No policy/reward/config/rollout/checkpoint/source edits; offline hypothetical shaping differences are not training improvements.',
            'Global phi delta uses only changed .85/4*.2 RR retention. Old .8 history constant and all other leg terms remain unchanged.',
            'First-prefix/unmatched before-state has no inferred candidate reward. Identical source trajectory is reused without counterfactual dynamics.'],
        tests=tests(),all_eligible_summary=summarize(eligible),third_RR_ground_summary=summarize(ground),
        selected_events=[r for r in rows if r['episode']==3 and r['tick'] in selected_ticks],
        ground_rows=ground)
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('tests','all_eligible_summary','third_RR_ground_summary','selected_events')},indent=2))


if __name__=='__main__':main()
