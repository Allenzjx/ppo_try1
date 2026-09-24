"""DRAFT: measured preparation proxies, never commands or contact evidence.

No hidden history: every input is current geometry/joints/contact, previously
observed physical history, or adjacent real observations. The actor already
observes these in422. Mount heights remain diagnostics, NOT posture rewards.
"""
from collections.abc import Mapping
import math

MODE = 'rr_capture_cooperative_preparation_v4'
WORKSPACE_WEIGHTS = dict(edge=.25, receiver=.5, fl_range=.125, rl_space=.125)


def member(value, key):
    return value[key] if isinstance(value, Mapping) else getattr(value, key)


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite measured value required: '+name)
    return float(value)


def clip(x):
    return min(1., max(0., x))


def aabb_clearance(bounds, obstacle):
    low = tuple(finite(x, 'collider min') for x in member(bounds, 'minimum_m'))
    high = tuple(finite(x, 'collider max') for x in member(bounds, 'maximum_m'))
    if len(low) != 3 or len(high) != 3 or any(a>b for a,b in zip(low,high)):
        raise ValueError('ordered current collider bounds required')
    front,back,right,left,bottom,top = (finite(member(obstacle,key),key) for key in
        ('front_x_m','back_x_m','right_y_m','left_y_m','bottom_z_m','top_z_m'))
    if not (front<back and right<left and bottom<top):
        raise ValueError('ordered physical obstacle planes required')
    delta = [max(x-b, a-y, 0.) for a,b,x,y in
             zip(low,high,(front,right,bottom),(back,left,top))]
    return math.sqrt(sum(x*x for x in delta))


def measure_preparation(*, observation, evaluation, rr_context, support_spec,
                        joint_limits, clearance_scale_m):
    """Reuse a small existing RL workspace share; no new task acceptance gate.

    RL wheel AABB clearance is a conservative instantaneous proxy, not whole
    linkage clearance or a collision classifier. Normal edge contact remains
    legal. FL range credit saturates at the existing10deg margin proxy, not a
    desired angle; at the accepted frame FL already has enough for this proxy.
    """
    tick = member(observation,'physics_tick')
    now = finite(member(observation,'simulation_time_s'),'time')
    if tick != evaluation.get('physics_tick') or not math.isclose(
            now, evaluation.get('simulation_time_s', -1.), rel_tol=0., abs_tol=1e-9):
        raise ValueError('preparation sensor/evaluator clocks disagree')
    margin = finite(clearance_scale_m,'existing task-space clearance scale')
    if margin <= 0.:
        raise ValueError('positive existing clearance scale required')
    legs, hist = evaluation['current_legs'], evaluation['history']
    rr,rl = legs['RR'],legs['RL']
    live = evaluation.get('valid') is True and evaluation.get('termination_reason') is None
    rl_swing = bool(rl.get('current_lift_valid') is True and rl.get('air') is True
        and rl.get('ground_contact') is False and rl.get('motion_continuation_allowed') is True)
    # This is geometric task relevance, not permission to unload a support.
    relevant = bool(live and hist['active_lift']['RR'] and hist['front_edge_crossed']['RR']
        and rr.get('ground_contact') is False and rr.get('within_top_xy') is True
        and rr.get('within_lateral_span') is True
        and (rr_context.get('rr_top_reachable') or rr_context.get('rr_current_bearing') or rl_swing))
    knee = finite(member(member(observation,'joints')['front_left_knee'],'position_deg'),'FL knee')
    lo,hi = joint_limits
    if not lo < hi or not lo <= knee <= hi:
        raise ValueError('FL range cannot override actual physical limits')
    negative,positive = knee-lo,hi-knee
    range_credit = clip(min(negative,positive)/10.)
    bounds = member(observation,'body_bounds_w_m').get('rear_left_wheel')
    if bounds is None or member(observation,'geometry_pose_aware') is not True:
        raise ValueError('current pose-aware RL wheel collider required')
    distance = aabb_clearance(bounds,member(observation,'obstacle'))
    space_credit = clip(distance/margin)
    retired = bool(rl_swing or hist['placed']['RL'])
    # Already earned preparation need not be performed again during real swing.
    if retired:
        range_credit=space_credit=1.
    return dict(schema='wlr50_clean.cooperative_preparation.v4', mode=MODE,
        observation_tick=tick, simulation_time_s=now, valid=bool(live),
        relevant=relevant, current_rr_bearing=bool(rr_context['rr_current_bearing']),
        capturable_AIR_preparation=bool(rr_context['rr_top_reachable'] and rr.get('air') is True),
        strong_transfer_permission_unchanged=True, air_is_support=False,
        fl_knee_actual_deg=knee, fl_knee_negative_margin_deg=negative,
        fl_knee_positive_margin_deg=positive, fl_range_credit=range_credit,
        fl_range_proxy_scale_deg=10., rl_wheel_clearance_lower_bound_m=distance,
        rl_clearance_scale_m=margin, rl_space_credit=space_credit,
        rl_whole_linkage_clearance_verified=False, rl_whole_linkage_clearance_m=None,
        rl_actual_swing=rl_swing, range_space_credit_retired_after_live_swing=retired,
        weights=dict(WORKSPACE_WEIGHTS),
        semantics='bounded_existing_RL_workspace_share;not_pose_target_or_contact_or_safety_gate')


def workspace_progress(edge, receiver, diagnostic):
    if diagnostic is None or diagnostic.get('relevant') is not True:
        return .5*edge+.5*receiver
    if diagnostic.get('valid') is not True:
        raise ValueError('relevant cooperative preparation measurement invalid')
    parts=dict(edge=edge,receiver=receiver,fl_range=diagnostic['fl_range_credit'],
               rl_space=diagnostic['rl_space_credit'])
    if any(not 0. <= finite(x,k) <= 1. for k,x in parts.items()):
        raise ValueError('bounded preparation progress required')
    return sum(WORKSPACE_WEIGHTS[k]*value for k,value in parts.items())


def fl_counterroll_sample(*, previous, current, nominal, actual_drive, dt_s,
                          residual_caps, force_noise_floor_n):
    """Small soft prior ONLY for measured nonprogress counterrolling in RR carry.

    A positive source suggestion is not proof of optimal traction. This is an
    explicitly declared task prior, not a mask/absolute-value transform. Actual
    forward RR progress OR legal descent retires this cost continuously.
    Legitimate source reverse/stop, FL AIR and earlier front work are untouched.
    """
    dt=finite(dt_s,'physical interval')
    if not 0 < dt <= 1/120+1e-9:
        raise ValueError('adjacent actual120Hz observations required')
    result=dict(eligible=False, raw_cost=0., supported_fl_counterroll_rad_s=0.,
        forward_or_legal_descent_m_s=0., lack_of_progress_fraction=0.,
        task_prior_not_proven_causal_traction=True, alters_action=False)
    task=current.task
    diag=task.get('cooperative_preparation')
    if not diag or not diag.get('relevant') or diag.get('rl_actual_swing'):
        return result
    ev=task['physical_evaluator'];rr=ev['current_legs']['RR'];fl=ev['current_legs']['FL']
    if not (ev.get('valid') is True and rr.get('current_lift_valid') is True
            and rr.get('air') is True and not diag['current_rr_bearing']
            and fl.get('air') is False and fl.get('bearing_verified') is True
            and fl.get('support') is True
            and (fl.get('ground_contact') is True or fl.get('top_surface_contact') is True)
            and finite(fl.get('bearing_force_n'),'FL force') >= force_noise_floor_n
            and nominal[8] > 0.):
        return result
    measured=finite(current.groups['actual_wheel_velocity_rad_s'][0],'FL measured canonical speed')
    reverse=min(max(0.,-actual_drive[8]),max(0.,-measured))
    # Existing physical residual capacity is a normalization, not a new target.
    cap=finite(residual_caps[8],'FL wheel residual cap')
    if cap<=0.:
        raise ValueError('positive physical FL wheel residual capacity required')
    before=previous.metrics['wheels'][3]['center'];after=current.metrics['wheels'][3]['center']
    forward=max(0.,(after[0]-before[0])/dt)
    descent=max(0.,(before[2]-after[2])/dt) if rr.get('within_top_xy') is True else 0.
    # The frozen measured wheel radius already used in slip diagnostics.
    expected=max(1e-12, nominal[8]*.04998999834060672)
    lack=1.-clip(max(forward,descent)/expected)
    result.update(eligible=True,raw_cost=clip(reverse/cap)*lack,
        supported_fl_counterroll_rad_s=reverse,forward_or_legal_descent_m_s=max(forward,descent),
        lack_of_progress_fraction=lack,source_nominal_FL_rad_s=nominal[8],
        final_FL_rad_s=actual_drive[8],actual_FL_rad_s=measured)
    return result
