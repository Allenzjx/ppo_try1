"""Read only: one sealed update after first384; no Torch/model/physics."""
import json
from collections import Counter
from pathlib import Path
from audit_first384_reward import ROOT, RUN, lines, stats
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor
from wlr50_clean.ppo.semantic_rear_policy_timing import rear_dependency


def main():
    completed = None
    for row in lines(RUN / 'optimizer_updates.jsonl'):
        if row['ppo_update'] == 1696:
            completed = row
            break
    assert completed and completed['global_policy_decisions'] == 221568
    audit = json.loads((RUN / 'rollouts/update_001696_likelihood.json').read_text())
    advantages = {}; exposure = Counter()
    for batch in audit['minibatches']:
        for indices, advantage in zip(batch['rollout_flat_indices'], batch['actual_advantage']):
            assert len(indices) == 1
            index = indices[0]; exposure[index] += 1
            if index in advantages:
                assert advantages[index] == advantage
            advantages[index] = advantage
    assert exposure == Counter({i: 5 for i in range(128)})
    sup = TaskStageSupervisor(ROOT / 'configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml')
    losses = {221230,221238,221248,221253,221339,221344,221423}
    reasons = []; rows = []
    keys = ('current_lift_valid','lift_established','ground_relative_lift_m',
            'body_control_evidence','observed_other_support_contacts',
            'motion_continuation_reason','motion_continuation_allowed','ground_contact',
            'air','top_contact','obstacle_pair_active','unsupported_free_lift_m')
    for raw in lines(RUN / 'residual_and_projection_audit.jsonl'):
        step = raw['global_policy_decision']
        if step > 221568:
            break
        applied = raw['applied_audit']; task = applied['semantic_task']
        ev = task['physical_evaluator']; rr = ev['current_legs']['RR']
        if step in losses:
            reasons.append(dict(global_decision=step, **{k:rr.get(k) for k in keys}))
        if step < 221441:
            continue
        dep = rear_dependency(task, sup.spec['support'])
        r = applied['reward_breakdown']
        assert abs(sup.physical_potential(ev) - r['potential_after']) < 1e-12
        placed = ev['history']['placed']['RR']
        retention = sup._current_capture_retention('RR',ev)
        rows.append(dict(global_decision=step,time_s=applied['sim_time_s'],tick=applied['physics_tick'],
            request_phase='P%02d'%(raw['policy_request']['stage_index']+1),
            end_phase=applied['end_phase_id'], RR_placed=placed,
            RR_gap_m=rr['clearance_m'],RR_top=dep['rr_top_contact'],RR_bearing=dep['rr_current_bearing'],
            RR_force_n=rr.get('bearing_force_n'),RR_air=rr['air'],RR_ground=rr['ground_contact'],
            RR_xy=rr['within_top_xy'],RR_retention=retention,
            RR_phi_share=.85/4*(.8+.2*retention) if placed else None,
            dependency=dep,RL_placed=ev['history']['placed']['RL'],
            RL_workspace_phi_share=.85/4*.1*sup._workspace_potential_progress('RL',ev),
            RL_gap_m=ev['current_legs']['RL']['clearance_m'],
            reward=raw['reward'],potential=r['potential_shaping'],phi_before=r['potential_before'],
            phi_after=r['potential_after'],weighted_families=r['families'],
            GAE_standardized=advantages[step-221441],terminal=raw['terminal'],
            input_rear_timing=raw['policy_request'].get('rear_policy_timing_observed_features')))
    assert len(rows) == 128 and len(reasons) == 7
    placed_rows = [r for r in rows if r['RR_placed']]
    lost = [r for r in placed_rows if not r['RR_bearing'] and not r['dependency']['rl_current_swing']]
    result = dict(scope='completed update1696 only (221441..221568); no pending update samples',
        run=str(RUN),currentQ_loss_reasons_first384=reasons,
        actual_phase_samples=dict(Counter(r['request_phase'] for r in rows)),
        placed_endpoint_count=len(placed_rows),placed_no_bearing_no_RL_swing_count=len(lost),
        placed_no_bearing_retention=stats(r['RR_retention'] for r in lost),
        placed_no_bearing_reward=stats(r['reward'] for r in lost),
        placed_no_bearing_GAE=stats(r['GAE_standardized'] for r in lost),
        first_placed=placed_rows[0] if placed_rows else None,
        initial_capture_and_loss=rows[max(0,rows.index(placed_rows[0])-2):rows.index(placed_rows[0])+6] if placed_rows else [],
        last_four=rows[-4:],
        rows=rows,
        limitations='Endpoint classifications explain the preceding action reward. GAE belongs to that raw-action sample; no recapture counterfactual or causal physical claim.')
    out=Path(__file__).with_name('P07_completed_capture_reward_audit.json')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','initial_capture_and_loss','last_four')},indent=2))


if __name__ == '__main__':
    main()
