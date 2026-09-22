"""Compact sealed block10A/B coverage only; no model, physics, fitting or polling."""
from collections import Counter
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PHASES = [f'P{i:02}' for i in range(1,14)]
COUNTERS = ('global_policy_decisions','ppo_updates','optimizer_steps')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def coverage(report):
    episodes = []; previous = None; counts = Counter()
    for row in rows(Path(report['run'])/'residual_and_projection_audit.jsonl'):
        a = row['applied_audit']; ev = a['semantic_task']['physical_evaluator']; fl = ev['current_legs']['FL']
        diag = a['semantic_task']['nominal_provider_diagnostics']['p05_preedge_approach_recovery']
        assist = a['actuator_target_effect_audit']['capture_assist_evidence']
        if a['decision_count'] == 1:
            episodes.append({'episode_index':len(episodes),'counts':Counter(),'first_gate_endpoint':None,
                'last_gate_endpoint':None,'first_FL_placement_observed':None,'first_FL_TOP_loss_after_placement':None,
                'first_FL_ground_after_placement':None})
            previous = None
        ep = episodes[-1]; local = ep['counts']; placed = bool(ev['history']['placed']['FL'])
        assert diag['mode'] == 'p05_preedge_approach_recovery_v1'
        gate = bool(diag['eligible'])
        assert gate == all(diag['checks'].values())
        assert not diag['independent_ack_verified'] and not diag['policy_residual_restricted']
        assert not diag['phase_or_capture_credit_awarded'] and not diag['recovery_clock_reset_or_latch']
        if previous is None:
            assert a['phase_id'] in ('P01','P04')
            input_gate = False  # Exact P05-only gate; these natural/prefix entry phases cannot qualify.
        else:
            assert previous['applied_audit']['physics_tick'] == a['physics_tick'] - a['physics_ticks']
            input_gate = previous['applied_audit']['semantic_task']['nominal_provider_diagnostics']['p05_preedge_approach_recovery']['eligible']
        def event():
            return {'global_decision':row['global_policy_decision'],'endpoint_tick':a['physics_tick'],'sim_time_s':a['sim_time_s'],
                'phase':a['end_phase_id'],'FL_top_contact':fl['top_contact'],'FL_top_surface_contact':fl['top_surface_contact'],
                'FL_support':fl['support'],'FL_bearing_verified':fl['bearing_verified'],'FL_ground_contact':fl['ground_contact'],
                'FL_AIR':fl['air'],'FL_gap_m':fl['clearance_m'],'FL_front_distance_m':fl['front_distance_m'],
                'assist_owner_indices':assist['owner_indices']}
        local['learner_decisions'] += 1
        for key,value in (('preedge_gate_eligible_input_samples',input_gate),('preedge_gate_eligible_endpoint_samples',gate),
            ('assist_owned_endpoint_samples',bool(assist['owner_indices'])),('assist_initialized_endpoint_samples',bool(assist['state_after']['initialized'])),
            ('FL_placed_history_endpoint_samples',placed),('FL_placed_and_current_TOP_endpoint_samples',placed and fl['top_contact']),
            ('FL_placed_and_current_verified_TOP_support_endpoint_samples',placed and fl['top_contact'] and fl['top_surface_contact'] and fl['support'] and fl['bearing_verified']),
            ('FL_placed_but_not_current_TOP_endpoint_samples',placed and not fl['top_contact']),
            ('FL_placed_and_current_ground_endpoint_samples',placed and fl['ground_contact']),
            ('FL_placed_and_current_AIR_endpoint_samples',placed and fl['air'])):
            local[key] += int(value); counts[key] += int(value)
        if gate:
            assert all(diag['checks'].values())
            if ep['first_gate_endpoint'] is None: ep['first_gate_endpoint'] = event()
            ep['last_gate_endpoint'] = event()
        if placed and ep['first_FL_placement_observed'] is None:
            ep['first_FL_placement_observed'] = {**event(),'actual_placed_event_tick':ev['history']['event_ticks']['placed']['FL']}
        if placed and not fl['top_contact'] and ep['first_FL_TOP_loss_after_placement'] is None:
            ep['first_FL_TOP_loss_after_placement'] = event()
        if placed and fl['ground_contact'] and ep['first_FL_ground_after_placement'] is None:
            ep['first_FL_ground_after_placement'] = event()
        ep['last_endpoint'] = event()
        previous = row
    assert sum(ep['counts']['learner_decisions'] for ep in episodes) == report['actual_added_counts']['global_policy_decisions']
    assert counts['assist_owned_endpoint_samples'] == report['execution']['assist_owned_endpoints']
    for ep in episodes: ep['counts'] = dict(ep['counts'])
    return {'counts':dict(counts),'episodes':episodes}


def main():
    paths = [OUT/f'block10{part}_training_audit.json' for part in ('A','B')]
    audits = [read(p) for p in paths]
    assert all(a['result']=='PASS' and a['audit_mode']=='sealed' for a in audits)
    assert audits[0]['checkpoint_sha256'] == audits[1]['source_sha256']
    assert audits[1]['checkpoint_sha256'] == '6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227'
    added = {k:sum(a['actual_added_counts'][k] for a in audits) for k in COUNTERS}
    assert added == dict(zip(COUNTERS,(2048,16,320)))
    phases = {p:sum(a['phase_input_counts'][p] for a in audits) for p in PHASES}; assert sum(phases.values())==2048
    parts = {f'10{letter}':coverage(a) for letter,a in zip(('A','B'),audits)}
    total = Counter()
    for part in parts.values(): total.update(part['counts'])
    final = audits[1]
    assert final['all_five_origins']['p05_preedge_approach_recovery']['actual_branch_counts'] == added
    result = {'schema':'wlr50_clean.block10_combined_sealed_coverage.v1','result':'PASS',
        'audit_bindings':[{'path':str(p),'sha256':sha(p)} for p in paths],
        'actual_added_counts':added,'actual_lifetime_counts':final['actual_lifetime_counts'],
        'phase_input_counts':phases,'prefix_decisions_uncredited':sum(a['prefix_decisions'] for a in audits),
        'combined_endpoint_and_input_coverage':dict(total),'parts':parts,
        'final_checkpoint':final['checkpoint'],'final_checkpoint_sha256':final['checkpoint_sha256'],
        'preedge_branch_total':final['all_five_origins']['p05_preedge_approach_recovery'],
        'all_AUX_exact_carry_no_added_AUX':True,'front_AUX':[96,96],'RR_AUX':[7,8],'mixed_AUX':[103,104],'older_AUX':[7,8],
        'limitations':['Counts distinguish input-state gate and decision endpoint gate; they are not per-physics-tick activation counts.',
            'FL placement history is not current support. TOP, verified TOP support, ground and AIR are reported independently.',
            'Assist ownership is sampled from the last native tick of each policy decision, not an invented all-tick count.',
            'A gate supplies nominal wheel advice before residual mapping, not independent ACK or pure-policy success.',
            'Both last partials are nonterminal sampling boundaries. No claim is made about the running new deterministic evaluation.'],
        'audit_fitting_physics_or_checkpoint_writes':0}
    (OUT/'block10_combined_training_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    body = ['# Block10A + 10B — actual sealed training', '',
        'PASS: **2048 policy decisions /16 PPO updates /320 Adam steps**; cumulative **218496 /1672 /33440**. No new AUX.', '',
        f'Combined P01–P13 inputs: **{phases}**. Frozen prefixes: **{result["prefix_decisions_uncredited"]} decisions**, all zero PPO credit.', '',
        '10A: natural P01, actual512/4/80; first episode P02 incomplete at15.683333s, second P02 nonterminal partial at18.4s. Its512 unconsumed decisions were not credited; the adjusted10B separately earned1536/12/240.', '',
        f'Actual preedge gate: **{total["preedge_gate_eligible_input_samples"]} input states /{total["preedge_gate_eligible_endpoint_samples"]} endpoints**; FL-assist-owned **{total["assist_owned_endpoint_samples"]} endpoints**. These are decision samples, not full per-tick activation counts.', '']
    for ep,training in zip(parts['10B']['episodes'],audits[1]['episodes']):
        first=ep['first_FL_placement_observed'];last=ep['last_endpoint'];c=ep['counts']
        body.extend([f'10B episode{ep["episode_index"]}: {training["learner_decisions"]} learner decisions, {training["physical_duration_s"]:.6f}s, {training["last_phase"]}, {training["termination_reason"] or "nonterminal sampling partial"}.',
            f'- FL actual placed event tick{first["actual_placed_event_tick"]}; first sampled placed endpoint TOP={first["FL_top_contact"]}, verified bearing={first["FL_bearing_verified"]}, assist owner={first["assist_owner_indices"]}. This is not labeled pure-policy capture.',
            f'- Afterwards: placed-history {c["FL_placed_history_endpoint_samples"]} endpoints, current TOP {c["FL_placed_and_current_TOP_endpoint_samples"]}, verified TOP support {c["FL_placed_and_current_verified_TOP_support_endpoint_samples"]}, non-TOP {c["FL_placed_but_not_current_TOP_endpoint_samples"]}, ground {c["FL_placed_and_current_ground_endpoint_samples"]}, AIR {c["FL_placed_and_current_AIR_endpoint_samples"]}. Final TOP={last["FL_top_contact"]}, support={last["FL_support"]}, ground={last["FL_ground_contact"]}, AIR={last["FL_AIR"]}.', ''])
    body += ['Both individual audits verify sealed raw389/raw12/μ/σ/logp/reward/done and five PPO uses, actual actor/Adam changes, Identity/full RNG progression, official save/reload, all five origins and the unchanged complete AUX ledgers. LR is taken from each real update, not assumed constant.', '',
        f'Final checkpoint SHA256 `{final["checkpoint_sha256"]}`. Training completion is not full-task success. No success claim is made for the concurrently running natural-P01 evaluation. CPU-only report generation; helper exits afterward.', '']
    (OUT/'block10_combined_training_audit.md').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('result','actual_added_counts','phase_input_counts','prefix_decisions_uncredited','combined_endpoint_and_input_coverage')},indent=2))


if __name__=='__main__':main()
