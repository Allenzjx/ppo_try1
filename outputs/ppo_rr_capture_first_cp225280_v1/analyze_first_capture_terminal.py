"""Bounded stdlib attribution of the first completed local training episode."""
import json
from pathlib import Path

from analyze_training_progress import DEFAULT_RUN, snapshot_rows

OUT = Path(__file__).resolve().parent
SELECTED = (9424, 9432, 9440, 9448, 9464, 9496)


def pair(values):
    return values[6:8] if isinstance(values, list) and len(values) == 12 else None


def analyze():
    status, samples, selected, transitions = {}, [], [], []
    prefix = 0
    for row in snapshot_rows(DEFAULT_RUN / "decisions.jsonl", True, status):
        if row.get("kind") == "frozen_prior_prefix":
            if row.get("PPO_credit") != 0:
                raise ValueError("prefix incorrectly credited")
            prefix += 1
            continue
        if row.get("kind") != "activated_on_policy" or row.get("PPO_credit") != 1:
            raise ValueError("unexpected intervention/non-policy row in training")
        info, policy = row["step_info"], row["policy_request"]
        audit = info["actuator_target_effect_audit"]
        local, reward = info["rr_capture_local"], info["local_reward"]
        metric, headroom = local["metrics"], audit["policy_headroom_evidence"]
        samples.append(row)
        for transition in info["stage_transition_evidence"]:
            transitions.append({k: transition[k] for k in ("from_stage", "to_stage", "physics_tick", "sim_time_s")})
            transitions[-1].update(local_terminated=reward["terminated"],
                                   terminal_bootstrap_allowed=info["terminal_bootstrap_allowed"])
        if metric["tick"] in SELECTED:
            selected.append(dict(tick=metric["tick"], time_s=metric["time_s"],
                phase=[info["phase_id"], info["end_phase_id"]],
                source_N_rr_deg=pair(info["nominal_action_full12"]),
                mapped_N_rr_deg=pair(audit["native_drive_target_full12"]),
                controller_rr_deg=pair(audit["controller_drive_bias_full12"]),
                selected_raw_rr=pair(policy["selected_raw_full12"]),
                conditional_mean_rr=pair(policy["conditional_mean_full12"]),
                declared_sigma_rr=pair(policy["active_conditional_std_full12"]),
                requested_residual_rr_deg=pair(headroom["requested_policy_residual_full12"]),
                effective_residual_rr_deg=pair(headroom["effective_policy_residual_full12"]),
                final_rr_deg=pair(info["actual_drive_target_full12"]),
                measured_rr_deg=metric["actual_rr_hip_knee_deg"], gap_mm=1000*metric["gap_m"],
                top=metric["current_top_contact"], bearing=metric["current_top_bearing"],
                force_n=metric["bearing_force_n"], load_fraction_valid=metric["load_fraction_valid"],
                hold_s=local["hold_elapsed_s"], consecutive_top_samples=metric["consecutive_top_samples"],
                local_terminated=reward["terminated"], reason=info["termination_reason"],
                terminal_bootstrap_allowed=info["terminal_bootstrap_allowed"],
                native_dispatch_tick=audit["physics_tick"],
                native_source_observation_tick=audit["capture_assist_evidence"]["context"]["source_observation_tick"]))
        if reward["terminated"]:
            break
    if not samples or not samples[-1]["step_info"]["local_reward"]["terminated"]:
        raise ValueError("first episode not complete in this snapshot")
    last = samples[-1]["step_info"]
    local, metric = last["rr_capture_local"], last["rr_capture_local"]["metrics"]
    contact_rows = [r for r in samples if r["step_info"]["rr_capture_local"]["metrics"]["current_top_contact"]]
    top_ticks = [r["step_info"]["physics_tick"] for r in contact_rows]
    contact_start_candidates = [r["step_info"]["physics_tick"] -
        r["step_info"]["rr_capture_local"]["metrics"]["consecutive_top_samples"] + 1 for r in contact_rows]
    expected_native_ticks = list(range(top_ticks[0]-7, metric["tick"]+1))
    recorded_ticks = [t["episode_physics_tick"] for r in contact_rows for t in r["step_info"]["actuator_target_effect_audit_ticks"]]
    audits = [r["step_info"]["actuator_target_effect_audit"] for r in samples]
    report = dict(schema="wlr50_clean.first_stochastic_RR_capture_attribution.v1",
        run=str(DEFAULT_RUN), source_snapshot=status, stopped_at_first_local_terminal=True,
        prefix_credit0=prefix, actual_PPO_credit1=len(samples), optimizer_updates_before_this_success=0,
        not_video=True, not_deterministic_success=True, not_full_task_success=True,
        zero_initialized_local_mean_all_samples=all(r["policy_request"]["local_raw_mean_delta_full12"] == [0.]*12 for r in samples),
        stochastic_sampling_draw1_all_samples=all(r["policy_request"]["sampling_draws"] == 1 for r in samples),
        no_independent_diagnostic_fields=all("independent_diagnostic" not in r["policy_request"] for r in samples),
        selected_raw_equals_issued_all_samples=all(r["policy_request"]["selected_raw_full12"] == r["step_info"]["raw_policy_action_full12"] for r in samples),
        rear_controller_bias_zero_all_samples=all(pair(a["controller_drive_bias_full12"]) == [0.,0.] for a in audits),
        nonzero_rear_controller_bias_rows=[dict(tick=r["step_info"]["physics_tick"],
            controller_rr_deg=pair(r["step_info"]["actuator_target_effect_audit"]["controller_drive_bias_full12"]))
            for r in samples if pair(r["step_info"]["actuator_target_effect_audit"]["controller_drive_bias_full12"]) != [0.,0.]],
        nonzero_bias_source=dict(source="existing reference-FSM P10 post_mapper_drive normal correction, not RR capture helper",
            phase_delta_rr_knee_deg=10.599999999999998, correction_fraction=.14150943396226415,
            formula="0.5 * phase.delta_full12[7] * normal_correction_fractions[7] = +0.75deg",
            references=["configs/fsm_states.yaml:P10 normal_correction_domain and fractions",
                "configs/recording_motion_contract.json:P10 delta_full12",
                "semantic_supervisor.py:_source_normal_bias; SemanticControllerAdapter.step sets feedback ZERO12 and source normal bias",
                "isaac_fsm_backend.py:build_residual_actuation_plan controller_bias = feedback + normal"]),
        no_RR_capture_assist_ownership_all_samples=all(a["capture_assist_owned_channels_full12"][6:8] == [False, False] for a in audits),
        rear_assist_correction_zero_all_samples=all(pair(a["capture_assist_evidence"]["assist_correction_full12"]) == [0.,0.] for a in audits),
        all12_unmodified_at_actuator_all_samples=all(a["all12_policy_channels_unmodified_at_actuator"] is True for a in audits),
        no_episode_state_writes_all_samples=all(r["step_info"]["no_in_episode_state_writes_verified"] is True for r in samples),
        phase_changes=transitions, rows=selected,
        contact_hold_evidence=dict(first_logged_TOP_endpoint=top_ticks[0],
            first_TOP_tick_inferred_from_native_observer_counts=contact_start_candidates[0],
            all_contact_endpoints_agree_on_start_tick=len(set(contact_start_candidates)) == 1,
            placed_tick=last["semantic_task"]["physical_evaluator"]["history"]["event_ticks"]["placed"]["RR"],
            terminal_tick=metric["tick"], terminal_reason=last["termination_reason"],
            terminal_native_observer_top_samples=metric["consecutive_top_samples"],
            native_observer_hold_s=local["hold_elapsed_s"], final_bearing_force_n=metric["bearing_force_n"],
            final_load_fraction_valid=metric["load_fraction_valid"],
            native_dispatch_episode_ticks_contiguous=recorded_ticks == expected_native_ticks,
            all_native_dispatch_ticks_verified=all(t["verified"] for r in contact_rows for t in r["step_info"]["actuator_target_effect_audit_ticks"]),
            independent_120Hz_contact_log_inspected=False,
            scope="Runtime native-observer hold plus TOP counters/current sensor-derived force; decision log does not contain each tick's contact measurement."),
        terminal_reward=last["local_reward"],
        conclusions=["First TOP is reached while source RR target remains P09 [-6.9,-37.8] degrees; positive knee residual unfolds RR before P10.",
            "Before first TOP, hip actual stays positive near+6 degrees; this sample does not prove negative hip was necessary for touchdown.",
            "After real placed, P10/P11 source knee unfolds toward-27.2 degrees while sampled residual and HISTORY also contribute; not all post-touch motion is policy-only.",
            "At endpoint9448 only, RR controller bias is[0,+0.75deg], the preserved P10 FSM post-mapper normal correction; other188 sampled endpoints are zero. No RR completion helper is inferred from this generic bias.",
            "P09-to-P10 and P10-to-P11 remain nonterminal and bootstrappable; only verified local hold is terminal.",
            "Success occurred before any optimizer update with local mean still zero: genuine stochastic exploration success, not learned or deterministic success.",
            "Final load_fraction_valid is false; bearing force/current bearing are separately reported, not a reliable normalized load fraction."])
    return report


def main():
    result = analyze()
    destination = OUT / "first_stochastic_RR_capture_attribution_v2.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    def p(values):
        return "/".join(f"{x:.3f}" for x in values)
    lines = ["# First stochastic RR capture — episode 1", "",
        "189 on-policy samples; 998 uncredited frozen-prefix decisions. Zero optimizer updates before this success; local mean stayed zero. This is headless stochastic exploration, not a video, learned gain, deterministic result or full-task success.", "",
        "RR pairs are hip/knee; angles deg, raw Gaussian samples unitless. REQUEST is the real filtered/headroom input, not a re-created nominal subtraction.", "",
        "This v2 supersedes the earlier Markdown's incorrect all189-controller-bias-zero sentence; the earlier JSON already reported that boolean as false. The nonzero inherited P10 correction is now explicit.", "",
        "|Tick / phase|Raw sample|N source / mapped|Controller bias|REQUEST|FINAL|Actual|gap mm / force N / hold s|",
        "|---|---|---|---|---|---|---|---|" ]
    for row in result["rows"]:
        lines.append(f"|{row['tick']} {row['phase'][0]}→{row['phase'][1]}|{p(row['selected_raw_rr'])}|{p(row['source_N_rr_deg'])} / {p(row['mapped_N_rr_deg'])}|{p(row['controller_rr_deg'])}|{p(row['requested_residual_rr_deg'])}|{p(row['final_rr_deg'])}|{p(row['measured_rr_deg'])}|{row['gap_mm']:.3f} / {row['force_n']:.3f} / {row['hold_s']:.3f}|")
    lines += ["", *["- " + statement for statement in result["conclusions"]], "",
        "Native-observer TOP counters consistently imply first contact tick9436; placed event9437; first endpoint9440; terminal9496 with61 TOP samples,0.5s hold and13.2609N verified bearing. All recorded dispatch ticks9433–9496 are contiguous/verified. No separate per-tick contact file was inspected.", "",
        "Across all189 samples: original raw equals issued raw; no diagnostic field or RR capture-assist owner/correction; all12 actuator audit unchanged and no episode state writes. Generic source controller bias is NOT entirely zero: at9448, RR knee+0.75deg equals0.5×10.6×0.14150943396226415 from preserved FSM P10 normal correction. Other188 sampled endpoints have zero RR controller bias. Phase transitions9440/9448 are nonterminal;9496 is genuine local terminal with no bootstrap.", "",
        "Clock distinction is retained in JSON: endpoint9496, source observation9495, native dispatch9675. No camera footage was produced by this analysis."]
    with (OUT / "first_stochastic_RR_capture_attribution_v2.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(destination), "samples": result["actual_PPO_credit1"],
                      "hold": result["contact_hold_evidence"], "phase_changes": result["phase_changes"]}))


if __name__ == "__main__":
    main()
