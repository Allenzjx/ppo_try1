"""One-pass, fixed-snapshot training evidence reader (standard library only).

No models, tensor files, robot imports or writes to a run. Decision endpoints
cannot reconstruct every 120 Hz contact event. Hold duration is explicitly the
logged native-observer result, not a duration inferred from sparse endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/ppo_rr_capture_first_cp225280_v1/train_first2048_ecf205e"
EPSILON = 1e-6


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def actual_pair(metrics):
    value = metrics.get("actual_rr_hip_knee_deg")
    return value if isinstance(value, list) and len(value) == 2 and all(map(number, value)) else None


def snapshot_rows(path, live, status):
    """Windows writer-safe open/seek/tell bound; never follow later appends."""
    status.update(path=str(path.resolve()), available=False, snapshot_bytes=None,
                  complete_rows=None, ignored_incomplete_tail_bytes=None)
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return
    with stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
        status.update(available=True, snapshot_bytes=size, complete_rows=0,
                      ignored_incomplete_tail_bytes=0)
        remaining = size
        while remaining:
            line = stream.readline(remaining)
            if not line:
                raise ValueError(f"{path}: file shrank inside its read snapshot")
            remaining -= len(line)
            if not line.endswith(b"\n"):
                if live:
                    status["ignored_incomplete_tail_bytes"] = len(line)
                    break
                raise ValueError(f"{path}: unfinished final line (use --live)")
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{path}: malformed complete JSON row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: JSON row is not an object")
            status["complete_rows"] += 1
            yield row


def local_after(row):
    if row.get("kind") == "frozen_prior_prefix":
        return mapping(row.get("local"))
    return mapping(mapping(row.get("step_info")).get("rr_capture_local"))


def rr_motion(before, after):
    a, b = mapping(before.get("metrics")), mapping(after.get("metrics"))
    result = dict(gap_descent_m=None, actual_hip_delta_deg=None, actual_knee_delta_deg=None)
    if number(a.get("gap_m")) and number(b.get("gap_m")):
        result["gap_descent_m"] = a["gap_m"] - b["gap_m"]
    qa, qb = a.get("actual_rr_hip_knee_deg"), b.get("actual_rr_hip_knee_deg")
    if (isinstance(qa, list) and isinstance(qb, list) and len(qa) == len(qb) == 2
            and all(number(x) for x in qa + qb)):
        result.update(actual_hip_delta_deg=qb[0]-qa[0], actual_knee_delta_deg=qb[1]-qa[1])
    return result


def likelihood_summary(run, update, samples):
    counts = mapping(update.get("counts"))
    index = counts.get("local_ppo_updates")
    result = dict(available=False, tensor_rollout_inspected=False,
                  tensor_storage_verification="N/A: no Torch/tensor deserialization",
                  update_number=index, saved_sample_coverage=None,
                  old_logp_matches_decision_receipts=None)
    if type(index) is not int:
        return result
    path = run / "rollouts" / f"likelihood_{index:04d}.json"
    result["path"] = str(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return result
    batches = data.get("minibatches")
    if not isinstance(batches, list):
        raise ValueError(f"{path}: missing minibatches")
    covered = Counter()
    comparable = mismatches = unknown = 0
    for batch in batches:
        indices, old = batch.get("rollout_flat_indices"), batch.get("old_log_probability")
        if not isinstance(indices, list) or not isinstance(old, list) or len(indices) != len(old):
            raise ValueError(f"{path}: malformed actual minibatch likelihood")
        for identities, old_logp in zip(indices, old):
            if not isinstance(identities, list) or not identities:
                raise ValueError(f"{path}: missing storage index identity")
            # Ambiguous identical obs/action rows remain ambiguous, not doubled.
            if len(identities) != 1:
                unknown += 1
                continue
            identity = identities[0]
            if type(identity) is not int or not 0 <= identity < len(samples):
                unknown += 1
                continue
            covered[identity] += 1
            expected = samples[identity]["old_logp"]
            if number(expected) and number(old_logp):
                comparable += 1
                mismatches += int(abs(expected-old_logp) > EPSILON)
            else:
                unknown += 1
    result.update(available=True, source=data.get("source"), minibatches=len(batches),
                  extra_random_draws=data.get("extra_random_draws"),
                  extra_model_forwards=data.get("extra_model_forwards"),
                  saved_sample_coverage=len(covered), observed_sample_exposures=sum(covered.values()),
                  unique_sample_exposure_range=[min(covered.values()), max(covered.values())] if covered else None,
                  old_logp_comparable_exposures=comparable, old_logp_mismatches=mismatches,
                  ambiguous_or_missing_exposures=unknown,
                  old_logp_matches_decision_receipts=(mismatches == 0 and unknown == 0) if comparable else None)
    return result


def analyze(run, *, live=False):
    run = Path(run).resolve()
    sources = {"decisions": {}, "updates": {}}
    # Updates are read first: a committed update cannot overtake the later
    # decision snapshot. Live rows after that update are labelled pending.
    updates = list(snapshot_rows(run / "updates.jsonl", live, sources["updates"]))
    counts, phases, physical = Counter(), Counter(), Counter()
    episodes, samples, issues = [], [], []
    last_tick = None
    episode = None
    for row in snapshot_rows(run / "decisions.jsonl", live, sources["decisions"]):
        kind, credit = row.get("kind"), row.get("PPO_credit")
        info, policy = mapping(row.get("step_info")), mapping(row.get("policy_request"))
        local = local_after(row)
        metrics = mapping(local.get("metrics"))
        tick = metrics.get("tick", row.get("physical_tick"))
        if episode is None or (type(tick) is int and type(last_tick) is int and tick < last_tick):
            episode = dict(index=len(episodes)+1, activation_tick=None, activation_time_s=None,
                           credited_samples=0, first_top_contact_endpoint=None,
                           first_top_bearing_endpoint=None, min_active_logged_gap_m=None,
                           max_native_observer_hold_s=None, local_success_endpoint=None,
                           first_active_actual_rr_reference=None, active_actual_rr_angle_ranges_deg=None,
                           last_tick=None, last_phase=None, last_metrics=None, termination_reason=None)
            episodes.append(episode)
        if type(tick) is int:
            last_tick = tick
            episode["last_tick"] = tick
        phase = info.get("phase_id", row.get("phase"))
        if phase is not None:
            episode["last_phase"] = phase
        if metrics:
            episode["last_metrics"] = metrics
        if local.get("active") is True and episode["activation_tick"] is None:
            if type(local.get("activation_tick")) is int:
                episode["activation_tick"] = local["activation_tick"]
                episode["activation_time_s"] = local.get("activation_time_s")
            else:
                counts["active_rows_missing_activation_clock"] += 1
        for key in ("current_top_contact", "current_top_bearing"):
            field = "first_top_contact_endpoint" if key == "current_top_contact" else "first_top_bearing_endpoint"
            if metrics.get(key) is True and episode[field] is None:
                episode[field] = tick
        gap, hold = metrics.get("gap_m"), local.get("hold_elapsed_s")
        if local.get("active") is True and number(gap):
            episode["min_active_logged_gap_m"] = gap if episode["min_active_logged_gap_m"] is None else min(gap, episode["min_active_logged_gap_m"])
        q = actual_pair(metrics)
        if local.get("active") is True and q is not None:
            if episode["first_active_actual_rr_reference"] is None:
                episode["first_active_actual_rr_reference"] = dict(tick=tick, actual_rr_hip_knee_deg=list(q),
                    source="first_logged_active_endpoint_actual_q_not_FINAL_or_exact_gate_tick")
                episode["active_actual_rr_angle_ranges_deg"] = dict(hip=[q[0], q[0]], knee=[q[1], q[1]])
            ranges = episode["active_actual_rr_angle_ranges_deg"]
            for joint, value in zip(("hip", "knee"), q):
                ranges[joint] = [min(ranges[joint][0], value), max(ranges[joint][1], value)]
        if number(hold):
            episode["max_native_observer_hold_s"] = hold if episode["max_native_observer_hold_s"] is None else max(hold, episode["max_native_observer_hold_s"])
        if local.get("local_success") is True and episode["local_success_endpoint"] is None:
            episode["local_success_endpoint"] = tick
        reason = info.get("termination_reason") or metrics.get("termination_reason")
        if reason:
            episode["termination_reason"] = reason
        diagnostic = isinstance(policy.get("independent_diagnostic"), dict) or kind in ("diagnostic", "direction_diagnostic")
        auxiliary = kind in ("auxiliary", "auxiliary_fit") or row.get("AUX_credit") == 1
        valid_credit = type(credit) is int and credit == 1 and kind == "activated_on_policy" and not diagnostic and not auxiliary
        if credit == 1 and not valid_credit:
            counts["invalid_or_intervened_PPO_credit_claims"] += 1
        if diagnostic:
            counts["diagnostic_rows_not_PPO"] += 1
        elif auxiliary:
            counts["auxiliary_rows_not_PPO"] += 1
        elif kind == "frozen_prior_prefix" and type(credit) is int and credit == 0:
            counts["prefix_decisions_credit0"] += 1
        elif not valid_credit:
            counts["unknown_or_unclassified_rows"] += 1
        if not valid_credit:
            continue
        counts["actual_on_policy_decisions"] += 1
        episode["credited_samples"] += 1
        phases[str(phase)] += 1
        before = mapping(row.get("before_local"))
        previous = mapping(before.get("metrics"))
        for key in ("current_top_contact", "current_top_bearing", "ground_contact"):
            if type(metrics.get(key)) is bool:
                physical[key + "_known_samples"] += 1
                physical[key + "_true_samples"] += int(metrics[key])
            else:
                physical[key + "_missing_samples"] += 1
        if type(previous.get("current_top_bearing")) is bool and type(metrics.get("current_top_bearing")) is bool:
            physical["bearing_transition_comparable_samples"] += 1
            physical["bearing_drops_between_decision_endpoints"] += int(previous["current_top_bearing"] and not metrics["current_top_bearing"])
            physical["bearing_gains_between_decision_endpoints"] += int(not previous["current_top_bearing"] and metrics["current_top_bearing"])
        motion = rr_motion(before, local)
        if number(motion["gap_descent_m"]):
            physical["gap_comparable_samples"] += 1
            physical["gap_descent_samples"] += int(motion["gap_descent_m"] > EPSILON)
        if number(motion["actual_hip_delta_deg"]):
            physical["actual_joint_motion_comparable_samples"] += 1
            negative, positive = motion["actual_hip_delta_deg"] < -EPSILON, motion["actual_knee_delta_deg"] > EPSILON
            physical["actual_hip_negative_samples"] += int(negative)
            physical["actual_knee_positive_samples"] += int(positive)
            physical["actual_hip_negative_and_knee_positive_samples"] += int(negative and positive)
        if q is not None and episode["first_active_actual_rr_reference"] is not None:
            reference = episode["first_active_actual_rr_reference"]["actual_rr_hip_knee_deg"]
            hip_bin, knee_bin = q[0]-reference[0] <= -5., q[1]-reference[1] >= 5.
            physical["actual_entry_relative_5deg_bin_known_samples"] += 1
            physical["actual_entry_relative_hip_le_minus5deg_samples"] += int(hip_bin)
            physical["actual_entry_relative_knee_ge_plus5deg_samples"] += int(knee_bin)
            physical["actual_entry_relative_hip_le_minus5_knee_ge_plus5deg_samples"] += int(hip_bin and knee_bin)
        if local.get("local_success") is True:
            physical["logged_local_success_samples"] += 1
        if number(hold):
            physical["hold_duration_known_samples"] += 1
            physical["positive_native_observer_hold_samples"] += int(hold > 0)
        native = mapping(info.get("actuator_target_effect_audit_summary"))
        if type(native.get("physics_ticks")) is int:
            physical["logged_native_physics_ticks"] += native["physics_ticks"]
        else:
            physical["missing_native_audit_samples"] += 1
        physical["all_native_ticks_verified_samples"] += int(native.get("all_ticks_verified") is True)
        old = row.get("old_logp")
        old = old[0] if isinstance(old, list) and len(old) == 1 else None
        receipt_logp = policy.get("selected_raw_log_probability")
        dimension = {'wlr50_clean.actual_rr_capture_local_request.v1':447,
                     'wlr50_clean.actual_rr_capture_local_request.v2':448}.get(policy.get('schema'))
        active = isinstance(row.get("observation"), list) and len(row["observation"]) == dimension and row["observation"][439] == 1.
        if dimension == 448:
            active = active and (row['observation'][447] in (0.,1.) and
                row['observation'][447] == float(previous.get('current_attempt_capture_eligible', -1)))
        consistent = bool(before.get("active") is True and active and policy.get("capture_active") is True
                          and number(old) and number(receipt_logp) and abs(old-receipt_logp) <= EPSILON)
        counts["active_observation_old_likelihood_verified_samples"] += int(consistent)
        if not consistent:
            counts["active_observation_or_old_likelihood_missing_or_mismatch"] += 1
        selected, issued = policy.get("selected_raw_full12"), info.get("raw_policy_action_full12")
        raw_known = (isinstance(selected, list) and isinstance(issued, list)
                     and len(selected) == len(issued) == 12 and all(number(x) for x in selected + issued))
        raw_matches = selected == issued if raw_known else None
        counts["selected_vs_issued_raw_known_samples"] += int(raw_known)
        counts["selected_vs_issued_raw_mismatches"] += int(raw_matches is False)
        samples.append(dict(phase=str(phase), old_logp=old, episode=episode["index"], tick=tick,
                            active_old_likelihood_verified=consistent, selected_vs_issued_raw=raw_matches,
                            native_all_ticks_verified=native.get("all_ticks_verified") is True))
    update_reports, offset = [], 0
    for update in updates:
        phase_counts = mapping(update.get("actual_phase_counts"))
        size = sum(phase_counts.values()) if phase_counts and all(type(n) is int and n >= 0 for n in phase_counts.values()) else None
        chunk = samples[offset:offset+size] if size is not None else []
        actual = dict(Counter(item["phase"] for item in chunk))
        report = dict(reported_branch_counters=mapping(update.get("counts")),
                      run_local_credited_sample_start=offset+1 if chunk else None,
                      run_local_credited_sample_end=offset+len(chunk) if chunk else None,
                      sample_count_from_actual_phase_report=size, available_decision_rows=len(chunk),
                      reported_phase_counts=phase_counts or None, decision_phase_counts=actual,
                      phase_counts_match=actual == phase_counts if size is not None else None,
                      active_old_likelihood_verified_samples=sum(x["active_old_likelihood_verified"] for x in chunk),
                      selected_vs_issued_raw_known_samples=sum(x["selected_vs_issued_raw"] is not None for x in chunk),
                      selected_vs_issued_raw_mismatches=sum(x["selected_vs_issued_raw"] is False for x in chunk),
                      native_all_ticks_verified_samples=sum(x["native_all_ticks_verified"] for x in chunk),
                      optimizer_steps=update.get("optimizer_steps"),
                      actor_parameters_changed=update.get("actor_parameters_changed"),
                      finite_nonzero_gradient_observed=update.get("finite_nonzero_gradient_observed"),
                      likelihood=likelihood_summary(run, update, chunk))
        update_reports.append(report)
        if size is not None:
            offset += size
    count_keys = ("actual_on_policy_decisions", "prefix_decisions_credit0", "diagnostic_rows_not_PPO",
                  "auxiliary_rows_not_PPO", "invalid_or_intervened_PPO_credit_claims", "unknown_or_unclassified_rows",
                  "active_observation_old_likelihood_verified_samples", "active_observation_or_old_likelihood_missing_or_mismatch",
                  "selected_vs_issued_raw_known_samples", "selected_vs_issued_raw_mismatches")
    decision_available = sources["decisions"]["available"]
    return dict(schema="wlr50_clean.rr_capture_training_progress_readonly.v1", run=str(run), live_snapshot=live,
                sources=sources, counts={k: counts[k] if decision_available else None for k in count_keys},
                actual_phase_counts=dict(phases) if decision_available else None,
                rr_physical_evidence=dict(physical) if samples else None,
                capture_opportunities_with_logged_activation=sum(ep["activation_tick"] is not None for ep in episodes) if decision_available else None,
                episodes=episodes if decision_available else None, updates=update_reports if sources["updates"]["available"] else None,
                completed_updates_observed=len(updates) if sources["updates"]["available"] else None,
                optimizer_steps_observed=sum(u["optimizer_steps"] for u in updates) if updates and all(type(u.get("optimizer_steps")) is int for u in updates) else (0 if sources["updates"]["available"] and not updates else None),
                pending_credited_samples_after_last_update=len(samples)-offset if decision_available else None,
                measurement_epsilon=EPSILON,
                small_direction_count_semantics="Numerically nonzero directions at1e-6, potentially noise; not meaningful candidate coverage.",
                entry_relative_5deg_semantics="Reporting bins ONLY, not control/reward/success thresholds; actual q relative to first logged active endpoint actual q, separately timestamped.",
                issues=issues,
                scope=["Only explicitly named run JSON/JSONL read; historical global counters are never added as new samples.",
                       "Counts are observed rows, not a physical-success claim; diagnostic/AUX rows cannot contribute PPO credit.",
                       "Motion is measured before/after actual q, never commanded target substituted for q.",
                       "Contact/drop counts use decision endpoints and are lower bounds; no full 120Hz contact reconstruction.",
                       "Hold seconds are logged native-observer state; reader does not independently revalidate all ticks.",
                       "Storage tensors/returns/advantages remain uninspected; JSON official minibatch likelihood evidence is separate.",
                       "Missing file/measurement means unavailable, never zero; live absence is not final failure."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, live=args.live)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT / "outputs"):
            raise ValueError("analysis output must be under outputs, never the live run")
        # No implicit replacement of a sealed report.
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
        print(json.dumps({"report": str(destination), "counts": result["counts"],
                          "completed_updates": result["completed_updates_observed"]}))
    else:
        print(text)


if __name__ == "__main__":
    main()
