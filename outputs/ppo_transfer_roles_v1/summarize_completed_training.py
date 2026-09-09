"""One-pass, N1 completed-run JSONL summary. Standard library only; no PT load.

Run ONLY after the parent confirms final lifecycle. Outputs are exclusive-create.
Coverage is optimized source-phase samples, not episode success or raw-tick replay.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

PHASES = [f"P{i:02}" for i in range(1, 14)]
LEGS = ("FR", "FL", "RR", "RL")
WRITES = ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
          "in_episode_force_or_impulse_writes", "in_episode_gravity_writes")
HEADER = ["run", "source_global", "end_global", "source_phase", "optimized_decisions",
          "physics_ticks", "nonterminal_handoffs", "terminal_handoffs", "native_effect_ticks",
          "own_phase_effect_ticks", "native_audit_anomaly_rows", "state_write_anomaly_rows", "sampled"]

def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)

def lines(path):
    if path.exists():
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    yield json.loads(line)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def audit_counts(info, count, *, prefix=False):
    result = Counter(physics_ticks=count)
    summary = info.get("actuator_target_effect_audit_summary", {})
    bad = summary.get("all_ticks_verified") is not True or summary.get("verified_tick_count") != count
    for source, target in (("actual_native_effect_tick_count", "native_effect_ticks"),
                           ("own_phase_request_effect_tick_count", "own_phase_effect_ticks")):
        value = summary.get(source)
        if type(value) is not int or not 0 <= value <= count:
            result[target+"_unavailable_rows"] += 1
            bad = True
        else:
            result[target] += value
    ticks = info.get("actuator_target_effect_audit_ticks")
    if ticks is None:
        result["compact_native_ticks_unavailable_rows"] += 1
        bad |= not prefix
    else:
        bad |= len(ticks) != count or any(x.get("verified") is not True for x in ticks)
    native = info.get("actuator_target_effect_audit")
    if native is None:
        result["final_native_audit_unavailable_rows"] += 1
        bad |= not prefix
    else:
        bad |= any(native.get(k) is not True for k in ("verified", "actual_mapping_matches_dispatch", "setter_dispatch_targets_equal"))
        bad |= native.get("target_dtype") != "torch.float32"
    result["native_audit_anomaly_rows"] += int(bad)
    write_bad = info.get("no_in_episode_state_writes_verified") is not True
    for key in WRITES:
        value = info.get(key)
        if type(value) is not int:
            result[key+"_unavailable_rows"] += 1
            write_bad |= not prefix
        elif value != 0:
            result[key+"_nonzero_rows"] += 1
            write_bad = True
    result["state_write_anomaly_rows"] += int(write_bad)
    return result


def new_episode(index, info, global_id):
    start = info.get("curriculum_start", {})
    return {"episode_index": index, "first_global": global_id, "decisions": 0, "physics_ticks": 0,
            "credit_start_tick": start.get("physics_tick", info["physics_tick"]-info["physics_ticks"]),
            "scope": info.get("task_result_scope", "full_task"), "curriculum_start": start,
            "phase_samples": Counter(), "events": [], "fl_reopening_observations": [],
            "fl_reopening_known_rows": 0, "fl_reopening_unavailable_rows": 0, "_seen": set(), "_handoffs": set(), "handoffs": [],
            "_reopening_states": {}}


def observe(ep, info, global_id):
    task = info["semantic_task"]
    hist = task["history"]
    ev = task["physical_evaluator"]
    current = ev.get("current_legs", {})
    events = [(x["leg"], x["event"], x["physics_tick"]) for x in hist.get("lift_attempt_events", [])]
    for kind, values in hist.get("event_ticks", {}).items():
        events.extend((leg, kind, tick) for leg, tick in values.items() if kind != "active_lift")
    names = {"whole_body_initial_clearance": "I", "qualified_measured_upward_lift": "Q",
             "front_edge_crossed": "C", "placed": "P", "qualification_revoked_ground_before_cross": "Q_revoked"}
    for leg, kind, tick in events:
        key = (leg, kind, tick)
        if key not in ep["_seen"]:
            ep["_seen"].add(key)
            ep["events"].append({"leg": leg, "kind": names.get(kind, kind), "physics_tick": tick,
                "first_observed_global": global_id,
                "credit": "policy" if tick > ep["credit_start_tick"] else "prefix_or_before_credit"})
    roles = task.get("transfer_roles", ev.get("transfer_roles", {}))
    for leg in ("FL", "FR"):
        current_leg = current.get(leg, {}); label = leg.lower()
        observations = ep.setdefault(label+"_reopening_observations", [])
        air = hist["placed"].get(leg) is True and current_leg.get("air") is True
        receivers = [x.get("receiver_workspace_state", {}) for x in roles.values() if x.get("diagonal_receiving_side") == leg]
        known = any(type(x.get("reopening_measured_response")) is bool for x in receivers)
        key = label+("_reopening_known_rows" if known else "_reopening_unavailable_rows"); ep[key] = ep.get(key, 0)+1
        reopening = any(x.get("reopening_measured_response") is True for x in receivers)
        for value, kind in ((air, "sampled_AIR_after_history_placed"), (reopening, "role_measured_reopening_response")):
            previous = (leg, kind)
            if value and not ep["_reopening_states"].get(previous, False):
                observations.append({"kind": kind, "global": global_id, "observation_tick": info["physics_tick"], "current_"+leg: current_leg})
            ep["_reopening_states"][previous] = value
    ep.update(last_global=global_id, last_tick=info["physics_tick"], last_active_stage=task["stage_id"],
        history_at_end={key: hist[key] for key in ("active_lift", "front_edge_crossed", "placed")},
        current_legs_at_end={leg: current.get(leg) for leg in LEGS},
        snapshot_task_success=info.get("task_success"), snapshot_full_task_success=info.get("full_task_success"),
        final_conditions_at_end={k:ev.get(k) for k in ("final_region_valid", "final_controlled", "final_support_available", "final_stable_for_s", "maximum_commanded_wheel_speed_rad_s", "measured_wheel_velocity_rad_s", "applied_wheel_command_rad_s")},
        body_motion_at_end={k:ev.get("goal_features", {}).get(k) for k in ("body_linear_speed_m_s", "body_angular_speed_rad_s", "maximum_wheel_speed_rad_s")},
        first_unfinished_stage=next((p for p in PHASES if p not in task["completed_stage_ids"]), None),
        incomplete_logged_goals={k: v for k, v in task.get("completion_values", {}).items() if isinstance(v, (float, int)) and v < 1.},
        physical_valid=ev.get("valid"), termination_reason=info.get("termination_reason"))


def summarize(run):
    final = read(run/"run_manifest.json")
    manifest = read(run/"training_manifest.json")
    allowed = ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY")
    need(final.get("lifecycle") in allowed and manifest.get("lifecycle") in allowed, f"not a completed training run: {run}")
    need(manifest["num_envs"] == 1, "this bounded helper supports N1 only")
    end, actual = manifest["global_policy_decisions"], manifest["actual_policy_decisions"]
    begin = end-actual
    updates = list(lines(run/"optimizer_updates.jsonl"))
    need(len(updates) == manifest["ppo_updates_this_run"] and updates[-1]["global_policy_decisions"] == end, "update boundary mismatch")
    need(all(x["global_policy_decisions"] == begin+128*(i+1) for i,x in enumerate(updates)), "non-contiguous optimized boundaries")
    need(all(b["ppo_update"] == a["ppo_update"]+1 for a,b in zip(updates, updates[1:])), "update index gap")
    prefix = Counter(); starts = []; prefix_phases = Counter()
    for row in lines(run/"prefix_evidence.jsonl"):
        need(row.get("policy_credit") is False, "prefix evidence lacks zero-credit binding")
        kind = row.get("kind", "")
        if kind in ("reset_only_prefix_decision", "checkpoint_prefix_decision"):
            prefix["decisions"] += 1; prefix_phases[row["phase_id"]] += 1
            prefix.update(audit_counts(row, row["physics_ticks"], prefix=True))
        elif kind == "policy_credit_start":
            starts.append({k: row["start"].get(k) for k in ("mode", "actual_phase", "physics_tick", "from_P01_current_policy")})
        elif kind.endswith("_prefix_result"):
            prefix["attempts"] += 1; prefix["accepted_attempts"] += int(row["accepted"])
    phases = {p: Counter() for p in PHASES}; authority = {p: {} for p in PHASES}; episodes = []; totals = Counter(); ep = None; expected = begin+1
    for row in lines(run/"residual_and_projection_audit.jsonl"):
        g = row["global_policy_decision"]
        if not begin < g <= end:
            totals["unoptimized_or_outside_rows_excluded"] += 1
            continue
        need(g == expected, "duplicate, missing or out-of-order policy decision"); expected += 1
        info = row["applied_audit"]; phase = info["phase_id"]
        need(phase in phases and type(row["terminal"]) is bool, "invalid phase/terminal")
        effective = info.get("actuator_target_effect_audit", {}).get("policy_headroom_evidence", {}).get("effective_policy_residual_full12")
        for label, values in (("raw_latent", row.get("raw_policy_action_full12")), ("filtered_requested", info.get("projected_residual_full12")),
                              ("headroom_effective", effective), ("nominal_canonical", info.get("nominal_action_full12")), ("final_drive_canonical", info.get("actual_drive_target_full12"))):
            ranges = authority[phase].setdefault(label, {})
            for index in (3, 8, 9, 10, 11):
                cell = ranges.setdefault(str(index), dict(count=0, unavailable=0, minimum=None, maximum=None, negative=0, positive=0, zero=0))
                if not isinstance(values, list) or len(values) != 12 or type(values[index]) not in (int, float) or not math.isfinite(values[index]):
                    cell["unavailable"] += 1; continue
                v = values[index]; cell["count"] += 1; cell["negative" if v < 0 else "positive" if v > 0 else "zero"] += 1
                cell["minimum"] = v if cell["minimum"] is None else min(v, cell["minimum"])
                cell["maximum"] = v if cell["maximum"] is None else max(v, cell["maximum"])
        if ep is None:
            ep = new_episode(len(episodes), info, g)
        need(info["decision_count"] == ep["decisions"]+1, "N1 episode decision discontinuity")
        ep["decisions"] += 1; ep["physics_ticks"] += info["physics_ticks"]; ep["phase_samples"][phase] += 1
        observe(ep, info, g)
        counts = audit_counts(info, info["physics_ticks"]); counts["optimized_decisions"] += 1
        for transition in info["semantic_task"].get("transition_evidence", []):
            key = (transition["from_stage"], transition["to_stage"], transition["physics_tick"])
            if key not in ep["_handoffs"] and key[2] > ep["credit_start_tick"]:
                ep["_handoffs"].add(key)
                ep["handoffs"].append(dict(from_stage=key[0], to_stage=key[1], physics_tick=key[2], observed_global=g, decision_terminal=row["terminal"]))
                counts["terminal_handoffs" if row["terminal"] else "nonterminal_handoffs"] += 1
                counts["handoff_bootstrap_contract_anomaly_rows"] += int(not row["terminal"] and info.get("terminal_bootstrap_allowed") is not True)
        counts["prefix_storage_violation_rows"] += int(any(info.get(k) is True for k in ("prefix_teacher_data_in_ppo_storage", "prefix_checkpoint_policy_data_in_ppo_storage")))
        phases[phase].update(counts); totals.update(counts)
        if row["terminal"]:
            ep.update(terminal=True, task_success=info["task_success"], full_task_success=info["full_task_success"], task_outcome_label=info.get("task_outcome_label"))
            episodes.append(ep); ep = None
    need(expected == end+1, "optimized audit shorter than published final boundary")
    completed = list(lines(run/"completed_episodes.jsonl"))
    for i, item in enumerate(episodes):
        need(i < len(completed) and completed[i]["episode_index"] == i and completed[i]["policy_decisions"] == item["decisions"], "completed episode mismatch")
        item["reported_duration_s"] = completed[i]["duration_s"]
    if ep is not None:
        ep["terminal"] = False; episodes.append(ep)
    for item in episodes:
        item.pop("_seen"); item.pop("_handoffs"); item.pop("_reopening_states")
        item["event_counts_by_leg_and_credit"] = dict(Counter(f'{x["leg"]}:{x["kind"]}:{x["credit"]}' for x in item["events"]))
        placements = sorted((x for x in item["events"] if x["leg"] == "FL" and x["kind"] == "P"), key=lambda x:x["physics_tick"])
        item["fl_first_placement"] = placements[0] if placements else None
        item["fl_first_policy_placement"] = next((x for x in placements if x["credit"] == "policy"), None)
    core = manifest["telemetry"]["core"]
    need(core["decisions"] == actual and core["physics_ticks"] == totals["physics_ticks"], "telemetry count mismatch")
    need(dict(core["phase_decisions"]) == {p:c["optimized_decisions"] for p,c in phases.items() if c["optimized_decisions"]}, "phase telemetry mismatch")
    need(core.get("prefix_behavior_decisions", 0) == prefix["decisions"] and core.get("prefix_physics_ticks", 0) == prefix["physics_ticks"], "prefix exclusion mismatch")
    coverage = [[run.name, begin, end, p]+[c[k] for k in HEADER[4:-1]]+[bool(c["optimized_decisions"])] for p,c in phases.items()]
    return coverage, {"run": str(run), "lifecycle": manifest["lifecycle"], "source_global": begin, "end_global": end,
        "actual_decisions": actual, "planned_decisions": manifest.get("planned_requested_policy_decisions"),
        "unconsumed_decisions": manifest.get("unconsumed_requested_policy_decisions"), "rounding_overrun": manifest.get("rounding_overrun"),
        "updates": len(updates), "optimizer_steps": sum(x["optimizer_steps"] for x in updates),
        "changed_actor_updates": sum(x["actor_parameters_changed"] is True for x in updates),
        "finite_gradient_updates": sum(x["finite_nonzero_gradient_observed"] is True for x in updates),
        "actor_chain_matches": all(a["actor_parameter_sha256_after"] == b["actor_parameter_sha256_before"] for a,b in zip(updates, updates[1:])),
        "wall_time_s": manifest["wall_time_s"], "totals": totals, "prefix_excluded": prefix, "prefix_phase_samples": prefix_phases,
        "prefix_attempt_results": [{"accepted": x["accepted"], "prefix_decisions": x["prefix_decisions"],
            "prefix_physics_ticks": x["prefix_physics_ticks"], "target_first_observed_decision": x.get("target_first_observed_decision"),
            "miss": {k:(x.get("miss") or {}).get(k) for k in ("reason", "termination_reason", "actual_phase", "physics_tick")}}
            for x in core.get("prefix_attempts", [])],
        "credit_starts": starts, "completed_episode_count": sum(x["terminal"] for x in episodes), "episodes": episodes,
        "completed_episode_rows_outside_boundary": len(completed)-sum(x["terminal"] for x in episodes),
        "checkpoints": manifest.get("checkpoints", []), "sampling": manifest.get("implemented_sampling"), "authority_by_source_phase": authority,
        "authority_scope": "Decision-source phase; raw is sampled latent, remaining vectors are decision-end snapshots, not every 120Hz tick. Filtered request is before headroom. FR knee index3 is canonical degrees relative to standing; wheel indices8:12 are canonical logical rad/s, NOT signed float32 native readback or measured q/velocity. Missing and unvisited ranges are unavailable, not zero. No success inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); root = Path(__file__).resolve().parent; output = args.output_dir.resolve()
    need(output == root or root in output.parents, "output must remain under outputs/ppo_transfer_roles_v1")
    paths = [output/"coverage_table.json", output/"training_block_summary.json"]
    need(not any(p.exists() for p in paths), "exclusive outputs already exist; use a new output subdirectory")
    runs = [p.resolve() for p in args.run]; need(len(set(runs)) == len(runs), "duplicate run")
    rows, summaries = [], []
    for run in runs:
        coverage, summary = summarize(run); rows.extend(coverage); summaries.append(summary)
    ordered = sorted(summaries, key=lambda x: x["source_global"])
    need(all(a["end_global"] <= b["source_global"] for a,b in zip(ordered, ordered[1:])), "overlapping run credit ranges")
    output.mkdir(parents=True, exist_ok=True)
    results = [{"header": HEADER, "rows": rows, "scope": "optimized source-phase counts; unvisited phase quality is unavailable"},
               {"schema": "ppo_transfer_roles_v1.completed_blocks.v1", "runs": summaries,
                "limits": ["No PT/torch/physics loaded; stored flags, not fresh physical revalidation.",
                    "Reopening AIR/role flags are decision-end observations, not every 120 Hz raw state.",
                    "Historical placement is not current support; event ticks retain prefix-vs-policy attribution."]}]
    for path, result in zip(paths, results):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"outputs": [str(p) for p in paths], "runs": len(runs), "optimized_decisions": sum(x["actual_decisions"] for x in summaries)}))


if __name__ == "__main__":
    main()
