"""Output-only, standard-library post-seal diagnostic; no simulator/model imports.

Prepared by static inspection. Do not execute until the owner confirms this run
has sealed and its Isaac process has exited. File length alone is not a seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/validation/20260923T0617478341071Z_g57ae41eb4b57_c5a28ad8bd2940ef9b5ade6b5e343d29"
HEAD_PREFIX = "57ae41eb4b57"
LEGS = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
LEG_FIELDS = ("air", "ground_contact", "top_surface_contact", "contact_surface", "support",
    "bearing_verified", "bearing_force_n", "obstacle_pair_active", "within_top_xy",
    "current_lift_valid", "lift_established", "front_distance_m", "clearance_m")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def finite_vector(value, size=3):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def fields(row, names):
    return {name: row.get(name) for name in names}


def lines(path, inventory):
    """One pass; digest the exact parsed bytes. Reject partial/non-object rows."""
    before = path.stat()
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), f"unsealed partial row: {path}:{count+1}")
            digest.update(raw)
            row = json.loads(raw)
            require(isinstance(row, dict), f"non-object row: {path}:{count+1}")
            count += 1
            yield row
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"stream changed while reading: {path}")
    inventory[path.name] = {"path": str(path), "rows": count, "sha256": digest.hexdigest()}


def read_json(path, inventory):
    data = path.read_bytes()
    inventory[path.name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    return json.loads(data)


def native_summary(row):
    audit = row["native_audit"]
    wheel = audit.get("rr_carry_wheel_evidence") or {}
    assist = audit.get("rr_capture_assist_evidence") or {}
    ctx, wc = assist.get("context") or {}, wheel.get("context") or {}
    output, previous = wheel.get("output_full12"), wheel.get("previous_final_wheel_rad_s")
    rate = [(output[8+i]-previous[i])*120. for i in range(4)] if finite_vector(output, 12) and finite_vector(previous, 4) else None
    return {"episode_post_tick": row["episode_physics_tick"],
        "native_dispatch_tick": audit.get("physics_tick"), "source_phase": row.get("source_phase_id"),
        "assist_pre_observation_tick": ctx.get("source_observation_tick"),
        "wheel_pre_observation_tick": wc.get("source_control_tick"),
        "source_nominal_full12": row.get("nominal_full12"),
        "raw_policy_full12": audit.get("raw_policy_action_full12"),
        "residual_mask_full12": audit.get("phase_mask_full12"),
        "projected_REQUEST_full12": row.get("projected_residual_full12"),
        "mapper_native_full12": audit.get("native_drive_target_full12"),
        "effective_policy_residual_full12": (audit.get("policy_headroom_evidence") or {}).get("effective_policy_residual_full12"),
        "native_actuator_targets": audit.get("actual_native_targets"),
        "native_joint_ids": {"servo": audit.get("servo_joint_ids"), "wheel": audit.get("wheel_joint_ids")},
        "native_verified": audit.get("verified"), "actual_target_source": audit.get("actual_target_source"),
        "rr_assist": assist, "wheel_projection": wheel,
        "wheel_final_rate_rad_s2_from_previous_FINAL_120Hz": rate,
        "pre_verified_FL_FR_RL_support": wc.get("verified_support_legs"),
        "pre_RR_current_TOP_bearing": ctx.get("current_top_bearing"),
        "pre_RR_TOP_is_not_all_ground_support": True}


def physical_summary(raw):
    wheels, contacts = raw.get("wheels") or {}, raw.get("contacts") or {}
    obstacle = raw.get("obstacle") or {}
    rr = wheels.get("rear_right_ankle") or {}
    bottom, center = rr.get("bottom_w_m"), rr.get("center_w_m")
    gap = bottom[2]-obstacle["top_z_m"] if finite_vector(bottom) and type(obstacle.get("top_z_m")) in (int, float) else None
    front = center[0]-obstacle["front_x_m"] if finite_vector(center) and type(obstacle.get("front_x_m")) in (int, float) else None
    quat, roll_pitch = (raw.get("base") or {}).get("orientation_wxyz"), None
    if finite_vector(quat, 4) and math.isclose(sum(v*v for v in quat), 1., abs_tol=1e-4):
        w, x, y, z = quat
        roll_pitch = [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x)))))]
    actual = raw.get("actual_full12")
    headroom = None
    if finite_vector(actual, 12):
        # Locked command_batch.py physical limits; NOT permission to extend assist.
        headroom = {leg: {joint: {"actual_deg": actual[2*i+j],
            "physical_command_bounds_deg": [-60., 210.] if j else [-135., 135.],
            "to_lower_deg": actual[2*i+j]-(-60. if j else -135.),
            "to_upper_deg": (210. if j else 135.)-actual[2*i+j]}
            for j, joint in enumerate(("hip", "knee"))} for i, leg in enumerate(LEGS)}
    return {"post_tick": raw["physics_tick"], "post_time_s": raw.get("simulation_time_s"),
        "base": raw.get("base"), "mass_COM": raw.get("center_of_mass"),
        "body_roll_pitch_deg_from_recorded_unit_wxyz": roll_pitch,
        "actual_full12_canonical_deg_rad_s": raw.get("actual_full12"),
        "actual_joint_physical_limit_headroom_not_search_budget": headroom,
        "commanded_full12": raw.get("commanded_full12"), "RR_gap_m": gap,
        "RR_front_distance_m": front, "obstacle_planes": obstacle,
        "joint_actual_command": {leg: {joint: (raw.get("joints") or {}).get(prefix+"_"+joint)
            for joint in ("hip", "knee")} for leg, prefix in LEGS.items()},
        "wheel_actual": {leg: wheels.get(prefix+"_ankle") for leg, prefix in LEGS.items()},
        "current_raw_contact_pairs": {leg: contacts.get(prefix+"_wheel") for leg, prefix in LEGS.items()},
        "stored_support_diagnostic_not_recomputed": raw.get("support"),
        "RR_hip_mount_w_m": None,
        "RR_hip_mount_unavailable_reason": "headless streams omit USD joint localPos0/mount transform; upper link origin is NOT substituted",
        "RR_upper_link_origin_w_m_not_hip_mount": ((raw.get("bodies") or {}).get("rear_right_upper") or {}).get("position_w_m")}


def make_anchor(raw, phase, receiver):
    com = raw.get("center_of_mass") or {}
    receiver0 = ((raw.get("wheels") or {}).get(LEGS[receiver]+"_ankle") or {}).get("center_w_m")
    com0 = com.get("position_w_m")
    valid = com.get("valid") is True and finite_vector(com0) and finite_vector(receiver0)
    dx, dy = (receiver0[0]-com0[0], receiver0[1]-com0[1]) if valid else (0., 0.)
    norm = math.hypot(dx, dy)
    valid = valid and norm > 1e-12
    return {"phase": phase, "receiver": receiver, "tick": raw["physics_tick"],
        "time_s": raw.get("simulation_time_s"), "valid": valid, "COM0_w_m": com0,
        "receiver0_w_m": receiver0, "base0_w_m": (raw.get("base") or {}).get("position_w_m"),
        "d0_xy": [dx/norm, dy/norm] if valid else None,
        "reason": "first scheduler entry; not exact physical transfer onset" if valid else "missing/invalid mass COM or receiver/degenerate planar direction"}


def fixed_projection(raw, anchor):
    if not anchor or not anchor.get("valid") or raw["physics_tick"] < anchor["tick"]:
        return None
    com = raw.get("center_of_mass") or {}
    point, velocity = com.get("position_w_m"), com.get("velocity_w_m_s")
    foot = ((raw.get("wheels") or {}).get(LEGS[anchor["receiver"]]+"_ankle") or {}).get("center_w_m")
    base = (raw.get("base") or {}).get("position_w_m")
    direction = anchor["d0_xy"]
    def project(value, origin):
        return sum((value[i]-origin[i])*direction[i] for i in range(2)) if finite_vector(value) and finite_vector(origin) else None
    return {"anchor_tick": anchor["tick"], "valid_COM": com.get("valid") is True,
        "COM_toward_receiver_m": project(point, anchor["COM0_w_m"]) if com.get("valid") is True else None,
        "COM_velocity_toward_receiver_m_s": sum(velocity[i]*direction[i] for i in range(2)) if com.get("valid") is True and finite_vector(velocity) else None,
        "receiver_motion_separate_m": project(foot, anchor["receiver0_w_m"]),
        "base_motion_separate_m": project(base, anchor["base0_w_m"]),
        "reference_only_not_transfer_validity_or_support_proof": True}


def diagnose(destination):
    inventory = {}
    run = read_json(SOURCE/"run_manifest.json", inventory)
    ev = read_json(SOURCE/"evaluation_manifest.json", inventory)
    require(run.get("lifecycle") in ("SUCCEEDED", "FAILED") and run.get("completed_at_utc"), "run is not sealed")
    require(run.get("command") == "eval" and ev.get("optimizer_updates_during_evaluation") == 0, "not read-only evaluation")
    require(ev.get("runtime_contract", {}).get("source_git_commit", "").startswith(HEAD_PREFIX), "wrong runtime")
    require(run.get("runtime_contract") == ev.get("runtime_contract"), "runtime manifests disagree")
    terminal = ev.get("observed_physics_ticks")
    require(type(terminal) is int and 0 < terminal <= 24000, "invalid continuous episode extent")
    milestones, entries, stage_rows = {terminal: ["terminal_or_evaluation_end"]}, {}, {}
    def add(tick, label):
        if type(tick) is int and 0 <= tick <= terminal:
            if label not in milestones.setdefault(tick, []):
                milestones[tick].append(label)
    add(terminal-240, "terminal_minus_2s")
    add(terminal-120, "terminal_minus_1s")
    for row in lines(SOURCE/"stage_transition_evidence.jsonl", inventory):
        tick, phase = row["physics_tick"], row.get("to_stage")
        if phase in ("P07", "P10", "P11", "P12", "P13") and phase not in entries:
            entries[phase] = tick; add(tick, "first_"+phase+"_entry"); stage_rows[tick] = row
        for event in ("active_lift", "front_edge_crossed", "placed"):
            exact = row.get("physical_history", {}).get("event_ticks", {}).get(event, {}).get("RR")
            if type(exact) is int:
                add(exact, "RR_history_"+event)
    # Retain compact actual decision endpoints only; no reconstruction of raw/HISTORY.
    decisions = {}
    for row in lines(SOURCE/"residual_and_projection_audit.jsonl", inventory):
        tick, task = row["physics_tick"], row.get("semantic_task") or {}
        current = task.get("physical_evaluator") or {}
        decisions[tick] = {"post_tick": tick, "post_time_s": row.get("sim_time_s"),
            "phase_start": row.get("phase_id"), "phase_end": row.get("end_phase_id"),
            "current_legs": {leg: fields((current.get("current_legs") or {}).get(leg, {}), LEG_FIELDS) for leg in LEGS},
            "history": task.get("history"), "potential": task.get("task_progress_potential"),
            "termination_reason": row.get("termination_reason"), "final_drive_full12": row.get("actual_drive_target_full12")}
    require(max(decisions, default=-1) == terminal, "decision tail differs from sealed extent")
    selected_native, seen, previous_row = {}, set(), None
    native_count = 0
    for row in lines(SOURCE/"native_tick_audit.jsonl", inventory):
        tick = row["episode_physics_tick"]
        require(tick == native_count+1, "non-contiguous native physics stream")
        native_count = tick
        audit = row["native_audit"]
        wheel, rr = audit.get("rr_carry_wheel_evidence") or {}, audit.get("rr_capture_assist_evidence") or {}
        state, ctx = rr.get("state_after") or {}, rr.get("context") or {}
        keys = []
        if wheel.get("envelope_active") is True:
            keys.append("first_RR_source_finished_wheel_envelope")
        if wheel.get("actual_projection_changed") is True:
            keys.append("first_actual_wheel_projection")
        if row.get("source_phase_id") == "P09":
            mode = state.get("mode_name")
            if mode not in (None, "WAIT"):
                keys.append("first_RR_assist_"+mode)
            travel = state.get("travel_used_deg")
            if type(travel) in (float, int) and travel > 20.:
                keys.append("first_RR_positive_knee_search")
            if type(travel) in (float, int) and travel >= 40.-1e-8:
                keys.append("first_RR_combined_search_budget_exhausted")
            if ctx.get("top_surface_contact") is True:
                keys.append("first_pre_dispatch_RR_TOP_contact")
            if ctx.get("current_top_bearing") is True:
                keys.append("first_pre_dispatch_RR_current_TOP_bearing")
        for key in keys:
            if key not in seen:
                seen.add(key); add(tick, key)
                if key.startswith("first_pre_dispatch_RR_"):
                    pre_tick = ctx.get("source_observation_tick")
                    add(pre_tick, key.replace("pre_dispatch", "observed"))
                    if previous_row and previous_row["episode_physics_tick"] == pre_tick:
                        selected_native[pre_tick] = native_summary(previous_row)
        if tick in milestones:
            selected_native[tick] = native_summary(row)
        previous_row = row
    require(native_count == terminal, "native tail differs from sealed extent")
    anchors, samples, raw_count = {}, [], -1
    for raw in lines(SOURCE/"physical_observations.jsonl", inventory):
        tick = raw["physics_tick"]
        require(tick == raw_count+1, "non-contiguous raw physics stream")
        raw_count = tick
        for phase, receiver in (("P07", "FL"), ("P10", "FR")):
            if entries.get(phase) == tick:
                anchors[phase] = make_anchor(raw, phase, receiver)
        if tick in milestones:
            samples.append({"labels": milestones[tick], "physical_post_step": physical_summary(raw),
                "dispatch_pre_step": selected_native.get(tick), "exact_decision_endpoint": decisions.get(tick),
                "stage_event": stage_rows.get(tick), "fixed_reference_projections": {
                    phase: fixed_projection(raw, anchors.get(phase)) for phase in ("P07", "P10")}})
    require(raw_count == terminal, "raw physical tail differs from sealed extent")
    for phase in ("P07", "P10"):
        anchors.setdefault(phase, {"valid": False, "reason": "phase not reached; no fabricated/reanchored direction"})
    report = {"schema": "output_only.sealed_fixed_COM_RR_response.v1", "source": str(SOURCE),
        "evaluation": fields(ev, ("checkpoint", "checkpoint_sha256", "task_success", "termination_reason", "duration_s", "policy_decisions")),
        "inventory": inventory, "anchors": anchors, "milestones": samples,
        "limits": ["Fixed scheduler-entry reference, never rolling reanchored; not proof of physical transfer validity or causality.",
            "Pre-dispatch context uses prior observed state; physical rows are after step. Native global dispatch tick is not episode tick.",
            "Historical placed is not current bearing. Raw contact pairs and exact decision evaluator support are separate.",
            "Decision endpoint current_legs may be absent at a non-15Hz milestone; never substituted with later state.",
            "True hip mount and unpersisted native actuator IDs are null. Native targets are not measured angular velocity.",
            "No actor forward, optimizer, GAE/reward credit, new physics, or task-acceptance recomputation.",
            "Windows live Length=0 is not evidence of an empty stream; only parsed sealed lines are counted."]}
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/"fixed_COM_RR_response.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    summary = ["# Sealed fixed CoM / RR response diagnostic", "", f"Source: `{SOURCE}`",
        f"Outcome: `{ev.get('termination_reason')}`; full success `{ev.get('task_success')}`; duration `{ev.get('duration_s')}` s.", "",
        "| Post tick | Milestone | RR gap m | RR front m | P07 fixed CoM m | P10 fixed CoM m |", "|---:|---|---:|---:|---:|---:|"]
    for sample in samples:
        p, projection = sample["physical_post_step"], sample["fixed_reference_projections"]
        summary.append(f"| {p['post_tick']} | {', '.join(sample['labels'])} | {p['RR_gap_m']} | {p['RR_front_distance_m']} | {(projection['P07'] or {}).get('COM_toward_receiver_m')} | {(projection['P10'] or {}).get('COM_toward_receiver_m')} |")
    summary += ["", "P07/P10 exact raw anchors and same-step wheel/servo/contact evidence are in JSON.", ""] + report["limits"]
    (destination/"fixed_COM_RR_response.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "milestones": len(samples), "anchors": anchors}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-confirmed-sealed", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    diagnose(arguments.output)
