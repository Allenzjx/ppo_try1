"""Build a bounded 120 Hz successful-N rear-transfer reference table.

This is an outputs-only, standard-library reader.  It never imports the
runtime, Torch, Isaac, or changes the sealed source.  Rows are exact logged
episode physics ticks; missing fields stay ``None`` and are never interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / (
    "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/"
    "20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source"
)
CONTRACT = REPO / "configs/recording_motion_contract.json"
CP225280 = REPO / (
    "outputs/ppo_rr_rl_timing_policy_learning_v1/"
    "CP225280_cooperative_RR_RL_window_audit.json"
)
OUT_JSON = Path(__file__).with_name("successful_N_P09late_P12_event_response.json")
OUT_MD = Path(__file__).with_name("successful_N_P09late_P12_event_response.md")

ORDER = (
    "FL_hip", "FL_knee", "FR_hip", "FR_knee",
    "RL_hip", "RL_knee", "RR_hip", "RR_knee",
    "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel",
)
JOINT_KEYS = {
    "FL_hip": "front_left_hip",
    "FL_knee": "front_left_knee",
    "FR_hip": "front_right_hip",
    "FR_knee": "front_right_knee",
    "RL_hip": "rear_left_hip",
    "RL_knee": "rear_left_knee",
    "RR_hip": "rear_right_hip",
    "RR_knee": "rear_right_knee",
}
WHEEL_KEYS = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}
CONTACT_KEYS = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}

# Exact episode-physics ticks from the sealed successful N source.  Selection is
# intentionally small but spans prerequisites, launches, responses, stops, and
# capture.  It is not a resampled trajectory.
EVENTS = {
    6152: "PRE_LATE_RR_LOADED_CONTEXT",
    6155: "RR_CROSSED_AND_PLACED_EVENT",
    6156: "P09_LATE_ATOMIC_FL_RL_LAUNCH",
    6160: "P09_LATE_EARLY_RESPONSE",
    6161: "P10_ENTRY_RR_KNEE_POSITIVE_STEP",
    6168: "P10_RR_KNEE_POSITIVE_ENDPOINT",
    6169: "P11_ENTRY_FR_RECEIVING_SPACE_STEP",
    6176: "P11_RESPONSE_FR_AIR_RR_LOADED",
    6177: "P12_ENTRY_RL_KNEE_POSITIVE_STEP",
    6201: "P12_RL_KNEE_POSITIVE_AND_FR_HIP_RESPONSE",
    6233: "P12_FOUR_WHEEL_REVERSE_BEGIN",
    6251: "RL_QUALIFIED_EVENT",
    6256: "REVERSE_AND_LOAD_RESPONSE",
    6272: "RR_SHORT_AIR_SNAPSHOT",
    6320: "RL_SWING_CONTINUATION",
    6497: "FIRST_REVERSE_STOP",
    6522: "FORWARD_RESUME",
    6652: "SECOND_STOP_BEFORE_RL_CROSS",
    6658: "RL_FRONT_EDGE_CROSSED_EVENT",
    6727: "RL_PLACED_EVENT",
    6728: "RL_CAPTURE_RESPONSE",
}


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full12(values: list[float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    if len(values) != 12:
        raise ValueError(f"expected Full12, got {len(values)}")
    return dict(zip(ORDER, values, strict=True))


def _contact(row: dict[str, Any], leg: str) -> dict[str, Any]:
    item = row["contacts"].get(CONTACT_KEYS[leg])
    if item is None:
        return {
            "contact_class": None,
            "ground_force_n": None,
            "obstacle_force_n": None,
            "total_normal_force_n": None,
        }
    ground = item.get("ground", {})
    obstacle = item.get("obstacle", {})
    gf = ground.get("normal_force_n")
    of = obstacle.get("normal_force_n")
    return {
        "contact_class": item.get("contact_class"),
        "ground_active": ground.get("active"),
        "obstacle_active": obstacle.get("active"),
        "ground_force_n": gf,
        "obstacle_force_n": of,
        "total_normal_force_n": None if gf is None or of is None else gf + of,
    }


def _geometry(row: dict[str, Any], leg: str) -> dict[str, float | None]:
    wheel = row["wheels"].get(WHEEL_KEYS[leg])
    obstacle = row.get("obstacle", {})
    if wheel is None or not wheel.get("center_w_m") or not wheel.get("bottom_w_m"):
        return {"front_distance_mm": None, "top_gap_mm": None}
    front = obstacle.get("front_x_m")
    top = obstacle.get("top_z_m")
    return {
        "front_distance_mm": None if front is None else 1000.0 * (wheel["center_w_m"][0] - front),
        "top_gap_mm": None if top is None else 1000.0 * (wheel["bottom_w_m"][2] - top),
    }


def _phase_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    wanted = {}
    for phase in contract["phases"]:
        if phase["state_id"] not in {"P09", "P10", "P11", "P12"}:
            continue
        waypoints = []
        for waypoint in phase.get("waypoints", []):
            if waypoint.get("kind") == "phase_entry" and not waypoint.get("changed_channels"):
                continue
            waypoints.append({
                "time_s": waypoint.get("time_s"),
                "kind": waypoint.get("kind"),
                "changed_channels": waypoint.get("changed_channels", []),
                "source_commands": waypoint.get("source_commands", []),
                "full12": _full12(waypoint.get("full12")),
            })
        wanted[phase["state_id"]] = {
            "state_name": phase["state_name"],
            "physical_purpose": phase["physical_purpose"],
            "start_full12": _full12(phase["start_full12"]),
            "end_full12": _full12(phase["end_full12"]),
            "waypoints": waypoints,
        }
    return {
        "path": str(CONTRACT),
        "sha256": _sha256(CONTRACT),
        "schema": contract["schema"],
        "reference_version": contract["reference_version"],
        "physics_hz": contract["physics_hz"],
        "full12_order": contract["full12_order"],
        "phases": wanted,
    }


def _cp225280_comparison() -> dict[str, Any]:
    audit = json.loads(CP225280.read_text(encoding="utf-8"))
    by_tick = {row["physics_tick"]: row for row in audit["selected_rows"]}
    chosen = []
    for tick, label in (
        (6752, "P09_ENTRY"),
        (6995, "RR_QUALIFIED"),
        (7979, "RR_CROSSED_AIR"),
        (10940, "TERMINAL"),
    ):
        row = by_tick[tick]
        chosen.append({
            "event": label,
            "physics_tick": tick,
            "time_s": row["time_s"],
            "phase": row["phase"],
            "FL_knee_actual_deg": row["front_knee_actual_deg"]["FL"],
            "FR_knee_actual_deg": row["front_knee_actual_deg"]["FR"],
            "wheel_final_target_rad_s": row["wheel_canonical_rad_s"]["final_target"],
            "wheel_actual_rad_s": row["wheel_canonical_rad_s"]["actual_measured"],
            "RR": row["RR"],
            "RL": row["RL"],
            "CoM_to_FR_local_window": row["com_to_FR_local_window"],
            "rear_policy_timing": row["rear_policy_timing"],
        })
    return {
        "path": str(CP225280),
        "sha256": _sha256(CP225280),
        "selected_rows": chosen,
        "interpretation_boundary": (
            "Different controller/policy geometry and episode. Descriptive comparison only; "
            "not a time-aligned counterfactual or causal attribution."
        ),
    }


def build() -> dict[str, Any]:
    manifest_path = SOURCE / "semantic_video_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("physical_task_success"):
        raise RuntimeError("selected reference is not the sealed successful N episode")
    if manifest.get("policy_sampling_mode") != "nominal_without_learned_residual":
        raise RuntimeError("selected reference is not N+0")

    selected = set(EVENTS)
    native: dict[int, dict[str, Any]] = {}
    for row in _json_lines(SOURCE / "native_tick_audit.jsonl"):
        tick = row["episode_physics_tick"]
        if tick in selected:
            audit = row["native_audit"]
            if any(abs(x) > 0.0 for x in audit["raw_policy_action_full12"]):
                raise RuntimeError(f"N reference has nonzero policy action at tick {tick}")
            native[tick] = {
                "source_phase": row["source_phase_id"],
                "source_nominal_full12": _full12(row["nominal_full12"]),
                "projected_residual_full12": _full12(row["projected_residual_full12"]),
                "final_drive_target_full12": _full12(audit["native_drive_target_full12"]),
                "raw_physics_tick": audit["physics_tick"],
                "audit_verified": audit["verified"],
            }
        if tick > max(selected):
            break

    physical: dict[int, dict[str, Any]] = {}
    for row in _json_lines(SOURCE / "physical_observations.jsonl"):
        tick = row["physics_tick"]
        if tick in selected:
            joints = {
                label: {
                    "actual_deg": row["joints"][key]["position_deg"],
                    "actual_velocity_deg_s": row["joints"][key]["velocity_deg_s"],
                    "command_deg": row["joints"][key]["command_deg"],
                }
                for label, key in JOINT_KEYS.items()
            }
            wheels = {
                label: {
                    "command_rad_s": row["wheels"][key]["command_rad_s"],
                    "actual_rad_s": row["wheels"][key]["velocity_rad_s"],
                }
                for label, key in WHEEL_KEYS.items()
            }
            physical[tick] = {
                "time_s": row["simulation_time_s"],
                "physical_commanded_full12": _full12(row["commanded_full12"]),
                "physical_actual_full12": _full12(row["actual_full12"]),
                "joints": joints,
                "wheels": wheels,
                "contacts": {leg: _contact(row, leg) for leg in ("FL", "FR", "RL", "RR")},
                "geometry": {leg: _geometry(row, leg) for leg in ("FL", "FR", "RL", "RR")},
                "support": row.get("support"),
                "base_position_w_m": row["base"]["position_w_m"],
                "base_linear_velocity_w_m_s": row["base"]["linear_velocity_w_m_s"],
                "base_angular_velocity_w_rad_s": row["base"]["angular_velocity_w_rad_s"],
                "center_of_mass_position_w_m": row["center_of_mass"]["position_w_m"],
                "center_of_mass_velocity_w_m_s": row["center_of_mass"]["velocity_w_m_s"],
            }
        if tick > max(selected):
            break

    transitions = []
    for row in _json_lines(SOURCE / "stage_transition_evidence.jsonl"):
        tick = row["physics_tick"]
        if 6150 <= tick <= 6728:
            transitions.append({
                "physics_tick": tick,
                "sim_time_s": row["sim_time_s"],
                "from_stage": row["from_stage"],
                "to_stage": row["to_stage"],
                "event_ticks": row["physical_history"]["event_ticks"],
            })
    if not transitions:
        raise RuntimeError("rear transfer transition evidence is absent")
    final_ticks = transitions[-1]["event_ticks"]
    if (final_ticks["front_edge_crossed"].get("RR") != 6155
            or final_ticks["placed"].get("RR") != 6155
            or final_ticks["active_lift"].get("RL") != 6251
            or final_ticks["front_edge_crossed"].get("RL") != 6658
            or final_ticks["placed"].get("RL") != 6727):
        raise RuntimeError("successful rear event ticks differ from the selected reference")

    missing = sorted(selected - native.keys()) + sorted(selected - physical.keys())
    if missing:
        raise RuntimeError(f"selected exact ticks missing: {missing}")

    rows = []
    for tick in sorted(selected):
        n = native[tick]
        p = physical[tick]
        # Both layers identify the same episode tick, but their logger boundaries
        # can straddle a source/servo update.  Preserve the difference instead of
        # silently asserting them equal or shifting either stream.
        alignment_delta = max(
            abs(a - b)
            for a, b in zip(
                n["final_drive_target_full12"].values(),
                p["physical_commanded_full12"].values(),
                strict=True,
            )
        )
        rows.append({
            "event": EVENTS[tick],
            "episode_physics_tick": tick,
            "native_vs_physical_command_max_abs_delta": alignment_delta,
            **n,
            **p,
        })

    artifacts = manifest["artifacts"]
    source_bindings = {}
    for name in (
        "native_tick_audit.jsonl",
        "physical_observations.jsonl",
        "stage_transition_evidence.jsonl",
        "video_policy_decisions.jsonl",
    ):
        source_bindings[name] = {
            "sha256": artifacts[name]["sha256"],
            "bytes": artifacts[name]["bytes"],
        }

    return {
        "schema": "outputs.successful_N_P09late_P12_event_response.v1",
        "scope": {
            "kind": "read_only_bounded_exact_tick_reference",
            "interpolation": False,
            "physics_or_model_execution": False,
            "source_reference_fact": (
                "Sealed successful N+0 natural-P01 episode. Commands, responses, and contacts "
                "below are logged facts at exact episode physics ticks."
            ),
            "user_priority_hypothesis": (
                "FR receiving space, RR load, FL usable travel/action, and coordinated RL "
                "knee/wheel retreat may form a useful mechanism; the reference does not prove "
                "that one fixed angle or pulse is necessary."
            ),
            "current_policy_boundary": (
                "CP225280 has a different controller/policy geometry. Its selected rows are "
                "descriptive only and are not a causal or time-aligned counterfactual."
            ),
        },
        "source": {
            "directory": str(SOURCE),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "runtime_git": manifest["runtime_contract"]["source_git_commit"],
            "policy_sampling_mode": manifest["policy_sampling_mode"],
            "optimizer_updates": manifest["optimizer_updates"],
            "physical_task_success": manifest["physical_task_success"],
            "physical_task_duration_s": manifest["physical_episode"]["physical_task_duration_s"],
            "artifacts": source_bindings,
        },
        "motion_contract": _phase_contract(),
        "stage_transition_event_evidence": transitions,
        "rows": rows,
        "CP225280_comparison": _cp225280_comparison(),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(data: dict[str, Any]) -> str:
    by_event = {row["event"]: row for row in data["rows"]}
    compact_events = [
        "PRE_LATE_RR_LOADED_CONTEXT",
        "P09_LATE_ATOMIC_FL_RL_LAUNCH",
        "P09_LATE_EARLY_RESPONSE",
        "P10_RR_KNEE_POSITIVE_ENDPOINT",
        "P11_RESPONSE_FR_AIR_RR_LOADED",
        "P12_RL_KNEE_POSITIVE_AND_FR_HIP_RESPONSE",
        "P12_FOUR_WHEEL_REVERSE_BEGIN",
        "REVERSE_AND_LOAD_RESPONSE",
        "FIRST_REVERSE_STOP",
        "SECOND_STOP_BEFORE_RL_CROSS",
        "RL_FRONT_EDGE_CROSSED_EVENT",
        "RL_CAPTURE_RESPONSE",
    ]
    lines = [
        "# Successful N: P09 late → P12 exact-tick event/command/response",
        "",
        "This is a bounded, read-only mechanism reference from the sealed successful N+0 episode. "
        "Every row is an exact 120 Hz episode tick; no interpolation or new physics was used. "
        "Angles are absolute canonical degrees, wheels are canonical rad/s, geometry is mm, and "
        "forces are measured normal force. The source-command column is the authored nominal; the "
        "JSON separately retains native drive target and physical commanded target at that tick. "
        "N is a mechanism reference, not a required pose.",
        "",
        "## Compact event table",
        "",
        "|event|tick / t|source command (selected)|measured response|contact / geometry context|",
        "|---|---:|---|---|---|",
    ]
    for event in compact_events:
        r = by_event[event]
        n = r["source_nominal_full12"]
        j = r["joints"]
        w = r["wheels"]
        c = r["contacts"]
        g = r["geometry"]
        cmd = (
            f"FL k={_fmt(n['FL_knee'],1)}, FR h/k={_fmt(n['FR_hip'],1)}/{_fmt(n['FR_knee'],1)}, "
            f"RL h/k={_fmt(n['RL_hip'],1)}/{_fmt(n['RL_knee'],1)}, "
            f"RR h/k={_fmt(n['RR_hip'],1)}/{_fmt(n['RR_knee'],1)}, "
            f"w={','.join(_fmt(n[x + '_wheel'],2) for x in ('FL','FR','RL','RR'))}"
        )
        response = (
            f"FL k={_fmt(j['FL_knee']['actual_deg'],1)}, FR h/k={_fmt(j['FR_hip']['actual_deg'],1)}/"
            f"{_fmt(j['FR_knee']['actual_deg'],1)}, RL k={_fmt(j['RL_knee']['actual_deg'],1)}, "
            f"RR k={_fmt(j['RR_knee']['actual_deg'],1)}; actual w="
            f"{','.join(_fmt(w[x]['actual_rad_s'],2) for x in ('FL','FR','RL','RR'))}; "
            f"vxy={_fmt(r['base_linear_velocity_w_m_s'][0],3)}/"
            f"{_fmt(r['base_linear_velocity_w_m_s'][1],3)}"
        )
        contact = (
            f"FR {c['FR']['contact_class']} {_fmt(c['FR']['total_normal_force_n'])}N; "
            f"FL {c['FL']['contact_class']} {_fmt(c['FL']['total_normal_force_n'])}N; "
            f"RR {c['RR']['contact_class']} {_fmt(c['RR']['total_normal_force_n'])}N "
            f"front/gap={_fmt(g['RR']['front_distance_mm'])}/{_fmt(g['RR']['top_gap_mm'])}; "
            f"RL {c['RL']['contact_class']} {_fmt(c['RL']['total_normal_force_n'])}N "
            f"front/gap={_fmt(g['RL']['front_distance_mm'])}/{_fmt(g['RL']['top_gap_mm'])}"
        )
        lines.append(
            f"|{event}|{r['episode_physics_tick']} / {_fmt(r['time_s'],3)}s|{cmd}|{response}|{contact}|"
        )

    lines += [
        "",
        "## What the successful source actually establishes",
        "",
        "- **FL action is not a single phase label.** At tick 6156 the P09-late atomic source changes "
        "FL hip/knee and the FL wheel together while also changing RL hip/knee. The FL knee and "
        "wheel remain active through the P10 and P11 RR/FR changes and into P12; the four-wheel "
        "reverse begins only at tick 6233. This is overlapping command evidence, not proof that one "
        "fixed FL pulse caused the body motion.",
        "- **The strong late group has prerequisites in this run.** Immediately before launch, RR is "
        "already crossed/placed and carrying TOP load. P10 then unfolds RR knee; P11 opens the FR "
        "hip while FR knee remains near the positive reference configuration. FR is allowed to go "
        "AIR during preparation; RR and FL retain real TOP support. The source therefore does not "
        "support treating an AIR RR as load-bearing.",
        "- **The selected response has measurable overlap, not just matching phase names.** From "
        "tick 6152 to 6201 the actual FL knee travels from about -12.0° to -33.1° while FL obstacle "
        "load grows from 5.93 N to 14.45 N; FR hip moves from about 0.5° to 8.9° while FR knee stays "
        "near +30° and FR unloads to AIR; RR remains obstacle-loaded near 14 N. At tick 6233, when "
        "the four-wheel reverse begins and RL is AIR, measured body vxy is +0.180/-0.120 m/s "
        "(negative y is the FR side in this run). This is the clearest selected concurrent "
        "front/FR-side motion signature, but it does not identify FL alone as causal.",
        "- **RL preparation overlaps body/wheel action.** P12 first drives RL knee positive while "
        "the prior FL/RR/FR configuration is still evolving. Four-wheel reverse starts at tick 6233 "
        "with FL and RR loaded; it is not an isolated RL-joint-only maneuver. The first stop at 6497, "
        "forward resume at 6522, and second stop at 6652 are explicit source commands. The second "
        "stop precedes RL edge crossing at 6658; capture follows with all four TOP at 6728.",
        "- **Response, not command sign, is the transferable fact.** Positive body-x and negative "
        "body-y (the FR side in this source) occur in parts of the window together with changing "
        "loads and RL unloading, but the logs do not identify one channel as independently causal. "
        "A new policy may reach the same physical result with different absolute angles.",
        "",
        "## CP225280 entrance contrast (descriptive, not causal)",
        "",
        "|event|tick / t|FL knee|FR knee|RR front/gap|RR TOP/load|late source|",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in data["CP225280_comparison"]["selected_rows"]:
        rr = r["RR"]
        timing = r["rear_policy_timing"]
        lines.append(
            f"|{r['event']}|{r['physics_tick']} / {_fmt(r['time_s'],3)}s|"
            f"{_fmt(r['FL_knee_actual_deg'])}|{_fmt(r['FR_knee_actual_deg'])}|"
            f"{_fmt(rr['front_distance_mm'])}/{_fmt(rr['gap_mm'])} mm|"
            f"{rr['current_TOP']}/{_fmt(rr['bearing_force_n'])}N|"
            f"p09={_fmt(timing['p09_source_time_s'],3)}s, wait={timing['p09_dependency_wait']}|"
        )
    lines += [
        "",
        "CP225280 never reaches the successful source's loaded P09-late entry: RR crosses while "
        "still AIR and remains 0 N; its P09 source time stalls at 5.4s and the loaded late group is "
        "not observed. FL knee is much more negative, "
        "and FR knee is negative rather than the successful reference's positive receiving geometry. "
        "That is a coupled entry/configuration difference—not evidence that copying +30° FR knee or "
        "any exact N angle is sufficient.",
        "",
        "## Evidence boundary",
        "",
        f"- N manifest: `{data['source']['manifest_sha256']}`; native 120 Hz audit: "
        f"`{data['source']['artifacts']['native_tick_audit.jsonl']['sha256']}`; physical observations: "
        f"`{data['source']['artifacts']['physical_observations.jsonl']['sha256']}`.",
        f"- Motion contract: `{data['motion_contract']['sha256']}`; CP225280 audit: "
        f"`{data['CP225280_comparison']['sha256']}`.",
        "- Contact class/force and geometry are copied from sealed logs. Missing values would be N/A; "
        "nothing is interpolated. User mechanism priorities remain hypotheses until a same-runtime "
        "physical diagnostic or policy evaluation produces the corresponding response.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate and print a compact summary only")
    args = parser.parse_args()
    data = build()
    markdown = render_markdown(data)
    if args.check:
        print(json.dumps({
            "schema": data["schema"],
            "rows": len(data["rows"]),
            "first_tick": data["rows"][0]["episode_physics_tick"],
            "last_tick": data["rows"][-1]["episode_physics_tick"],
            "source_success": data["source"]["physical_task_success"],
        }, sort_keys=True))
        return
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown, encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
