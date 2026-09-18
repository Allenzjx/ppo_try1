"""Extract sealed, continuous contact/free-rise evidence; not task re-evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "residual_RR_diagnosis.json"
data = json.loads(source.read_text(encoding="utf8"))
result = {
    "schema": "wlr50_clean.rr_free_lift_fixture.v1",
    "purpose": "positive/negative qualification component replay fixture, not physics or full-success re-evaluation",
    "source": str(source),
    "existing_minimum_lift_gain_m": 0.008,
    "new_rule_status": "proposal_not_implemented",
    "no_original_labels_changed": True,
    "runs": {},
}
for name, run in data["runs"].items():
    rows = run["rows"]
    by_tick = {r["tick"]: r for r in rows}
    groups = []
    for row in rows:
        if not row["RR_air"]:
            continue
        if not groups or groups[-1][-1]["tick"] + 1 != row["tick"]:
            groups.append([row])
        else:
            groups[-1].append(row)
    segments = []
    for group in groups:
        before = by_tick.get(group[0]["tick"] - 1)
        if before is None:
            continue
        maximum = max(group, key=lambda r: r["RR_bottom_z"])
        reaches = [r["tick"] for r in group if r["RR_bottom_z"] - before["RR_bottom_z"] >= .008]
        segments.append({
            "air_start_tick": group[0]["tick"], "air_end_tick": group[-1]["tick"],
            "contact_baseline_tick": before["tick"], "contact_baseline_z_m": before["RR_bottom_z"],
            "preceding_ground_contact": before["RR_ground"], "preceding_obstacle_surface": before["RR_surface"],
            "maximum_free_rise_m": maximum["RR_bottom_z"] - before["RR_bottom_z"],
            "maximum_free_rise_tick": maximum["tick"],
            "first_free_8mm_tick": reaches[0] if reaches else None,
            "max_AIR_above_obstacle_top_m": maximum["RR_gap"],
        })
    keys = ("tick", "time_s", "RR_ground", "RR_obstacle", "RR_air", "RR_surface",
            "RR_bottom_z", "RR_gap", "RR_front_distance", "RR_bearing_N", "RR_bearing_verified",
            "RR_load_fraction", "RR_obstacle_force_w_n", "RR_qd", "FL_air", "bearing_by_leg")
    result["runs"][name] = {
        "sealed_source": run["source"], "bounds": run["bounds"],
        "air_segments": segments,
        "continuous_physics_rows": [{key: row[key] for key in keys} for row in rows],
    }
out = ROOT / "rr_lift_qualification_replay_fixture.json"
with out.open("x", encoding="utf8") as stream:
    json.dump(result, stream, indent=2, ensure_ascii=False)
print(out)
