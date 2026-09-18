"""Re-evaluate only RR components on sealed observations, with no physical writes."""
import itertools
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, load_task_spec, P09_FREE_AIR_LIFT_MODE

index = json.loads((ROOT / "outputs/diagnostics_v1/residual_RR_diagnosis.json").read_text(encoding="utf8"))
base = load_task_spec(ROOT / "configs/ppo_task_first_recovery_v1/stage_task_spec.yaml")
new = deepcopy(base)
new["p09_lift_semantics"] = P09_FREE_AIR_LIFT_MODE
receipt = {"schema": "wlr50_clean.RR_v3_sealed_input_component_replay.v1",
    "no_physics": True, "original_labels_changed": False,
    "scope": "bounded RR evaluator only; front histories are not seeded, no full-success re-evaluation",
    "old_mode": base["p09_lift_semantics"], "new_mode": P09_FREE_AIR_LIFT_MODE, "runs": {}}
for name, start, end in (("CP177152", 4292, 5432), ("retained_zero", 5100, 6160)):
    source = Path(index["runs"][name]["source"])
    old_ev, new_ev = TaskEvaluator(spec=base), TaskEvaluator(spec=new)
    rows = []
    with (source / "physical_observations.jsonl").open(encoding="utf8") as stream:
        for raw in map(json.loads, itertools.islice(stream, start, end+1)):
            old_state, new_state = old_ev.observe(raw), new_ev.observe(raw)
            rr, old_rr = new_state["current_legs"]["RR"], old_state["current_legs"]["RR"]
            rows.append(dict(tick=raw["physics_tick"], air=rr["air"], surface=rr["contact_surface"],
                ground=rr["ground_contact"], old_initial=old_rr["initial_now"], new_initial=rr["initial_now"],
                old_qualified=old_rr["lift_established_now"], new_qualified=rr["lift_established_now"],
                old_current_valid=old_rr["current_lift_valid"], new_current_valid=rr["current_lift_valid"],
                old_crossed=old_state["history"]["front_edge_crossed"]["RR"],
                new_crossed=new_state["history"]["front_edge_crossed"]["RR"],
                unsupported_free_lift_m=rr["unsupported_free_lift_m"],
                ground_relative_lift_m=rr["ground_relative_lift_m"],
                bottom_vz_m_s=rr["wheel_bottom_vz_m_s"],
                new_termination_reason=new_state["termination_reason"]))
    old_events = [e for e in old_ev.snapshot["history"]["lift_attempt_events"] if e["leg"] == "RR"]
    new_events = [e for e in new_ev.snapshot["history"]["lift_attempt_events"] if e["leg"] == "RR"]
    old_q = [e["physics_tick"] for e in old_events if e["event"] == "qualified_measured_upward_lift"]
    new_q = [e["physics_tick"] for e in new_events if e["event"] == "qualified_measured_upward_lift"]
    by_tick = {row["tick"]: row for row in rows}
    if name == "CP177152":
        assert old_q == [4633, 4940], old_q
        assert new_q == [4633], new_q
        assert not by_tick[4731]["new_current_valid"]
        assert not by_tick[4900]["new_initial"] and not by_tick[4940]["new_qualified"]
        assert not by_tick[5428]["new_crossed"]
    else:
        assert old_q == new_q == [5434], (old_q, new_q)
        assert all(row["old_current_valid"] == row["new_current_valid"] for row in rows)
        assert by_tick[6155]["new_crossed"]
    receipt["runs"][name] = dict(source=str(source), bounds=[start,end], row_count=len(rows),
        old_qualification_events=old_q, new_qualification_events=new_q,
        old_RR_events=old_events, new_RR_events=new_events, rows=rows)
out = ROOT / "outputs/diagnostics_v1/rr_v3_evaluator_replay_receipt.json"
with out.open("x", encoding="utf8") as stream:
    json.dump(receipt, stream, indent=2, ensure_ascii=False)
print(json.dumps({name: {k:v for k,v in result.items() if k not in ("rows", "old_RR_events", "new_RR_events")}
                  for name,result in receipt["runs"].items()}, indent=2))
