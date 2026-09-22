"""Bounded counterfactual Phi only; prints JSON, never edits records or runs actor/physics."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from wlr50_clean.ppo.semantic_supervisor import (
    TaskStageSupervisor, load_task_spec, _current_rr_receiver_preparation_retired,
    RR_WORKSPACE_RETIREMENT_MODE, RR_WORKSPACE_RETIREMENT_MODE_V2,
)

OUT = ROOT/"outputs/ppo_p05_hip_only_continuation_v1"
OLD_RUN = ROOT/"runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff"
BLOCK08 = ROOT/"runs/ppo_p05_hip_only_continuation_v1/train/20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812"
SPEC = ROOT/"configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stats(values):
    return {"n":len(values), "min":min(values), "max":max(values), "mean":sum(values)/len(values),
            "exactly_nonzero":sum(v != 0. for v in values), "above_1e12":sum(abs(v)>1e-12 for v in values)}


def main():
    v2 = load_task_spec(SPEC)
    assert v2["rr_postcross_workspace_semantics"] == RR_WORKSPACE_RETIREMENT_MODE_V2
    v1 = deepcopy(v2); v1["rr_postcross_workspace_semantics"] = RR_WORKSPACE_RETIREMENT_MODE
    old = object.__new__(TaskStageSupervisor); old.spec = v1
    new = object.__new__(TaskStageSupervisor); new.spec = v2
    input_hash = hashlib.sha256()
    unchanged = 0

    def compare(ev, *, source_index, learner_decision=None, recorded_phi=None):
        nonlocal unchanged
        before = digest(ev)
        input_hash.update(bytes.fromhex(before))
        rr, history = ev["current_legs"]["RR"], ev["history"]
        assert ev["valid"] is True and ev["termination_reason"] is None
        old_gate = _current_rr_receiver_preparation_retired(v1, "RR", ev)
        new_gate = _current_rr_receiver_preparation_retired(v2, "RR", ev)
        old_phi, new_phi = old.physical_potential(ev), new.physical_potential(ev)
        assert digest(ev) == before
        unchanged += 1
        delta = new_phi-old_phi
        expected_delta = (0. if history["placed"]["RR"] else
            .85/4*.1*.5*(float(new_gate)-float(old_gate))*(1.-ev["transfer_roles"]["RR"]["workspace_progress"]))
        assert abs(delta-expected_delta) < 1e-12
        if recorded_phi is not None:
            assert abs(old_phi-recorded_phi) < 1e-12
        return {"source_endpoint_index":source_index, "learner_global_decision":learner_decision,
            "input_tick":ev["physics_tick"], "currentQ":rr["current_lift_valid"],
            "historyQ":history["active_lift"]["RR"], "historyCross":history["front_edge_crossed"]["RR"],
            "historyPlaced":history["placed"]["RR"], "lift_established":rr["lift_established"],
            "body_control_evidence":rr["body_control_evidence"], "contact_mode":rr["contact_mode"],
            "ground":rr["ground_contact"], "legal_xy":rr["within_top_xy"] and rr["within_lateral_span"],
            "gap_mm":1000*rr["clearance_m"], "retired_v1":old_gate, "retired_v2":new_gate,
            "receiver_progress":ev["transfer_roles"]["RR"]["workspace_progress"],
            "phi_v1":old_phi, "phi_v2":new_phi, "phi_delta_v2_minus_v1":delta,
            "recorded_v1_phi_match":None if recorded_phi is None else True,
            "evaluator_object_unchanged":True}

    same41 = []
    old_path = OLD_RUN/"residual_and_projection_audit.jsonl"
    with old_path.open("rb") as stream:
        for index, line in enumerate(itertools.islice(stream, 1101)):
            if index < 1060:
                continue
            row = json.loads(line)
            ev = row["applied_audit"]["semantic_task"]["physical_evaluator"]
            assert ev["physics_tick"] == 8*(index+1) and not row["terminal"]
            same41.append(compare(ev, source_index=index))
    assert len(same41) == 41
    precapture = [r for r in same41 if not r["historyPlaced"]]
    assert len(precapture) == 30
    assert [r["input_tick"] for r in precapture if r["currentQ"]] == [8688,8696,8704,8720]
    assert sum(not r["retired_v1"] and r["retired_v2"] for r in precapture) == 26
    assert all(r["retired_v2"] for r in precapture)

    block_rows = []
    previous, previous_index, previous_phi = None, None, None
    block_path = BLOCK08/"residual_and_projection_audit.jsonl"
    with block_path.open("rb") as stream:
        for index, line in enumerate(itertools.islice(stream, 1024)):
            row = json.loads(line)
            assert row["global_policy_decision"] == 213377+index
            a = row["applied_audit"]
            if a["decision_count"] == 1:
                previous = None
            # Use the preceding endpoint only within the same real episode.
            # It is this learner's actual input, never the learner's future endpoint.
            if previous is not None and previous["history"]["front_edge_crossed"]["RR"]:
                block_rows.append(compare(previous, source_index=previous_index,
                    learner_decision=row["global_policy_decision"], recorded_phi=previous_phi))
            task = a["semantic_task"]
            previous = task["physical_evaluator"]
            previous_index, previous_phi = index, task["task_progress_potential"]
    assert index == 1023
    assert len(block_rows) == 389
    assert all(r["currentQ"] and r["retired_v1"] and r["retired_v2"] for r in block_rows)
    assert all(r["phi_delta_v2_minus_v1"] == 0. for r in block_rows)
    assert unchanged == 430
    a, b = [next(r for r in same41 if r["input_tick"] == tick) for tick in (8704,8712)]
    shaping = {version:5*(.9985*b["phi_"+version]-a["phi_"+version]) for version in ("v1","v2")}
    result = {
        "schema":"wlr50_clean.rr_receiver_v2_actual_states_readonly.v1",
        "analysis":"counterfactual potential on identical recorded physical evaluator states; not new reward/GAE in PPO",
        "source_head_now":subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"], check=True,
            capture_output=True,text=True).stdout.strip(),
        "current_supervisor_sha256":hashlib.sha256((ROOT/"src/wlr50_clean/ppo/semantic_supervisor.py").read_bytes()).hexdigest(),
        "current_task_spec_sha256":hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "mode_v1":RR_WORKSPACE_RETIREMENT_MODE, "mode_v2":RR_WORKSPACE_RETIREMENT_MODE_V2,
        "old41_source":str(old_path), "old41_endpoint_index_range":[1060,1100],
        "old41_input_tick_range":[8488,8808], "old41_rows":41, "old41_precapture_rows":30,
        "old41_precapture_retired_v1":sum(r["retired_v1"] for r in precapture),
        "old41_precapture_retired_v2":sum(r["retired_v2"] for r in precapture),
        "old41_precapture_false_to_true":26,
        "old41_precapture_phi_delta":stats([r["phi_delta_v2_minus_v1"] for r in precapture]),
        "old41_all_phi_delta":stats([r["phi_delta_v2_minus_v1"] for r in same41]),
        "old41_postcapture_phi_delta":stats([r["phi_delta_v2_minus_v1"] for r in same41 if r["historyPlaced"]]),
        "transition_8704_to_8712":{"before":a,"after":b,"gamma":.9985,"potential_weight":5,
            "dt_s":8/120,"full_potential_shaping_v1":shaping["v1"],
            "full_potential_shaping_v2":shaping["v2"],
            "shaping_delta_v2_minus_v1":shaping["v2"]-shaping["v1"],
            "event_and_time_costs_not_recomputed":True},
        "block08_source":str(block_path), "block08_rows_read_bounded":1024,
        "block08_actual_postcross_inputs":389, "block08_currentQ_true":389,
        "block08_phi_delta":stats([r["phi_delta_v2_minus_v1"] for r in block_rows]),
        "block08_learner_global_range":[block_rows[0]["learner_global_decision"],block_rows[-1]["learner_global_decision"]],
        "block08_selected_identity_digest":digest([(r["learner_global_decision"],
            r["source_endpoint_index"],r["input_tick"]) for r in block_rows]),
        "all430_input_ev_digest_chain":input_hash.hexdigest(),
        "all430_evaluator_objects_currentQ_and_history_unchanged":True,
        "old41_record_columns":["input_tick","currentQ","historyPlaced","retired_v1","retired_v2",
            "phi_v1","phi_v2","phi_delta_v2_minus_v1"],
        "old41_records":[[r[k] for k in ("input_tick","currentQ","historyPlaced","retired_v1","retired_v2",
            "phi_v1","phi_v2","phi_delta_v2_minus_v1")] for r in same41],
        "actor_evaluations":0,"optimizer_steps":0,"new_policy_decisions":0,
        "simulator_executed":False,"old_reports_modified":False,
        "limitations":["No evaluator replay: the exact already-recorded physical states are held fixed.",
            "The old41 v1 comparison is current-spec counterfactual, not the old run's original reward definition.",
            "Block08 v1 Phi matches its originally recorded input potential; every selected v2 Phi is exactly unchanged.",
            "No physical improvement or 50mm-hover repair is established by this calculation."]
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
