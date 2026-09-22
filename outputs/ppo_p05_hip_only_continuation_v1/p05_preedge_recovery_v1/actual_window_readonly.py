"""Bounded actual-state predicate audit; no live objects, actor, optimizer or physics."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
name = "wlr50_clean.ppo.preedge_actual_window_review"
item = importlib.util.spec_from_file_location(name, HERE / "candidate/src/wlr50_clean/ppo/semantic_supervisor.py")
s = importlib.util.module_from_spec(item); sys.modules[name] = s; item.loader.exec_module(s)
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.reference.motion_contract import load_motion_contract

SOURCE = ROOT / "runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1521094182530Z_g336b7c56d2f0_3cdf838115e44829b66ad9585ad92ba7/source"
FIRST, LAST = 5992, 6480
# Include the next ACK context: it describes the current row's source observation.
with (SOURCE / "capture_assist_ticks.jsonl").open("rb") as stream:
    raw = list(itertools.islice(stream, FIRST-2, LAST+1))
assert all(line.endswith(b"\n") for line in raw)
rows = {row["episode_physics_tick"]: row for row in map(json.loads, raw)}
assert sorted(rows) == list(range(FIRST-1, LAST+2))
with (SOURCE / "video_policy_decisions.jsonl").open("rb") as stream:
    decisions = [json.loads(line) for line in itertools.islice(stream, FIRST//8-1, LAST//8)]
tasks = {d["end_tick"]: d["step_info"]["semantic_task"] for d in decisions}
assert sorted(tasks) == list(range(FIRST, LAST+1, 8))
entry = next(event for event in tasks[FIRST]["transition_evidence"] if event["to_stage"] == "P05")
assert entry["physics_tick"] == 2400 and entry["sim_time_s"] == 20.
entries = {"P01": 0, **{event["to_stage"]: event["physics_tick"]
    for event in tasks[FIRST]["transition_evidence"]}}
assert set(entries) == {"P01", "P02", "P03", "P04", "P05"}
spec = s.load_task_spec(HERE / "candidate/configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml")
contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
provider = s.NominalMotionProvider(contract, spec=spec, fsm_spec=load_fsm_spec(ROOT / "configs/fsm_states.yaml"))
layers = []
for phase, tick in entries.items():
    motion = MotionExecutor(physics_hz=120., servo_rate_limit_deg_s=contract.servo_rate_limit_deg_s,
                            initial_full12=contract.phase(phase).start_full12)
    provider._start_source_motion(motion, contract.phase(phase))
    # Pure reconstruction of an unheld source clock; no source poses/actions are
    # deployed and no measurements are generated or altered.
    motion._tick_index = FIRST-1-tick
    layers.append(dict(stage=phase, motion=motion, sample=motion.tick(),
                       ticks=FIRST-tick, advanced_this_tick=True))
provider._continuous_layers = layers
failure_counts = Counter()
truth_counts = Counter()
results, decision_crosschecks = [], 0
for tick in range(FIRST, LAST+1):
    row, next_row = rows[tick], rows[tick+1]
    context = next_row["dispatch"]["capture_assist_evidence"]["context"]
    assert row["phase"] == "P05" and context["source_observation_tick"] == tick
    assert context["physical_valid"] is True
    assert context["source_endpoint_issued"] is True and context["source_unfold_dispatched"] is True
    assert math.isclose(row["sim_time_s"], tick/120., rel_tol=0., abs_tol=1e-9)
    previous_sample = layers[-1]["sample"]
    for layer in layers:
        # Production explicitly advances P01..P05 after their endpoints.
        assert provider._sequence_permission(layer, tasks[FIRST], None) is True
        layer["sample"] = layer["motion"].tick()
        layer["ticks"] += 1
    provider.endpoint_issued = layers[-1]["sample"].endpoint_issued
    provider.elapsed_s = layers[-1]["sample"].elapsed_s
    value = deepcopy(tasks[FIRST])
    value.update(stage_elapsed_s=row["sim_time_s"]-entry["sim_time_s"], termination_reason=None)
    ev = value["physical_evaluator"]
    # Fields absent from the per-physics capture row come from its NEXT dispatch
    # context, whose source_observation_tick is checked above. The accepted
    # evidence-status enum is exactly _all_stage_finish(current.load_fraction_valid).
    ev.update(physics_tick=tick, simulation_time_s=row["sim_time_s"], valid=True,
        termination_reason=None, current_legs=row["current_legs"],
        physical_evidence_status="VERIFIED" if all(v["load_fraction_valid"] for v in row["current_legs"].values())
            else "CONTACT_BEARING_UNVERIFIED")
    ev["history"]["placed"] = row["placed_history"]
    ev["history"]["active_lift"]["FL"] = context["qualified_FL"]
    ev["history"]["front_edge_crossed"]["FL"] = context["crossed_FL"]
    reconstructed = provider._p05_preedge_recovery_status(value,
        prior_source_endpoint_issued=previous_sample.endpoint_issued, prior_observation_tick=tick-1)
    if tick in tasks:
        full_task = tasks[tick]
        assert full_task["physical_evaluator"]["current_legs"] == row["current_legs"]
        assert full_task["stage_elapsed_s"] == value["stage_elapsed_s"]
        assert full_task["nominal_provider_diagnostics"]["capture_continuation"]["source_observation_tick"] == tick
        direct = provider._p05_preedge_recovery_status(full_task,
            prior_source_endpoint_issued=previous_sample.endpoint_issued, prior_observation_tick=tick-1)
        assert direct == reconstructed
        decision_crosschecks += 1
    failure_counts.update(reconstructed["reasons"])
    truth_counts.update(key for key, passed in reconstructed["checks"].items() if passed)
    results.append(dict(tick=tick, time_s=row["sim_time_s"], age_s=value["stage_elapsed_s"],
        eligible=reconstructed["eligible"], reasons=reconstructed["reasons"],
        prior_source_endpoint=previous_sample.endpoint_issued, source_endpoint=provider.endpoint_issued,
        P05_source_tick=layers[-1]["sample"].tick_index, fresh_wheel_owners=reconstructed["fresh_source_wheel_owners"],
        gap_m=ev["current_legs"]["FL"]["clearance_m"],
        front_m=ev["current_legs"]["FL"]["front_distance_m"],
        other_supports=reconstructed["observed_other_support_contacts"]))
eligible = [row for row in results if row["eligible"]]
after = [row for row in results if row["tick"] >= 6000]
def bounds(key):
    values = [row[key] for row in after]
    return {"min": min(values), "max": max(values)}
report = dict(schema="wlr50_clean.p05_preedge_actual_window_readonly.v1",
    source=str(SOURCE), candidate_source_sha256=hashlib.sha256((HERE / "candidate/src/wlr50_clean/ppo/semantic_supervisor.py").read_bytes()).hexdigest(),
    capture_selected_bytes_sha256=hashlib.sha256(b"".join(raw)).hexdigest(),
    fixed_ticks=[FIRST, LAST], capture_including_neighbor_ticks=[FIRST-1, LAST+1],
    count=len(results), direct_full_task_crosschecks=decision_crosschecks,
    P05_entry=entry["physics_tick"], P05_entry_time_s=entry["sim_time_s"],
    first_eligible=eligible[0] if eligible else None, eligible_count=len(eligible),
    post_window_start_count=len(after), post_window_start_eligible=sum(row["eligible"] for row in after),
    failed_check_counts=dict(failure_counts), passed_check_counts=dict(truth_counts),
    observed_FL_gap_m=bounds("gap_m"), observed_FL_front_distance_m=bounds("front_m"),
    actual_source_endpoint_flags_true_all=True, actual_previous_dispatch_endpoint_verified_all=True,
    consecutive_actual_episode_and_context_ticks=True,
    source_reconstruction=dict(stage_entry_ticks=entries,
        semantics="pure source-clock calculation from actual phase entry ticks; P01..P05 _sequence_permission always true",
        atomic_groups="MotionExecutor emits only at exact source ticks, does not repeat endpoint groups",
        all_fresh_wheel_owner_counts_zero=all(not row["fresh_wheel_owners"] for row in results),
        P05_source_tick_range=[results[0]["P05_source_tick"], results[-1]["P05_source_tick"]],
        raw_advanced_this_tick_flag_persisted=False),
    snapshots=[row for row in results if row["tick"] in (5992,5999,6000,6001,6008,6240,6480)],
    actual_candidate_control_applied=False, physical_success_claimed=False,
    limits=["Offline predicate evaluation on actual logged states, not counterfactual physics.",
        "Per-physics task fields reconstructed only from same-source-tick capture/next-ACK context; checked exactly against 62 full saved task endpoints.",
        "Actual source layer mutable object/advanced_this_tick is not serialized; its behavior is proven by unchanged code and exact source clock reconstruction.",
        "Candidate source endpoint gate still does not independently read ACK; this audit additionally finds recorded capture-assist endpoint dispatch evidence true."])
print(json.dumps(report, indent=2, allow_nan=False))
