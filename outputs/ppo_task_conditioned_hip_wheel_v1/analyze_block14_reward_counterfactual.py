"""Offline same-state reward analysis only: no actor, buffer, GAE, PPO or physics."""
from __future__ import annotations
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CANDIDATE = OUT / "candidate"
RUN = ROOT / "runs/ppo_fl_capture_quality_v1/train/20260918T1037290906944Z_ge73542cb57ad_befe799ac0b0461ca2c4f9189be96966"
sys.path.insert(0, str(ROOT / "src"))


def module(name):
    full = "wlr50_clean.ppo._counterfactual_" + name
    path = CANDIDATE / "src/wlr50_clean/ppo" / (name + ".py")
    spec = importlib.util.spec_from_file_location(full, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[full] = value
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                mean=sum(values)/len(values), sum=sum(values)) if values else dict(count=0)


def contact(row):
    return (row.get("top_contact") is True and row.get("top_surface_contact") is True
            and row.get("support") is True and row.get("bearing_verified") is True
            and row.get("air") is False and row.get("ground_contact") is False)


def main():
    supervisor, quality = module("semantic_supervisor"), module("semantic_task_quality")
    manifest = json.loads((RUN / "training_manifest.json").read_text())
    contract = manifest["runtime_contract"]
    lock_path = ROOT / "configs/environment_lock.json"
    for relative in ("configs/environment_lock.json", "src/wlr50_clean/sensing/geometry.py",
                     "src/wlr50_clean/sensing/sensor_reader.py"):
        assert sha(ROOT / relative) == contract["files"][relative]
    obstacle = json.loads(lock_path.read_text())["obstacle"]
    front, bottom = obstacle["front_face_x_m"], obstacle["bottom_z_m"]
    planes = (front, front + obstacle["length_m"], obstacle["center_y_m"] + obstacle["width_m"]/2,
              obstacle["center_y_m"] - obstacle["width_m"]/2, bottom, bottom + obstacle["height_m"])
    source_spec = ROOT / "configs/ppo_fl_capture_quality_v1/stage_task_spec.yaml"
    source_reward = ROOT / "configs/ppo_fl_capture_quality_v1/reward_config.yaml"
    for path in (source_spec, source_reward):
        assert sha(path) == contract["files"][path.relative_to(ROOT).as_posix()]
    old, new = (supervisor.TaskStageSupervisor.__new__(supervisor.TaskStageSupervisor) for _ in range(2))
    old.spec = supervisor.load_task_spec(source_spec)
    new.spec = supervisor.load_task_spec(CANDIDATE / "configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml")
    rows = [json.loads(line) for line in (RUN / "residual_and_projection_audit.jsonl").read_text().splitlines()]
    assert len(rows) == 512 and [r["global_policy_decision"] for r in rows] == list(range(185345, 185857))
    reward_values = yaml.safe_load(source_reward.read_text())
    target_values = yaml.safe_load((CANDIDATE / "configs/ppo_task_conditioned_hip_wheel_v1/reward_config.yaml").read_text())
    gamma, phi_weight = reward_values["gamma"], reward_values["potential_weight"]
    geometry_beta = target_values["family_weights"]["body_stability"] * target_values["task_space_quality"]["geometry_fraction"]
    assert gamma == target_values["gamma"] == manifest["runner_config"]["algorithm"]["gamma"] == .9985
    assert phi_weight == target_values["potential_weight"] == 5. and geometry_beta == .03
    states, geometry = [], defaultdict(list)
    for raw in rows:
        audit = raw["applied_audit"]
        task, reward = audit["semantic_task"], audit["reward_breakdown"]
        ev = task["physical_evaluator"]
        old_phi, new_phi = old.physical_potential(ev), new.physical_potential(ev)
        assert abs(old_phi - task["task_progress_potential"]) < 1e-12
        assert abs(old_phi - reward["potential_after"]) < 1e-12 or raw["terminal"]
        legs = {}
        for leg in ("FL", "FR"):
            c = ev["current_legs"][leg]
            legs[leg] = dict(current_contact_support=contact(c), air=c["air"], placed=ev["history"]["placed"][leg],
                clearance_m=c["clearance_m"], old_retention=old._current_capture_retention(leg, ev),
                new_retention=new._current_capture_retention(leg, ev))
        expected_delta = .85/4*.2*sum(v["new_retention"]-v["old_retention"] for v in legs.values() if v["placed"])
        assert abs(new_phi-old_phi-expected_delta) < 1e-12
        item = dict(decision=raw["global_policy_decision"], tick=audit["physics_tick"], time_s=audit["sim_time_s"],
            request_phase=audit["phase_id"], end_phase=task["stage_id"], terminal=raw["terminal"],
            termination_reason=audit["termination_reason"], old_phi=old_phi, new_phi=new_phi,
            phi_delta=new_phi-old_phi, legs=legs,
            nearest_rear_front_distance_m=max(ev["current_legs"][p]["front_distance_m"] for p in ("RL", "RR")),
            stored_reward=reward)
        states.append(item)
        sample = quality.task_space_quality_sample(task,
            dict(sim_time_s=audit["sim_time_s"], obstacle_planes_world_m=planes), quality.SPACE_CONFIG)
        sample.update(decision=item["decision"], tick=item["tick"], request_phase=item["request_phase"],
                      known_single_physics_tick_dt_s=1/120,
                      endpoint_single_tick_cost=(geometry_beta*sample["raw_geometry_cost"]/120 if sample["valid"] else None))
        geometry[task["stage_id"]].append(sample)
    pairs, missing, loss, regain = [], [], [], []
    for before, after in zip(states, states[1:]):
        reward = after["stored_reward"]
        if (before["terminal"] or after["tick"]-before["tick"] != 8
                or abs(after["time_s"]-before["time_s"]-reward["elapsed_physics_s"]) > 1e-10):
            missing.append(after["decision"])
            continue
        assert abs(before["old_phi"] - reward["potential_before"]) < 1e-12
        terminal = after["terminal"]
        old_next, new_next = (0., 0.) if terminal else (after["old_phi"], after["new_phi"])
        old_shaping = phi_weight*(gamma*old_next-before["old_phi"])
        new_shaping = phi_weight*(gamma*new_next-before["new_phi"])
        assert abs(old_shaping-reward["potential_shaping"]) < 1e-12
        p = dict(from_decision=before["decision"], to_decision=after["decision"],
            from_tick=before["tick"], to_tick=after["tick"], dt_s=reward["elapsed_physics_s"],
            request_phase=after["request_phase"], terminal=terminal,
            old_phi_before=before["old_phi"], old_phi_after=old_next,
            new_phi_before=before["new_phi"], new_phi_after=new_next,
            old_full_shaping=old_shaping, new_full_shaping=new_shaping,
            full_shaping_delta=new_shaping-old_shaping,
            FL_before=before["legs"]["FL"], FL_after=after["legs"]["FL"],
            FR_before=before["legs"]["FR"], FR_after=after["legs"]["FR"],
            nearest_rear_before_m=before["nearest_rear_front_distance_m"],
            nearest_rear_after_m=after["nearest_rear_front_distance_m"])
        fl_delta = lambda state: (state["legs"]["FL"]["new_retention"]-state["legs"]["FL"]["old_retention"]
                                 if state["legs"]["FL"]["placed"] else 0.)
        p["FL_capture_share_shaping_delta_decomposition"] = phi_weight*.85/4*.2*(
            (0 if terminal else gamma*fl_delta(after))-fl_delta(before))
        pairs.append(p)
        if p["request_phase"] == "P06":
            a, b = p["FL_before"]["current_contact_support"], p["FL_after"]["current_contact_support"]
            if a and not b: loss.append(p)
            if not a and b: regain.append(p)
    assert not missing and not any(s["terminal"] for s in states)
    by_phase = {}
    for phase in sorted({p["request_phase"] for p in pairs}):
        group = [p for p in pairs if p["request_phase"] == phase]
        by_phase[phase] = dict(pairs=len(group), delta=stats([p["full_shaping_delta"] for p in group]),
                              old=stats([p["old_full_shaping"] for p in group]),
                              new=stats([p["new_full_shaping"] for p in group]))
    geo_summary = {}
    for phase, samples in geometry.items():
        valid = [s for s in samples if s["valid"]]
        geo_summary[phase] = dict(endpoint_samples=len(samples), valid_samples=len(valid),
            valid_distance_m=stats([s["separation_lower_bound_m"] for s in valid]),
            raw_cost=stats([s["raw_geometry_cost"] for s in valid]),
            known_endpoint_single_tick_cost_sum=sum(s["endpoint_single_tick_cost"] for s in valid),
            full_120Hz_integral=None, full_integral_missing_reason="seven intermediate body AABBs per decision not logged",
            minimum_distance_example=min(valid, key=lambda s:s["separation_lower_bound_m"]) if valid else None)
    p06 = [p for p in pairs if p["request_phase"] == "P06"]
    report = dict(schema="wlr50_clean.counterfactual_reward_analysis.v1",
        classification="offline same-real-state counterfactual reward analysis; not rollout, GAE, intervention or optimizer replay",
        source_run=str(RUN), source_audit_sha256=sha(RUN / "residual_and_projection_audit.jsonl"),
        source_contract_head=contract["source_git_commit"], source_states=len(states), adjacent_pairs=len(pairs),
        complete_old_global_phi_reproduced_states=len(states), complete_old_shaping_reproduced_pairs=len(pairs),
        gamma=gamma, potential_weight=phi_weight, gamma_frequency="once per policy decision, not each physics tick",
        policy_dt_s=8/120, physics_dt_s=1/120, geometry_beta_per_s=geometry_beta, original_terminal_count=0,
        source_missing_first_predecision_snapshot=True,
        new_first_transition_reward=None, no_GAE_or_value_or_policy_change_computed=True,
        obstacle_planes_world_m=planes, obstacle_source="same-run SHA-verified frozen environment_lock + geometry + sensor_reader",
        candidate_sources={str(p.relative_to(ROOT)):sha(p) for p in (
            CANDIDATE / "src/wlr50_clean/ppo/semantic_supervisor.py",
            CANDIDATE / "src/wlr50_clean/ppo/semantic_task_quality.py",
            CANDIDATE / "configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml",
            CANDIDATE / "configs/ppo_task_conditioned_hip_wheel_v1/reward_config.yaml")},
        by_request_phase=by_phase,
        p06_FL_contact_loss=dict(count=len(loss), full_shaping_delta=stats([p["full_shaping_delta"] for p in loss]),
                                transitions=loss),
        p06_FL_contact_regain=dict(count=len(regain), full_shaping_delta=stats([p["full_shaping_delta"] for p in regain]),
                                  first=regain[0] if regain else None),
        p06_first_last=[p06[0],p06[-1]],
        endpoint_geometry_by_actual_state_phase=geo_summary,
        limits=["Captured contact loss occurs between 15Hz endpoints; exact first 120Hz loss tick is not reconstructed.",
                "AABB separation is a conservative geometric lower bound, not exact mesh separation.",
                "Endpoint zero cost does not prove zero unlogged within-interval integral.",
                "Only same-state reward sensitivity; changed policy trajectories and advantages remain unknown."])
    destination = OUT / "block14_reward_counterfactual.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report":str(destination), "pairs":len(pairs), "loss_count":len(loss),
        "loss_delta":report["p06_FL_contact_loss"]["full_shaping_delta"], "p06":by_phase["P06"],
        "geometry":{k:{"n":v["endpoint_samples"],"min":v["valid_distance_m"].get("minimum"),
                        "cost":v["known_endpoint_single_tick_cost_sum"]} for k,v in geo_summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
