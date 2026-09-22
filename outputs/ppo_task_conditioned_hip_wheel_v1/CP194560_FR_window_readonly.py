"""One completed live FR prefix; reuse prior extracted references, never old raw."""
import hashlib
import json
from pathlib import Path
from video_fr_preview_readonly import extract

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1117298582122Z_g97ecd305afb5_26ae0caab77f4c1bab164b45635744d4"
REFERENCE = OUT / "CP192512_FR_window_readonly.json"
END = 1795
old = json.loads(REFERENCE.read_text(encoding="utf-8"))
candidate = extract(RUN / "source", END, with_kinematics=True)
assert candidate["FR_events"] == dict(active_lift=23, front_edge_crossed=1785, placed=1795)
started = json.loads((RUN / "run_manifest.started.json").read_text())
assert started["arguments"]["seed"] == 4001 and started["arguments"]["stochastic_policy"] is False
assert Path(started["arguments"]["checkpoint"]).name == "checkpoint_step_000194560.pt"


def prefix_receipt(path):
    digest = hashlib.sha256(); count = size = 0; last = None
    with path.open("rb") as stream:
        for line in stream:
            assert line.endswith(b"\n"), "required closed prefix incomplete"
            r = json.loads(line); last = r["physics_tick"]
            assert last <= END, "refuse reading beyond the closed window"
            digest.update(line); count += 1; size += len(line)
            if last == END: break
    assert last == END
    return dict(path=str(path), last_tick=END, complete_lines=count, prefix_bytes=size,
        prefix_sha256=digest.hexdigest(), scope="consumed complete prefix only; not sealed whole-file hash")


prefixes = {name: prefix_receipt(RUN / "source" / name) for name in
    ("physical_observations.jsonl", "stage_transition_evidence.jsonl")}
records = {key: old["records"][key] for key in ("current_B", "CP192512_det")}
records["CP194560_det_PPO_LIMITEDAUX"] = candidate
differences = {}
for name in ("current_B", "CP192512_det"):
    b = records[name]
    differences[name] = dict(duration_s=candidate["duration_s"]-b["duration_s"],
        rate_RMS_fraction=candidate["rate_RMS_rad_s"]/b["rate_RMS_rad_s"]-1,
        peak_tilt_fraction=candidate["peak_tilt_rad"]/b["peak_tilt_rad"]-1,
        body_minimum_z_difference_m=candidate["body_collider_minimum_world_z_m"]["minimum"]-b["body_collider_minimum_world_z_m"]["minimum"],
        body_mean_z_difference_m=candidate["body_collider_minimum_world_z_m"]["mean"]-b["body_collider_minimum_world_z_m"]["mean"],
        body_capture_z_difference_m=candidate["body_collider_minimum_world_z_m"]["last"]-b["body_collider_minimum_world_z_m"]["last"],
        FR_safe_air_mean_gap_difference_m=candidate["FR_gap_first_safe_before_cross_m"]["mean"]-b["FR_gap_first_safe_before_cross_m"]["mean"],
        actual_body_advance_difference_m=candidate["kinematics"]["body_forward_displacement_m"]-b["kinematics"]["body_forward_displacement_m"])
runtime = started["runtime_contract"]
old_run = Path(records["CP192512_det"]["source"]).parent
old_runtime = json.loads((old_run / "run_manifest.started.json").read_text())["runtime_contract"]
changed = sorted(k for k in set(runtime["files"]) | set(old_runtime["files"])
    if runtime["files"].get(k) != old_runtime["files"].get(k))
result = dict(schema="CP194560.provisional_complete_FR_event_window.v1", records=records,
    differences=differences, candidate_analyzed_through_tick=END,
    training_label="PPO+LIMITEDAUX", label_source="root-provided checkpoint provenance; no aux causal attribution",
    candidate_whole_episode_outcome=None, candidate_future_outcome_analyzed_or_claimed=False,
    active_source_bounded_prefix_receipts=prefixes,
    reused_reference_artifact=dict(path=str(REFERENCE), sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()),
    comparison_identity=dict(physics_seed=4001, mode="deterministic_conditional_mean",
        candidate_runtime_content_sha256=runtime["runtime_content_sha256"],
        CP192512_runtime_content_sha256=old_runtime["runtime_content_sha256"],
        changed_runtime_files_vs_CP192512=changed,
        current_B_comparability_caveat=old["comparison_identity"]),
    semantics=old["semantics"])
with (OUT / "CP194560_FR_window_readonly.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
for name, r in records.items():
    print(name, json.dumps(dict(duration=r["duration_s"], angles=r["kinematics"]["angle_RMS_rad"],
        rate=r["rate_RMS_rad_s"], peak=r["peak_tilt_rad"], body=r["body_collider_minimum_world_z_m"],
        gap=r["FR_gap_first_safe_before_cross_m"], advance=r["kinematics"]["body_forward_displacement_m"])))
print("DIFFERENCES", json.dumps(differences))
print("CHANGED_RUNTIME", changed)
print("PREFIXES", json.dumps(prefixes))
