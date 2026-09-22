"""Bounded live FR event window; no growing-file full hashes or future reads."""
import hashlib
import json
from pathlib import Path
from video_fr_preview_readonly import extract

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/"runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1006004011698Z_g97ecd305afb5_78f9ffd75d504f0e8447f44e61d134ce"
REFERENCE=OUT/"CP189952_FR_window_readonly.json"
old=json.loads(REFERENCE.read_text(encoding="utf-8"))
candidate=extract(RUN/"source",1738,with_kinematics=True)
assert candidate["FR_events"]==dict(active_lift=24,front_edge_crossed=1729,placed=1738)
started=json.loads((RUN/"run_manifest.started.json").read_text())
assert started["arguments"]["seed"]==4001 and started["arguments"]["stochastic_policy"] is False


def prefix_receipt(path,end):
    h=hashlib.sha256();n=0;size=0;last=None
    with path.open("rb") as stream:
        for line in stream:
            assert line.endswith(b"\n"),"required window has incomplete line"
            r=json.loads(line);tick=r["physics_tick"]
            assert tick<=end,"expected exact complete event endpoint"
            h.update(line);n+=1;size+=len(line);last=tick
            if tick==end:break
    assert last==end
    return dict(path=str(path),last_tick=end,complete_lines=n,prefix_bytes=size,
        prefix_sha256=h.hexdigest(),scope="only consumed complete prefix through tick1738; not sealed whole-file hash")


prefixes={name:prefix_receipt(RUN/"source"/name,1738) for name in ("physical_observations.jsonl","stage_transition_evidence.jsonl")}
records={"current_B":old["records"]["current_B"],"CP189952_det":old["records"]["CP189952_det"],"CP192512_det":candidate}
differences={}
for label in ("current_B","CP189952_det"):
    b=records[label]
    differences[label]=dict(duration_s=candidate["duration_s"]-b["duration_s"],
        rate_RMS_fraction=candidate["rate_RMS_rad_s"]/b["rate_RMS_rad_s"]-1,
        peak_tilt_fraction=candidate["peak_tilt_rad"]/b["peak_tilt_rad"]-1,
        body_minimum_z_difference_m=candidate["body_collider_minimum_world_z_m"]["minimum"]-b["body_collider_minimum_world_z_m"]["minimum"],
        body_mean_z_difference_m=candidate["body_collider_minimum_world_z_m"]["mean"]-b["body_collider_minimum_world_z_m"]["mean"],
        body_capture_z_difference_m=candidate["body_collider_minimum_world_z_m"]["last"]-b["body_collider_minimum_world_z_m"]["last"],
        FR_safe_air_mean_gap_difference_m=candidate["FR_gap_first_safe_before_cross_m"]["mean"]-b["FR_gap_first_safe_before_cross_m"]["mean"],
        actual_body_advance_difference_m=candidate["kinematics"]["body_forward_displacement_m"]-b["kinematics"]["body_forward_displacement_m"])
old_B_run=Path(records["current_B"]["source"]).parent
old_runtime=json.loads((old_B_run/"run_manifest.started.json").read_text())["runtime_contract"]
runtime=started["runtime_contract"]
diff_files=sorted(k for k in set(runtime["files"])|set(old_runtime["files"]) if runtime["files"].get(k)!=old_runtime["files"].get(k))
expected=["configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml","src/wlr50_clean/ppo/semantic_cli.py","src/wlr50_clean/ppo/semantic_migration.py","src/wlr50_clean/ppo/semantic_training.py"]
assert diff_files==expected
result=dict(schema="CP192512.provisional_complete_FR_event_window.v1",records=records,differences=differences,
    candidate_analyzed_through_tick=1738,candidate_whole_episode_outcome=None,
    final_video_not_sealed_for_this_analysis=True,candidate_future_outcome_analyzed_or_claimed=False,
    active_source_bounded_prefix_receipts=prefixes,
    reused_reference_artifact=dict(path=str(REFERENCE),sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()),
    comparison_identity=dict(physics_seed=4001,mode="deterministic_conditional_mean",
        candidate_runtime_content_sha256=runtime["runtime_content_sha256"],reference_runtime_content_sha256=old_runtime["runtime_content_sha256"],
        runtime_hash_equal=False,changed_runtime_files=diff_files,
        execution_profile_only_diff="training_budgets.full_episode 100000 ->131072, git-diff verified",
        unchanged_physical_files_and_action_observation_reward_stage_configs=True),
    semantics=dict(rate_RMS="sqrt(time-integral((wrapped adjacent120Hz roll_rate^2+pitch_rate^2)/2)/T); NOT RMS/T",
        angle_RMS="sqrt(trapezoidal time mean of measured Euler angle squared)",
        geometry="real body collider lowest world z; AABB conservative lower bound, not base z/exact mesh gap",
        causality="FR event-complete only, differing durations; slower motion and body/FR clearance changes reported, not causal attribution or full-episode victory",
        missing_mount="RR/RL hip mount heights unmeasured by this bounded extractor; not inferred from body/CoM"))
with (OUT/"CP192512_FR_window_readonly.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
for label,r in records.items():
    print(label,json.dumps(dict(duration=r["duration_s"],angles=r["kinematics"]["angle_RMS_rad"],rate=r["rate_RMS_rad_s"],peak=r["peak_tilt_rad"],
        body=r["body_collider_minimum_world_z_m"],gap=r["FR_gap_first_safe_before_cross_m"],advance=r["kinematics"]["body_forward_displacement_m"],
        rear=r["kinematics"]["rear_joints"])))
print("DIFFERENCES",json.dumps(differences))
print("PREFIXES",json.dumps(prefixes))
