"""Short measured residual-workspace responses, not training or success gates."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .semantic_cli import PROJECT_ROOT, runtime_contract, version_paths
from .semantic_training import jsonable, seed_training_rngs, sha256_file, write_json


def parser():
    result=argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--run-dir",type=Path,required=True)
    result.add_argument("--expected-head",required=True)
    result.add_argument("--seed",type=int,default=1001)
    result.add_argument("--from-phase",choices=tuple(f"P{i:02}" for i in range(6,14)),default="P06")
    result.add_argument("--teacher-offset-decisions",type=int,default=0)
    result.add_argument("--decisions",type=int,default=64)
    result.add_argument("--raw-magnitude",type=float,default=.5)
    return result


def response_actions(magnitude):
    if not math.isfinite(magnitude) or not 0 < magnitude <= 1:
        raise ValueError("workspace probe raw magnitude must be finite in (0,1]")
    # A declared diagnostic stimulus, not a learned action or success recipe.
    return (magnitude,-magnitude)*4+(magnitude,)*4


def _line(stream,record):
    stream.write(json.dumps(jsonable(record),allow_nan=False,separators=(",",":"))+"\n")


def run_segments(app,args,contract,*,backend=None):
    from .semantic_env import SemanticEpisodeEnv
    from .semantic_prefix import PrefixCreditCore,PrefixRequest,PrefixSemanticIsaacBackend,ZERO12
    from .semantic_legacy_evaluation import measured_observation
    from .semantic_training import verified_native_effect
    config=version_paths("v3")[2]
    old_config=version_paths("v2")[2]
    if backend is None:
        backend=PrefixSemanticIsaacBackend(app,prefix_request=PrefixRequest(
            args.from_phase,teacher_offset_decisions=args.teacher_offset_decisions),
            execution_profile=config/"execution_profile.yaml",task_spec_path=config/"stage_task_spec.yaml",
            audit_actuator_target_effect=True)
    stimulus=response_actions(args.raw_magnitude)
    results=[]
    for label,profile,raw in (("nominal_zero",config/"execution_profile.yaml",ZERO12),
                              ("old_range",old_config/"execution_profile.yaml",stimulus),
                              ("new_range",config/"execution_profile.yaml",stimulus)):
        segment=args.run_dir/label
        segment.mkdir(exist_ok=False)
        phase_counts=Counter()
        collect=False
        response_ticks=0
        effect_ticks=0
        peak_native_delta=[0.]*12
        with (segment/"prefix_evidence.jsonl").open("x",encoding="utf-8") as prefix_stream, \
             (segment/"native_response_120hz.jsonl").open("x",encoding="utf-8") as tick_stream, \
             (segment/"response_decisions.jsonl").open("x",encoding="utf-8") as decision_stream:
            def observe(before,after,projection):
                nonlocal response_ticks,effect_ticks
                if not collect:
                    return
                audit=after.info["actuator_target_effect_audit"]
                response_ticks+=1
                effect_ticks+=int(audit["changed_target_channel_count"]>0)
                actual=audit["actual_native_targets"]
                nominal=audit["counterfactual_native_targets"]
                actual12=tuple(actual["servo_position_rad"])+tuple(actual["wheel_velocity_rad_s"])
                nominal12=tuple(nominal["servo_position_rad"])+tuple(nominal["wheel_velocity_rad_s"])
                for index,(a,b) in enumerate(zip(actual12,nominal12,strict=True)):
                    peak_native_delta[index]=max(peak_native_delta[index],abs(a-b))
                _line(tick_stream,{"policy_optimizer_credit":False,"source_phase":before.state_id,
                    "actual_phase":after.state_id,"physics_tick":after.physics_tick,"sim_time_s":after.sim_time_s,
                    "nominal_full12":before.nominal_action_full12,"raw_stimulus_full12":raw,
                    "projected_residual_full12":projection.safe_projected_residual_full12,
                    "applied_full12":projection.applied_action_full12,"native_audit":audit,
                    "physical_observation":measured_observation(after.info["raw_observation"])})
            core=SemanticEpisodeEnv(backend,action_config=profile,
                reward_config_path=config/"reward_config.yaml",observation_schema_path=config/"observation_schema.json",
                collect_trace=False,tick_observer=observe)
            def sink(record):
                _line(prefix_stream,record)
                if record["kind"]!="reset_only_prefix_decision":
                    prefix_stream.flush()
            credit=PrefixCreditCore(core,evidence_sink=sink)
            credit.reset(seed=args.seed)
            initial=measured_observation(core.frame.info["raw_observation"])
            start_reference=dict(credit._start_reference)
            collect=True
            last_info=None
            for _ in range(args.decisions):
                step=credit.step(raw)
                last_info=step.info
                verified_native_effect(last_info,raw)
                if (last_info["actuator_target_effect_audit_summary"]["all_ticks_verified"] is not True
                        or last_info["no_in_episode_state_writes_verified"] is not True):
                    raise RuntimeError("workspace response lost native audit or no-state-write contract")
                phase_counts[last_info["phase_id"]]+=1
                _line(decision_stream,{"policy_optimizer_credit":False,**last_info})
                if step.terminated:
                    break  # Keep task failures; they are not interface failures.
            collect=False
            final=measured_observation(core.frame.info["raw_observation"])
        record={"segment":label,"seed":args.seed,"action_profile":str(profile),
            "action_profile_sha256":sha256_file(profile),"task_spec_sha256":sha256_file(config/"stage_task_spec.yaml"),
            "raw_stimulus_full12":raw,"start":start_reference,
            "requested_response_decisions":args.decisions,"actual_response_decisions":sum(phase_counts.values()),
            "response_physics_ticks":response_ticks,"phase_decisions":dict(phase_counts),
            "actual_native_effect_ticks":effect_ticks,"peak_absolute_native_delta_full12":peak_native_delta,
            "native_units":"first8 physical servo radians; last4 physical wheel rad/s",
            "initial_observation":initial,"final_observation":final,
            "termination_reason":last_info["termination_reason"],"task_outcome_label":last_info["task_outcome_label"],
            "prefix_decisions":credit.prefix_decisions,"prefix_physics_ticks":credit.prefix_ticks,
            "reset_wall_time_s":credit.reset_wall_time_s,"roll_in_wall_time_s":credit.roll_in_wall_time_s,
            "optimizer_updates":0,"optimizer_steps":0,"optimized_policy_decisions":0,
            "physical_response_is_not_a_success_or_improvement_claim":True}
        record["artifacts"]={name:{"path":str(segment/name),"sha256":sha256_file(segment/name)} for name in (
            "prefix_evidence.jsonl","native_response_120hz.jsonl","response_decisions.jsonl")}
        write_json(segment/"segment_manifest.json",record)
        results.append(record)
    return {"schema":"wlr50_clean.semantic_workspace_response.v1","runtime_contract":contract,
        "segments":results,"same_raw_old_new":True,"same_version_nominal_supervisor_all_segments":True,
        "independent_physical_resets":3,"optimizer_updates":0,"optimized_policy_decisions":0,
        "comparison_requires_inspecting_actual_initial_states":True,"automatic_training_gate":False}


def _require_no_parallel_simulator():
    if os.name!="nt":
        return
    query=("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne "+str(os.getpid())+
        " -and ($_.Name -match '^(kit|isaac-sim)(\\.exe)?$' -or "
        "($_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(isaacsim|isaaclab|omni\\.kit|wlr50_clean\\.ppo)')) } | Select-Object -ExpandProperty ProcessId")
    result=subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",query],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.stdout.strip() or result.stderr.strip():
        raise RuntimeError("another Python/Isaac task is active or process inspection failed")


def main(argv=None):
    args=parser().parse_args(argv)
    response_actions(args.raw_magnitude)
    if not 1<=args.decisions<=128 or not 0<=args.teacher_offset_decisions<1800:
        raise ValueError("probe needs 1-128 response decisions and a bounded teacher offset")
    root=(PROJECT_ROOT/"runs/ppo_semantic_v3/interface_checks").resolve()
    args.run_dir=args.run_dir.resolve()
    if args.run_dir==root or not args.run_dir.is_relative_to(root):
        raise ValueError("workspace probe must use a fresh v3 interface_checks child")
    _require_no_parallel_simulator()
    contract=runtime_contract(expected_head=args.expected_head,semantic_version="v3")
    args.run_dir.mkdir(parents=True,exist_ok=False)
    lifecycle={"schema":"wlr50_clean.semantic_workspace_run.v1","lifecycle":"RUNNING",
        "runtime_contract":contract,"arguments":jsonable(vars(args)),"optimizer_updates":0,
        "started_at_utc":datetime.now(timezone.utc).isoformat()}
    write_json(args.run_dir/"run_manifest.started.json",lifecycle)
    app=None
    try:
        import torch
        import tensordict
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=False).app
        app.update()
        seed_training_rngs(args.seed)
        result=run_segments(app,args,contract)
        if runtime_contract(expected_head=args.expected_head,semantic_version="v3")!=contract:
            raise RuntimeError("runtime changed during physical workspace response")
        write_json(args.run_dir/"workspace_probe_manifest.json",result)
        lifecycle.update(lifecycle="SUCCEEDED",result_manifest=str(args.run_dir/"workspace_probe_manifest.json"))
        return 0
    except BaseException as exc:
        lifecycle.update(lifecycle="FAILED",error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        lifecycle["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
        write_json(args.run_dir/"run_manifest.json",lifecycle)
        if app is not None:
            app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":
    raise SystemExit(main())
