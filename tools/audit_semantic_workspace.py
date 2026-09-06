"""Historical-state actuator workspace audit, not a simulation or causal probe.

Only frozen command limits and pure bounded feedback are imported. The mapper
is never advanced; Isaac, policy, evaluator and robot construction are absent.
Outputs are exclusive-create and cannot rewrite a historical run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_COMMAND_SIGN, WHEEL_FORWARD_SIGN,
    WHEEL_VELOCITY_LIMIT_RAD_S, SERVO_REFERENCE_VELOCITY_DEG_S, servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.ppo.action_projection import load_action_projection_config
from diagnose_semantic_continuity import SOURCES, rows

FOCUS = ("P07", "P08", "P09")
OLD_REVISION = "d19c655713bfec6571032650af0b8b2faaf66521"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def held_native_projection(*, previous_final, native, controller_bias,
                           previous_residual, request, residual_rate, ticks,
                           limits, servo, residual_bounds):
    """Independent channel envelope, frozen native/state inputs except slew.

    The 8-tick calculation holds native/control feedback fixed. It is a static
    command-space sensitivity, NOT the next 8 ticks of live controller/physics.
    """
    final, residual = previous_final, previous_residual
    for _ in range(ticks):
        residual += clamp(request-residual, -residual_rate/120., residual_rate/120.)
        # Same residual-space safety interval used by ActionProjector; reserve
        # is around source nominal, not the post-mapper target or measured q.
        residual = clamp(residual, *residual_bounds)
        if servo:
            final = bounded_drive_feedback_step(previous_deg=final, native_deg=native,
                bias_deg=controller_bias+residual,
                maximum_delta_deg=SERVO_REFERENCE_VELOCITY_DEG_S/120.,
                lower_deg=limits[0], upper_deg=limits[1])
        else:
            final = clamp(native+controller_bias+residual, *limits)
    return final


def stat(record, key, value):
    record.setdefault(key, []).append(float(value))


def summarize(values):
    return {"min": min(values), "max": max(values), "mean": sum(values)/len(values)}


def audit(csv_path, config_path):
    config_bytes = config_path.read_bytes()
    new_config = yaml.safe_load(config_bytes)
    old_text = subprocess.check_output(["git", "show", f"{OLD_REVISION}:configs/ppo_semantic_v2/execution_profile.yaml"], cwd=ROOT, text=True)
    old_config = yaml.safe_load(old_text)
    old_res, new_res = old_config["residual"], new_config["residual"]
    projector_limits = load_action_projection_config().safety_limits_full12
    assert new_config["physics_hz"] == old_config["physics_hz"] == 120.
    assert new_config["decision_hz"] == old_config["decision_hz"] == 15.
    caps = new_res["phase_caps_full12"]
    assert set(caps) == {f"P{i:02d}" for i in range(1,14)}
    assert all(len(v)==12 and all(float(x)>0 for x in v) for v in caps.values())
    historical = {}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["phase_source"] in FOCUS:
                historical[(item["role"], int(item["tick"]))] = item
    records = []
    for role, source in SOURCES.items():
        buckets, totals = {}, {}
        previous_residual, previous_phase = (0.,)*12, None
        for item in rows(source / "native_tick_audit.jsonl"):
            phase, tick = item["source_phase_id"], item["episode_physics_tick"]
            totals[phase] = totals.get(phase,0)+1
            native = item["native_audit"]
            current_residual = item["projected_residual_full12"]
            observed = historical.get((role,tick))
            if observed is not None:
                assert native["verified"] and native["actual_mapping_matches_dispatch"] and native["setter_dispatch_targets_equal"]
                assert phase == observed["phase_source"]
                for i, channel in enumerate(FULL12_ORDER):
                    state = buckets.setdefault((phase,channel),{})
                    servo = i<8
                    lower,upper = servo_limits_deg(channel) if servo else (-WHEEL_VELOCITY_LIMIT_RAD_S,WHEEL_VELOCITY_LIMIT_RAD_S)
                    nominal = item["nominal_full12"][i]
                    safety_lower,safety_upper = projector_limits[i]
                    residual_bounds = (min(0.,safety_lower-nominal),max(0.,safety_upper-nominal))
                    mapped = native["native_drive_target_full12"][i]
                    controller_bias = native["controller_drive_bias_full12"][i]
                    postmapper = mapped+controller_bias
                    drive = float(observed["drive_"+channel])
                    measured = float(observed[("q_"+channel+"_deg") if servo else ("dq_"+channel+"_rad_s")])
                    previous_final = native["previous_final_drive_servo_deg"][i] if servo else drive
                    old_cap = old_res["initial_servo_cap_deg" if servo else "initial_wheel_cap_rad_s"]
                    new_cap = float(caps[phase][i])
                    residual_rate = float(new_res["servo_rate_deg_s" if servo else "wheel_rate_rad_s2"])
                    raw = native["raw_policy_action_full12"][i]
                    for key,value in {
                        "nominal":nominal,"mapper_native":mapped,"postmapper_without_current_residual":postmapper,
                        "actual_drive":drive,"measured":measured,"tracking_error_abs":abs(drive-measured),
                        "physical_float32_target":float(observed["native_target_"+channel]),
                        "nominal_negative_hard_margin":nominal-lower,"nominal_positive_hard_margin":upper-nominal,
                        "nominal_negative_projector_reserve_margin":max(0.,nominal-safety_lower),
                        "nominal_positive_projector_reserve_margin":max(0.,safety_upper-nominal),
                        "postmapper_negative_hard_margin":postmapper-lower,"postmapper_positive_hard_margin":upper-postmapper,
                        "actual_negative_hard_margin":drive-lower,"actual_positive_hard_margin":upper-drive,
                        "abs_raw":abs(raw),"abs_projected_residual":abs(current_residual[i]),
                        "observed_old_cap_utilization":abs(current_residual[i])/old_cap,
                        "native_effect_fraction":float(native["changed_channels_full12"][i]),
                        "RR_front_m":float(observed["RR_front_m"]),"RR_clearance_m":float(observed["RR_clearance_m"]),
                        "FL_AIR_fraction":float(observed["FL_AIR"]=="True"),
                        "FL_actual_load_fraction":float(observed["FL_load_fraction"] or 0.),
                    }.items(): stat(state,key,value)
                    for label,cap in (("old",old_cap),("new",new_cap)):
                        negative = min(cap,max(0.,postmapper-lower)); positive = min(cap,max(0.,upper-postmapper))
                        stat(state,label+"_eventual_negative_cap_after_hard",negative)
                        stat(state,label+"_eventual_positive_cap_after_hard",positive)
                        stat(state,label+"_eventual_hard_retained_span_fraction",(negative+positive)/(2*cap))
                        reserve_negative = min(negative,-residual_bounds[0])
                        reserve_positive = min(positive,residual_bounds[1])
                        stat(state,label+"_negative_cap_projector_postmapper_hard_intersection",reserve_negative)
                        stat(state,label+"_positive_cap_projector_postmapper_hard_intersection",reserve_positive)
                        stat(state,label+"_intersection_retained_span_fraction",(reserve_negative+reserve_positive)/(2*cap))
                        for steps in (1,8):
                            endpoints = [held_native_projection(previous_final=previous_final,native=mapped,
                                controller_bias=controller_bias,previous_residual=previous_residual[i],
                                request=sign*cap,residual_rate=residual_rate,ticks=steps,
                                limits=(lower,upper),servo=servo,residual_bounds=residual_bounds) for sign in (-1.,1.)]
                            assert lower-1e-8 <= endpoints[0] <= endpoints[1]+1e-8 <= upper+1e-8
                            span = endpoints[1]-endpoints[0]
                            stat(state,f"{label}_held_native_{steps}tick_target_span",span)
                            stat(state,f"{label}_held_native_{steps}tick_span_over_requested",span/(2*cap))
                    if previous_phase==phase and native["policy_request_phase"]==phase:
                        request = math.tanh(raw)*old_cap
                        stat(state,"eligible_requested_projection_difference_abs",abs(request-current_residual[i]))
                        stat(state,"eligible_requested_projection_modified_fraction",float(abs(request-current_residual[i])>1e-8))
                    stat(state,"tick",tick)
            previous_residual,previous_phase = current_residual,phase
        for number in range(1,14):
            phase = f"P{number:02d}"
            for i,channel in enumerate(FULL12_ORDER):
                servo=i<8; lower,upper=servo_limits_deg(channel) if servo else (-WHEEL_VELOCITY_LIMIT_RAD_S,WHEEL_VELOCITY_LIMIT_RAD_S)
                observed=buckets.get((phase,channel)); rate=new_res["servo_rate_deg_s" if servo else "wheel_rate_rad_s2"]
                record={"role":role,"phase":phase,"channel":channel,"canonical_unit":"deg" if servo else "rad/s",
                    "physical_target_unit":"rad" if servo else "rad/s",
                    "physical_sign":(SERVO_COMMAND_SIGN if servo else WHEEL_FORWARD_SIGN)[channel],
                    "status":"observed_focus_window" if observed else ("outside_P07_P09_focus" if phase in totals else "phase_not_reached_no_live_data"),
                    "observed_source_phase_total_ticks":totals.get(phase),"focused_ticks":len(observed["tick"]) if observed else None,
                    "old_cap":old_res["initial_servo_cap_deg" if servo else "initial_wheel_cap_rad_s"],
                    "new_cap":caps[phase][i],"hard_lower":lower,"hard_upper":upper,
                    "projector_safety_lower":projector_limits[i][0],"projector_safety_upper":projector_limits[i][1],
                    "projector_reserve_deg":projector_limits[i][0]-lower if servo else None,
                    "residual_slew_per_120hz_tick":rate/120.,"residual_slew_per_15hz_decision":rate/15.,
                    "frozen_final_servo_slew_per_tick":SERVO_REFERENCE_VELOCITY_DEG_S/120. if servo else None,
                    "frozen_final_servo_slew_per_decision":SERVO_REFERENCE_VELOCITY_DEG_S/15. if servo else None,
                    "source_run":str(source),"new_config_sha256":hashlib.sha256(config_bytes).hexdigest(),
                    "envelope_semantics":"static_per_channel_fixed_native_and_controller_bias_no_physics_or_contact_prediction",
                    "measured_geometry_semantics":"observed_association_not_FK_or_causal_gain"}
                if observed:
                    for key,values in observed.items():
                        for suffix,value in summarize(values).items(): record[key+"_"+suffix]=value
                records.append(record)
    assert len(records)==312
    assert sum(r["status"]=="observed_focus_window" for r in records)==72
    return records,{"schema":"wlr50_clean.residual_workspace_audit.v1","old_revision":OLD_REVISION,
        "new_config_path":str(config_path),"new_config_sha256":hashlib.sha256(config_bytes).hexdigest(),
        "projector_definition_sha256":sha(ROOT/"configs/ppo_action_projection.yaml"),
        "source_continuity_csv_sha256":sha(csv_path),"rows":len(records),"observed_phase_channel_rows":72,
        "focus_phases":list(FOCUS),"uses_isaac":False,"uses_forward_kinematics":False,"additional_simulation_steps":0,
        "definitions":{
            "projection_ratios":"span remaining / full requested 2*cap; raw physical hard margin remains signed",
            "intersection":"Requested cap intersected with source nominal residual-space projector safety reserve and post-mapper absolute actuator headroom. Signed raw margins and hard-only spans are retained separately.",
            "held_native":"Frozen current native target and controller bias. Residual slew and original nominal-centered 2-degree safety reserve are applied algebraically; final servo step reuses bounded_drive_feedback_step. No mapper advance or robot write.",
            "120hz_15hz":"1 or 8 controller ticks at 120Hz; 15Hz values are available target movement budgets, not joint-body achievement.",
            "missing":"Outside focus or unreached phases have empty measured statistics, never manufactured zero.",
            "real_evidence":"Reported float32 native target ranges and geometry are actual historical data. New-cap envelopes are analytical, not live verified.",
            "forbidden_claim":"No geometric controllability or stable success claim is supported without the next real physics probe."}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/ppo_semantic_v3/execution_profile.yaml")
    parser.add_argument("--continuity-csv",type=Path,default=ROOT/"outputs/ppo_semantic_v3/reports/continuity_diagnosis.csv")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/ppo_semantic_v3/metrics")
    parser.add_argument("--update-derived-audit",action="store_true",help="Update only the hash-verified audit owned by this tool, never a source run")
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    csv_path=out/"residual_workspace_audit.csv"; manifest_path=out/"residual_workspace_audit_manifest.json"
    mode="x"
    if csv_path.exists() or manifest_path.exists():
        if not args.update_derived_audit or out.resolve()!=(ROOT/"outputs/ppo_semantic_v3/metrics").resolve():
            raise FileExistsError("use a fresh output directory; source evidence is immutable")
        old_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if (old_manifest.get("schema")!="wlr50_clean.residual_workspace_audit.v1"
                or old_manifest.get("csv_sha256")!=sha(csv_path)
                or old_manifest.get("source_continuity_csv_sha256")!=sha(args.continuity_csv)):
            raise ValueError("refuse to replace unowned or modified derived audit")
        mode="w"
    records,manifest=audit(args.continuity_csv,args.config)
    columns=list(dict.fromkeys(key for record in records for key in record))
    with csv_path.open(mode,encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(records)
    manifest["csv_sha256"]=sha(csv_path)
    with manifest_path.open(mode,encoding="utf-8") as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
    print(json.dumps({"csv":str(csv_path),"manifest":str(manifest_path),"rows":len(records),"live_probe_performed":False}))


if __name__=="__main__": main()
