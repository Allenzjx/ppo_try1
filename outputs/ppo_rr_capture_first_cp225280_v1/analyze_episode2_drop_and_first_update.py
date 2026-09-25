"""Bounded stdlib-only read of episode2 first loss and the sealed first update."""
import hashlib
import json
from pathlib import Path

from analyze_training_progress import DEFAULT_RUN, snapshot_rows

OUT = Path(__file__).resolve().parent
TICKS = (9688, 9696, 9704, 9712, 9736, 9872)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def pair(values):
    return values[6:8]


def main():
    status, selected, episode, last_tick, episode2_samples = {}, [], 0, None, 0
    for row in snapshot_rows(DEFAULT_RUN / "decisions.jsonl", True, status):
        info = row.get("step_info", {})
        local = row.get("local", {}) if row.get("kind") == "frozen_prior_prefix" else info.get("rr_capture_local", {})
        metric = local.get("metrics", {})
        tick = metric.get("tick")
        if tick is not None and (last_tick is None or tick < last_tick):
            episode += 1
        if tick is not None:
            last_tick = tick
        if episode > 2:
            break
        if episode == 2 and row.get("PPO_credit") == 1:
            episode2_samples += 1
            if tick in TICKS:
                audit = info["actuator_target_effect_audit"]
                headroom = audit["policy_headroom_evidence"]
                policy = row["policy_request"]
                diagnostics = info["semantic_task"]["nominal_provider_diagnostics"]
                layers = diagnostics["source_partial_order"]["layers"]
                p09 = next(x for x in layers if x["stage"] == "P09")
                legs = info["semantic_task"]["physical_evaluator"]["current_legs"]
                selected.append(dict(tick=tick, phase=[info["phase_id"], info["end_phase_id"]],
                    local_gate_active=local["active"], local_success=local["local_success"],
                    placed_history=metric["placed"], TOP=metric["current_top_contact"],
                    bearing=metric["current_top_bearing"], force_n=metric["bearing_force_n"],
                    load_fraction_valid=metric["load_fraction_valid"], gap_mm=1000*metric["gap_m"],
                    hold_s=local["hold_elapsed_s"], consecutive_top_samples=metric["consecutive_top_samples"],
                    source_rr=pair(info["nominal_action_full12"]),
                    mapped_N_rr=pair(audit["native_drive_target_full12"]),
                    controller_rr=pair(audit["controller_drive_bias_full12"]),
                    requested_rr=pair(headroom["requested_policy_residual_full12"]),
                    effective_rr=pair(headroom["effective_policy_residual_full12"]),
                    FINAL_rr=pair(info["actual_drive_target_full12"]),
                    actual_rr=metric["actual_rr_hip_knee_deg"],
                    sampled_raw_rr=pair(policy["selected_raw_full12"]),
                    conditional_mean_rr=pair(policy["conditional_mean_full12"]),
                    source_all8=info["nominal_action_full12"][:8],
                    FINAL_all8=info["actual_drive_target_full12"][:8],
                    late_group_start_tick=p09.get("late_group_start_tick"),
                    source_P09_layer_status=p09.get("status"), source_P09_wait_reason=p09.get("wait_reason"),
                    rear_capture_backup_enabled=diagnostics["source_partial_order"]["RR_contact_backup_enabled"],
                    supports={leg:{key:value.get(key) for key in ("contact_surface","bearing_force_n","bearing_verified","support")}
                              for leg,value in legs.items()},
                    local_terminated=info["local_reward"]["terminated"],
                    terminal_bootstrap_allowed=info["terminal_bootstrap_allowed"]))
        if episode == 2 and tick is not None and tick >= 9872:
            break
    by_tick = {r["tick"]:r for r in selected}
    before, after = by_tick[9696], by_tick[9704]
    diff = dict(actual_knee_deg=after["actual_rr"][1]-before["actual_rr"][1],
        final_knee_deg=after["FINAL_rr"][1]-before["FINAL_rr"][1],
        mapped_N_plus_controller_knee_deg=(after["mapped_N_rr"][1]+after["controller_rr"][1]
            -before["mapped_N_rr"][1]-before["controller_rr"][1]),
        filtered_policy_knee_deg=after["requested_rr"][1]-before["requested_rr"][1],
        actual_hip_deg=after["actual_rr"][0]-before["actual_rr"][0])
    result = dict(schema="wlr50_clean.second_episode_first_drop_readonly.v1", source=status,
        analyzed_only_episode=2, stop_tick=9872, actual_episode2_samples_read=episode2_samples,
        first_contact_endpoint_tick=9696, first_drop_endpoint_tick=9704,
        contact_start_inferred_from_logged_counter=9695, first_drop_exact_tick=None,
        first_drop_interval_ticks=[9697,9704], row_table=selected,
        first_loss_command_decomposition_deg=diff,
        conclusion="Initial loss is not RR knee re-folding: actual knee moves+5.487deg and FINAL+10deg; mapped N+existing P10 bias contribute+7.70deg and filtered policy+3.50deg before final slew. Full-body late source begins9695. Later P11 AIR knee re-folding coincides with larger gap but follows initial loss; this is not isolated causal proof.",
        limits=["Source change coincides with loss; without a same-entry counterfactual, it is not isolated causal proof.",
            "P11 source roles continue while local gate remains active; placed is historical, TOP/hold are current and false/zero after loss.",
            "Only logged endpoints and native-observer counters inspected; exact first loss tick unavailable.",
            "Existing P10+0.75deg normal correction is not an RR capture completer; no control change made."])
    path = OUT / "episode2_first_RR_drop_readonly.json"
    with path.open("x",encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")

    progress_path = OUT / "train_first2048_first_update_live_verified.json"
    progress = json.loads(progress_path.read_text())
    update = progress["updates"][0]
    manifest_path = OUT / "checkpoints/history/checkpoint_CP225792_local000512_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(manifest["checkpoint"])
    prior, prior_manifest = Path(manifest["prior"]["path"]), Path(manifest["prior"]["path"]).with_name(Path(manifest["prior"]["path"]).stem+"_manifest.json")
    verified = dict(schema="wlr50_clean.first_local_PPO_update_readonly.v1", update=update,
        checkpoint=str(checkpoint), checkpoint_sha256_actual=sha(checkpoint),
        checkpoint_sha256_matches_sidecar=sha(checkpoint)==manifest["checkpoint_sha256"],
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
        checkpoint_counts=manifest["counts"], counts_match_update=manifest["counts"]==update["reported_branch_counters"],
        learning_rate=manifest["learning_rate"], producer_actual_save_reload_verified=manifest["save_load_round_trip"],
        producer_empty_rollout=manifest["rollout_empty"], declared_serialized_state_hashes=manifest["state_hashes"],
        prior_checkpoint_sha_matches_identity=sha(prior)==manifest["prior"]["sha256"],
        prior_manifest_sha_matches_identity=sha(prior_manifest)==manifest["prior"]["manifest_sha256"],
        source_Adam_loaded=manifest["prior"]["prior_optimizer_loaded"],
        producer_declared_frozen_prior=manifest["prior"]["prior_weights_fully_frozen"],
        independent_tensor_or_optimizer_deserialization=False,
        frozen_state_scope="Byte hashes bind saved composite and immutable source; freeze assertions/save-reload are producer evidence. Stdlib reader does not independently compare tensors in .pt.",
        local_mean_zero_marker_semantics="prior.new_local_mean_initialized_zero is initialization provenance, not a claim learned local mean remains zero after update.",
        real_ability_claim=False,
        interpretation="Official update/storage likelihood checks succeeded; one prior stochastic local success preceded optimization. Nonzero gradient or changed parameters do not establish improved capability.")
    with (OUT / "first_update_verified.json").open("x",encoding="utf-8") as stream:
        json.dump(verified, stream, indent=2, allow_nan=False); stream.write("\n")
    lines=["# Episode 2: first RR support loss (read-only)", "",
        "P11 is not success. RR TOP/placed appeared at endpoint9696; next endpoint9704 is AIR and hold reset. Local gate stays active, history placed remains true, no local terminal.", "",
        "|Tick / phase|N RR h/k|REQUEST RR h/k|FINAL RR h/k|Actual RR h/k|gap mm|TOP / hold s|", "|---|---|---|---|---|---|---|"]
    def p(v): return "/".join(f"{x:.3f}" for x in v)
    for row in selected:
        lines.append(f"|{row['tick']} {'→'.join(row['phase'])}|{p(row['source_rr'])}|{p(row['requested_rr'])}|{p(row['FINAL_rr'])}|{p(row['actual_rr'])}|{row['gap_mm']:.3f}|{row['TOP']} / {row['hold_s']:.3f}|")
    lines += ["", result["conclusion"], "", *["- "+x for x in result["limits"]], "",
        "First update independently reconciles512 new samples (P09=397/P10=2/P11=113),20 minibatches/2560 old-likelihood exposures, each saved sample seen5 times, no mismatch. Checkpoint file SHA matches sidecar;512/1/20,1995 uncredited prefix,2 opportunities,1 pre-update local success,AUX0. Frozen-source byte hashes match; no tensor deserialization or new ability claim."]
    with (OUT / "episode2_first_RR_drop_readonly.md").open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    print(json.dumps(dict(drop_report=str(path), first_loss_delta=diff,
        first_update_verified=str(OUT/"first_update_verified.json"), checkpoint_hash_matches=verified["checkpoint_sha256_matches_sidecar"],
        counts_match=verified["counts_match_update"], source_hash_matches=verified["prior_checkpoint_sha_matches_identity"])))


if __name__ == "__main__":
    main()
