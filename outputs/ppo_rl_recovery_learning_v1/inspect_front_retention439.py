"""Bounded, zero-update inspection of the latest student's retention candidate.

No physics, optimizer step, teaching deployment, or checkpoint publication.
All teaching pairs are the sealed student's executed pre-intervention requests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "staged_front_retention439"))

import data_contract as data
import front_retention439 as kernel

BRANCH = ROOT / "outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2"
P10_RUN = ROOT / "runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0731401970292Z_g72e63592bdf4_8abca665b0f64439bdd233a9d189e513"


def prepare_inputs(device):
    import torch
    binding = data.verify_reviewed_source_files()
    selected = data.plan_probe_rows(data.iter_jsonl(Path(binding["source_decisions"]["path"])), binding)
    observations, provenance = [], []
    seen = set()
    # These saved observations are real pre-action learner inputs, not inferred
    # from endpoint diagnostics. P13 was not reached; it is not manufactured.
    for path in sorted((P10_RUN / "rollouts").glob("rollout_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        rows = saved["observations"]["policy"].reshape(-1, 439)
        selected_indices = []
        for index, row in enumerate(rows):
            phase = int(row[:13].argmax())
            if phase in (9, 10, 11) and phase not in seen:
                observations.append(row.tolist())
                selected_indices.append({"flat_time_major_index": index, "phase": f"P{phase+1:02}"})
                seen.add(phase)
        if selected_indices:
            provenance.append({"path": str(path.resolve()), "sha256": data.file_sha256(path), "rows": selected_indices})
        if seen == {9, 10, 11}:
            break
    if seen != {9, 10, 11}:
        raise RuntimeError("real P10/P11/P12 invariance inputs missing")
    selected["invariance_provenance"] = provenance
    selected["invariance_actual_rows_by_phase"] = {"P10": 1, "P11": 1, "P12": 1, "P13": 0}
    train = kernel.tensors([r["observation"] for r in selected["train_rows"]], device=device)
    hold = kernel.tensors([r["observation"] for r in selected["holdout_rows"]], device=device)
    invariant = kernel.tensors(observations, device=device)
    return selected, train, hold, invariant


def make_runner(source):
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    from wlr50_clean.ppo.semantic_training import construct_semantic_runner
    from wlr50_clean.ppo.semantic_rear_owner_profile import REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT
    device = source["runner_config"]["device"]
    return construct_semantic_runner(_shape_env(439, device), seed=source["seed"], device=device,
        policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
        initialize_actor=False)[0]


def main():
    import torch
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, digest
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    pointer = json.loads((BRANCH / "checkpoints/checkpoint_last_pointer.json").read_text())
    checkpoint = Path(pointer["checkpoint"])
    if data.file_sha256(checkpoint) != pointer["checkpoint_sha256"] or data.file_sha256(Path(pointer["manifest"])) != pointer["manifest_sha256"]:
        raise RuntimeError("latest immutable checkpoint changed")
    source = checkpoint_metadata(checkpoint)
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3", experiment_id="rr_rl_timing_policy_learning_v1")
    if contract != source["runtime_contract"]:
        raise RuntimeError("read-only inspection requires source-compatible current runtime")
    runner = make_runner(source)
    infos = training.load_semantic_checkpoint(runner, checkpoint, contract=contract, seed=source["seed"])
    deps = kernel._numeric_dependencies()
    before = kernel._protected_state(runner, source["seed"], deps)
    selected, train, hold, invariant = prepare_inputs(str(runner.device))
    # Compare the pure numeric path against the actual official deterministic
    # actor, with no random draw. Official actor may refresh a distribution cache.
    with torch.inference_mode():
        exact = all(torch.equal(runner.alg.actor(batch, stochastic_output=False),
                                kernel.distribution(runner.alg.actor, batch, deps=deps)["mean"])
                    for batch in (train, hold, invariant))
    if not exact:
        raise RuntimeError("staged kernel differs from actual official deterministic actor")
    report = kernel.inspect(runner.alg.actor, train,
        [r["target_raw"] for r in selected["train_rows"]], hold,
        [r["target_raw"] for r in selected["holdout_rows"]], invariant, seed=source["seed"])
    after = kernel._protected_state(runner, source["seed"], deps)
    if before != after or training.parameter_hash(runner.alg.actor) != infos["actor_parameter_sha256"]:
        raise RuntimeError("read-only inspection changed learned/protected state")
    # Record per-phase/channel task-unit differences; these are current-model
    # responses to recorded observations, not a new closed-loop rollout.
    response = {}
    with torch.no_grad():
        for label, batch, rows in (("train", train, selected["train_rows"]), ("holdout", hold, selected["holdout_rows"])):
            dist = kernel.distribution(runner.alg.actor, batch, deps=deps)
            targets = torch.tensor([r["target_raw"] for r in rows], dtype=torch.float32, device=runner.device)
            delta = dist["caps"] * (targets.tanh() - dist["mean"].tanh())
            response[label] = {phase: {
                "mean_target_minus_current_REQUEST_full12": delta[[i for i,r in enumerate(rows) if r["phase"] == phase]].mean(0).cpu().tolist(),
                "max_abs_target_minus_current_REQUEST_full12": delta[[i for i,r in enumerate(rows) if r["phase"] == phase]].abs().amax(0).cpu().tolist()}
                for phase in data.TARGET_PHASES}
    report.update(source_checkpoint=pointer, current_head=args.expected_head,
        selected_data_receipt=selected["receipt"], invariance_provenance=selected["invariance_provenance"],
        invariance_actual_rows_by_phase=selected["invariance_actual_rows_by_phase"],
        official_deterministic_mean_bitwise_equal=exact, protected_state_before=before,
        protected_state_after=after, same_input_request_differences=response,
        physical_steps_added=0, optimizer_steps_added=0, actual_fit_executed=False)
    training.write_json(output, report)
    print(json.dumps({"report": str(output), "sha256": data.file_sha256(output),
        "train_half_MSE": report["initial_train_raw_half_MSE"],
        "holdout_half_MSE": report["initial_holdout_raw_half_MSE"],
        "gradient_l2": report["selected_gradient_l2"], "fit_executed": False}))


if __name__ == "__main__":
    main()
