"""Prepared, not yet executed: bounded CPU-only parameter differences.

Run in a fresh process only when the root agent has stopped Isaac:
  python outputs/ppo_transfer_roles_v1/compare_role_checkpoint_parameters.py \
    --before <mapped_initial_372.pt> --after <later_372.pt> --output <new_report.json>

Only trusted local checkpoints: torch.load(weights_only=False) is the official
RSL checkpoint format. No actor construction/forward, optimizer step, checkpoint
write, RNG restoration, or simulator import. Output is exclusive JSON under this
experiment's outputs directory. No physical-output/variance/causal conclusions.

Naming checked against the installed RSL 5.0.1 PPO.save, MLPModel, and MLP:
actor_state_dict / critic_state_dict; hidden_dims=[256,256] gives mlp.0,2,4.
Actor output is [2,12], so mlp.4 rows 0:12 are base mean, rows 12:24 log sigma;
the subsequent Unflatten has no state. SemanticHistoryMLPModel adds no parameter.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

OUTPUT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_ROOT.parents[1]
PARAMETER_NAMES = tuple(f"mlp.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parameter_digest(state):
    # Exact semantic_training.parameter_hash order/formula. Here strict state
    # keys prove all entries are parameters, with no hidden normalizer buffers.
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def load_verified_parameters(path, torch):
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, policy_version_from_metadata
    from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_DIM

    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    manifest = checkpoint_metadata(path)  # Existing exact file SHA/sidecar/roundtrip checks.
    require(policy_version_from_metadata(manifest) == HISTORY_POLICY, "requires the history-conditioned policy")
    policy = manifest["policy_contract"]
    require(policy.get("observation_layout") == ROLE_OBSERVATION_LAYOUT
            and policy.get("observation_dimension") == ROLE_OBSERVATION_DIM == 372,
            "both checkpoints must have the explicitly bound role372 layout")
    config = manifest["runner_config"]
    require(all(config[role]["obs_normalization"] is False for role in ("actor", "critic")),
            "requires identity observation normalizers")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping), "checkpoint payload must be a mapping")
    infos = payload.get("infos")
    expected = {key: value for key, value in manifest.items()
                if key not in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip")}
    require(isinstance(infos, Mapping) and canonical(infos) == canonical(expected),
            "embedded metadata differs from verified manifest")
    for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps"):
        require(type(infos.get(key)) is int and infos[key] >= 0, f"invalid counter: {key}")
    require(payload.get("iter") == infos["ppo_updates"], "saved iteration differs from embedded PPO counter")
    models = {}
    for role, outputs in (("actor", 24), ("critic", 1)):
        state = payload.get(f"{role}_state_dict")
        require(isinstance(state, Mapping) and set(state) == set(PARAMETER_NAMES),
                f"unexpected {role} keys; refusing to guess parameter/head grouping")
        shapes = ((256, 372), (256,), (256, 256), (256,), (outputs, 256), (outputs,))
        for name, shape in zip(PARAMETER_NAMES, shapes, strict=True):
            tensor = state[name]
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                    and tensor.dtype == torch.float32 and tuple(tensor.shape) == shape
                    and bool(torch.isfinite(tensor).all()), f"unexpected/nonfinite {role}.{name}")
        require(parameter_digest(state) == infos[f"{role}_parameter_sha256"],
                f"{role} tensors do not match the saved parameter fingerprint")
        models[role] = dict(state)
    # Optimizer and RNG payloads are neither restored nor evaluated; releasing
    # the container retains only the small model tensors and JSON metadata.
    del payload
    current = path.stat()
    require((current.st_size, current.st_mtime_ns) == signature, "checkpoint changed during this read")
    return models, dict(infos), manifest["checkpoint_sha256"]


def differences(before, after, selections, torch):
    rows = []
    for name, selector, label in selections:
        left, right = before[name][selector], after[name][selector]
        delta = right.to(torch.float64) - left.to(torch.float64)
        squared = float(delta.square().sum())
        maximum = float(delta.abs().max())
        require(math.isfinite(squared) and math.isfinite(maximum), "nonfinite difference reduction")
        rows.append({"state_key": name, "selection": label, "shape": list(left.shape),
                     "changed": not torch.equal(left, right),
                     "before_nonzero_elements": int(torch.count_nonzero(left)),
                     "after_nonzero_elements": int(torch.count_nonzero(right)),
                     "changed_elements": int(torch.count_nonzero(delta)),
                     "parameter_elements": left.numel(), "difference_l2": math.sqrt(squared),
                     "difference_max_abs": maximum})
    return {"tensor_or_slice_count": len(rows), "changed_tensor_or_slice_count": sum(row["changed"] for row in rows),
            "changed_elements": sum(row["changed_elements"] for row in rows),
            "difference_l2": math.sqrt(sum(row["difference_l2"]**2 for row in rows)),
            "difference_max_abs": max(row["difference_max_abs"] for row in rows), "parameters": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    before_path, after_path = args.before.resolve(strict=True), args.after.resolve(strict=True)
    output = args.output.resolve()
    require(before_path != after_path, "provide two distinct checkpoint paths")
    require(all(path.is_file() and path.suffix == ".pt" for path in (before_path, after_path)), "inputs must be .pt files")
    require(output.is_relative_to(OUTPUT_ROOT) and output.suffix == ".json", "output must be experiment-local JSON")
    require(not output.exists(), "output already exists; refusing overwrite")
    for path in (before_path, after_path):
        require(path.with_name(path.stem + "_manifest.json").is_file(), f"missing manifest: {path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    old, old_info, old_sha = load_verified_parameters(before_path, torch)
    new, new_info, new_sha = load_verified_parameters(after_path, torch)
    for key in ("policy_contract", "runner_config", "runtime_contract", "seed", "normalizer_state_sha256"):
        require(canonical(old_info[key]) == canonical(new_info[key]), f"comparison changes bound {key}")
    require(old_info.get("stage") == "initial_v3_warm_start", "before must be the mapped pre-update initial checkpoint")
    evidence = old_info.get("observation_append_evidence", {})
    require(evidence.get("appended48_columns_zero_initialized") is True
            and evidence.get("original324_columns_and_all_other_tensors_preserved") is True,
            "initial checkpoint lacks the mapped append48 receipt")
    for role in ("actor", "critic"):
        require(int(torch.count_nonzero(old[role]["mlp.0.weight"][:, 324:])) == 0,
                f"initial {role} role-input columns are not zero")
    require(all(new_info[key] > old_info[key] for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")),
            "after must be a later updated checkpoint, not an initial or reversed comparison")

    all_keys = [(key, Ellipsis, "all") for key in PARAMETER_NAMES]
    shared = [(key, Ellipsis, "all") for key in PARAMETER_NAMES[:4]]
    mean = [(key, slice(0, 12), "rows[0:12]") for key in PARAMETER_NAMES[4:]]
    log_std = [(key, slice(12, 24), "rows[12:24]") for key in PARAMETER_NAMES[4:]]
    role_columns = [("mlp.0.weight", (slice(None), slice(324, 372)), "all_rows,columns[324:372]")]
    report = {
        "schema": "ppo_transfer_roles_v1.cpu_parameter_group_difference.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same-layout checkpoint parameter differences only; no inference or optimization",
        "inputs": [{"checkpoint": str(path), "checkpoint_sha256_verified": sha,
                    "stage": info["stage"], "global_policy_decisions": info["global_policy_decisions"],
                    "ppo_updates": info["ppo_updates"], "optimizer_steps": info["optimizer_steps"]}
                   for path, info, sha in ((before_path, old_info, old_sha), (after_path, new_info, new_sha))],
        "policy_contract": old_info["policy_contract"],
        "actor_all_unique_tensors": differences(old["actor"], new["actor"], all_keys, torch),
        "actor_shared_layers": differences(old["actor"], new["actor"], shared, torch),
        "actor_base_mean_head": differences(old["actor"], new["actor"], mean, torch),
        "actor_log_sigma_head": differences(old["actor"], new["actor"], log_std, torch),
        "critic_all_unique_tensors": differences(old["critic"], new["critic"], all_keys, torch),
        "appended_role_input_columns": {
            "columns": [324, 372], "column_interval": "half_open", "initial_zero_verified": True,
            "actor": differences(old["actor"], new["actor"], role_columns, torch),
            "critic": differences(old["critic"], new["critic"], role_columns, torch),
        },
        "limits": ["Head groups count tensor slices; mlp.4 weight/bias each appear in two disjoint head groups.",
                   "Appended-column statistics overlap first-layer/shared/all-parameter groups; do not sum them as independent tensors.",
                   "Nonzero appended weights demonstrate parameter updates, not beneficial use of role inputs or task contribution.",
                   "Shared hidden layers affect both outputs; head weight changes are not output/std/variance changes.",
                   "Mean is the pre-rho base head; no forward pass or history-conditioned output was calculated.",
                   "No task success, failure cause, PPO credit, simulator action, optimizer step or RNG restoration is implied."],
        "diagnostic_optimizer_steps": 0, "checkpoint_writes": 0, "actor_forward_calls": 0,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"written": str(output), "before_global": old_info["global_policy_decisions"],
                      "after_global": new_info["global_policy_decisions"], "diagnostic_optimizer_steps": 0}))


if __name__ == "__main__":
    main()
