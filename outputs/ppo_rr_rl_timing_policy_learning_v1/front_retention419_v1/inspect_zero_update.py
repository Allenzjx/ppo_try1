"""Bounded read-only CP221568 front-retention inspection; never fit or save.

The same-runtime CP220544 actor is queried on sealed CP221568 learner
observations.  Its conditional means are offline references, not executed
actions, demonstrations, or physical-success labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HEAD = "44219b4fdc4d36d33be489b833c03b897766045b"
TEACHER = ROOT / ("outputs/ppo_rr_rl_timing_policy_learning_v1/checkpoints/history/"
                  "checkpoint_rear_recapture_CP220544_g44219b4fdc4d.pt")
TEACHER_SHA = "7bd9db99a70ab4a9bdf6c251c48f638d8dd21bdc9a854766c8b0ebbc7bb75382"
TEACHER_MANIFEST_SHA = "85fed85652e1a45738785bf23316ce4130ad0012aaa6b4a52b442a989ce83e5a"
STUDENT = ROOT / ("outputs/ppo_rr_rl_timing_policy_learning_v1/branches/"
                  "ancestor220544_recapture_v2/checkpoints/history/"
                  "checkpoint_step_000221568.pt")
STUDENT_SHA = "1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5"
STUDENT_MANIFEST_SHA = "b27d373edb11ed5d9455bc3d6b32dda4657c9b08b08f91d9d42de0d425686b75"
NATURAL_RUN = ROOT / ("runs/ppo_rr_rl_timing_policy_learning_v1/train/"
    "20260923T2129444443920Z_g44219b4fdc4d_b6fa467c3d27489dbb54d4adb117c1a2")
P07_RUN = ROOT / ("runs/ppo_rr_rl_timing_policy_learning_v1/train/"
    "20260923T2051315554325Z_g44219b4fdc4d_44bd3200f2bc41509ecdda378efbc8f5")
P10_RUN = ROOT / ("runs/ppo_rr_rl_timing_policy_learning_v1/train/"
    "20260923T2112002124392Z_g44219b4fdc4d_719169cdf58947d3bd595c9e8efd9674")
OUTPUT = HERE / "front_retention419_zero_update_inspection.json"
REPORT = HERE / "front_retention419_zero_update_inspection.md"
WHEELS = ("FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def sidecar(path):
    return path.with_name(path.stem + "_manifest.json")


def load_kernel():
    path = HERE / "front_retention419.py"
    spec = importlib.util.spec_from_file_location("_front_retention419_bound", path)
    require(spec is not None and spec.loader is not None, "kernel import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parameter_hash(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_copy(kernel, checkpoint):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    obs = TensorDict({"policy": torch.zeros(1, 419),
                      "critic": torch.zeros(1, 419)}, batch_size=[1])
    actor = kernel.SemanticRearPolicyTimingHistoryMLPModel(
        obs, {"actor":["policy"], "critic":["critic"]}, "actor", 12,
        hidden_dims=(256, 256), activation="elu", obs_normalization=False,
        distribution_cfg={"class_name":"HeteroscedasticGaussianDistribution",
                          "init_std":.15, "std_type":"log"},
        observation_layout="role410_rear_policy_timing_v1",
        exploration_std_temperature=.25)
    actor.load_state_dict(bundle["actor_state_dict"], strict=True)
    actor.eval()
    return actor


def rollout(path, expected_policy, expected_runtime):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    require(bundle.get("schema") == "wlr50_clean.semantic_on_policy_rollout.v1"
            and bundle.get("policy_contract") == expected_policy
            and bundle.get("runtime_contract") == expected_runtime,
            f"rollout contract differs: {path}")
    value = bundle["observations"]["policy"].flatten(0, 1).detach().cpu().clone()
    require(tuple(value.shape) == (128, 419) and bool(torch.isfinite(value).all()),
            f"rollout observation shape/value differs: {path}")
    phases = value[:, :13]
    require(bool((((phases == 0) | (phases == 1)).all())
                 and (phases.sum(-1) == 1).all()), "phase one-hot changed")
    return value


def td(value, indices):
    selected = value[indices]
    return TensorDict({"policy":selected, "critic":selected.clone()},
                      batch_size=[len(selected)])


def spaced(rows, count):
    require(len(rows) >= count > 0, "not enough source rows")
    if count == 1:
        return [rows[0]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))]
            for index in range(count)]


def ids(update, base, indices):
    return [f"rollout_{update:06d}:flat_{index:03d}:global_{base+index+1}"
            for index in indices]


def summary(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return {"signed_mean": tensor.mean(0).tolist(),
            "absolute_mean": tensor.abs().mean(0).tolist(),
            "signed_minimum": tensor.amin(0).tolist(),
            "signed_maximum": tensor.amax(0).tolist(),
            "maximum_absolute": tensor.abs().amax(0).tolist()}


def main():
    require(not OUTPUT.exists() and not REPORT.exists(), "inspection output already exists")
    require(sha(TEACHER) == TEACHER_SHA and sha(sidecar(TEACHER)) == TEACHER_MANIFEST_SHA,
            "teacher checkpoint binding differs")
    require(sha(STUDENT) == STUDENT_SHA and sha(sidecar(STUDENT)) == STUDENT_MANIFEST_SHA,
            "student checkpoint binding differs")
    teacher_meta, student_meta = read(sidecar(TEACHER)), read(sidecar(STUDENT))
    runtime, policy = student_meta["runtime_contract"], student_meta["policy_contract"]
    require(runtime["source_git_commit"] == HEAD
            and teacher_meta["runtime_contract"] == runtime
            and teacher_meta["policy_contract"] == policy
            and student_meta["save_load_round_trip"] is True
            and teacher_meta["save_load_round_trip"] is True,
            "teacher/student are not exact same-runtime round trips")
    for run in (NATURAL_RUN, P07_RUN, P10_RUN):
        manifest = read(run / "run_manifest.json")
        require(manifest.get("lifecycle") == "SUCCEEDED"
                and manifest.get("runtime_contract") == runtime,
                f"run is not sealed under the bound runtime: {run}")

    paths = {
        1689: P07_RUN / "rollouts/rollout_001689.pt",
        1690: P07_RUN / "rollouts/rollout_001690.pt",
        1691: P07_RUN / "rollouts/rollout_001691.pt",
        1692: P07_RUN / "rollouts/rollout_001692.pt",
        1693: P10_RUN / "rollouts/rollout_001693.pt",
        1694: P10_RUN / "rollouts/rollout_001694.pt",
        1695: NATURAL_RUN / "rollouts/rollout_001695.pt",
        1696: NATURAL_RUN / "rollouts/rollout_001696.pt",
    }
    rows = {update: rollout(path, policy, runtime) for update, path in paths.items()}
    phases = {update:(value[:, :13].argmax(-1) + 1).tolist()
              for update, value in rows.items()}
    require(Counter(phases[1695]) == {1:2, 2:126}
            and Counter(phases[1696]) == {2:128},
            "natural-P01 source coverage differs")
    train_index = [0] + spaced([i for i,p in enumerate(phases[1695]) if p == 2], 32)
    validation_index = spaced([i for i,p in enumerate(phases[1696]) if p == 2], 32)
    train, validation = td(rows[1695], train_index), td(rows[1696], validation_index)

    phase_sources = {7:[], 9:[], 12:[]}
    base = {1689:220544, 1690:220672, 1691:220800, 1692:220928,
            1693:221056, 1694:221184}
    for update in range(1689, 1695):
        for index, phase in enumerate(phases[update]):
            if phase in phase_sources:
                phase_sources[phase].append((update, index))
    selected = {7:phase_sources[7], 9:spaced(phase_sources[9], 8),
                12:spaced(phase_sources[12], 8)}
    invariance_value = torch.stack([rows[update][index]
        for phase in (7, 9, 12) for update,index in selected[phase]])
    invariance = TensorDict({"policy":invariance_value,
        "critic":invariance_value.clone()}, batch_size=[len(invariance_value)])
    invariance_ids = {f"P{phase:02d}":[
        f"rollout_{update:06d}:flat_{index:03d}:global_{base[update]+index+1}"
        for update,index in selected[phase]] for phase in (7, 9, 12)}

    entry_rng = torch.get_rng_state().clone()
    kernel = load_kernel()
    teacher, student = actor_copy(kernel, TEACHER), actor_copy(kernel, STUDENT)
    actor_before = {"teacher":parameter_hash(teacher),
                    "student":parameter_hash(student)}
    with torch.no_grad():
        teacher_train = kernel.distribution(teacher, train)
        teacher_validation = kernel.distribution(teacher, validation)
    inspected = kernel.inspect(student, train, teacher_train["mean"],
        validation, teacher_validation["mean"], invariance)
    with torch.no_grad():
        student_train = kernel.distribution(student, train)
        student_validation = kernel.distribution(student, validation)

    gradient = torch.tensor(inspected["selected_gradient_full256x2"], dtype=torch.float32)
    leaf = kernel.first_layer(student).weight[:, :2].detach().clone()
    with torch.no_grad():
        before = kernel.distribution(student, invariance)
        hypothetical = kernel.distribution(student, invariance, leaf - gradient)
    exact = {key:torch.equal(before[key], hypothetical[key])
             for key in ("mean", "log_sigma", "sigma", "request")}
    actor_after = {"teacher":parameter_hash(teacher),
                   "student":parameter_hash(student)}
    torch.set_rng_state(entry_rng)

    sets = {}
    for name, student_dist, teacher_dist, row_ids in (
            ("train", student_train, teacher_train, ids(1695, 221312, train_index)),
            ("validation", student_validation, teacher_validation,
             ids(1696, 221440, validation_index))):
        mu_delta = student_dist["mean"] - teacher_dist["mean"]
        request_delta = student_dist["request"] - teacher_dist["request"]
        jvp = inspected["frozen_first_gradient_response"][name][
            "requested_residual_per_unit_lr_full12"]
        sets[name] = {
            "row_ids": row_ids,
            "phase_counts": dict(Counter(
                f"P{int(value)+1:02d}" for value in
                (train["policy"][:, :13].argmax(-1) if name == "train"
                 else validation["policy"][:, :13].argmax(-1)).tolist())),
            "same_continuous_episode_as_other_partition": True,
            "wheel_student_conditional_mu_mean": student_dist["mean"][:, 8:12].mean(0).tolist(),
            "wheel_teacher_conditional_mu_mean": teacher_dist["mean"][:, 8:12].mean(0).tolist(),
            "wheel_student_minus_teacher_conditional_mu": summary(mu_delta[:, 8:12]),
            "wheel_student_request_mean": student_dist["request"][:, 8:12].mean(0).tolist(),
            "wheel_teacher_reference_request_mean": teacher_dist["request"][:, 8:12].mean(0).tolist(),
            "wheel_student_minus_teacher_request_rad_s": summary(request_delta[:, 8:12]),
            "wheel_request_JVP_along_negative_initial_gradient_per_unit_lr":
                summary(torch.tensor(jvp)[:, 8:12]),
        }

    result = {
        "schema":"wlr50_clean.front_retention419_zero_update_inspection.v1",
        "scope":"read-only CPU actor copies; no fit, optimizer, checkpoint write, or physical run",
        "teacher_semantics":"same-runtime CP220544 conditional raw mean queried on student observations including HISTORY; offline reference, not executed action or success label",
        "channel_order_full12":list(kernel.CHANNELS),
        "wheel_order":list(WHEELS),
        "teacher":{"checkpoint":str(TEACHER), "sha256":TEACHER_SHA,
                   "manifest_sha256":TEACHER_MANIFEST_SHA},
        "student":{"checkpoint":str(STUDENT), "sha256":STUDENT_SHA,
                   "manifest_sha256":STUDENT_MANIFEST_SHA,
                   "counters":{key:student_meta[key] for key in
                    ("global_policy_decisions", "ppo_updates", "optimizer_steps")}},
        "runtime_head":HEAD,
        "source_rollouts":[{"update":update, "path":str(path),
                            "sha256":sha(path),
                            "phase_counts":dict(Counter(f"P{p:02d}" for p in phases[update]))}
                           for update,path in paths.items()],
        "partition_rule":"train=P01 first row plus 32 source-order-even P02 rows from rollout1695; validation=32 source-order-even P02 rows from rollout1696",
        "partition_limitations":["train and validation are correlated portions of one continuous episode",
            "P01 has one training row and no independent validation row",
            "there are no P03-P06 learner rows; none are synthesized"],
        "sets":sets,
        "initial_selected_gradient_l2":inspected["selected_gradient_l2"],
        "initial_selected_gradient_by_phase_column_l2":
            inspected["selected_gradient_by_phase_column_l2"],
        "raw_half_MSE_directional_derivative_per_unit_lr":
            inspected["frozen_first_gradient_response"]["raw_half_MSE_per_unit_lr"],
        "actual_same_input_invariance":{"row_ids":invariance_ids,
            "phase_counts":{key:len(value) for key,value in invariance_ids.items()},
            "hypothetical_full_unit_negative_gradient_leaf_step_bitwise_equal":exact,
            "claim_scope":"these sealed P07/P09/P12 inputs only; same-input, not future trajectory"},
        "actor_parameter_sha256_before":actor_before,
        "actor_parameter_sha256_after":actor_after,
        "actor_unchanged":actor_before == actor_after,
        "cpu_rng_restored":bool(torch.equal(torch.get_rng_state(), entry_rng)),
        "PPO_decisions_added":0, "PPO_updates_added":0,
        "PPO_optimizer_steps_added":0, "auxiliary_updates_added":0,
        "fit_called":False, "optimizer_constructed":False,
        "checkpoint_written":False, "teacher_deployed":False,
        "physical_success_claimed":False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Front-retention419 zero-update inspection", "",
        "This is a read-only CPU sensitivity check. No fit, optimizer step, checkpoint, PPO/AUX credit, or physics was produced.", "",
        f"- Teacher actor: `{TEACHER_SHA}`; student actor: `{STUDENT_SHA}`.",
        "- Teacher targets are offline same-runtime conditional means on the student's actual 419-column observations (including HISTORY). They were not executed and are not success labels.",
        "- Train uses one P01 plus 32 P02 rows from rollout1695; validation uses 32 P02 rows from rollout1696. Both partitions are from one continuous episode; P01 has no independent validation. P03-P06 learner data is absent and was not invented.",
        f"- Selected gradient L2: `{result['initial_selected_gradient_l2']:.9g}`; phase-column L2 P01/P02: `{result['initial_selected_gradient_by_phase_column_l2']}`.",
        "- Actual same-input protection rows: P07=1, P09=8, P12=8. A hypothetical full unit step along the initial negative gradient leaves mean/log-sigma/sigma/request bitwise equal on those rows; this is not a closed-loop trajectory claim.",
        "- Teacher/student actors are hash-identical before/after the inspection and CPU RNG was restored.", "",
        "## Wheel request sensitivity", "",
        "Values below are `cap*tanh(raw mean)` in rad/s. JVP is per unit learning rate along the frozen initial negative-gradient direction; it is not an applied update.", "",
        "| set | student−teacher request signed mean FL/FR/RL/RR | initial request JVP signed mean FL/FR/RL/RR | JVP max abs FL/FR/RL/RR |",
        "|---|---|---|---|",
    ]
    for name in ("train", "validation"):
        value = sets[name]
        lines.append("| " + name + " | "
            + ", ".join(f"{x:.7g}" for x in value["wheel_student_minus_teacher_request_rad_s"]["signed_mean"])
            + " | " + ", ".join(f"{x:.7g}" for x in value["wheel_request_JVP_along_negative_initial_gradient_per_unit_lr"]["signed_mean"])
            + " | " + ", ".join(f"{x:.7g}" for x in value["wheel_request_JVP_along_negative_initial_gradient_per_unit_lr"]["maximum_absolute"])
            + " |")
    lines += ["", "Full row IDs, source hashes, conditional-mean deltas, JVP ranges, and invariance evidence are in the paired JSON."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(OUTPUT), "report":str(REPORT),
        "selected_gradient_l2":result["initial_selected_gradient_l2"],
        "wheel_train":sets["train"]["wheel_request_JVP_along_negative_initial_gradient_per_unit_lr"],
        "wheel_validation":sets["validation"]["wheel_request_JVP_along_negative_initial_gradient_per_unit_lr"],
        "invariance":exact, "actor_unchanged":result["actor_unchanged"],
        "cpu_rng_restored":result["cpu_rng_restored"]}, indent=2))


if __name__ == "__main__":
    main()
