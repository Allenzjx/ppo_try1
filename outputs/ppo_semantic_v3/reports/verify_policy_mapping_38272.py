"""Read-only source weights/rollouts; CPU policy equivalence, not task success."""
from pathlib import Path
import json
import torch
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_training import (
    construct_semantic_runner, load_semantic_checkpoint, parameter_hash,
    sha256_file, write_json,
)
from wlr50_clean.ppo.semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, map_gaussian_actor_state,
)

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000038272.pt"
EXPECTED = "f8b6e19233283c0678f336072d5febffe94329c097b5662bcb1b400e5b0ddde3"
assert sha256_file(CHECKPOINT) == EXPECTED
metadata = json.loads(CHECKPOINT.with_name(CHECKPOINT.stem + "_manifest.json").read_text())
paths = [
    "runs/ppo_semantic_v3/train/20260906T0819495765677Z_g64abc5357d00_b22cdf1222cd4e759de72734f0d5fa7a/rollouts/rollout_000201.pt",
    "runs/ppo_semantic_v3/train/20260906T0819495765677Z_g64abc5357d00_b22cdf1222cd4e759de72734f0d5fa7a/rollouts/rollout_000264.pt",
    "runs/ppo_semantic_v3/train/20260906T0748083849450Z_g64abc5357d00_525e380c18ae4039ad93d764e7a06f50/rollouts/rollout_000200.pt",
    "runs/ppo_semantic_v3/train/20260906T0636173277212Z_g64abc5357d00_6614a15089014b2ebef51695b9b8cc22/rollouts/rollout_000172.pt",
]
torch.set_num_threads(1)
observations, bindings = [], []
for relative in paths:
    path = ROOT / relative
    rollout = torch.load(path, map_location="cpu", weights_only=False)
    assert rollout["runtime_contract"] == metadata["runtime_contract"]
    values = rollout["observations"]["policy"].reshape(-1, 324)
    assert values.shape == (128, 324) and torch.isfinite(values).all()
    assert torch.equal(values[:, :13].sum(-1), torch.ones(128))
    bindings.append({"path": str(path), "sha256": sha256_file(path), "observation_count": 128})
    observations.append(values)
inputs = torch.cat(observations)

class ObservationEnv:
    num_envs, num_actions = 1, 12
    cfg = {"semantic_version": "v3", "evaluation": True}
    def get_observations(self):
        return TensorDict({"policy": inputs[:1], "critic": inputs[:1].clone()}, batch_size=[1])

source, _ = construct_semantic_runner(ObservationEnv(), seed=1001, device="cpu",
    policy_version=LEGACY_POLICY, initialize_actor=False)
load_semantic_checkpoint(source, CHECKPOINT, contract=metadata["runtime_contract"], seed=1001)
target, _ = construct_semantic_runner(ObservationEnv(), seed=1001, device="cpu",
    policy_version=STATE_DEPENDENT_POLICY, initialize_actor=False)
source_actor_hash = parameter_hash(source.alg.actor)
mapped, proof = map_gaussian_actor_state(source.alg.actor.state_dict(), target.alg.actor.state_dict())
target.alg.actor.load_state_dict(mapped, strict=True)
target.alg.critic.load_state_dict(source.alg.critic.state_dict(), strict=True)
batch = TensorDict({"policy": inputs, "critic": inputs.clone()}, batch_size=[len(inputs)])
with torch.inference_mode():
    before, after = source.alg.actor(batch), target.alg.actor(batch)
    values_before, values_after = source.alg.critic(batch), target.alg.critic(batch)
    for model in (source.alg.actor, target.alg.actor):
        model.distribution.update(model.mlp(model.get_latent(batch)))
    sigma_before, sigma_after = source.alg.actor.output_std, target.alg.actor.output_std
    kl = target.alg.actor.get_kl_divergence(source.alg.actor.output_distribution_params,
                                           target.alg.actor.output_distribution_params)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sigma_before, sigma_after, rtol=1e-6, atol=1e-7)
    assert torch.equal(values_before, values_after)
    assert torch.isfinite(kl).all() and kl.abs().max() <= 1e-6
assert parameter_hash(source.alg.actor) == source_actor_hash
assert sha256_file(CHECKPOINT) == EXPECTED
result = {
    "schema": "wlr50_clean.actual_checkpoint_policy_mapping_cpu.v1",
    "source_checkpoint": str(CHECKPOINT), "source_checkpoint_sha256": EXPECTED,
    "source_global_policy_decisions": 38272, "source_ppo_updates": 264,
    "observation_count": len(inputs), "rollouts": bindings,
    "phase_counts": {f"P{i+1:02d}": int((inputs[:, :13].argmax(-1) == i).sum()) for i in range(13)},
    "mapping": proof, "mean_max_abs_difference": float((before-after).abs().max()),
    "sigma_max_abs_difference": float((sigma_before-sigma_after).abs().max()),
    "value_max_abs_difference": float((values_before-values_after).abs().max()),
    "kl_max_abs": float(kl.abs().max()),
    "source_actor_parameter_sha256": source_actor_hash,
    "target_actor_parameter_sha256": parameter_hash(target.alg.actor),
    "critic_preserved_exact": parameter_hash(target.alg.critic) == metadata["critic_parameter_sha256"],
    "optimizer_updates_added": 0, "policy_decisions_added": 0, "physics_steps_added": 0,
    "checkpoint_written": False, "stochastic_action_sampled": False,
    "scope": "512 archived physical observations, CPU forward equivalence; not task success or bitwise trajectory equivalence",
}
write_json(Path(__file__).with_name("policy_mapping_38272_cpu.json"), result)
print(json.dumps({k: v for k, v in result.items() if k not in ("mapping", "rollouts")}, indent=2))
