# Actual front rehearsal AUX — independent CPU audit

Result: **PASS for state/lineage integrity; physical recovery is not established by this audit.** The actual CUDA fit completed32 accepted/32 attempted independent SGD updates. This audit only read its sealed report and both real checkpoints on CPU; no optimization, GPU/Isaac launch or bound-helper modification.

- Source: `checkpoints/history/checkpoint_step_000209920.pt`, SHA256 `11476243212e12e5e6aa0591de4b00f41f75b1f718e142a862112d66736255ac`.
- AUX: `checkpoints/history/checkpoint_aux_frontrehearsal_step_000209920_v1.pt`, SHA256 `3c45e7325210487431ebe8b5d1cbd92b480734e3d53829b04429869c4773e3e8`.
- Source/report/sidecar/embedded metadata match. Actor and critic hashes were independently recomputed from actual saved tensors.
- **Exactly512 scalar actor weights changed**, all within `mlp.0.weight[:,0:2]`. Its remaining99,072 scalars, every other actor parameter/buffer and all critic tensors are exactly equal. Maximum selected weight change0.11915213.
- Complete Adam state/groups/moments/steps hash is identical. Effective PPO LR remains1e-5. Full Python, NumPy, Torch CPU **and serialized CUDA RNG/count** metadata are identical. Full saved runner config remains identical, including `cuda:0`; Identity normalizer/runtime/policy contract unchanged.
- PPO counters remain **209920 decisions /1605 updates /32100 Adam steps**, with +0 credit from AUX. All original origins, branch counts, immutable migrations and the prior AUX **7 accepted/8 attempted** ledger are exactly preserved.
- Only the current RR branch gains `front_rehearsal_auxiliary`, event1 with32/32. Its original keys match the source after removing that one new child. Actual source/data/helper/fit-report bindings match; all three current helper files still match the recorded SHA256s. No old ledger was rewritten.
- The actual publisher records official source-device save and a distinct fresh-runner official reload. Both actual payloads and reported reload bindings pass this independent read. `checkpoint_last_pointer.json` still references the original209920 source SHA, not AUX.

Saved-state training MSE decreased0.00171491958→0.00163514633; independent selected validation MSE0.00142229965→0.00134894589. Across all32 accepted proposals and both datasets, cumulative maximum full-Gaussian KL was0.30428012 forward /0.29407475 reverse; maximum |Δlogσ|0.17214823. Thus this is **not mean-only** learning.

Maximum requested-residual changes (not a promise of final actuator change): FL knee0.240358°, FR knee0.059084°, FL wheel0.00325745rad/s. All12 channels remain inside the declared3°/.15rad/s cumulative bounds. All32 recorded real P03–P06 and synthetic P03–P13 same-input distribution checks passed; the actual parameter whitelist independently supports that algebraic invariance. No real P13 coverage or unchanged future physical trajectory is claimed.

The local target data remain the earlier first-episode actual random raw actions that preceded verified FR capture and continuation; neither stored means nor transformed actuator targets were substituted. Later failures are not relabeled successful.

**Pending:** current natural-P01 AUX evaluation and the **next genuine PPO update/save carrying this new real AUX ledger**. Only synthetic normal-save carry was previously tested; this audit does not promote that to a real-training result. CPU audit processes exited; the root's current video was not interrupted.
