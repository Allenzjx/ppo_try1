# Actual deterministic-P02 AUX event2 audit

**PASS. Actual saved artifact, not a synthetic fit.** CPU-only independent source/target tensor and ledger checks; no fitting or simulator use by this audit.

- Source CP211968 `5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881` → unique AUX candidate `27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55`. Official source-device CUDA save and fresh independent official reload are recorded true; actual tensor hashes and embedded/sidecar values match. Latest pointer still names the pre-AUX source.
- Actual event2 **32 accepted /32 attempted**. Exactly **256 scalars**, all in `actor.mlp.0.weight[:,1]`, changed. All388 non-P02 columns (including P01), other actor tensors and every critic tensor are identical.
- Full Adam state, LR1e-5, Identity, source CUDA:0 runner config, full saved CPU/CUDA RNG, runtime/policy and PPO counters are identical. Counters remain **211968 decisions /1621 PPO /32420 Adam**; AUX adds0 PPO. All historical lineage fields and all3 origins persist.
- Event1 is byte-equivalent as parsed canonical data, including its original source/data/helper/report bindings; previous AUX7/8 ledger is intact. Front ledger now has event1=32/32 + event2=32/32 = **64/64**. Historical credit is not relabelled.
- Actual train raw MSE **0.000360060017556 → 0.000232219405007**; validation **0.000361353944754 → 0.000233187442063**. These are fixed historical-state losses, not physical success.
- Maximum cumulative REQUEST shifts across all32 steps and both sets: FL knee 0.303457975°, FR hip 0.0700193048°, FR knee 0.102660775°; largest wheel 0.00594808906 rad/s. Maximum |Δlogσ| 0.148748875; full Gaussian KL forward/reverse 0.341333018/0.331660947. Thus this is not mean-only learning.
- Reported real P01, real P03–P06 and synthetic P01/P03–P13 same-input entire-Gaussian checks pass. Independently, exact parameter support establishes no change for any same input with P02 one-hot0. It does **not** prove future-state or closed-loop trajectory invariance.

Inputs remain explicitly field-reconstructed historical deterministic P02 data, not directly saved389 or new on-policy samples. Their bounded admission and final source-specific inspection hashes match the real event. Current physical evaluation is separate. **Ordinary real PPO carry after event2 is not yet verified** (the earlier event1 carry was verified separately).

No frozen helper or production file was modified. Audit JSON gives full12 maxima and lineage keys; this CPU process exits after the report.
