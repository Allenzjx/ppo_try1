# Block14 first 128 — actual receiving-profile audit PASS

Scope: only the immutable CP197248, rollout/update1506 and the first 128 request rows. CPU read-only reconstruction; no model forward, optimizer step, GPU RNG replay or Isaac invocation. No production/checkpoint edits.

CP197120 → **CP197248 / 1506 PPO updates / 30120 optimizer steps** is genuinely +128 decisions, +1 PPO update, +20 optimizer steps, +0 AUX. Phase coverage: **P01 2, P02 126; P03–P13 0**, no terminal decision. CP197248 SHA256: `f05f522486b0f6f3e2ed4f8e01fd88bf4949e6e2ebadeb302dbb30e620b667ff`.

The official migration plan revalidated against actual HEAD/runtime; the persisted receiving-sigma receipt exactly matches. Both source/target saved actor, critic and full Adam tensor hashes match their manifests. First update's actor-before hash is exactly the source actor hash. All 12 Adam parameter states retain their settings and advance by 20 steps; the actual optimizer has **one parameter group**, LR **1e-5** before and after. Identity state, original branch/origin, quantity receipt and AUX **7 accepted / 8 attempted** remain intact. Every inherited branch counter advances by 128/1/20; only full-episode spending increases by 128 (now 102656, phase_suffix 84480, smoke 0).

Full actor/critic/Adam/Identity hashes and source RNG restoration are checked by the live loader before sampling. The surviving state and first-update hash corroborate that path. There is **no independently saved pre-update Adam/RNG memory snapshot**, so this audit does not claim an independent GPU RNG replay.

New empty-storage enforcement is in the loader (`step=0`, no pending action); the saved `128×1×372` / `128×1×12` rollout contains exactly new decisions 197121–197248, new profile/runtime, natural P01 and no prefix. All recorded raw samples, conditional means, sigmas and old log-probabilities match storage exactly. All 12 sigmas are finite/positive; each request records one draw and zero extra forwards/draws. The actual phase residual mask is all ones; all **1024/1024 physics ticks** have verified native target audits.

The 20 actual optimizer minibatches use each saved sample five times. CPU algebra reconstructs the same current mean (maximum error 0), sigma (7.45e-9), current log-probability (3.81e-6) and Gaussian entropy (reported −26.31820087 vs reconstructed −26.31820164). First-minibatch ratio deviation from 1 is at most 5.72e-6; recorded mean KL is .01376999 and clip fraction .28125. No optimizer was replayed.

**Receiving ×3 activated in 0 samples**, because this batch only reached P01/P02. This verifies the unchanged branch of the new profile in a real run, not physical P10–P12 activation or improved support/task success. That later activation remains to be observed. Exact evidence and paths are in `block14_first128_receiving_audit.json`.
