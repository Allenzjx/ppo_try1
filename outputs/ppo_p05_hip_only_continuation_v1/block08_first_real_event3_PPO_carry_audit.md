# First genuine PPO carry after mean-head AUX event3

**PASS. Actual sealed update1633** from mean3 source CP213376 `945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3` to ordinary CP213504 `dd1899ffef011ae09d387f97ed9b678b7edcf98ccfd831da1b4797c8d420ae77`. New **128 decisions /1 PPO /20 Adam**; cumulative **213504 /1633 /32660**. No AUX added.

The actual checkpoint-policy prefix used the frozen source mean3 actor: 298 decisions/2384 ticks across 1 starts, reaching P04 before credited learning. Every prefix record explicitly has policy_credit=false, and the exact128 raw samples exclude all prefix actions. This is a P04-initialized suffix course, not a full P01 learned success.

The complete three-event front AUX96/96 dictionary, old AUX7/8, all3 origins and immutable migrations remain exactly equal to source. Real PPO changes actor/Adam; all12 Adam states advance20, LR1e-5 and Identity persist. RNG includes all source state kinds/CUDA count and its actual saved state advances normally; embedded/sidecar and checkpoint hashes verify, official save/reload is recorded true.

Actual input phase counts: {'P04': 1, 'P05': 127}. Credited physics ticks 1024; assist-owned endpoints 0, terminals 0. Native projection/dispatch and all12 residual permissions verify. FL-only assist, if active, stays separately observed and is not relabeled a raw Gaussian sample.

389 observations and raw12/μ/σ/logp/value/reward/done exactly match the synchronous stream. Each original sample is used5 times in20 minibatches; CPU logp maximum difference 3.81469727e-06. No physical success or P03+ invariance is claimed; PPO can change trunk and sigma after the mean-only AUX.

Evidence: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812`, first sealed rollout1633, actual source/target checkpoints and likelihood/optimizer/prefix logs. CPU-only audit; no simulator/GPU/fit/production or frozen-helper edits, no checkpoint writes. Helper exits after this report.
