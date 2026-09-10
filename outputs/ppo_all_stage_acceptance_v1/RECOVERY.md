# All-stage acceptance v1 — saved Block3 snapshot; Block4 P06 pending

Source branch base: 285307603f43bfc2846dada5f3ede73b0dc5f536 (user output-only commit).
Current branch: ppo/all-stage-acceptance-v1. Preserve the existing unrelated untracked filename.
Source runtime: d7479d9fc41cacd98740a10fb47c2fee64fd74b7.
Verified source: outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496.pt.
Source SHA256: e568ccd4573035fb3de734a09ac6afe1a7e212fb0c4c055d2d80e78f8e84279a.
Counters before this revision: 138496 decisions / 1047 PPO updates / 20940 optimizer steps.
First all-stage block complete: new1024decisions/8PPOupdates/160optimizer steps.
Second block stopped normally at a verified update boundary: actual P06
384decisions/3PPOupdates/60optimizer steps from139520; lifecycle
STOPPED_AT_VERIFIED_UPDATE_BOUNDARY. Its original1536 request left1152 unconsumed,
now reallocated to P07. Two completed P09 FALL episodes had no RR C/P; the P06
tail is nonterminal, not success. Prior block/history files are unchanged.
Third block P07 stopped at its verified update boundary: original1152 requested,
actual128decisions/1PPOupdate/20optimizer steps/1004physics ticks;1024 unconsumed.
Five completed P09 episodes:14FALL,14BODY_COLLISION,39BODY_COLLISION,50FALL,
11BODY_COLLISION. Six accepted frozen-FSM prefixes excluded4434decisions/35472ticks;
last reset/prefix after the fifth terminal adds no credited tail or sixth PPO episode.
Those1024 unconsumed decisions are now the ordinary-resume Block4 P06 preparation.
Cumulative new credit through three completed blocks:1536decisions/12updates/
240optimizer steps/12252physics ticks. This is a SAVED snapshot, not live totals.
Latest saved policy in this snapshot:140032decisions/1059PPOupdates/21180optimizer steps, real
save/load_round_trip=true; no full naturalP01 evaluation has run yet.
Checkpoint: outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140032.pt.
Recorded SHA256: 76924f10db37163e9d6ea88185a2a2f74bf6a7cdfea5f7e50778411def27c494.
Canonical manifest: same history folder checkpoint_step_000140032_manifest.json.
Recorded manifest SHA256: 211d06dfaeb68a20330a071a58da27762be8ba071615c0031ea695110b2daf56.
These hashes are copied from existing bound receipts, not a new hash/PT check.
Current committed runtime: a8b1484631156bad6e6ef0c96bd79a2fbf12e308.
Active run: runs/ppo_all_stage_acceptance_v1/train/20260909T0532461768313Z_ga8b148463115_50c0b9ea678a413c87c11ebbefe70f6e.
Its run_manifest.started.json records P06/1024decisions, phase_suffix, offset0,
frozen_fsm prefix, N1, seed1001, checkpoint every4updates, ordinary resume from140032,
new_mdp_warm_start=false. Block4 completion is PENDING; no live collection or
updates are counted in this saved snapshot. No fresh Adam reset.
P07's earlier stop request was fulfilled at140032, not a process kill. Its original
1152 request and the initial short-episode rationale remain in training_manifest.
Prefix is real reset-only teacher
execution and is excluded from policy credit. The final mutable pointer may
already be newer than this explicit snapshot: verify checkpoint_last_pointer.json.
Do not start a second Isaac process or edit production while it is sampling.
Initial migrated checkpoint was actually saved/reloaded (round_trip=true), with
source counters138496/1047; it contains no newly learned update. See manifests/training_manifest.json.
Runtime9e3e3cd diagnostic1 was interrupted after3dec/24ticks/0.2s for a measured
Python-vertex hot path; all records preserved, no task failure verdict or updates.
Exact float64 vectorization passed166 targeted integration tests (including56 new
geometry tests). Earlier full integration376 passed before this performance-only
and error-provenance patch. These receipts overlap; do not add counts blindly.
Diagnostic2 completed under2f942e8, freshP01/seed2001/B-zero,900dec/60s/P09:
runs/ppo_all_stage_acceptance_v1/diagnostics/20260909T0246187023798Z_g2f942e824f82_a77db36275dd4bc4be422482922fde92.
Observed actual geometry:849612 body+4*12708 wheel vertices, .08-.09s/tick.
FR/FL real Q/C/P, RR only initial lift; no RR crossing/placement. External diagnostic
window, not internal task terminal. RR final contact was ambiguous/unverified.
No B sample is credited to PPO. Full report: reports/B2_completed/prior_diagnostic_summary.md.

Implementation: versioned common physical acceptance, pose-aware collider measurement,
actual-transfer response maturity, captured-layer retirement, targeted P03 wheel
counter-authority and bounded local recovery. No physical assets/dynamics changes.
Existing HISTORY372/12/N1/120–15Hz/gamma=.9985/lambda=.99 retained.
Migration validates weights/critic/std/identity-normalizer/RNG/counters, then resets Adam
moments to lr=3e-5 and collects an empty new rollout. No historical rollout reused.

Completed first-block command (historical invocation, do not rerun):
`& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead a8b1484631156bad6e6ef0c96bd79a2fbf12e308 -SemanticVersion v3 -ExperimentId all_stage_acceptance_v1 -Stage full_episode -FromPhase P01 -Decisions 1024 -Seed 1001 -Device cuda:0 -NewMdpWarmStart -Checkpoint outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496.pt`
Completed P06 invocation (historical request1536; actually stopped after384, do not rerun):
`& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead a8b1484631156bad6e6ef0c96bd79a2fbf12e308 -SemanticVersion v3 -ExperimentId all_stage_acceptance_v1 -Stage phase_suffix -FromPhase P06 -Decisions 1536 -Seed 1001 -Device cuda:0 -CheckpointIntervalUpdates 4 -Checkpoint outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000139520.pt`
Completed P07 invocation (historical request1152; actual128, do not rerun):
`& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead a8b1484631156bad6e6ef0c96bd79a2fbf12e308 -SemanticVersion v3 -ExperimentId all_stage_acceptance_v1 -Stage phase_suffix -FromPhase P07 -PrefixSource frozen_fsm -TeacherOffsetDecisions 0 -Decisions 1152 -Seed 1001 -Device cuda:0 -CheckpointIntervalUpdates 4 -Checkpoint outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000139904.pt`
Current Block4 P06 invocation (already running, do not duplicate):
`& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead a8b1484631156bad6e6ef0c96bd79a2fbf12e308 -SemanticVersion v3 -ExperimentId all_stage_acceptance_v1 -Stage phase_suffix -FromPhase P06 -PrefixSource frozen_fsm -TeacherOffsetDecisions 0 -Decisions 1024 -Seed 1001 -Device cuda:0 -CheckpointIntervalUpdates 4 -Checkpoint outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140032.pt`
Reallocated first-check curriculum: P01/1024 COMPLETED + P06/384 COMPLETED +
P07/128 COMPLETED + P06/1024 RUNNING/PENDING + P10/1024 PLANNED + P05/512 PLANNED
=4096. P07's original1152 request spent128 and returned1024 to P06 preparation.
Only1536 is consumed in this snapshot; remaining2560 is planned, not credited.
After Block4's verified boundary retain P10/1024 and P05/512, then reload naturalP01
at the first-check4096. No full P01 evaluation/video/success claim yet.
Cumulative credited phase P01..P13:9,450,22,10,286,610,7,33,109,0,0,0,0.
Actual phase ticks:72,3600,176,80,2281,4872,56,264,851,0,0,0,0; not8*N.
All12252 native ticks have recorded verification;35 nonterminal handoffs,0 saved
native/state-write/bootstrap/prefix-storage anomalies. Twelve PPO updates are
shared across phases, not twelve independent updates per phase. Ten completed
training episodes include7FALL/3BODY_COLLISION; two older optimized nonterminal
tails total367decisions. No tail is called a completed task.
Prefix total5778decisions/46224ticks across9accepted attempts is excluded. Compact
prefix rows lack full native effect/four-write fields, so unavailable is not zero.
Current-PPO FRQ/C/P4each; FLQ5/revoke2/C3/P3; RR I13/Q7/revoke2 and RL I12/Q2/
revoke2, no rear C/P. Inherited prefix FR/FL Q/C/P8each are separately attributed
and do not certify PPO achievement or current support. P06→P07 twice and P07→P08/
P08→P09 seven times each are recorded advances, not proof all original guards
were simultaneously met; qualified takeover is separate from hard chain completion.
Fixed evidence: block3_summary/training_block_summary.json, all three completed
coverage_table.json files, and metrics/cumulative_after_block3.json. No raw reread,
PT load or rehash was used. Do not infer newer live totals from this file.
P10–P13 physical coverage remains UNVERIFIED; synthetic tests are not live success.
Do not claim old 0/8 different-checkpoint development attempts as a policy success rate.
Do not call the previous B1 missing-Q verdict positive proof of wheel-only climbing.
