# C76,416 natural-P01 evaluation — FL capture remains incomplete

Final execution: **SUCCEEDED / exit0**, but **task_success=false**. This is a genuine incomplete task, not an interface, sensor or video failure.

Run `runs/ppo_semantic_v3/validation/20260906T1908330051233Z_gc34262abffc1_fd1eba3dea714afca139357945d47440` uses unchanged `c34262abffc16847ff32d15ecbf790dd60803e0a`, seed2001, N1, natural P01 and deterministic heteroscedastic mean from immutable `checkpoint_step_000076416.pt`. There are no teacher actions or optimizer updates. Final `evaluation_manifest.json` reports **648 decisions /5,184 ticks /43.2s**, `window_ended_before_task_terminal=false`, terminal **P05 INCOMPLETE_CONTROLLER_BLOCKED**, stage_age30.000000000000004s. Shared physical evaluation is valid, successfalse, physical termination_reason=null/reason empty.

Phase P01–P13 ledger is **[1,193,3,1,450,0,0,0,0,0,0,0,0]**. Every policy action executes8 ticks. This evaluation never visits P06 or the rear carry/headroom window, so the observed P05 failure is not proof of a causal effect from that unexecuted rear correction or a test of P13 stopping.

## First unfinished physical task and contact evidence

FR genuinely qualifies at47, crosses at1,562 and places at1,576. Raw exact FR-wheel/obstacle contacts at1,575/1,576 are active and pair-verified, with world-Z forces41.5870933533N/7.8683032990N. Terminal FR remains TOP, load0.4287563109, front+103.286151mm.

FL genuinely qualifies at1,661 and crosses at2,700, but **never places**. Raw FL is continuously AIR for3,588 samples from1,597 through5,184. During every P05 physical tick1,585–5,184, the FL obstacle pair has0 active samples and0 nonzero normal-force samples. The closest absolute bottom/top gap after the wheel center crosses the front plane is **+5.271797mm at2,784**, front+4.638377mm, AIR, pair-verified but inactive and force0. A verified inactive sensor pair is not a touchdown.

Terminal FL is still AIR, front+5.655411mm, clearance+6.397297mm, within top XY/top geometrytrue, current TOPfalse, load0. Its `placed_FL=.85` is the unchanged partial qualification/crossing/geometry score, not a placement latch or85% success. The first unfinished physical task is therefore **actual FL surface capture/contact**, not an exact historical entry angle or a required replay duration. No RR/RL hard Q/C/P events exist.

The final advice still contains servo targets `[22.8,-13.4,0,45.9,6.9,0,0,0]` and zero wheel advice; actual dispatch remains live, including nonzero servo deviations and wheel targets. Missing contact is not caused by the recorder cutting off a completed nominal file. This report does not infer why the learned policy fails to touch down or prove that suffix training caused early-phase forgetting.

## Evidence integrity and limits

All5,185 raw observations (ticks0–5,184) are finite and contiguous. Body detected/real-pair-active/persistent flags are allfalse. All5,184 per-tick native audits are contiguous, verified, staged/dispatched equal, actual reconstruction equal and same-tick counterfactual true; all four state-write categories are0. Decision summaries independently total5,184 verified ticks, own-phase5,180, excluding four ordinary handoff-hold ticks. Terminal bootstrapping is false. No tensor/LP reanalysis or checkpoint rehash was performed.

Latest completed natural-P01 is now **C76,416/P05 incomplete**. C73,088 and all previous A/B/C failures remain preserved. This result adds0 training decisions and no suffix/full success, paired stability superiority or successful new video. The separately proposed post-mapper headroom correction, front-wheel range candidate and GPU-copy candidate have no new live result credited here.
