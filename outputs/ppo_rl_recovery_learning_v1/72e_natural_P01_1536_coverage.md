# Sealed natural P01 course: 1,536 optimized decisions

Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T0938059083357Z_g72e63592bdf4_889c2d62e7fe433ea0dc3e99787a26ca`  
Runtime: `72e63592bdf412d375abfda29b03cf456fbf4f7e`; seed 1001; learning rate 1e−5.

**+1,536 real learner decisions, +12 PPO updates (1744–1755), +240 Adam steps. Zero teacher prefix.** One natural P01 episode spans all 12 complete updates; no reset, no terminal, no completed task. The lifecycle `SUCCEEDED` means the requested training block finished, **not obstacle traversal success**. Reads were confined to this sealed run; the new deterministic video was not inspected.

| Policy-request stage | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized samples | 2 | 292 | 4 | 10 | 477 | 458 | 1 | 1 | 291 | 0 |

These are request-stage counts, not endpoint labels; ordinary transitions did not end the episode.

## Real events and post-P06 coverage

FR crossed at **19.7583 s**, placed **19.8167 s**. FL crossed **52.3083 s**, placed **52.4083 s**. P05 had soft-entered P06 at 52.3333 s with no placement credit; the late real FL contact completed the pending task, without rewinding. The earlier [bounded P06 wheel-path note](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_rl_recovery_learning_v1/72e_natural_P06_through228683_readonly.md) remains unchanged.

P06→P07: **82.8667 s**; P07→P08: **82.9333 s**; P08→P09: **83.0000 s**. This establishes reachability of the rear preparation segment in this episode, not adequate rear-task sampling: P07 and P08 each received only one learner decision.

RR did obtain a **new measured lift at tick 10211 / 85.0917 s**, revoked on returning to ground at **10301 / 85.8417 s**. P09 contains **11 currently-qualified RR endpoints**, **14 RR AIR endpoints**, and **277 RR GROUND endpoints** out of 291. The earlier P05 RR lift at 47.1417 s was separately revoked at 47.5167 s; the stale first event timestamp is not used to credit the later attempt.

Across P07–P09 (293 samples), RL stayed GROUND at every decision endpoint. **RR/RL front crossing, legal top bearing, placement, capture-ready AIR, qualified RL lift, and all later RL-transfer/capture/P13 windows: zero.** Counts are sampled endpoints, not inferred continuous durations.

From the first P09 decision endpoint (83.0667 s) to the final endpoint, body-forward position changed **−81.663 mm** and mass-weighted CoM world xyz changed **(−95.036, −32.581, −11.370) mm**. This is actual retreat, not a claim of successful side-directed transfer.

## Budget-end physical state

End: **global 229120, tick 12288, 102.4 s, P09 age 19.4 s**, nonterminal. First currently unfinished task: **restore and sustain RR measured lift, then carry/cross/capture**; the observed brief lift did not become a crossing or placement.

| Leg | Current contact | Bearing force N | Front distance mm | Wheel-bottom gap to top mm |
| --- | --- | ---: | ---: | ---: |
| FL | AIR | 0.000 | +161.924 | +116.870 |
| FR | TOP | 13.456 | +302.398 | +0.009 |
| RL | GROUND | 12.450 | −359.961 | −49.998 |
| RR | GROUND | 3.069 | −318.997 | −50.008 |

FR/FL `placed=true` is history; it does **not** mean FL currently bears. Both rear current-lift flags are false. P09 source clock **96** holds before its pending knee waypoint because `current_free_lift_before_pending_knee` is false; this is a presently lost lift condition, not evidence that RR never lifted at all. There was no recorded safety or task terminal; the fixed training budget ended.

Final source wheels are all zero; final canonical FL/FR/RL/RR targets **[−1.044913, −0.037561, −0.194999, +0.002476] rad/s**, actual **[−1.046499, −0.171579, −0.098120, +0.042518] rad/s**. These end-state data are separate from the prior P06 +0.3 nominal window and are not assigned single-cause responsibility for retreat.

Checkpoint: `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000229120.pt`  
Counters: **229120 / 1755 / 35100**; SHA256 **a02bfd50e17cad3bebe083493b0da9611cfe607fa18dad6a4ed83abdecb91f1c**; save/reload roundtrip already verified by the parent task. Finite nonzero gradients and changed actor hashes are recorded. This in-training trajectory spans policy updates and is **not** the new checkpoint's frozen deterministic evaluation or proof of learned rear recovery.
