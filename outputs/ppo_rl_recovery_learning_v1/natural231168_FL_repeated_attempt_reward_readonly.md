# CP231168 natural FL repeated attempts — bounded read-only reward audit

## Result

**This sealed run does not show net-positive FL lift/revoke reward farming.** There is no per-lift event bonus. All 16 qualification→GROUND-revocation windows have negative total reward; all 16 revocation-containing decision intervals lose global potential. The observed behaviour is repeated unsuccessful FL lift/capture attempts, with a separate **relative critic/GAE credit concern**, not a proven positive reward cycle.

No reward, advantage, policy, controller, process, or training state was modified. No Torch, PXR, model, rollout tensor, or new simulation was used.

## Scope and endpoint accounting

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T1257554297942Z_g59e868f3e223_8a6cf8fd89554916abdc8a6b9f6fe1ac` (CP230144→231168, HEAD 59e868f3e223).
Read 1,024 complete decision rows, existing reward breakdowns/event ticks, and two pre-update advantage JSON records. Snapshots used open→seek(END)→tell, not Windows directory Length. JSON report records exact byte extents and hashes.

Two actual episode segments: 243 decisions (P02 terminal), then **781 decisions**, globals 230388–231168. Second segment ends at tick 6248 / **52.066667 s**, still P05, **nonterminal**. It is an incomplete collection tail, not a logged P05 episode termination: no −40 failure event was received in this second segment. Update 1761 occurs during that segment, so its whole trajectory is not one frozen actor.

P05 request-phase accounting starts after tick 2008 (global 230639); the preceding P04→P05 transition is excluded from this P05 sum. It did not set done or reset potential. All FL qualification and revocation events are within this 530-decision window.

| Actual logged sum | P05: 530 decisions / 35.333333 s | Repeated revoke-endpoint cycles 2–16: 456 decisions / 30.4 s |
|---|---:|---:|
| Potential before → after | 0.313755 → 0.292122 | 0.271719 → 0.278529 |
| 5 × endpoint potential change | -0.108165 | 0.034051 |
| γ<1 potential decay | -1.311497 | -1.119807 |
| Potential shaping total | -1.419662 | -1.085757 |
| Time cost | -0.706667 | -0.608000 |
| Terminal event / counterroll / other weighted families | 0 / 0 / 0 | 0 / 0 / 0 |
| **Total reward** | **-2.126328** | **-1.693757** |
| Discounted total reward | -1.397308 | -1.096825 |

The complete second segment totals **-1.475702** (includes real FR progress before P05). Stored float32 reward sum differs from full-precision breakdown by only 1.298e-7 in P05.

## Qualification and revocation evidence

16 fresh FL qualifications, 16 matching `qualification_revoked_ground_before_cross`; no FL crossing or placement. Qualifications measured only ~8–9.6 mm upward motion with wheel-bottom still ~41–43 mm **below** obstacle top; qualification is not top clearance.

- Qualification-containing decisions together: **+3.208862**.
- Revocation-containing decisions together: **-5.968928**; all 16 negative, all 16 global-potential drops.
- Every inclusive qualification→revocation decision window is net negative (range -0.486663 to -0.098850).
- 13/15 nonoverlapping revoke-endpoint→next-revoke-endpoint cycles are negative. Two are positive (attempt 4: +0.061236; attempt 16: +0.007610), but their final global potentials are higher by +0.026623 and +0.005309 respectively. These are **not identical-state closed cycles** and do not establish farmable positive loop reward. Combined repeated-cycle reward is −1.693757.
- All 16 individual event times, enclosing decision rewards/potential deltas, old values, geometry, and cycle sums are in the JSON. Enclosing intervals include up to 8 physics ticks, so their entire delta must not be attributed solely to the event.

Representative intervals:

| Attempt | Qualified → revoked physics tick | Qualification decision reward | Revocation decision reward | Inclusive window total |
|---:|---:|---:|---:|---:|
| 1 | 2011 → 2564 | 0.065389 | -0.311982 | -0.486663 |
| 4 | 3233 → 3303 | 0.254502 | -0.334825 | -0.267534 |
| 5 | 3446 → 3769 | 0.281272 | -0.446493 | -0.400550 |
| 10 | 4607 → 5028 | 0.164980 | -0.349231 | -0.467065 |
| 16 | 6188 → 6213 | 0.186940 | -0.348614 | -0.177428 |

Final endpoint: FL active-lift=false, crossed=false, placed=false; current `air=true`, `ground_contact=false`, no TOP/contact/bearing, front distance **−159.478 mm**, gap **−49.515 mm**. Thus “all attempts GROUND-revoked” does **not** mean the final snapshot itself is GROUND.

## Reward implementation versus critic

Current `semantic_reward.py:593–600` computes:
`5 × (0.9985 × phi_after − phi_before) + terminal_event − 0.02 × dt − counterroll_cost`, plus weighted nonpositive quality families. The only positive event is complete-task success (+40); terminal failure is −40. `touchdown_events` is a diagnostic count, not a bonus. RR retention replacement is logged `scope=unchanged` throughout this second episode.

For every P05 decision the formula error is exactly 0 and adjacent endpoint potential continuity error is exactly 0. Undiscounted shaping telescopes to `5(φ_end−φ_start)−5(1−γ)Σφ_after`; discounted shaping also telescopes (error 1.110e-16). There is no observed phase/qualification reset injecting an extra reward. This establishes actual global-potential withdrawal, not a claim that every internal FL feature is uniquely isolated by this audit.

| Existing audit | P05 samples | Mean reward | Raw GAE mean; positive count | Stored normalized advantage positive | Mean old V → mean return |
|---|---:|---:|---|---:|---|
| Update 1761 | 18 | 0.001499 | -0.126888; 0/18 | 18/18 | -4.756114 → -4.883002 |
| Update 1762 | 512 | -0.004206 | 0.275333; 455/512 | 318/512 | -5.685959 → -5.410626 |

Update 1761 contains the earlier P02 terminal loss; rollout-wide normalization makes all 18 early P05 advantages positive even though raw GAE is negative. Update 1762 contains only P05 and all its recorded returns remain negative; 455/512 raw GAE values are positive because returns are less negative than old V. This may reinforce parts of an unsuccessful movement relative to the current critic; it is **not evidence of positive event payout or demonstrated cycle exploitation**. Exact tail V and per-event raw GAE are **N/A in inspected JSON**; the audit says official nonterminal bootstrap was used but does not log last V separately. They were not reconstructed from assumptions or tensors.

Update 1762 JSON ranges: old V **[-6.114383, -4.948180]**; returns **[-5.846047, -4.937942]**; raw GAE **[-0.255825, 0.642045]**. All returns are negative. Available first/last decision samples:

| Global decision | End tick | Stored old V | Immediate reward |
|---:|---:|---:|---:|
| 230657 | 2160 | -5.012225 | 0.046344 |
| 230658 | 2168 | -5.021176 | -0.078676 |
| 230659 | 2176 | -5.024043 | 0.005465 |
| 231166 | 6232 | -5.583035 | 0.076160 |
| 231167 | 6240 | -5.572873 | -0.027260 |
| 231168 | 6248 | -5.554733 | 0.035142 |

Individual return/raw GAE for these six rows and exact tail bootstrap V are **N/A in inspected JSON**. The last stored old V (−5.554733) is the critic value for the final action's input state, **not** a logged bootstrap value at the resulting tail state. A negative immediate reward can coexist with positive GAE when the multi-step, bootstrapped return is less negative than old V; this is not by itself a bug or a reason to force advantage signs.


## Actionable conclusion

Do not add a new repeated-lift penalty or change advantage based on this audit alone. Existing actual reward already makes the repeated attempts net costly and withdraws potential on revocation. Preserve this unsuccessful physical evidence; if investigating learning credit later, inspect the already-saved terminal continuation and exact bootstrap/critic accounting at a safe boundary. This report imposes no additional gate on current rear-leg work.

Evidence limit: decision endpoints, logged per-physics event ticks, and embedded reward sample audits are available; a separate complete 120 Hz joint/contact trajectory was not read or claimed. Read-only companion: [full numeric JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_rl_recovery_learning_v1/natural231168_FL_repeated_attempt_reward_readonly.json).
