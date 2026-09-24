# RR capture reward review — frozen fa4b98ed

Read-only scope: run `20260923T1918580156414Z_gfa4b98ed506e_73c2c47d6e63471ebbcd876e45516dd5`, completed updates 1693–1696 only. Decisions 221057–221568 (512 actual learner samples), no prefix credit, no pending update. No model forward, Torch, simulator, reward/advantage edits or production changes. Both bounded base-Python helpers exited successfully.

## Findings

1. **Not a direct sigma-jitter-cost conflict.** For the first 384 samples, actual weighted body/contact/smoothness/regularization totals are all zero. Smoothness diagnostics sum to −5.54676 but have weight zero. Actual reward totals −1.54403 = potential shaping −1.03203 + elapsed-time cost −0.512. Noise could indirectly disturb physical support; this audit does not establish that cause.

2. **RR receiver retirement is working in the observed pre-capture window.** All 320 crossed/legal-XY/non-ground/unplaced endpoints retain the receiver share. There are zero retirement-loss events. Postcross unload credit is also fixed at 1; actual loading does not lose unload credit. Current capture proximity increases as a positive gap decreases toward zero; actual TOP samples receive the independent contact half of the existing capture share. No evidence here supports changing those two retirement rules.

3. **There are current-support qualification cliffs, not a hidden height target.** Seven current-RR-qualification losses remove exactly 0.03984375 of total Phi (about −0.19922 before discount/other changes). RR remains established, non-ground AIR, legally crossed; `body_control_evidence=false` in all seven. Four endpoints have only FR as an observed other support; three have FR+FL currently but the existing short support-history condition is false. Thus this is a real support-evidence rule, not proven detector failure or a reason to weaken safety. Four events occur while RR is descending; the small positive capture increment is outweighed by that qualification loss. Two of the seven actual standardized advantages are nevertheless positive, so immediate penalty cannot be equated with the PPO learning sign.

4. **Confirmed post-placement recapture/retention signal gap.** Once `history.placed.RR` is true, RR bypasses current lift/capture and uses `.8 + .2*retention`. For RR, retention checks only platform XY and lower gap boundary; it accepts any positive AIR gap at full credit and never checks present TOP/bearing. Within completed update 1696, 103 sampled endpoints have RR placed history; 99 lack current RR bearing while RL is not in a qualified swing. All 99 have retention exactly 1, RR total-Phi contribution exactly 0.2125, despite gaps ranging −0.323 to +58.265 mm. They retain the entire direct RR share. This is a concrete mismatch with P12's current-support dependency, not proof of the physical cause of bounce.

| Actual action endpoint | State | RR force | RR Phi share | Reward | Stored standardized GAE |
|---|---|---:|---:|---:|---:|
| 221466 / 70.3333 s | First placed sampled endpoint, real TOP/bearing | 0.29630 N | 0.2125 | +0.123849 | +2.14210 |
| 221467 / 70.4000 s | AIR, gap +0.366 mm | 0 N | 0.2125 | −0.006274 | −0.11974 |
| 221470 / 70.6000 s | AIR, gap +21.523 mm | 0 N | 0.2125 | −0.006274 | −0.56986 |
| 221471 / 70.6667 s | AIR, gap +28.553 mm | 0 N | 0.2125 | −0.006274 | −0.63466 |

These are decision endpoints, not a claim about the first physics contact tick. The first real placed action has positive reward and advantage: **contact is not shown to be punished**. On the following input, public timing is already RL preparation (`[carry,handoff,prep,swing]=[0,0,1,0]`). `public_timing` prioritizes started late transfer/P10+ over RR carry, so losing current bearing does not reopen `rr_carry_capture` or its observed-state exploration gate.

There is an **indirect existing signal**: with no current RR support and no qualified RL swing, new-mode RL potential retains only its .1 workspace share. This avoids rewarding fresh RL unloading from historical RR placement alone. If RL has not yet earned non-workspace progress, however, that clamp produces no additional loss. It does not replace an RR recapture target.

## GAE and interpretation limits

First 384 inputs: P07=1, P08=1, P09=382; fourth update: P09=26, P10=3, P11=99. These ordinary transitions are nonterminal. Actual minibatch likelihood audits expose each raw-action sample five times; the reported advantages are the values actually supplied to PPO, not recomputed counterfactuals.

| Completed update | Raw GAE mean (return minus old value) | Raw GAE positive / 128 |
|---|---:|---:|
| 1693 | +0.446752 | 96 |
| 1694 | +0.820116 | 127 |
| 1695 | +0.807912 | 128 |

For the 160 descending pre-capture endpoints, 55 stored standardized advantages are positive and 105 negative; average −0.28680. This correlation is not causal evidence that descent itself is discouraged. At Phi≈0.62, the unchanged discounted potential contributes approximately −0.00465 even at unchanged state; elapsed time adds −0.001333/decision. Potential shaping uses `5*(0.9985*Phi_after-Phi_before)`, with zero terminal Phi and ±40 success/termination event. Negative instantaneous reward is therefore not synonymous with negative progress or negative GAE. Nonterminal 128-step chunks bootstrap from the current critic; these finite observations cannot establish an optimal-policy conflict. All recomputed stored endpoint Phi values match exactly.

## Smallest later-version correction to test — not implemented

Only the new rear-timing mode should change. Preserve actual RR placed history and its .8 leg credit. While RL is neither placed nor in a currently qualified ongoing AIR swing, reopen an explicitly observed RR recapture/hold task when current RR bearing needed for transfer is lost. Reuse the existing .2 retention share for measured current contact/bearing plus continuous legal task-space recovery, rather than full credit at arbitrary positive height. Saturate at usable measured support; do not reward ever-larger force, add a second touchdown event, invent bearing, require zero body motion, rewind source cursors or command a new teacher trajectory. Any exploration-state meaning change must remain identical between sampling/current log-prob audits and receive a versioned fresh-rollout migration.

Necessary targeted positive/negative tests: real TOP+bearing with a valid front bridge; TOP pair but unverified/insufficient force; history-placed AIR with gap 0/12/58 mm; illegal XY/ground/invalid evaluator; loss and reacquisition without a second placed bonus; valid ongoing RL AIR not frozen by RR support loss; RL placed/finished continuation; old experiment semantics unchanged. Test the continuous potential, task-state bits and scheduler dependency together; preserve existing physical success/failure predicates.

Source locations: `semantic_supervisor.py:1311` (placed branch and RL dependency), `:1463` (current retention), `:1495` (current lift credit), `:961` (functional-RR support evidence); `semantic_rear_policy_timing.py:18` (measured dependency), `:46` (public timing priority); `semantic_reward.py:436` (discounted shaping/terminal treatment).

Evidence files beside this report: `P07_first384_reward_audit.json` and `P07_completed_capture_reward_audit.json`. The actual course and subsequent deterministic baseline remain unchanged and continue independently of this diagnostic.
