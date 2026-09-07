# C103168 final — FL capture incomplete; real tracking-reference use confirmed

Run `runs/ppo_semantic_v3/validation/20260907T0147389603120Z_g42b91e857a0a_a502f3cc872645a496b2fa1d6968b077`, runtime42b91e857a0a2da256dee85e8092642ca8e64aeb, saved103168/N1seed2001/naturalP01/fixed deterministic mean. Root confirmed session56302 exit0/PID gone. Final lifecycleSUCCEEDED is execution completion, **not task success**.

Actual **660 decisions /5273 physics ticks /43.941666667s**, P05 `INCOMPLETE_CONTROLLER_BLOCKED`, taskfalse, physicalvalidtrue/nullhardfailure, optimizer0/windowEndedfalse. Stage age30.008333333s. Final decision contains1tick;659 full8tick intervals plus1 precisely explain5273, not a missing-write or codec failure.

Scope: final manifests, one complete660-row decision pass, one whitelist-field scan of all5273 native rows, and terminal raw5273. No repeated hashes/PT/GPU/Python/Isaac, production/test changes or new workspace-revision conclusions.

## Task and phase evidence

Source-phase counts P01–P13: **[1,203,4,1,451,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 start at8/1632/1664/1672 ticks (.066667/13.6/13.866667/13.933333s). No P06–P13 behavior was exercised.

| Leg | Qualified active lift | Front crossing | Placed | Final current state / load |
|---|---:|---:|---:|---|
| FR | 48 | 1644 | 1659 | TOP / .347264298 |
| FL | 1754 | 2791 | **Absent** | AIR / 0 |
| RR | — | — | — | GROUND / .213330188 |
| RL | — | — | — | GROUND / .439405515 |

First unfinished task is **genuine FL capture/placement**, `placed_FL=.85`: .35qualified lift + .35crossing + .15top geometry, with no sustained trueTOP-contact term. The early RL initial-clearance hint at10 is not a hard qualification. FL already crossed legally; this is not an unqualified crossing failure or a fixed historical joint/clock entry requirement.

Of451 P05 decision endpoints, FL is AIR449, TOP0, obstacle-active0. The first two precede its uninterrupted lift. Final evaluator records3581 consecutiveAIR samples ending5273 (streak start1693), topcount0/within_topXYtrue. This is recorded evaluator history, not an independently reparsed full rawcontact interval.

Nearest absolute top gap among qualified+crossed+topXY **decision endpoints** is **+16.587552996mm at3016**, front+5.836251806mm, AIR/load0. This is not a claim about every unparsed120Hz geometry sample. Terminal raw independently confirms FL bottomz.066635616692m against top.05m: **+16.635616692mm** gap, front+5.441311429mm, AIR, verified inactive ground/obstacle pairs with0N. Thus geometric overlap alone did not become actual capture.

Final rawFR obstacle pair is active/verified, normal8.971070996N; RL/RR ground normals11.351406097/5.511076927N. All wheelgeometry verified. SupportFR/RL/RR count3, valid, CoM inside/+24.222089mm margin, but not FLsupport. Finalrawfinite; bodycollision detected/active/persistentfalse and penetration0. Body linear/angular speed=.023441947m/s/.053535579rad/s. These scoped safety measurements are not a paired stability claim.

## Native completeness and true120Hz reference usage

Decision aggregation and the independent native-log scan agree on **5273 verified/actual-target-effect ticks**. All native rows are contiguous episode1–5273; four in-episode state-write totals0. Detailed audit flags setter=dispatch, actual-mapping=dispatch and same-tick counterfactual alltrue. Own-phase request effect total5269 excludes four phase-handoff holds. All660 decision physical snapshots are valid, no finite fallback or decision-clock anomaly.

All5273 reference receipts independently verify, with0 checked clock inconsistencies: dispatch=auditphysicaltick, previousACK=dispatch−1, previousfeedbacksample+1=mapperfeedbacktick=previousACKwritecount. Native command clock includes settling offset; it is not incorrectly equated to episodeclock.

**Reference was actually used on1306 physical ticks, comprising2729 channel-use samples.** Counts come directly from each channel's `reference_used=true`, not from the mode being enabled or the decision-end flag:

| Source phase | Ticks with at least one used reference |
|---|---:|
| P01 | 0 |
| P02 | 395 |
| P03 | 8 |
| P04 | 2 |
| P05 | 901 |

Per-channel use: RLhip796, FRknee1305, FLknee12, FLhip616; other servo channels0. First/last used episode ticks9/5273. Only **1 of660 decision-end snapshots** has any reference_used=true. The original4tick feedback cadence versus8tick policy boundaries explains why endpoint-only inspection is inadequate; it must not be interpreted as no live reference use.

## One actual feedback/target example, not an improvement proof

P05 episode1709 / native dispatch1888 / previousACK1887, FLhip:

- Actual measured physical q=.101337097585rad; nominal4.8°, actual canonical error−.926836436°. Previous filtered **REQUEST**=+4.542589552°; active reference uses that same request, not previous effective displacement.
- Original desired diagnostic correction c0=−7.414691489°; reference-adjusted desired c1 is clipped/bounded to+10°. The computational-reference physical input is.022053953332rad; the actual measured-q evidence remains separately recorded.
- Previous mapper compensation0°, original1.25°/tick slew yields native drive6.05° (=4.8+1.25), not an immediate+10° jump. Current effective combined post-mapper bias+4.542589552° produces canonical target10.592589552° before standing-pose conversion.
- Actual dispatched float32 servo target=.186260506511rad; standing pose+.079351564°. Its audit and independent mapping check pass. Same-history zero-**current**-residual counterfactual=.142627283931rad retains previous-ACK reference/history and slew; it is **not an executed old-mode c0 comparison**.

This proves that the new reference participates in a real scheduled feedback tick and actual native dispatch, while retaining bounded compensation slew. It does not prove that reference use caused the final clearance, improved stability, or solved FL placement. No second mapper/physics trial was run to assert such causality.

**Conclusion:** FL legitimately lifts/crosses but never gains true capture; P05 times out with positive gap/zero load. The native/reference route is exercised and audited, with no demonstrated execution mismatch. C101376/P06, C93184/P05, oldA and all previous failed/incomplete results remain unchanged. The separately developed workspace-potential revision receives no evidence credit here, and no full/suffix success, successful video or paired improvement is claimed.
