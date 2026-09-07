# Block 24 finalized: checkpoint-policy P06 curriculum to 87,040

Both training and run manifests finalized **SUCCEEDED execution**. Root confirmed session99244 closed/exit0 and final pointer/sidecar agreement. This is not physical task success: six completed episodes fail BODY_COLLISION in P09, followed by an already-optimized, nonterminal P06 tail.

Run: `runs/ppo_semantic_v3/train/20260906T2211507063967Z_ga580fc2add81_e539358648084c70bf6b28037515480c`.

Runtime `a580fc2add8137ea599df558f74d34ec9c50ef97`; v3/N1/seed1001/P06/offset0, `prefix_source=checkpoint_policy`, explicit NewMdpWarmStart from immutable84,992, requested2,048, checkpoint cadence4. Finalized at `2026-09-06T18:39:15.314765-04:00`. Only this completed run was parsed; no new evaluation, checkpoint tensors, GPU/Python/Isaac execution, or duplicate file hashing was used.

## Actual optimizer and budget ledger

Actual=requested=planned **2,048 decisions**, unconsumed0, rounding0; **16 PPO updates /320 optimizer steps**. Source84,992/629/12,580 becomes **87,040/645/12,900**. Recorded training-loop wall time is **1,487.5502931000665s**.

Final spending is full_episode33,280 / phase_suffix43,648 / smoke0, with original origin10,112 preserved. `33,280 + 43,648 + 10,112 = 87,040`. Across24 finalized blocks this is **76,928 new decisions /601 PPO updates /12,020 optimizer steps** since origin10,112/44/880. Earlier stopped budgets are not retrospectively relabeled consumed.

Final immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000087040_manifest.json` records checkpoint SHA `66fdb6b0ca8be83c31566a0aa6701834d5d8159d4614b4f43251b00a51388729` and `save_load_round_trip=true`. Five immutable saves are85,120/85,504/86,016/86,528/87,040. Existing receipt values were read; hashes were not recomputed here.

The16 optimizer records are updates630–645, each advancing128 decisions and20 optimizer operations. Every record reports changed actor parameters and finite nonzero gradients. The source-to-first actor link and all15 adjacent links match. Source actor `8eb3d28a16b8da10a7660d3ae2cbe2b8642bb7b14b043909183663297263d56c` reaches final `ded137cc5567b5df36678c25b1d28587d08bb67e23da9c0d017dd001e9fad164`, matching the final sidecar. Final critic is `3ab4ca3d4a901d9f366a44352f82287f1aed07899bb5f120a1401335d7f50e22`. Configured5 epochs ×4 minibatches agrees with16×20=320 operations. All recorded update-end LRs are1e-5; this is not an unlogged per-minibatch LR measurement.

## Policy phase and episode ledger

One complete policy pass verifies contiguous globals84,993–87,040 and phase counts **P06=1,971, P07=10, P08=6, P09=61; every other phase=0**. These are PPO requests, not initialization samples.

| Episode | Global range | Credited decisions / ticks | End phase / original episode time | Outcome |
|---|---|---:|---|---|
| 0 |84,993–85,325 |333 /2,659 |P09 /45.625s |BODY_COLLISION |
| 1 |85,326–85,597 |272 /2,173 |P09 /41.575s |BODY_COLLISION |
| 2 |85,598–85,951 |354 /2,831 |P09 /47.058333s |BODY_COLLISION |
| 3 |85,952–86,276 |325 /2,593 |P09 /45.075s |BODY_COLLISION |
| 4 |86,277–86,573 |297 /2,370 |P09 /43.216667s |BODY_COLLISION |
| 5 |86,574–86,931 |358 /2,859 |P09 /47.291667s |BODY_COLLISION |
| 6, unfinished |86,932–87,040 |109 /872 |P06 /30.733333s |Nonterminal tail |

Six terminals total1,939 decisions; the109-decision tail is not a seventh failure or success. All six completed outcomes are physically valid recorded task failures, not a software lifecycle failure. This compact training audit does not independently reconstruct contact forces/body-pair persistence from an unavailable full120Hz raw sensor recording and does not assign collision causality to a particular command.

All **16,357 compact native ticks** verify and have actual effect; own-phase effect16,332. All four recorded in-episode root-pose/root-velocity/force-or-impulse/gravity write counters sum to0, and all per-decision no-write attestations are true. Six genuine short terminal intervals at globals85,325/85,597/85,951/86,276/86,573/86,931 contain3/5/7/1/2/3ticks: their27 missing nominal ticks explain `2,048×8−27=16,357` exactly.

All18 P06→P07→P08→P09 transitions in the six completed episodes are nonterminal, `time_outs=false`, `terminal_bootstrap_allowed=true`; no phase-label episode/GAE cut is introduced. All policy rows retain `checkpoint_policy_initialized_suffix` and `prefix_checkpoint_policy_data_in_ppo_storage=false`.

## Seven real frozen-C prefixes, all excluded from PPO

The prefix stream has one bootstrap, seven starts,2,464 initialization decision rows, seven accepted results and seven credit-start records. Each attempt uses352 deterministic decisions /2,816ticks and reaches P06 at23.4666666667s, with176.5333333333s remaining and no fallback. Aggregate initialization phase counts are P01=7/P02=1,379/P03=21/P04=7/P05=1,050. There are **19,712 prefix ticks**, all native-verified/actual-effect, own19,684, four write categories0, and every initialization record is `policy_credit=false`.

All seven source/provenance records bind the same immutable checkpoint84,992, source629updates, full actor hash `8eb3d28a…63d56c`, and independent frozen parameter/buffer storage. The final checkpoint curriculum still binds this source actor even though PPO's main actor changed16times. Equal prefix length does not imply bitwise-identical physical states: the first FL crossing occurs at2655, later terminal histories report2648. No change of frozen source weights is inferred from this physical variation.

Physical core totals are **4,512 decisions /36,069 episode ticks**, exactly2,464+2,048 and19,712+16,357 across seven natural resets. These totals are not one episode duration. Reset telemetry records26.942712999880314s; roll-in telemetry records864.1623346996494s. First initialization precedes the training loop, later reset/roll-in work overlaps its timer; these quantities must not be added/subtracted to invent an optimizer-only throughput.

The first128 saved-update seam, preserved separately in `checkpoint_prefix_first_live_85120.md`, proves actual carry of all12 residuals through the existing P05→P06 bridge and correctly excludes its incoming hold from own-phase effect. Credit begins without resetting task history/clock/bridge. Current fixed policy is used only for initialization, never its samples/log probabilities for PPO storage.

## Rear-leg history and current physical state

Every episode begins PPO credit with source-policy FR/FL Q/C/P already earned. Recorded FR Q45/C1594/P1608 and FL Q1693/C2655-or-2648/P2816 are **prefix-owned**, not learning achieved in this block. No completed episode or tail earns RR/RL hard qualification, front-crossing, or placement. RR has only one initial-clearance event in completed episode4 at5137; initial clearance is not qualification. Numerous RL initial-clearance events also remain below hard Q/C/P.

All six completed terminals have current FL AIR/load0 and RR AIR with negative clearance (roughly−47.316 to−43.532mm), despite historical front placement. This distinguishes active support from event history without claiming it alone caused the collision.

The final tail is tick3688/P06 age7.266667s, still nonterminal. FL is current TOP/load.498042404, FR is AIR/load0, RL AIR/front−429.949921mm/clearance−39.166557mm/load0, RR GROUND/front−437.383782mm/clearance−49.780226mm/load.501957596. Both rear histories have no hard Q/C/P. No P10–P13 policy samples were produced in this block.

## Migration boundary and claim limits

The existing first-live report records source→initial actor/learned std, critic, identity-normalizer and complete RNG equality. Adam intentionally changes from source effective1e-5 to fresh3e-5, with old moments/rollout/physical state discarded and original counts/budgets retained. The final identity-normalizer hash remains `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`. Initial and final topology/current sampling/provenance are consistent; no actor architecture, reward, nominal motion, physical configuration, or frozen-A revision is credited by this reset-only implementation.

This block exercised genuine checkpoint-policy initialization and PPO updates; it did **not** achieve suffix/full-task success or demonstrate paired stability improvement. Root has started a separately reloaded natural-P01 evaluation of87,040; it is not read or prefilled here. Latest completed full-P01 result remains C84,992/P12 incomplete. All earlier failures, unused budgets, and initial bounded evidence remain preserved.
