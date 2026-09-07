# Block28 final — tracking-reference diagnostic boundary103168

Run `runs/ppo_semantic_v3/train/20260907T0134080107746Z_g42b91e857a0a_a862752b4ec84772b24f061befcd1676`, runtime42b91e857a0a2da256dee85e8092642ca8e64aeb, N1seed1001/naturalP01/full_episode. Root confirmed session5759 exit0/PID gone. Both final manifests record **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**.

## Actual budget and migration

Actual **1792 decisions /14 PPO updates /280 optimizer steps**, source101376/757/15140 → **103168/771/15420**. Original planned4096 leaves **2304 unconsumed**, rounding0. Final `requested_policy_decisions=1792` is the finalized actual stop amount; `planned_requested_policy_decisions=4096` preserves the original plan. Training-loop wall723.9258038001135s. Full spending45312/suffix47744/smoke0/origin10112;28 finalized blocks since10112/44/880 add **93056 decisions /727 updates /14540 optimizer steps**.

The preserved `stop_after_update.request.json` explicitly requests a complete saved update before a separately versioned workspace-potential-density revision. This is a controlled update-boundary stop, not an optimizer/interface failure, task-success gate or completion of the original4096 budget. No future workspace revision or C103168 evaluation credit is added.

This block introduced the explicit **new-MDP previous-ACK requested-servo tracking reference**, not an ordinary exact-runtime resume. `tracking_reference_first_101504.md` preserves the bounded source/initial receipts: actor including learned state-dependent std, critic, identity-normalizer, RNG, counters and spent budget were preserved; old rollout/physical state were not inherited; fresh Adam initialLR3e−5 replaces sourceLR1e−5. It also limits the same-state initial comparison to logical actions, not changed native dynamics. The earlier implementation and first128 reports remain unchanged. No claim that this control revision itself is PPO-learned improvement is made.

## Policy phase and episode ledger

The complete1792-row pass is contiguous global101377–103168 and matches final telemetry. Policy/core ticks=14336;2 naturalP01 starts, no teacher or checkpoint-prefix budget.

| Episode | Global range | Decisions / ticks | P01–P06 source-phase counts | Outcome |
|---|---|---|---|---|
| 0 | 101377–102333 | 957 /7656 | 1,205,4,1,146,600 | P06 INCOMPLETE_CONTROLLER_BLOCKED,63.8s |
| 1, unfinished tail | 102334–103168 | 835 /6680 | 1,189,4,1,162,478 | P06,55.666667s, **nonterminal** |

Full phase vector P01–P13: **[2,394,8,2,308,1078,0,0,0,0,0,0,0]**. One completed episode and one optimized unfinished tail are not two failures or any task success. All physical-evaluator snapshots are valid; terminal hard-failure reason isnull. Neither segment reaches P07–P13.

| Segment | FR Q/C/P ticks | FL Q/C/P ticks | RR/RL hardQ/C/P |
|---|---|---|---|
| Episode0 | 42/1661/1680 | 1788/2802/2849 | All absent |
| Tail1 | 46/1522/1547 | 1642/2692/2853 | All absent |

RR initial-clearance hints occur27/11 times in the completed/tail segment respectively; each also has one early RL initial hint. **None is a hard qualification.** Front placement is actual current-policy history, not teacher credit, but FL is AIR/load0 at both endpoints and FR has returned toGROUND.

First missing task is measured P06 `rear_approach=0` at both endpoints. Episode0 RRfront−668.114852mm/AIR/gap−46.639944mm and RLfront−560.309321mm/GROUND remain448.114852/340.309321mm behind the existing−220mm workspace bound. Tail RRfront−665.977013mm and RL−595.236626mm are bothGROUND, with load.043618544/.528435259; deficits445.977013/375.236626mm. TailFL AIR/front−81.172701mm/gap+34.787869mm/load0, FR GROUND/load.427946197. These are real unfinished task states, not evidence of current obstacle-top capture from historical placed bits.

## Native, continuity, reference and optimizer checks

All1792 intervals contain8ticks, yielding **14336 verified/native-effect ticks**, own-phase effects14326. Four in-episode state-write totals0, every audit/no-write flagtrue, no sequence/nonfinite-distribution/physical-validity anomaly or finite fallback. All10 phase changes remain nonterminal/bootstrap-allowed/not timeouts.

All1792 detailed decision-end native audits select `previous_ack_requested_servo_reference_v1` and report independently verified previous-ACK reference. This endpoint count is not a count of every per-channel active feedback-reference use across14336 compact ticks; no such higher-resolution activity claim is invented. Native audit evidence contains no reported mismatch, but does not establish physical improvement.

All14 optimizer receipts cover PPO758–771, global101504–103168 at128-decision increments, with20 optimizer steps each (280total). All actors change and gradients are finite/nonzero, norm extrema1.0003439493–1.4142136024. Source-before/13 adjacent/final hash links and counters have0 violations. Update-end LR≈1e−5 is recorded separately from initial freshAdam3e−5; it is not a claim about every minibatch.

Source actor `529984f97c26de203c796617545b36c18e690b67280a2079c8d1e2bf4e079dd6` becomes `91794a7d6bab065dedcbb7c1806189e05117e9d7b6300c7517e135c7d792c4fd`, matching the final manifest/sidecar. The3 saved boundaries101504/102400/103168 record roundtriptrue and identical identity-normalizer hash. Latest pointer and immutable `checkpoints/history/checkpoint_step_000103168.pt` sidecar agree on source run/counters/budget; recorded CP SHA `005d8794fc4fbae5616c478321ddd1d0e2d8f3082fd94caec22d3acbb6a6c0c3`, pointer-recorded sidecar SHA `0c5ce571870dad9e18018dab8d9dfff1f742af7675d4d996335a562343232838`. No rehash or PT/GPU loading was performed here.

**Conclusion:** a valid1792-decision diagnostic training block was saved at a complete update boundary, with one P06 task noncompletion and an unfinished P06 tail; no rear transfer or whole-task success. The early stop preserves2304 unused planned decisions. C103168 naturalP01 fixed-mean evaluation is separately running/pending and was not read. Latest completed full evaluation remains C101376/P06 incomplete; older C93184/A results and all28-block history remain intact.
