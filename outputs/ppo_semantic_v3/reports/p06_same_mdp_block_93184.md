# Block26 final — same-MDP P06 checkpoint-prefix training to93184

Final run: `runs/ppo_semantic_v3/train/20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf`, runtime2677995544c974c03d8b1e41d77375b45e323c9e, N1/seed1001/P06offset0/checkpoint_policy. Root confirms session10060 CLOSED/exit0 and no remaining PID162828. Both finalized manifests record `SUCCEEDED` execution. This report performs one final complete policy ledger pass and separate prefix/update/completed-episode receipt checks; no PT loading, fresh hashes, GPU, simulator or subsequent evaluation reads.

## Actual completed work and save

Actual=requested=planned **4,096 decisions /32 PPO updates /640 optimizer steps**, unconsumed0, rounding0. Source89,088/661/13,220 becomes **93,184/693/13,860**. Recorded training-loop wall is **2385.0470698999707s**. Stage spending is full_episode35,328/phase_suffix47,744/smoke0, with the original10,112 origin retained. Across26 finalized blocks since10,112/44/880, additional optimized work totals **83,072 decisions /649 PPO updates /12,980 optimizer steps**.

Latest pointer and matching immutable sidecar reference `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000093184.pt`, recorded SHA `f82d1a53f1580b730d7472d18ebd966d4503f83165a0e8d6d416dc2c1f3e0656`. Pointer-recorded sidecar SHA is `58546c94fad050444b3996abf752d2f7d63f84981bc79c0d49fbd02151e077ab`. Its source run, counters, stage ledger and final actor agree, and save/load roundtrip istrue. This report compares recorded receipts rather than recomputing hashes. All9 published saves (89216,89600,90112,90624,91136,91648,92160,92672,93184) record roundtriptrue and unchanged identity-normalizer hash.

This is **ordinary same-MDP resume**, not a fresh-Adam/new-MDP boundary. Final ancestry binds source89088/661/13220 and `resume_migration=null`, with the same runtime fingerprint. The retained first128 report `p06_same_mdp_first_89216.md` records the source network/Adam/normalizer and ordinary-resume distinction. The source actor, including learned heteroscedastic std, begins the first update unchanged; source Adam is resumed rather than reset. Physical initialization remains a real fresh episode plus reset-only frozen policy prefix, not restoration of an old physical snapshot. The earlier `p06_prefix_rr_entry_contrast_first_two.md` remains a fixed1219-row contrast, not overwritten by this final ledger.

## Policy phases and complete episode ledger

The4,096 policy rows are exactly contiguous **global89089–93184**. Phase counts P01–P13 are **0,0,0,0,0,3025,20,8,589,1,3,450,0**. There are7 completed episodes and one optimized nonterminal tail:

| Episode | Credited global range | Decisions | Policy physics ticks | End tick / full episode time | Outcome |
|---|---|---:|---:|---|---|
| 0 | 89089–89449 | 361 | 2882 | 5682 /47.350000s | P09 BODY_COLLISION |
| 1 | 89450–90307 | 858 | 6864 | 9664 /80.533333s | P12 INCOMPLETE |
| 2 | 90308–90629 | 322 | 2575 | 5375 /44.791667s | P09 BODY_COLLISION |
| 3 | 90630–90988 | 359 | 2865 | 5665 /47.208333s | P09 BODY_COLLISION |
| 4 | 90989–91337 | 349 | 2786 | 5586 /46.550000s | P09 BODY_COLLISION |
| 5 | 91338–92118 | 781 | 6241 | 9041 /75.341667s | P09 INCOMPLETE |
| 6 | 92119–92718 | 600 | 4800 | 7600 /63.333333s | P06 INCOMPLETE |
| 7 | 92719–93184 | 466 | 3728 | 6528 /54.400000s | P09 nonterminal tail |

The7 completed episodes sum3,630 decisions;466 remaining optimized decisions are neither an eighth failure nor a success. The independent completed-episode ledger agrees on every terminal count/time. Four physical terminal reasons are `TASK_FAILURE_BODY_COLLISION`; the three incomplete episodes and unfinished tail have valid physical evaluation and null hard-failure reason. Both fresh-P01 task-success count and checkpoint-initialized suffix-success count are0. P13 has no samples.

## Rear-leg credit and first unfinished goals

All front-leg Q/C/P present at the2800-tick credit boundaries came from the fixed prefix. They are not credited as new full-P01 PPO achievements. RR qualifications below happen strictly after takeover:

| Episode | RR hard event evidence | Actual first unfinished goal / current state |
|---|---|---|
| 0 | initial5648 only; no Q/C/P | placed_RR .255267; collision while RR AIR/load0, FL AIR |
| 1 | **Q5903/C6026/P6027** | Later P12 placed_RL0; RR has returned GROUND/front−267.238mm/load.152664, RL GROUND/front−312.712mm/load.532737 |
| 2 | No RR initial or Q/C/P | placed_RR .302313; body collision |
| 3 | One initial hint; no Q/C/P | placed_RR .272126; body collision |
| 4 | No RR initial or Q/C/P | placed_RR .081023; body collision |
| 5 | **Q5942**, no cross/place | placed_RR .35 at P09 deadline; RR still qualified and AIR+68.138571mm but front−398.640823mm, load0 |
| 6 | Seven initial hints, no Q/C/P | rear_approach .203984786 at P06 deadline; RR/RL front−419.004/−413.287mm |
| 7, unfinished | Two initial hints, no Q/C/P | placed_RR .334105; RR AIR/front−175.729mm/clearance−44.711mm/load0 |

Only episode1 completes a real RR placement; episode5 obtains a true above-top lift qualification but not crossing. **No RL hard Q/C/P occurs in any of the8 episodes.** Initial-clearance hints and continuous task values are not hard qualifications. RR's persistent completed-history bit after episode1 placement is not current support: its final state is GROUND. Episode5 shows that above-top AIR alone is insufficient when the wheel remains far behind the front plane.

FL is not TOP at any of the7 terminal endpoints: it is AIR/load0 except episode5, where it is actually GROUND/load.162764/clearance−46.811mm. TailFL is AIR/load0/clearance+144.176mm; tailRL is GROUND/load.511587. These current conditions are distinct from the prefix's historical placement. The compact training logs do not supply a complete raw body-pair force/contact-point/CoM trajectory, so this ledger preserves the actual collision classification without inventing its mechanics.

## Prefix, native and continuous credit accounting

All8 prefix results are accepted, missnull, each **350 decisions /2800ticks**. Each starts policy credit in current P06 at23.333333s, with176.666667s remaining on the original episode horizon. Their source remains checkpoint89088 and frozen actor receipt `733b1c1efaa2fddd8c5754f2ec94cb288f951fc0934503fb02acee51f33312bd` throughout the block, while the trainable actor updates normally. No fallback is recorded.

Prefix totals are **2800 decisions /22400ticks**, with phase counts P01–P05=8,1544,32,8,1208 and no credited P06 decision. Every prefix log row is policy_creditfalse. Prefix native summaries verify/effect all22400ticks, own22368, four state-write counters0. The prefixes legitimately execute the frozen actor's nonzero actions; exclusion from PPO is not a claim that their actions are zero. Every policy audit row separately has both teacher/checkpoint-prefix-in-PPO-storage flagsfalse and `task_result_scope=checkpoint_policy_initialized_suffix`.

Policy native summaries verify/effect **32741ticks**, own32709, with all four root-pose/root-velocity/force-or-impulse/gravity writes0. Exactly5 terminal decisions are short: g89449/2ticks,90629/7,90988/1,91337/2,92118/1; their deficits6+1+7+6+7=27 explain4096×8−27=32741. All24 cross-phase decisions remain nonterminal, timeoutfalse, bootstraptrue: the task phases do not create artificial GAE boundaries. Real collision/deadline terminals retain their real terminal semantics.

Physical core including prefixes is **6896 decisions /55141ticks**, exactly4096+2800 and32741+22400, not extra optimized credit. Recorded reset wall27.6551572997123s and prefix roll-in wall985.7647374998778s are separate telemetry fields. Do not add them to loop wall as disjoint durations: the first prefix precedes the training loop, while later reset/prefix work occurs inside it.

## Optimizer and immutable continuity

All32 update records are contiguous662–693 with global increments128 and20 actual optimizer steps each (=5 epochs×4 minibatches). Every actor update changes parameters and records finite nonzero gradients; source→first-before,31 adjacent links and final-after→manifest actor agree. Observed gradient extrema are1.0002508919–1.4142136547. All update-end learning-rate records are1e−5; this does not assert a constant rate for every minibatch. Final actor is `fc9e30f050fde45629ee4964cf352c7f5dc7f26fd1265c7b746a01fbb3e5dfc5`; final critic `20fb832b985a0ac88db76655c678bc44306aca03e01cf91ed5565d2b6ddd62e7`; identity-normalizer remains `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

The same final policy-stream pass checks finite12-dimensional old means/positive finite standard deviations and recomputes diagonal-Gaussian old log probabilities in scalar CPU arithmetic: maximum absolute error1.456163938e−6. No optimizer, policy inference or alternative training is run for this check.

**Conclusion:** the entire requested same-MDP optimizer block completed and saved93184 with recorded roundtriptrue. It contains two RR qualifications, one true RR crossing/placement episode, four P09 body collisions, three incomplete episodes and an unfinished P09 tail; **no suffix/full success**. Those local task events do not establish stability improvement or prove a single control parameter caused the differences. The next reloaded C93184 natural-P01 evaluation is outside this report and remains pending here; latest completed full evaluation is still C89088 P06 incomplete. Existing first128, first-two-episode, prior block and failed A/B/C records are preserved unchanged.
