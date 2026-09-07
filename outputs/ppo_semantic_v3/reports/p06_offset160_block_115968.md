# Finalized block34 — P06 offset160, checkpoint115,968

Run `train/20260907T0531193843603Z_gf4bfe2560bfd_4b60e01ba2614a4a90ad4c6a7c06a5e1`, frozen HEAD `f4bfe2560bfd228541f7829d830fa441054eab1d`, N1/seed1001/frozen-FSM P06/offset160. Both final manifests record **SUCCEEDED execution**; root confirmed exit0. This is full training-request completion, not task success.

Source113920/855/17100 → **115968/871/17420**; actual=requested=planned **2048 decisions /16 PPO updates /320 optimizer steps**, unused0/rounding0. Budget full53504 remains, suffix50304→52352, smoke0/origin10112; `53504+52352+10112=115968`. The34 finalized blocks add **105856 decisions /827 PPO updates /16540 optimizer steps** since10112/44/880.

Only this completed run's JSON policy/native/prefix/update/episode streams and checkpoint sidecars were audited with PowerShell. No PT load, Python, tests, simulator, repeated checkpoint hash, or new P07 run was read. The earlier fixed [first16/handoff report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_offset160_first_113936.md) remains unchanged.

## Policy versus teacher accounting

Exactly2048 policy rows are contiguous **g113921–115968**, and **all2048 are P06**; P01–P05 and P07–P13 each have0 policy samples. All rows explicitly retain `prefix_teacher_data_in_ppo_storage=false`. No current-policy phase transition or suffix/full success occurred.

All**five** physical prefixes were accepted, miss=null/no fallback. Each has608 decisions/4864ticks: P01=1/P02=207/P03=4/P04=1/P05=235/P06=160. Thus **3040 teacher decisions /24320 ticks** are excluded from policy credit. Every prefix record is policy_credit=false, every teacher raw and projected policy residual is0, all24320 native ticks verify and no-state-write checks pass. This does not mean the teacher's nominal controller exerted no physical control.

Each actual P06 target first appeared attick3584; offset160 consumed another1280ticks. Every credit start is at4864/40.533333333s, with159.466666667s of the whole-task horizon remaining, not a reset-to-zero clock. At all five saved handoff snapshots, RR/RL remainGROUND, front−338.263508/−352.447670mm and no hardQ/C/P; FL is actuallyTOP/load.192344755. Teacher FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 are preserved throughout every policy row but never attributed to this block's PPO. The first handoff's independent exact contact and CoM evidence remains in the bounded report linked above.

PPO contributes**16356 physics ticks**; adding24320 prefix ticks yields core**40676 ticks** and2048+3040=**5088 decisions**, exactly the final telemetry. The recorded training-loop wall is**1824.8636562000029s**. Core telemetry separately records roll-in wall1231.783564399695s and reset wall23.788317599799484s. These fields have their original scopes; they are not added or subtracted to invent exclusive policy-processing time, because the loop includes later resets/prefixes while initial setup precedes it.

## Episodes, deadlines and tail

| Episode | Policy globals | Policy decisions /ticks | Final physical tick /time | Outcome |
| --- | --- | --- | --- | --- |
| 0 | 113921–114361 | 441 /3521 | 8385 /69.875s | P06 INCOMPLETE |
| 1 | 114362–114802 | 441 /3521 | 8385 /69.875s | P06 INCOMPLETE |
| 2 | 114803–115243 | 441 /3521 | 8385 /69.875s | P06 INCOMPLETE |
| 3 | 115244–115684 | 441 /3521 | 8385 /69.875s | P06 INCOMPLETE |
| 4, tail | 115685–115968 | 284 /2272 | 7136 /59.466666667s | P06, **nonterminal** |

The four completed records each say `INCOMPLETE_CONTROLLER_BLOCKED`, physicalvalid=true, physical failure=null, P06 age40.008333333s. Each final policy interval has**1 tick**, atg114361/114802/115243/115684; these explain `16356=2048×8−4×7`. They are real task deadlines rather than fabricated rollout truncations. The284-decision tail is already optimized but not terminal: P06 age29.6s/reasonnull. It is not a fifth failure or success.

## First missing task and current support

No credited row or retained hard history records **RR/RL qualification, crossing or placement**. RR's initial-clearance flag appears at one decision endpoint inepisode0 and zero endpoints inepisodes1–4; that is not a count of hard-qualified attempts. None reaches P07. The first incomplete task remains physical rear approach/workspace, not an old fixed joint entry or a codec/interface condition.

| Episode | End rear_approach | End RR /RL front mm | Closest sampled RR /RL front mm | FL TOP /AIR decision endpoints |
| --- | ---: | --- | --- | --- |
| 0 | .241408515 | −409.647871 /−333.518859 | −297.724348 /−318.735447 | 11 /430 |
| 1 | .320172924 | −389.956769 /−311.072465 | −301.515710 /−306.272175 | 13 /428 |
| 2 | .528050799 | −337.987300 /−257.437852 | −276.161712 /−251.692083 | 10 /431 |
| 3 | .586994011 | −323.251497 /−240.891006 | −288.602975 /−239.816007 | 11 /430 |
| 4, tail | .604297656 | −318.925586 /−253.850606 | −271.898490 /−250.178095 | 16 /268 |

Closest positions above are **decision-end sampled maxima**, not unobserved120Hz geometric extrema or proof of dynamic feasibility. Fractional completion is not a satisfied rear task.

All five end snapshots have **FL AIR/load0**, with bottom gaps+82.937275/+86.581518/+76.523547/+104.395186/+95.358258mm respectively. The teacher's placed latch does not imply continued support. FR remainsTOP at every listed endpoint, loads.452000171/.457753524/.386328766/.410302207/.444788590. RL remainsGROUND, loads.547999829/.449689079/.484881045/.454872080/.462912477. RR is AIR/load0/bottomgap−49.699555mm inepisode0, andGROUND inepisodes1–4, loads.092557397/.128790189/.134825712/.092298933. Low AIR does not satisfy above-top qualification.

These are valid, executed exploration outcomes. The scalar approach changes or terminal wheel commands cannot isolate a unique physical cause or establish improvement. For example, the final tail's actual canonical wheels are `[−.214552536,−.244625323,+.024510615,+.690595156]`rad/s; this is an actual-command observation, not residual magnitude inferred from a final target or an automatic reason to alter a cap.

## Native, masks, clocks and writes

All**16356 policy native ticks** record verified/effect/own-request-effect=true. Compact tick rows, endpoint summaries and policy interval lengths agree; handoff-hold count0 because no credited phase change occurred. Endpoint float32 setter/mapping/same-tick counterfactual evidence is verified and bound to the stored raw request. Selected raw/mean/std/action/value/log-probability/reward/drive scalars are finite, and all physical evaluator snapshots are valid/null hard failure.

The**2044 nonterminal rows** permit bootstrap. Four true task terminals have bootstrap=false and potential_after0. time_outs=false throughout, no finite fallback, and no artificial phase terminal is observed. This validates recorded masks and clocks; no PT/GAE tensor replay was performed.

All four state-write totals—root pose, root velocity, force/impulse and gravity—are**0**, and all no-state-write checks pass. Global sequence, clock `episode_tick=4864+credited_ticks`, simulation time, compact command-tick relation and raw request binding have0 discrepancies. Every policy row retains the teacher front Q/C/P event ticks exactly. These checks do not turn incomplete tasks into physical success or create another optimizer gate.

## Same-MDP updates and immutable saves

This block uses ordinary source-bound resume113920, not a new-MDP or policy-distribution migration. The earlier new-MDP ancestry retained in sidecars is not another fresh-Adam reset. The selected return profile remains gamma.9985/lambda.99/rollout128. The fixed initial/handoff report records the source path/runtime/Adam receipt; no redundant source tensor/hash audit was added here.

All**16 updates856–871** advance exactly128 decisions/20 optimizer steps. Every actor changes with finite nonzero gradients; source→first, all15 adjacent and final→result actor fingerprints match. Checked PPO diagnostics are finite. Update-end LR values are1e−5 and its floating representation1.0000000000000003e−5; this is not a per-minibatch LR claim. Recorded value-loss range is.005569743703–380.381924438.

Saved114048/114432/114944/115456/115968 sidecars all record roundtrip=true, correct corresponding update actor and this source_run, unchanged identity-normalizer fingerprint and resume ancestry113920. Their final budget is full53504/suffix52352/origin10112. Final actor fingerprint is `09faa93fb17fccb24ad91c5f5e371c620ee1e1f6ed9fcd44a143422cc3d04053`.

The actual pointer names `checkpoint_step_000115968.pt` and its immutable sidecar, counters115968/871/17420. Recorded CP SHA is **`a4c6320ed34755b97b4c6f06a9e5a78eda76c3a1e6350cef4b0f126eb8ed145f`**, sidecar SHA **`188e398ec6c227673edb07be0ade8781292cc195d861dd5df97b1c9d72cff84f`**. These are existing verified-save receipts, not newly recomputed file hashes.

Latest completed natural-P01 evaluation remains **C113920/P05 incomplete**. No later P07 data, future count, full/suffix success, paired stability gain or successful video is inferred. Report and master append finished; read/write scope is closed.
