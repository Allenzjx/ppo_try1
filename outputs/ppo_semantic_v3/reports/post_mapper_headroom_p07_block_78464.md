# Post-mapper headroom P07 block — finalized78,464

**Final execution SUCCEEDED / exit0; task successes0.** The requested2,048-decision training block is complete, with four P09 incomplete episodes and an already-optimized nonterminal tail. This is neither suffix nor natural-P01 success.

Run `runs/ppo_semantic_v3/train/20260906T1928297576378Z_ge99fde1b3e83_639e48bdc9c4499eae7e839939fe0e71`, HEAD `e99fde1b3e8366f0ff1d484140b82877df745f0c`, uses N1/seed1001/P07offset0 with explicit NewMdpWarmStart from immutable76,416. This report inspected only the completed block, once streaming its final policy ledger; no Python/Isaac/GPU execution, production/history modification, tensor load or repeated checkpoint hash.

## Final accounting and migration boundary

Final run/training manifests agree: actual=requested=planned **2,048 decisions**, unconsumed0, rounding0; **16 PPO updates/320 optimizer steps**, source76,416/562/11,240 → **78,464/578/11,560**. Recorded `wall_time_s=1933.4150088999886` is retained with its actual producer meaning: the training-loop wall includes subsequent resets/teacher prefixes, while the first prefix occurs before that loop. It is not a measured pure policy-compute time or total process wall with all five prefixes included.

Final stage spending is full_episode29,184/phase_suffix39,168/smoke0; origin10,112 remains. Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000078464.pt` records SHA **d09fa40910d8d7986d9813d3f8659b45c84d8b19daceffc70f6477a8de2ae644**, `save_load_round_trip=true`. Root verified process exit0 and final pointer78,464/578/11,560/roundtrip. This report reads the saved receipt and does not claim an additional actual hash computation.

The earlier fixed `post_mapper_headroom_p07_initial_128.md` remains unchanged. Its source/initial receipts preserve full actor including learned heteroscedastic std,critic,identity normalizer,RNG,policy/runner contract,76,416 counters,29,184/37,120 budget and10,112 origin. Adam is reset from source state/LR1e-5 to fresh moments/initial3e-5; rollout and physical state are not inherited. This was a new execution-MDP version, not exact-resume equivalence of physical outputs. It corrects where servo2° headroom is computed, retaining real physical hard limits and final slew; no reward,nominal,hard task predicate or frozen-A change is credited to the actor. The separately developed front-wheel1.2/GPU-copy prototypes remain unpublished/unwired.

## Complete policy and episode ledger

All2,048 policy rows are contiguous globals76,417–78,464, without rows beyond the completed boundary. Policy-request phases P01–P13 are **[0,0,0,0,0,0,5,5,2038,0,0,0,0]**. No P10–P13 policy experience was collected.

Q/C/P are actual hard qualified-lift/front-cross/placement events. Episode clocks below include the reset-only prefix.

| Episode | Policy globals | Decisions / policy ticks | Last physical tick / seconds | Outcome | RR Q / C / P |
|---|---|---:|---|---|---|
|0 |76,417–76,868 |452 /3,616 |9,568 /79.733333 |P09 incomplete,age30s |6,405 /none /none |
|1 |76,869–77,320 |452 /3,616 |9,568 /79.733333 |P09 incomplete,age30s |6,423 /none /none |
|2 |77,321–77,772 |452 /3,616 |9,568 /79.733333 |P09 incomplete,age30s |6,108 /6,200 /none |
|3 |77,773–78,224 |452 /3,616 |9,568 /79.733333 |P09 incomplete,age30s |6,043 /6,170 /none |
|4, nonterminal tail |78,225–78,464 |240 /1,920 |7,872 /65.6 |P09,termination=null |6,306 /none /none |

Four completed episodes total1,808 decisions; tail240 is already optimized,not a fifth failure or success. Every completed reason is `INCOMPLETE_CONTROLLER_BLOCKED`, with physical evaluator validtrue and physical termination_reason=null. There are no collision or wheel-only terminal classifications in this block.

The first unfinished task is RR placement. Episodes0/1 never cross RR; terminal RR is AIR/load0 at front−161.272105/−250.108179mm,clearance+18.504136/+6.184460mm. Episodes2/3 have genuine crossing history but no placement and subsequently retreat behind the edge: episode2 endsAIR/front−225.922015mm/clearance+6.712736mm/load0; episode3 endsGROUND/front−347.104829mm/clearance−48.019456mm/load0.029237532. Their `placed_RR=.7` is historical qualification+crossing credit,not current capture. FL is currentlyAIR/load0 at episodes0/1/3,whereas episode2 terminalFL is realTOP/load0.140769404; do not collapse these into a uniform support story. RL has no hard Q/C/P in any policy segment.

The tail retains RR qualification6,306 but no crossing/placement. At its boundary RR is AIR,front−108.987954mm/clearance−3.039366mm/load0,FL isAIR/load0, and `placed_RR=0.5092710807949838`. A historical qualified event does not mean current positive clearance,front crossing or support. There is no unrecorded future tail outcome.

## Teacher exclusion and continuous native execution

Five actual reset-only prefixes each744 decisions/5,952ticks contribute **3,720 teacher decisions/29,760ticks**. The prefix stream contains3,720 decision records plus five start/result/credit-start records each. All teacher raw/projected residuals are zero,all records carry policy_creditfalse,all29,760 native ticks verify,no in-episode state writes. FR Q71/C1,665/P1,695 and FL Q2,461/C3,115/P3,583 precede policy takeover and are excluded from PPO learning/success credit. First actual P07 takeover data in the fixed initial report shows RR/RL GROUND with no hard Q/C/P; it is not an injected ready-lift pose.

PPO **16,384ticks=2,048×8**,with no short intervals. Total physical core including teachers is5,768 decisions/46,144ticks. All16,384 individual per-tick audits agree with their summaries and have verified/actual-native-effect true. Own-phase effects16,374 exclude10 incoming handoff-hold ticks. All policy records identify teacher_initialized_suffix,teacher data absent from PPO storage,and verified zero root-pose/root-velocity/force-or-impulse/gravity writes.

All10 ordinary P07→P08/P08→P09 transitions have terminalfalse,time_outsfalse,terminal-bootstrap-allowedtrue. They remain continuous through the shared physical clock and official PPO done/GAE path,not independent phase episodes. This checks the emitted flags and unchanged producer path; tensor-valued GAE was not independently reloaded. The first128 report independently reconstructs headroom and float32 targets at its128 decision ends,including both necessary new-space clipping and an actual old-logical-preclip counterexample. That bounded mathematical reconstruction is not enlarged into a claim of a full16,384-tick offline rerun here; the full-block claim is the actual online per-tick verified audit above.

## Actual optimization and final scope

All16 optimizer records563–578 have20 steps,changed actor parameters,finite nonzero gradients and128-global increments. Their15 adjacent actor hash links match; first before-hash equals preserved source actor0f4edbbd…7494,final after-hash equals saved actorcc5caa5d…36ea. Final identity normalizer remains source hashc230b0db…4552. Runner configuration is5 epochs×4 minibatches; all recorded update-end LR values are1e-5,not a statement that every minibatch used that LR. Five immutable save receipts are76,544/76,928/77,440/77,952/78,464,with the first128 report kept as its original time slice.

Twenty-one finalized blocks now add **68,352 decisions/534 PPO updates/10,680 optimizer steps** since origin10,112/44/880. Latest completed full natural-P01 evaluation remains **C76,416/P05 incomplete**; no C78,464 evaluation is invented. Root subsequently started same-MDP natural-P01 block22 from78,464,requested4,096,in run `20260906T2007180722147Z_ge99fde1b3e83_4cec420c803f4493a1cdb29abf662bd0`,without NewMdpWarmStart. It is launch-only here: this report does not read its logs or add its planned samples. No A/probe-success gate,paired stability improvement or new successful video is claimed.
