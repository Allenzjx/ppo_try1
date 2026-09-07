# Headroom-version natural-P01 block — finalized82,560

**Execution SUCCEEDED / exit0; task successes0.** Actual requested training budget completed:4,096 decisions/32 PPO updates/640 optimizer steps. Five completed task episodes comprise one P09 incomplete and four P09 body collisions; a sixth episode contributes an optimized,nonterminal392-decision tail. No full-task or suffix success is claimed.

## Run, exact resume and actual budget

Run `runs/ppo_semantic_v3/train/20260906T2007180722147Z_ge99fde1b3e83_4cec420c803f4493a1cdb29abf662bd0`,HEAD `e99fde1b3e8366f0ff1d484140b82877df745f0c`,N1/seed1001/full_episode,naturalP01,starts from immutable78,464. Actual started arguments have `new_mdp_warm_start=false` and `policy_distribution_migration=false`; no new warm-start or prefix artifact is created. Source/final runtime-content hashes match. This is the selected same-MDP exact-resume path,not a fresh Adam reset or another execution/reward change. The previous block's migration ancestry is provenance,not a new migration performed here.

Final run/training manifests: actual=requested=planned4,096; unconsumed0; rounding_overrun0; `wall_time_s=1553.7430133000016`. Source78,464/578/11,560 → **82,560/610/12,200**,with actual **32×20=640** optimizer steps. Final spending is full_episode33,280/phase_suffix39,168/smoke0; unchanged origin10,112. Root independently confirmed process exit0 and final saved pointer/counters/roundtrip.

Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000082560.pt` records SHA **cc8d148603553b4fd0fe0019bae0fb51bc7fde116451f4274c961399614391f5**,`save_load_round_trip=true`. This report reads the receipt,without repeated file hashing or tensor loading. It does not invent an independently measured process-total wall time from the recorded training-loop wall.

## Complete phase and episode ledger

One complete policy-stream pass verifies4,096 contiguous globals78,465–82,560,with no later row. P01–P13 policy-request counts are **[6,1102,22,6,892,1570,5,5,488,0,0,0,0]**. P10–P13 have0 samples,not missing values inferred from future experience.

| Episode | Policy globals | Decisions / physics ticks | Last seconds / stage | Actual outcome |
|---|---|---:|---|---|
|0 |78,465–79,502 |1,038 /8,304 |69.2 /P09 |INCOMPLETE_CONTROLLER_BLOCKED |
|1 |79,503–80,284 |782 /6,250 |52.083333 /P09 |BODY_COLLISION |
|2 |80,285–80,872 |588 /4,699 |39.158333 /P09 |BODY_COLLISION |
|3 |80,873–81,517 |645 /5,159 |42.991667 /P09 |BODY_COLLISION |
|4 |81,518–82,168 |651 /5,207 |43.391667 /P09 |BODY_COLLISION |
|5, nonterminal tail |82,169–82,560 |392 /3,136 |26.133333 /P06 |termination=null |

Five terminal episodes total3,704 decisions; final392 decisions are already optimized but are neither a sixth failure nor a success. Completed-episode records independently agree with the policy stream. Episode0's physical evaluator is valid with null physical failure; the four body-collision episodes are valid sensor/evaluator results with `TASK_FAILURE_BODY_COLLISION`,not interface errors. This bounded ledger preserves their classification; it does not independently reconstruct raw exact body-pair forces absent from these compact training files.

All six starts are actual naturalP01,first policy action at physics_tick8/decision_count1. There is no teacher prefix or injected later-phase credit boundary. FR/FL hard Q/C/P in this block are therefore current policy-segment events rather than teacher history:

| Episode | FR qualified / crossed / placed ticks | FL qualified / crossed / placed ticks | RR qualified / crossed / placed ticks |
|---|---|---|---|
|0 |58 /1,440 /1,455 |1,542 /2,574 /2,654 |5,525 /5,610 /none |
|1 |47 /1,436 /1,450 |1,555 /2,578 /2,654 |none /none /none |
|2 |37 /1,527 /1,534 |1,628 /2,652 /2,734 |none /none /none |
|3 |44 /1,489 /1,490 |1,585 /2,573 /2,687 |none /none /none |
|4 |53 /1,553 /1,581 |1,678 /2,675 /2,775 |none /none /none |
|5, tail |51 /1,482 /1,505 |1,602 /2,486 /2,711 |none /none /none |

RL has no hard Q/C/P in any segment. Episode0 genuinely crosses RR but never places it: terminalRR isAIR/front−275.748905mm/clearance+19.609050mm/load0,despite historical `placed_RR=.7` partial progress. FL is then realTOP/load0.068578912. The four collision terminals have FL AIR/load0 and RR AIR/load0 with clearances about−45.57…−46.66mm; none has RR hard qualification. These facts do not identify one causal mechanism for collision or establish that the headroom revision improved or worsened it.

At the optimized tail boundary,phaseP06 has `rear_approach=0`; RR isGROUND/front−488.535039mm/clearance−50.264737mm/load0.222947229 and FL is currentTOP/load0.258650657. No tail termination or later rear progress is projected.

## Native execution, continuity and optimization

Total physics ticks **32,755=4,096×8−13**. Four actual collision-ending partial decisions explain the difference exactly: global80,284 executes2ticks (−6),80,872 executes3 (−5),81,517 and82,168 execute7 each (−1/−1). There are no other short intervals. Policy and physical-core counts match; teacher decisions/ticks are0,confirmed by naturalP01 start evidence and absence of a prefix path/artifact,not by interpreting a missing suffix-only metadata field as an error.

All32,755 individual native tick records agree with their summaries and are verified/actual-native-effect true; own-phase effect32,710 excludes45 actual incoming handoff-hold ticks. Every policy row records verified zero in-episode root-pose,root-velocity,force-or-impulse and gravity writes. All45 ordinary phase transitions remain terminalfalse,time_outsfalse,terminal-bootstrap-allowedtrue. No phase-label GAE terminal mask is introduced; terminal done remains the actual task outcome. This audits records and their established producer path,not a new numerical tensor GAE calculation.

All32 optimizer records579–610 have128-global increments,20 optimizer steps,changed actor parameters and finite nonzero gradients. All31 adjacent actor hash links match; initial before-hash equals source actorcc5caa5d…36ea,final after-hash equals saved605d1772…17ad. Identity normalizer hash remains sourcec230b0db…4552. Runner configuration retains5 epochs×4 minibatches; all32 logged update-end LR values are1e-5,with no claim about unseen per-minibatch rates. Thirty-two rollout artifacts exist but were not loaded. Nine immutable checkpoint receipts span78,592 through82,560.

Twenty-two finalized blocks now add **72,448 decisions/566 PPO updates/11,320 optimizer steps** since10,112/44/880. All earlier stopped budgets,failed outcomes and bounded reports remain. Latest completed saved-checkpoint naturalP01 evaluation remains **C76,416/P05 incomplete**; root has started C82,560 evaluation separately,not read or prefilled here. No probe/A-success gate,new physical success,paired stability superiority or successful video is claimed. Report work stops at this finalized block.
