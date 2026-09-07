# C89088 final — P06 rear workspace remains incomplete

Completed evaluation: `runs/ppo_semantic_v3/validation/20260906T2316032937892Z_g2677995544c9_7a1d828322994ae5a270a90d2aac627c`. Saved89088, runtime2677995544c974c03d8b1e41d77375b45e323c9e, naturalP01/N1/seed2001, fixed deterministic actor mean, no teacher. Final run lifecycle is `SUCCEEDED`; root confirms session90638 CLOSED/exit0. **Actual950 decisions /7,600 physics ticks /63.333333s, P06 `INCOMPLETE_CONTROLLER_BLOCKED`, taskfalse, optimizer updates0.** The shared physical evaluator is valid, successfalse and physical termination reasonnull. This is task noncompletion at the P06 deadline, not a hard physical failure or an execution/codec failure.

## Phase ledger and exact first missing goal

Actual policy-request counts P01–P13 are **1,193,4,1,151,600,0,0,0,0,0,0,0**. The five continuous nonterminal transitions occur at8/1552/1584/1592/2800, entering P02/P03/P04/P05/P06 respectively. Every transition has null termination, timeoutfalse and bootstraptrue. P06 starts at2800/23.333333s and ends at7600/63.333333s: **40.0s stage age**, exactly its configured task deadline. No P07–P13 policy samples are present. Terminal bootstrap isfalse; stored `time_outs=false` is not reinterpreted as the200s global timeout.

P06's configured completion predicate is `rear_approach`. Current `TaskStageSupervisor.predicate` defines it as `min(workspace_RL, workspace_RR)`; each workspace uses measured front distance within **[−0.22,+0.06]m** and the measured lateral span. Both rear wheels satisfy lateral span, but both are still behind the lower workspace bound:

| Leg | Final front distance (mm) | Deficit to−220mm (mm) | Workspace value | Bottom-to-top clearance (mm) |
|---|---:|---:|---:|---:|
| RR | −236.756679 | 16.756679 | 0.932973284 | −49.734483 |
| RL | −226.803657 | 6.803657 | 0.972785372 | −50.540021 |

The final completion value is therefore **rear_approach0.9329732840471127**, limited by RR. This is the existing measured workspace goal, not a historical joint/velocity/force/clock-entry template. A single decision-stream pass found the same terminal values are the best decision-end RR/RL distances and best pair progress within this P06 episode. This is a15Hz decision-end statement, not an unexamined120Hz trajectory extremum. The last four snapshots advance toward the boundary: pair progress7576/.925957460,7584/.928229112,7592/.930646438,7600/.932973284. The task deadline arrives before the goal is met; no criterion is changed here.

## Historical qualification versus current measured support

FR genuinely earns **Q45/C1563/P1577** and FL **Q1669/C2604/P2793** in this natural-P01 policy episode. RR has no initial-clearance event and no hardQ/C/P. RL has10 initial-clearance events (including the initial P01 hint), but no hardQ/C/P. Initial hints and historical front placement are not rear qualification or current support.

The final raw observation is finite, tick7600, and agrees with the endpoint evaluator:

| Leg | Current classification | Actual pair force (N) | Normalized load | Current geometry/contact |
|---|---|---:|---:|---|
| FL | OBSTACLE / evaluator TOP | Obstacle normal12.867535; force_z12.770314 | 0.473384 | Front+377.778593mm; clearance−0.045213mm; top geometry+pair true |
| FR | AIR | Ground0 / obstacle0 | 0 | Front+504.720989mm; clearance+20.832359mm; no top contact |
| RR | GROUND | Ground14.314496 | 0.526616 | Front−236.756679mm; no obstacle pair |
| RL | AIR | Ground0 / obstacle0 | 0 | Front−226.803657mm; no obstacle pair |

FR's AIR streak is236 samples and RL's6 samples at the endpoint; FL has4,773 consecutive evaluator top samples. These are current sensor histories, not assumptions based on placed bits. The inactive RL ground pair still contains a recorded contact-point value; with activefalse/zero force, that stale/aggregate point is **not** evidence of current support. Final exact base_link–Obstacle pair is verified but inactive, force[0,0,0], active history[false,false,false], with no body detection/persistence and geometry penetration0. This report reads the final raw row rather than inventing a complete raw contact audit. Shared physical failure remainsnull.

Final base position is[.606514454,.024287798,.084893420]m; recorded base velocity[.013114410,.005882642,.020286804]m/s, linear speed.024862575m/s and angular speed.047474170rad/s. Valid mass-weighted CoM position is[.597434789,−.098417036,.156296097]m. These low instantaneous rates and absence of collision termination do **not** establish full-task stability improvement: this run stops in an earlier phase than C87040 and does not reach its collision window.

## Actual wheel controls at the terminal seam

Canonical wheel order is FL,FR,RL,RR, rad/s:

| Component | FL | FR | RL | RR |
|---|---:|---:|---:|---:|
| Nominal | +0.3 | +0.3 | +0.3 | +0.3 |
| Projected residual | −0.318750175 | +0.506907105 | +0.320162685 | +0.043995997 |
| Actual canonical drive | −0.018750175 | +0.806907105 | +0.620162685 | +0.343995997 |

The terminal front residuals are both within the **old±.6** range. FR actual+.806907105 is nominal+.3 plus residual+.506907105, **not** proof of residual authority beyond+.6. No new full-trajectory range scan is performed or inferred from this endpoint. The last three preterminal selected snapshots7576/7584/7592 record `live_endpoint_tail` and four nominal wheels+.3; the final diagnostic becomes `terminal_no_tail` while the last already-dispatched nominal/actual commands remain recorded. It is not a new terminal action or a reset to zero. The persistent rolling suggestion has not silently vanished, but these arithmetic facts alone do not prove why the measured rear workspace remained short.

## Native evidence and attribution limits

One completed decision-ledger pass confirms950 sequential decisions and7,600 ticks; every interval is8ticks. The independent native ledger contains7,600 sequential rows with verified true, actual mapping agreement and setter/dispatch equality. Decision summaries report **7,600 verified/effect ticks, own-phase7,595**, and all four in-episode root-pose/root-velocity/force-or-impulse/gravity writes are0. No teacher credit or optimizer update is present in this evaluation. The terminal raw row is finite; no claim of a separate full raw-stream finite/contact scan is made.

**Conclusion:** C89088 does not complete the existing P06 rear-workspace task before its40s stage deadline. RR is16.757mm and RL6.804mm short of the existing lower bound; neither rear leg has hard Q/C/P. Physical evaluation remains valid with no hard failure. The absence of collision in this shorter, earlier-phase rollout is not improved whole-task stability, and the range revision is not proved causally beneficial or harmful by this result. C87040's P09 collision and all older failed/incomplete outcomes remain intact.

Root separately starts same-HEAD ordinary resume89088 into P06/offset0 checkpoint-policy-prefix training, planned4096, run `train/20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf`. This is a new in-progress training block, **not** a new-MDP migration; its prefix actions will remain excluded from policy credit. This report does not inspect that run or add planned counts, successes or a future evaluation outcome.
