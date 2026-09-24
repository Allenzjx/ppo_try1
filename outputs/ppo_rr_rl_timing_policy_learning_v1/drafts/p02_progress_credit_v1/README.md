# P02 measured-progress credit v1 — outputs-only review draft

Status: Root applied the baseline supervisor/integration candidate after sealing CP221696 and stopping Isaac. This subagent did not apply production changes. Root's first installed-code run reported127/128 tests passing; the new fresh-pre-cross lift case correctly exposed that event_ticks retains the FIRST event even after legitimate requalification. `followup_fresh_pre_cross_lift.patch` is an unapplied, narrow follow-up for root review. No new physical success is claimed. The **419→422** public observation plan remains unchanged.

Files:
- `progress_credit_draft.py`: pure, standard-library credit kernel.
- `test_progress_credit_draft.py`:12 targeted kernel cases.
- `saved_CP221568_credit_replay.json`: bounded saved-log ledger replay only.
- `production.patch`: archived baseline candidate (now applied by root at the idle boundary).
- `integration_tests.patch`: archived baseline real-TaskEvaluator regression candidate (now applied by root).
- `followup_fresh_pre_cross_lift.patch`: narrow root-review follow-up; preserves both stale-post-cross counterexample and genuine fresh-pre-cross qualification.
- `test_supervisor_integration_draft.py` and `run_staged_supervisor_tests.py`: execute those supervisor hunks in memory only, with Torch/Isaac imports forbidden.
- Pair/terminal evidence: `../../CP221568_vs_v7_P02_first_divergence_readonly.md`.

Mode: `p02_measured_progress_credit_v1`, locked stage-task-spec key `p02_progress_credit_mode`; execution profile uses the same key via the migration agent. Existing versions without this mode retain the old behavior.

## Why this small mechanism

The saved CP221568 P02 terminates at22.433s while FR has advanced16.126mm over3s (7.221mm last1s), clear/AIR with measured supports, no physical safety terminal and stall=false. It is10.346mm short of the unchanged accepted approach band. Source/controller/wheel masks are not missing. The goal is to avoid a deadline interrupting useful approach, not to award completion or enlarge residuals.

This candidate earns a small distance-denominated credit only from **new geometric best progress** toward the existing approach boundary. Credit drains with physical time and is capped. Repeated back-and-forth over already visited geometry, wheel spin alone, phase progress changes from lifting, or a single old lift event cannot refill it.

## State, units, update

Let r=max(0, existing_accepted_approach_lower_bound-current_FR_front_distance), in metres. The existing bound is approach_min_m−xy_measurement_tolerance_m (currently−0.010m), not a new acceptance threshold.

Two persistent values within P02:
1. best_remaining_m = smallest measured r so far.
2. credit_m = nonnegative opportunity balance.

Each120Hz tick:
- new_progress=max(0,old_best−r); best=min(old_best,r).
- Only eligible current physical motion earns new_progress.
- credit=clip(credit + earned_progress − drain_m_s*dt,0,capacity_m).
- At the unchanged local deadline, unfinished P02 can continue only if current eligibility AND credit>0; clocks, nominal, policy and physical stepping continue.
- Never reset best because of a backwards excursion, temporary ground, empty credit, or a failed earning tick. New episode/P02 exit resets only this ledger, not action/HISTORY/GAE.
- Track the geometric best on all valid P02 measurements even when ineligible: later becoming eligible cannot retroactively claim unsafe/ground motion. Invalid measurements must not be fed as geometry; existing evaluator invalid/safety termination has priority.

Candidate capacity is0.0075m, drain0.00125m/s, derived from the current P02 stall contract:
existing minimum_potential_change0.01 ×3 equally averaged completion predicates ×existing approach shaping distance0.25m =7.5mm; divided by existing6s stall window gives1.25mm/s. This keeps the previous diagnostic's physical progress scale explicit. The literal0.25 currently lives in approach predicate code; integration should centralize or explicitly version this dependency rather than silently drift.

This is NOT “add6s at timeout”: a stalled candidate receives no new credit, cannot renew it, and an earned full bucket lasts at most6s without new best progress. New useful forward motion can replenish it; arbitrarily slow/noisy drift below the drain cannot sustain it. At the recorded terminal the total remaining possible new distance is10.346mm, so even with full7.5mm credit the kernel alone has at most(7.5+10.346)/1.25≈14.28s of further opportunity before either reaching the unchanged boundary or exhausting credit. The200s global cap remains authoritative.

## Eligibility uses real current evidence, not wheel animation

Proposed integration gate (root review required, no invented force/dwell/angle thresholds):
- P02, entry valid, evaluator valid, no evaluator/independent safety abort.
- Existing qualified measured FR lift event from `history.active_lift.FR` / `history.event_ticks.active_lift.FR`.
- Current FR AIR, no ground contact, current active attempt, within lateral span, and existing `clear_FR` predicate=1.
- Before crossing/placement, the existing evaluator revokes active_lift on real ground and restores it only after fresh measured qualification; its event_ticks intentionally keeps the FIRST event. Use the current active_lift+active_attempt+AIR here, so a real re-lift is not incorrectly rejected due to an old first-event timestamp. After crossing/placement, that historical bit can survive ground, so require the recorded first event to lie in the current uninterrupted AIR span; otherwise fail closed for P02 credit. This is not a new FR latch or a claim that stale history proves current lift. The separate current_eligible slot exposes this exact permission.
- Count current other-leg `support` values from verified evaluator contact evidence against the existing `support.minimum_other_supports` (currently2). AIR legs do not count; no new force threshold or static support-margin test.
- New reward/credit is geometric FR approach progress, not wheel angular velocity or body roll. Existing body/safety rules unchanged.

Normal physical completion and the one-forward-transition-per8-ticks scheduler run first. If all existing P02 goals become complete on a nondecision physics tick, permit only the already-pending next decision-boundary handoff (at most7 ticks), not a new long recovery, so real TOP/capture is not rejected merely because it is no longer AIR. This pending-completion case is directly observable through phase_progress=1, stage/time and existing valid completion, not an extra latch. Add a real supervisor regression for that boundary.

## Observation and migration

Append exactly3 fields after current419, with actor and critic identical:
- index419: `p02_best_remaining_m`, metres, fixed scale1.
- index420: `p02_progress_credit_fraction`, credit_m/capacity_m, [0,1].
- index421: `p02_current_continuation_eligible`, exact current physical permission boolean (not proxy contact flags).

All three are0 outside P02. The third slot matters because exact same-AIR qualified event/current sensor permission cannot be reconstructed exactly from old419 historical lift bits and contact proxies. Logging it but hiding it from the network would not satisfy observability. Current r, phase age, total age, stage/progress and safety remain in existing observations. Two ledger values plus current permission make the control decision visible; this does not claim the complete robot/contact process becomes Markov.

Expand first actor/critic input matrices with zero columns, preserve all419 old columns, hidden layers/outputs, sigma, actual LR and compatible full Adam moments; zero only new moment columns. Identity normalizer/fixed scales remain. Preserve latest sealed CP and lineage, reset no mean head, do not switch front/rear models. Clear unfinished rollout at the legal boundary and recollect under422. No update count is earned by migration. Same control in training/DET; no P02-only eval hack. Migration implementation belongs to root/migration agent, not this draft.

## Narrow production integration locations

- `semantic_supervisor.py:TaskStageSupervisor.__init__`: gated ledger state only for new mode; validate mode/config with load_task_spec.
- Current `update`: after real evaluation/normal completion scheduling and before local-deadline branch, observe current physics geometry once; retain physical/global terminal priority. Existing nominal local allowance formula can remain; this is an additional explicit earned-opportunity permission at its exhaustion.
- Expose `task['p02_progress_credit']` snapshot with the three short codec keys `best_remaining_m`, `credit_fraction`, `current_eligible`, plus mode, exact lower bound/r/best/credit, new progress, drain/cap units, eligibility reasons and whether legacy deadline was suppressed. Read this task snapshot directly; no backend parallel info copy is required. All clocks and source owners unchanged.
- Observation/schema/actor loader/policy contract/migration must explicitly accept422 and keep parent419 slices; do not bypass old validators or reuse an unrelated rear timing slot.
- No reward, actuator, wheel mask, geometry, nominal or physics changes are required for this deadline fix.

## Tests and evidence status

Executed pure draft unittest with base Python:12 passed in0.017s. Initial two failures were exact-float assertions (3e−18 /0.030000000000000002), corrected to tolerance; physical counterexamples retained. Subsequently all production.patch hunks were applied to an in-memory module (not the source file) and14 real-TaskEvaluator integration cases passed, in about2.7s including process startup. A meta-path guard forbids Torch/Isaac imports. No model or simulator was run.

The staged supervisor implementation derives dt from actual simulation_time_s; initial tick0 gives zero dt/credit; repeated same-tick reads are idempotent and inconsistent same-tick clocks are rejected. The enclosing TaskEvaluator retains its existing contiguous120Hz requirement. Integration passed current AIR/lift/clear/support, old crossed-event→GROUND→AIR rejection, missing sensor, unilateral support, lateral/clearance failures, global/body safety priority, unchanged P03 approach boundary, and real TOP completion waiting only for the next decision tick even with empty credit. These tests are synthetic sensor semantics, not proof of real stability or completion.

Passed kernel cases: genuine new forward progress; bounded stall expiry; repeated-old-space oscillation; no-progress wheel/AIR; backwards motion; subfloor noise; ineligible current contact/support/clearance input; no retroactive unsafe credit; global/safety override; no phase/contact award; phase-local state reset; public state/nonfinite rejection.

Still REQUIRED after actual production application (the candidate in-memory tests already cover the supervisor items below, but actual installed-code/422 wiring must be rerun):
- Actual TaskEvaluator positive current AIR/lift/clear/support versus ground→AIR with stale historical event, missing contact validity, AIR other leg, lost clearance/lateral and actual safety abort.
- Credit permits the recorded valid cutoff case but old/no-mode profile still ends at its old budget.
- Repeated ground/re-lift/position oscillation cannot renew historical credit; safe fresh physical episode can earn it normally.
- P03 exact acceptance threshold unchanged; valid physical completion immediately before a decision boundary proceeds, not cut off.
- At global200s stop regardless credit; normal stage transition no done/GAE reset and no actuator/HISTORY reset.
- Actual actor/critic422 encode all3 receipt values with exact old419 prefix; migrated network conditional Gaussian preserved at zero appended fields; optimizer/normalizer/fresh-rollout contract enforced.
- NaturalP01 fixed-checkpoint DET/real training must establish later outcome; saved-input replay does not prove new physics.

Saved CP221568 offline ledger replay (not full evaluator re-execution; real eventt24 and recorded current AIR/gap/contacts, lateral validity from existing task snapshots) gives terminal credit7.5mm, eligible=true, best_remaining10.345625mm. It demonstrates the draft would not discard this measured opportunity; it cannot predict post-terminal success. No continuation or controller repair is claimed to have been learned by the policy.
