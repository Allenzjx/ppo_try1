# P09 current-TOP confirmation and next-decision handoff — proposal only

Reviewed against runtime HEAD `e24a3c2630b0b95439a7a71fe6a8a3f610a9a385`. Nothing in `src`, `configs`, or `tests` was edited. No Isaac, actor forward, checkpoint load/update, or live-rollout inspection was performed for this proposal. The active real training block must finish or stop at a verified complete-update boundary before any implementation/migration.

## Confirmed defect and scope

`TaskStageSupervisor.observe_and_update()` first evaluates current physical measurements, then performs at most one phase transition on ticks divisible by 8, and only afterwards applies deadlines. Its RR local exception currently requires `rr_capture_transfer_context()["rr_capture_recovery_allowed"]`, which means a reachable state plus assist mode `DESCEND`. A real first contact normally changes the next committed assist state to `HOLD`, so the exception disappears exactly while the evaluator is accumulating the two required TOP samples or awaiting the next normal decision tick.

The sealed synthetic counterexample report establishes this failure for actual placement at modulo-8 ticks 1, 3, and 7, and for the first qualified TOP after a weak obstacle pair. It also shows that contact at a decision tick, or contact before the local deadline, does hand off correctly. These are real production-function synthetic counterexamples, not the cause of the current Isaac run's failure and not new physical success evidence.

Do not change the evaluator's placement predicate, `minimum_top_samples`, source action clock, RR assist trajectories, action ownership, final writer, phase-transition cadence, PPO terminal/GAE code, or safety/global-200-second precedence to fix this scheduling defect.

## Minimal two-part exception

Enable only the new RR experiment with an explicit task-spec mode. Historical N/FSM specifications omit the mode and retain old behavior. Both backend observation construction and supervisor deadline construction must call the same context helper with the same immutable configured window.

Common evidence for either new exception:

- Current valid evaluator and no physical/task termination.
- Current sensor-confirmed RR TOP, verified finite bearing force at or above the existing 0.2 N floor, AIR=false, ground=false, obstacle pair active, current legal TOP XY/lateral span.
- Current functional RR qualification plus earned crossing history; neither old placed nor high wheel position substitutes for this.
- At least two *other* currently verified supports, using the existing force/noise predicate.
- A validated, initialized, non-retired assist snapshot that currently owns RR indices 6/7 in `DESCEND`, `HOLD`, or `BLOCKED`. `BLOCKED` is needed when the final bounded descent step exhausts its search budget on the same physics step that first lands; this exception permits HOLD/confirmation, not another descent step. WAIT/RELEASE/RELEASED do not qualify.
- Supervisor additionally requires the existing fresh committed-feedback envelope. Native dispatch ticks are not the same clock as episode observation ticks; do not compare them directly. The existing envelope already records both.

1. **Contact confirmation:** allow an unfinished real TOP capture while existing cumulative public `hold_elapsed_s <= (minimum_top_samples + physics_hz/decision_hz)/physics_hz`. Current values are `(2+8)/120 = 0.0833333333 s`. Do not reset this clock on contact toggles or phase changes. A long prior weak/edge contact can consume this budget; the exception does not renew it.
2. **Completed handoff pending:** if real `history.placed.RR` has been earned, entry and the original completion goals are currently valid, and current measured TOP/bearing still satisfy the common evidence, allow only non-decision ticks of P09 to reach its next normal modulo-8 transition. This narrow completed-case exception is not capped by the already-spent confirmation clock: a fully earned current placement should not be discarded because earlier weak contact spent 83 ms. It contains no latch or new timer. With continuous evidence it lasts at most 7 physical ticks; at the next modulo-8 tick the existing transition executes first. Losing current TOP/bearing immediately removes it. AIR with old placed and even `placed_RR==1` never qualifies.

The second exception is intentionally distinct from “all goals == 1”: the old-placed-AIR counterexample has that value. It is also distinct from granting unlimited HOLD after a deadline. If contact disappears on a decision tick, no completed-contact exception remains; any continued AIR descent must independently satisfy its already-finite real search rules, or the episode terminates incomplete. Global 200 s and physical failures always retain priority.

## Proposed production touch points

`RR_TOP_handoff_candidate.patch.txt` contains a reviewable **unapplied candidate**, not a release-ready migration:

- `semantic_rr_capture_context.py`: opt-in window validation/calculation; one shared pure predicate; three extra diagnostic fields; broaden existing `rr_capture_recovery_allowed` to existing descent OR the two new current-contact cases.
- `semantic_supervisor.py`: validate/store the immutable configured window; supply stage/entry/completion facts to the context. Existing fresh-feedback and local/global timeout order stay unchanged.
- `semantic_backend.py`: calculate the identical configured window and pass it to observation construction. No actuation code is changed.
- new-experiment `stage_task_spec.yaml`: explicit opt-in only. Original N/FSM configs unchanged.

No changes to assist `advance()` or `hold_elapsed_s` are required. Do not change source P10/P11 behavior as part of this narrow patch. Whether those later actions preserve actual captured support is a separate physical test.

## Observation and migration

Existing 410 dimensions suffice without hidden mutable state:

- Assist positions 389–402 already contain mode, initialization, cumulative `hold_elapsed_s` (index 398), retired flag, etc.
- Task positions 403–409 already contain current TOP (405), current bearing (406), and `rr_capture_recovery_allowed` (408).
- Phase, entry-related physical state, completion progress, Q/cross/placed histories and timing exist in the legacy prefix. The resulting exception permission is explicitly exposed through bit 408, not hidden behind these inputs.

This is still an **MDP/terminal and one-boolean semantic change**, not numerical trajectory equivalence. Bit 408 formerly meant live descent only; it now also means narrowly bounded current-TOP confirmation/handoff. Scalar order/scales, 410x12 network shape, raw Gaussian sample/log likelihood, HISTORY, projection, action transform and assist state shape remain unchanged. No architecture, optimizer, learning-rate, normalizer or mean-head reset is needed. Preserve all exact model/Adam/LR/normalizer/RNG and prior AUX/migration/count metadata; discard old incomplete rollout and collect fresh data under a frozen new runtime.

The existing knee-v3 migration allowlist does **not** authorize this change: it intentionally forbids task-spec and backend/context changes, and its source hash is the zero-update 26db publication. Add a narrowly scoped, separately named same-410 migration/receipt from the *actual latest complete learned checkpoint* after the current block, with selected config hashes and source/runtime hashes. Do not modify its prior receipt or rerun that old migration. Publishing a semantically migrated checkpoint grants zero new decisions/updates. Runtime profile revision and factor/receipt/load/save binding must describe this new handoff mode. The candidate patch does not implement that migration and must not be launched alone.

## Minimal directed checks

`test_rr_top_handoff_proposal_draft.py` is an unexecuted test draft targeting the candidate API; it is expected not to pass on the unchanged e24a production code. After implementation, use these plus existing finite-search/safety/observation tests, not a new broad release gate:

- Expired P09 captures at modulo8 1/3/7 survive real TOP count1→2 and continue to P10 on the next modulo8=0; no fabricated placement, no phase change before that tick.
- Weak pair alone gets no new waiver; its following first actual TOP gets finite confirmation, then real sample2 and handoff.
- A completed current TOP can wait <=7 ticks even if the old cumulative HOLD clock exceeds the confirmation budget; unfinished TOP cannot borrow that completed-case permission.
- Ground, AIR+historical placed, wall/EDGE, invalid/low/NaN bearing, insufficient other support, unqualified/cross-missing state, stale/missing feedback, wrong assist owner/released/retired state get no new waiver.
- The cumulative confirmation window is not renewed by contact toggles. Exactly-on-boundary allowance and first-over-boundary denial are checked.
- Global 200 s, body collision and all physical safety failures still win; phase changes do not become `done`; the helper does not mutate assist/evaluator/history.
- Backend and supervisor seven-bit contexts agree for the same current input; mask/log probability/HISTORY remain untouched; absent opt-in gives byte-equivalent old context behavior except additional diagnostics.

This corrects an evidence-backed scheduler race only. It cannot make the current airborne RR descend, fix the sustained FL policy reverse, or establish FR receiving support/RL transfer; do not report those effects without physical data.
