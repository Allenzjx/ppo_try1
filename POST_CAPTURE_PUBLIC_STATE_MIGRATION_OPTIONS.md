# Post-capture preparation / paused RL source lane: read-only design check

No implementation, dimensions, targets, migration source counters, or new training credit are selected here. The active suffix block and its stop request must seal first. Its first completed episode is not a full PPO success, and the next partial prefix earns no policy credit.

## What the existing code actually exposes

- `semantic_supervisor._continuous_advisory` already keeps source layers with `motion`, `ticks`, `sample`, `last`, `touched` and `advanced_this_tick`. A disallowed layer reuses its last sample without ticking its source motion. Later channel owners still take precedence. `_sequence_permission` currently specializes P07–P09, not a public RR-bearing-conditioned P10–P12/RL preparation lane.
- Some source ticks are present in nominal diagnostics, but the 410 actor codec does **not** encode a separate paused RL source cursor/owner. Existing `task_times` are actual time, remaining task time and stage elapsed time; they are not equivalent to independently consumed lane time. Current/previous nominal and mapper history do not uniquely recover a paused source cursor or its pending owner.
- RR fields389–402 expose14 assist scalars;403–409 expose seven current task booleans. RR current bearing, reachable workspace and historical placement remain distinct. Reusing these slots for new lane state would silently change their meanings and is not an append migration.

## Smallest defensible state contract, to finalize after control design

Keep public task phase and global time running normally. Pause only the specific RL preparation/source lane when its current physical permission is absent; do not reset the phase, consume pending source events while paused, or catch up skipped commands on resume.

Any new persistent variable that changes the next target must be observable. Likely categories are lane/preparation mode, actual consumed source cursor, and any persistent ownership selector. If the proposed controller also keeps a latched entry target, retained target, finite pause/recovery budget or progress reference that cannot be reconstructed from the existing410 input, that value must also be public. If event index, mode or ownership is exactly derivable from a public cursor and immutable source plan, do not duplicate hidden state unnecessarily. **No feature count is fixed yet.**

Permission can use existing measured RR TOP/bearing plus other real support and the current preparation state, not the `placed` latch alone. Loss of current support may pause or change the declared lane, but does not erase legitimate lift/contact history or fabricate bearing. If a finite pause/exposure clock is introduced, expose its controlling elapsed/budget state; the original global task deadline must keep advancing.

Pausing a servo/source lane is not a whole-body stop. Preserve unrelated wheel/leg owners, source stop ownership, all residual channels, mapper/HISTORY/filter continuity and the single physical dispatch. Do not blanket-zero wheels, restore entry joints, or count source-time permission as independent actuator ACK. Actual dispatcher audit and same-tick counterfactual must consume the same public lane snapshot. Required pre/post state and cursor advancement should be tied to the authoritative physical/dispatch sequence so repeated reads do not advance it twice.

## Existing compatible expansion machinery

`semantic_rr_capture_migration.zero_append_rr_training_state` is a working **specific389→410** implementation, not a generic arbitrary-width loader. It deep-copies the full bundle, verifies six named MLP parameters per actor/critic, copies the first matrix, appends zero columns, and zero-appends only first-layer Adam `exp_avg`/`exp_avg_sq` columns. It preserves scalar Adam steps, parameter groups, actual LR, all other actor/critic parameters, iteration, Identity and full training RNG. Original actor/critic first-layer Adam IDs0/6 are checked against the complete official parameter ordering, not assumed blindly.

`load_rr_capture_migration` loads the old source runner, restores the mapped full optimizer and RNG into an empty target storage, then official save/fresh reload is available in `publish_rr_capture_checkpoint`. This is the correct mechanical pattern for a **new410→410+K** factor once K and semantics are fixed. The old389 function and its old migration receipt must not be repurposed, and the current same410 identity validator cannot accept a wider actor through a waiver.

Preserve all410 positions/scales and append only the declared new state. Use an explicit new layout/profile/policy-contract identity and strict codec fields, keeping the existing410 profile loadable. Current `SemanticRRCaptureHistoryMLPModel` validates exact410; changing its global dimension in place would invalidate historical loading. A new registered appended-input actor can retain the same raw12 Gaussian head, old389 HISTORY-center and old372 state-dependent sigma kernel. No zeroed mean head, new optimizer, warm-start reset or copied physical state is needed.

A new factor must bind the **actually latest saved checkpoint on the existing ancestor output branch after sealing**, preserve all earlier origins/AUX/v2–v8 receipts and its real cumulative counters, and keep main latest640 separate. The source learned410 first-layer columns and old Adam columns remain byte-exact; only new columns/moments are zero. Full-state hashes naturally change because matrix shapes change. At zero learning, raw mean/std and critic output must agree for the same old410 prefix plus varied new public states within an explicit numerical tolerance; do not claim an unchanged closed-loop trajectory when the controller changes.

## Necessary narrow integration and verification

New observation schema/encoder and actor/profile/distribution registration; authoritative backend lane state; the actual nominal/controller change; training load/save/request-audit registration; prefix dimension/provenance support; and explicit branch routing for the new formal receipt/dimension. Existing CLI branch guards require410 and a current v6/v7/v8 receipt, so they must be extended narrowly rather than accepting arbitrary dimensions or falling back to an old receipt.

Focused checks should cover: (1) same old state/different paused cursor produces explicit distinct inputs; pause/resume does not consume/drop source actions or freeze unrelated wheel owners, and current support loss is not hidden by `placed`; (2)410-column and full Adam/RNG preservation with zero new columns/moments, arbitrary appended-state raw Gaussian/value continuity, official save/fresh reload; (3) fresh real rollout only, ordinary phase changes nonterminal, prefix excluded, all state/receipt carried by a normal save. These tests establish contracts, not physical RL completion. No A/B five-success or pre-training deterministic-success gate is proposed.
