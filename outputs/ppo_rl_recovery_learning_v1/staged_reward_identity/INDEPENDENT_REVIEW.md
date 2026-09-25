# Independent bounded review — reward identity candidate

Scope: read-only review of `complete_reward_identity.apply_patch`, the v2
reward candidate, migration module, wiring, README, and proposed tests.  No
production file was changed and no Python, Torch, model, or Isaac process was
started.  The active CP226048 video was not read.

## Result

No blocking implementation defect was found in the reviewed boundary.

The reward change is local to decision-endpoint Phi.  It does not write
`task_progress_potential`, any 439 observation group, history, contact state,
or task completion.  Its scope requires historical `placed.RR`, excludes
placed/current-swing RL, validates the live evaluator, and reconstructs the
existing RR retention share using the same `.85 / 4 * .2` global weight,
XY/gap scale, legal AIR/TOP conditions, TOP-sample fraction, and force floor.
The applied replacement is exactly

`delta_phi = (.85 / 4 * .2) * (new_retention - old_retention)`.

Both endpoints receive that replacement once per policy decision, so the
reward difference is `potential_weight * (gamma*delta_after - delta_before)`;
it is not repeated per 120 Hz sample.  Terminal Phi is zeroed before reading a
possibly invalid current endpoint.  The v2 tests already cover input
immutability, unchanged 439 encoding, legal AIR/TOP identity, no false contact
from force/history, terminal handling, and one-versus-eight-tick invariance.

One semantic boundary should be explicitly accepted: `old_legal` includes
legal AIR and geometric TOP even when verified bearing is absent.  In those
states `new == old`, so this revision deliberately does **not** change the
reward for a bearing loss that remains in the legal AIR/TOP region.  The new
dense preparation exists only after the placed RR falls outside that old legal
region (ground/outside/too-low geometry).  This matches the candidate README;
if “drop load” was intended to mean every no-bearing TOP tick, the code does
not implement that broader rule.

The same439 migration is also internally consistent: it selects the actual
complete learned checkpoint by explicit checkpoint/sidecar hashes and matching
`last_update`; permits exactly six reviewed runtime paths; reconstructs and
revalidates the frozen f6d owner439 ancestry; preserves actor, critic, complete
Adam/options/steps, LR, Identity normalizers, RNG, route, counters, stage
usage, and prior receipts; clears old rollout/transition state; publishes
zero migration credit without pointer promotion; and allows ordinary
descendants to advance weights, counters, RNG, LR, and stage usage while the
receipt/route/kernel ancestry remains immutable.  Loader dispatch, namespace
validation, and normal-save carry are all present.

## Only the necessary remaining tests

1. Add one end-to-end endpoint-transition test for
   `TOP+bearing -> ground/outside -> closer -> TOP+bearing`, asserting the full
   reward difference equals `potential_weight*(gamma*delta_after-delta_before)`
   at every edge and that task/contact/history inputs remain unchanged.
2. Add the adjacent `ground/outside -> RL current swing` edge.  The correction
   intentionally becomes zero at swing entry, which contributes
   `-potential_weight*delta_before`; assert the chosen representative total Phi
   still reflects the intended handoff, or explicitly accept that bounded
   shaping drop.  A standalone “RL swing has delta zero” test does not exercise
   this transition.
3. At the legal boundary, run the already-prepared actual-CP226048 publication,
   independent reload, and one synthetic 128-row ordinary descendant test.
   These are sufficient for full-state identity/receipt carry; no broader new
   release gate is recommended.

This revision cannot affect or explain the active video's first RR lift:
before historical RR placement its delta is exactly zero.
