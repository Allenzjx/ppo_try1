# Same448 source-owner repair: cold rebind draft

Scope: outputs-only patch preparation; no production application, Torch import,
checkpoint load, optimizer update or Isaac execution has been performed here.
The 14 standard-library contract/patch tests passed against live 5f8487b source.

## Allowed change

Source HEAD is exactly 5f8487b76f27eb165a329a0b6f3096f54ebb4d48.
The checkpoint must be v2/448 at an empty-rollout complete update. Rebind uses
the latest valid complete checkpoint, not a fixed old decision count.

Exactly these tracked file hashes may differ in the new runtime:

- src/wlr50_clean/ppo/semantic_rr_capture_deferred_late.py
- src/wlr50_clean/ppo/semantic_rr_capture_local.py
- configs/ppo_rr_capture_first_cp225280_v1/local_training.json

The local training JSON may only add
source_tracking_owner_revision = pending_source_tracking_inheritance_v1.
Task/layout/actor/distribution, MODE, source prior, all control configuration,
HISTORY, physics and library versions remain identical. Tests/docs must stay
outputs-only for this precise three-file repair.

Apply the staged route patch only after the live episode/rollout naturally
finishes and saves its complete update. Integrate the independent source-owner
repair and one config field in the same cold version. Commit first; the route's
existing clean-tree/runtime contract still applies. Do not apply this patch
while Isaac is running.

## State preservation

The ordinary load remains exact-runtime only. The explicit rebind route checks
the old/new allowed-difference contract, provided checkpoint AND sidecar SHA,
embedded info equality, actual tensor/Adam hashes, complete-update boundary,
actual effective LR and exact runner configuration. It then invokes the existing
strict load with the checkpoint's real old runtime.

The existing resume runner necessarily constructs its model/optimizer objects
before loading state; rebind neither constructs another Adam nor resets it.
It verifies that the optimizer object and live parameter ordering remain
unchanged by load, and verifies all loaded learned/moment/step hashes. The
default config LR must not overwrite the actual checkpoint LR. Identity
normalizers, full Python/NumPy/Torch CPU/CUDA RNG and the original 447-to-448
local_migration receipt are preserved.

A separate local_control_rebinds receipt is appended with source/destination
hashes, exactly preserved counts, state hashes and zero new learning credit.
New publication has the same CP counts and a new HEAD suffix. Old CPs stay intact.
No mean scaling, optimizer migration, AUX or PPO update occurs in rebind.

## Tests

Safe now (stdlib only; does not import production modules or Torch):

    & 'C:\Program Files\Python313\python.exe' outputs\ppo_rr_capture_first_cp225280_v1\staged_same448_rebind\test_same448_rebind_stdlib.py

Deferred until parent explicitly confirms the normal safe boundary:

    & 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs\ppo_rr_capture_first_cp225280_v1\staged_same448_rebind\test_same448_rebind_cpu_draft.py --isaac-stopped

The six CPU tensor tests use a temporary, explicitly non-publishable fixture
copied from the actual sealed state with only its runner device metadata
self-consistently changed to CPU. They exercise the route's real strict loader
and learned state. Saved CUDA RNG is also restored/checked if present; therefore
they must not run alongside Isaac. This is not a physical validation and the
fixture is not a new compatible deployment checkpoint.

The stdlib staged-source hunk test is meant to run BEFORE applying the patch.
After integration, rerunning that one pre-application test will correctly fail
because the patch no longer uniquely applies; the pure contract cases remain
valid. CPU tests read the new functions from the staged patch and use the actual
production loader, so can run before or after integration at the cold boundary.

## Cold publish command

After source/config fix is committed, read the latest complete pointer once.
Use its actual CP and both SHA values; do not paste the old local2048 example.
The published runtime must be the actual committed repair HEAD.

    python -m wlr50_clean.ppo.semantic_rr_capture_local rebind --expected-head <NEW_REPAIR_HEAD> --run-dir runs/ppo_rr_capture_first_cp225280_v1/rebind_same448_<UNIQUE> --checkpoint <LATEST_COMPLETE_CP> --checkpoint-sha256 <POINTER_CP_SHA> --manifest-sha256 <POINTER_MANIFEST_SHA>

Run in the existing Isaac environment and the same device saved in runner_config
(default cuda:0). Rebind launches no Isaac application. It creates a new sealed
checkpoint at the same decision/update count with the new HEAD suffix. Inspect
its manifest and pointer, then start only a fresh compatible rollout.

Follow-up video tools must explicitly understand the new
control_contributions.source_tracking_owner_revision before formal export.
This task adds the route manifest field only; video export tools are untouched.

