# UNAPPLIED — pending late tracking ownership correction

Status: outputs-only candidate. Production/runtime/checkpoint/reward/observation/optimizer are untouched; current Isaac rollout continues unchanged. Apply only at the parent-selected complete update/version boundary.

## Confirmed defect

The original source sample647 (held while late648 waits) declares no servo tracking. The full12 late648 declares all8servo tracking, even though FR/RR goal values do not change. The current wrapper suppresses FL/RL names but accidentally inherits FR/RR names from that unconsumed event; actual existing FR capture ownership removes FR, leaving RR hip/knee newly tracked. Normal wheel-stop864 later ends this transient permission. This is unintended tracking activation, **not** loss of an old continuously tracked pending RR, not a residual mask, and not proof of a single physical failure cause.

## Minimal patch

- Save previous_sample.tracking_servo_names at wrapper creation, alongside existing original logical source values.
- Emit exactly that preceding source tracking responsibility while the late event remains unconsumed. In the actual selected contract it is empty, before and after the separate wheel-only stop.
- Keep the carrier clock, pending event, targets, stop group timing/ownership, P12 guard and all12 policy channels unchanged. Do not write FINAL into nominal, reset HISTORY, extend tracking, or add a descent target.
- Receipt adds the retained tracking names and their source. Unchanged independent P10/P11 source layers still own their own actions. This change is not a global servo-tracking disable.

Files: tracking_owner.patch is a minimal production-file diff (checked, NOT applied); semantic_rr_capture_deferred_late.py is an UNAPPLIED runnable candidate for isolated stdlib tests. Its header warning is deliberately not part of the production patch.

## Verification

Python313 test_tracking_owner_fix.py: **6/6 PASS**. Real Recording/MotionExecutor cases cover sample647 empty vs late648 full tracking, no new tracking throughout500 carrier ticks, stop864 exactly once with original wheel owners, preservation of hypothetical already-legal preceding responsibility, untouched independent P10 RRk/P11 FRhip action sources, and no added policy/HISTORY/actuator/learning APIs. git apply --check passed; no patch was applied.

These are wiring tests, not a floating-body success test. No Torch/PXR/Isaac import or model operation was performed. Candidate module SHA256: 97bf67f2d25696df0c65f9ac0aa9d88875a193ff0f9e321a5401e35e90b6bd9b.

Physical evidence is in ../v2_episode1_tracking_owner_followup.md/json and the10-row ../v2_episode1_min_gap_source_response_readonly.md/json. The close-to-contact gap rebound had fixed N/mappedN; retain that distinction. Parent owns later same448 state/Adam-compatible version publication and fresh collection; this candidate performs no migration.
