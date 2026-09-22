# Capture feedback revision: read-only consumer compatibility check

Scope: proposed same-389 HOLD→AIR local progress-window correction, plus a nonnumeric snapshot-level `feedback_revision`. Current runtime inspected: `0001c3138b0b`. No source/config/script edits and no Isaac execution. The correction itself remains the capture agent's work; migration/optimizer preservation remains the migration agent's work.

## Result

**No additional production consumer edit is required merely for the additive snapshot string or the local HOLD→AIR window reset**, provided the revision is emitted consistently by the same `HipOnlyCaptureAssist` class and stays outside its 12 numeric feature names. There is one concrete existing workflow restriction: a frozen checkpoint prefix must start from the **saved/reloaded new-runtime checkpoint**, not directly from old-runtime provenance relabeled with a new runtime hash.

| Consumer | Actual behavior | Compatibility consequence |
|---|---|---|
| `semantic_capture_assist.py:33–37` / observation encoder `semantic_observation.py:384–396` | Encodes only the explicit 12 `CAPTURE_ASSIST_FEATURE_NAMES`, followed by five explicit continuation features. | Snapshot metadata is not converted to float and does not extend 389 inputs. Existing window-baseline/elapsed changes remain observable numeric state. |
| `semantic_residual_adapter.py:226–260` | Uses `advance`, the returned snapshot transform and numeric ownership fields, then existing final clamp/slew and one native dispatch. | New string metadata passes through the receipt. No adapter-key whitelist rejects it; it cannot become a wheel mask or physical state write. |
| `actuator_target_effect.py:175–208` | Rebuilds only the 12 numeric state values into a fresh `HipOnlyCaptureAssist`, reruns `advance` on the recorded same pre-dispatch feedback, compares the **full** `state_after` JSON, and reconstructs target/owner effects. | With dispatch/replay on the same revised class, the new fixed metadata and rebound-window transition agree automatically. Keep exact comparison; do not strip revision from it to accept mixed-version receipts. |
| `semantic_backend.py:398–400` | Emits `self._capture_assist.snapshot()` and the complete receipt. The assist is constructed/reset as a runtime object, not loaded from optimizer state. | Additive metadata is automatically logged. No backend schema edit is necessary. |
| `semantic_video.py:446–467` | Copies the complete assist snapshot and dispatch receipt to JSONL; does not numerically flatten them. | New string survives unchanged. Historical videos/logs do not need rewriting. |
| Checkpoint-policy prefix core | Steps the same ordinary backend/observation/audit path; bootstrap binds numeric observations/history/task state; handoff opens credit without resetting the physical core. | Same correction applies to prefix and learner, with no special capture snapshot import or reset at the credit seam. Prefix samples remain uncredited. |

A pure in-memory check against the existing code confirmed that adding `feedback_revision` leaves the 12 numeric features and inactive transform exactly unchanged and round-trips through JSON. This is a wiring check, not proof that the corrected rebound controller has been deployed or physically validated.

## Concrete prefix binding restriction

`semantic_checkpoint_prefix_policy.py:124–132` already requires P05/389 prefix provenance to have:

- equal source/effective/target policy contracts;
- `effective_runtime_content_sha256 == source_runtime_content_sha256`;
- an actual saved 389 source checkpoint.

Therefore, applying a controller-runtime migration and immediately trying to install an old-checkpoint/new-runtime prefix binding fails with `P05 prefix requires an exact saved389 checkpoint/runtime, not an implicit remapping`. The intended save/reload-migrated-checkpoint boundary satisfies the existing contract; **no prefix validation relaxation is needed**. This finding was sent to the migration agent. It is separate from the numeric snapshot compatibility question.

The full snapshot JSON equality also deliberately prevents an old receipt without the revision from being silently validated as a new-controller receipt. Preserve old runtime/source artifacts for old-run provenance; do not rewrite old results to make them pass the new revision.

## Physical/task semantics

The proposed transition can be derived from the already-observable prior mode HOLD plus current absence of both TOP-surface and obstacle-pair contact. Reinitializing only the current progress-window baseline/elapsed does not create a hidden mode bit or require a new policy input.

Capture context reads real evaluator history/current measurements. The assist transforms only FL hip/knee actuator targets and has no setter for `placed_FL`, TOP contact, load, task completion, stage success or physical state. P05/P06 task predicates remain supervisor/evaluator-owned. A reset of this local progress baseline therefore cannot award placement or pretend a suspended FL bears load.

For the proposed bounded fix, the capture agent must retain cumulative hip entry/target, fixed knee anchor, finite total travel, mechanical margin, actual tracking and physical/XY/support checks; ordinary AIR ticks, BLOCKED ticks and phase handoffs must not repeatedly restart the window. Those are controller-change correctness conditions, not additional downstream consumer edits. P07 release and the existing nominal/history/rollout credit boundaries remain unchanged.

## Limits of this check

This verifies consumer shapes, metadata handling and existing provenance constraints. It does not certify Adam/RNG/Identity-normalizer byte preservation in a migration that has not yet been executed. No `NewMdpWarmStart` is required by these snapshot consumers; use the separately reviewed same-389 explicit controller migration and freshly collected post-boundary rollouts.
