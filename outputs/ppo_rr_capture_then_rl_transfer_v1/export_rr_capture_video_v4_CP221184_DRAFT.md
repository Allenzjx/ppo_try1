# CP221184 / v4 export draft — not run

Only output text was authored. No Python/helper/test/FFmpeg/Isaac process was started and no live trace was scanned. The prior 26db exporter and every existing movie remain unchanged. The draft is pinned to the already-published migrated CP221184 SHA `2fed192f…672f90`, HEAD `632295bc…c3ff`, and this one new source run. It must be reviewed and tested after the sole video process exits; it is **not yet a verified exporter or a delivered video**.

The new script imports the hash-pinned older exporter to retain its run-sealing, artifact hashes, complete native-frame validation, full decode, normal-speed encoding, fixed canvas, detail trim and visibly frozen historical-N comparison. Only the pair's candidate-label AST assignment is replaced; no original file is changed. Checkpoint provenance validates the immutable actual v4 plan/publication and carried ancestors/AUX. It states real prior learning **640 decisions / 5 PPO updates / 100 Adam steps**, with **zero** migration and evaluation optimization.

The source writer retains the static field `rr_capture_assist.first_revision_wheel_guidance='off'`. That is explicitly preserved as a legacy first-revision declaration, **not** asserted to be the effective v4 mode. Effective support-wheel v4 comes from checkpoint/runtime migration plus every actual independently verified native wheel receipt. The script does not edit or relabel the original source manifest.

## Existing evidence locations

- `capture_assist_ticks.jsonl`: every episode physics tick; measured canonical wheel speeds, leg contact/history, body/CoM, actual final canonical drive and servo-assist state. Its `dispatch` whitelist does **not** include the new wheel receipt or native joint IDs.
- That whitelist also requests `independent_policy_residual_effective_full12`, but the current adapter does not produce that top-level key: do not slice it or interpret `null` as zero. Read the actual effective pre-wheel-projection residual from `native_audit.policy_headroom_evidence.effective_policy_residual_full12`.
- `native_tick_audit.jsonl` → `native_audit.rr_carry_wheel_evidence`: complete source proof, support/geometry context, original candidate, desired floor, previous FINAL, output, controller delta, envelope and change flags. `rr_carry_wheel_context_and_previous_FINAL_independently_verified` and `previous_final_drive_wheel_rad_s` provide the independent pre-dispatch check. The same audit contains raw policy, residual permission mask, final native actuator targets and the same-state zero-current-policy native counterpart.
- Join those two streams using `episode_physics_tick`; retain the receipt's pre-observation and physical-dispatch clocks separately from post-step measured state. In **each JSON row of `native_tick_audit.jsonl`**, `row['nominal_full12']` is the actual before-frame source nominal. It is a **top-level sibling** of `row['native_audit']`, not `row['native_audit']['nominal_full12']` and not a separately recomputed N. This matches `PhysicalEvaluationRecorder`'s persisted writer and the script's `native_row.get('nominal_full12')`.
- The seven task bits in the resulting frame include next-state X409 eligibility for the following dispatch. It must **not** be forced equal to the preceding action receipt's envelope bit on transition ticks.
- Exact native actuator IDs are absent from these two persisted writer schemas: report `null`, not guessed IDs. Native target signs are checked using the existing canonical mapping. Measured canonical qdot remains distinct from native actuator target and wheel-end displacement.

Future outputs are one full ≤200 s continuous movie, one contiguous task-window-through-terminal detail (or clearly labelled predecessor tail if RR was never reached), and a historical-N comparison. Whether RR/RL were reached and whether the task succeeded are derived only after sealing. No success, fresh paired B, stability improvement or projection-caused traction is predeclared.

Static v3 compatibility check: the pinned old `validate_historical_rr_snapshot` checks schema, metadata, finite 14-scalar layout and ownership/clock types; it contains **no** travel ≤20°, total elapsed ≤12 s, or fixed `knee_hold_deg` check. Its isolated revision constant is set to the v3 revision before use. The adapter then explicitly permits combined travel ≤40° and elapsed ≤32 s, with derived hip exposure ≤12 s and the v3 dynamic-knee search metadata. A never-initialized WAIT state with travel/elapsed 0 is valid after a physical dispatch. Neither capture-row collection nor detail selection requires any RR milestone: a genuine P05-only failure exports its full episode and a clearly labelled predecessor terminal tail, not a fabricated RR attempt.

After sealing and review, the intended command is:

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_rr_capture_then_rl_transfer_v1/export_rr_capture_video_v4_CP221184.py --destination outputs/ppo_rr_capture_then_rl_transfer_v1/video_review/CP221184_v4_deterministic
```

The destination must not already exist. Do not run this command while the current simulator/viewport process is active.
