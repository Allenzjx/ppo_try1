# Bounded CPU regression receipt

Executed selected existing tests against the current worktree with `C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe`. No production/test files changed and no Isaac process launched. CUDA visibility was set to the empty string **inside this new Python process** before pytest imports; OMP/MKL threads were restricted to1. This did not alter the active video process or global environment. Pytest cache provider was disabled.

Successful receipt: `cpu_regression_20260914_worktree_import.xml`, timestamp `2026-09-14T00:40:14.285472-04:00`, execution6.739s, exit0,32 parameterized cases, no failures/errors/skips. These CPU/fixture checks do not replace live B0/C0 physical or media results.

| Existing module | Selected behavior and boundary |
|---|---|
| test_successful_fsm_derived_nominal.py | Identical input/history uses one N irrespective of policy role; unfinished predecessor remains concurrent without entry-vector restoration; source P03 logical correction retained |
| test_actuator_target_effect.py | Same-prestate counterfactual retains controller feedback, uses previous final drive, and does not advance mapper/write articulation a second time; these are target-interface tests, not actual tracking proof |
| test_semantic_transfer_roles.py | Dynamic two-contact transfer with AIR receiver allowed; receiver motion cannot invent CoM translation; static angle/instant low load insufficient; body collision and wheel-only failure cannot be masked by roles |
| test_semantic_observation_reward_env.py | Eight-physics-tick hold and Full12 request/mask; ordinary phase transition does not reset potential/create reward; transfer attitude discount; real terminal handling and no-bootstrap including HARD_JOINT_LIMIT signal; BODY_COLLISION stops repeat at actual terminal tick |
| test_semantic_video_fsm_reference_v2.py | A reader recipe and shared B/C config routing; real CPU saved HISTORY372 reload retains actor/critic/Adam/normalizer/RNG/budgets and deterministic action kernel; partial real terminal preserved without extra PRE/POST physics; incomplete run is not success; encoder failure reaches finally/finalize and keeps physical success distinct from invalid media |

Coverage limits: no exact existing selected test changes reward weights and then compares an otherwise frozen actor on identical observation; reload/action-kernel identity does not imply that intervention was tested. The hard-limit case here checks terminal propagation and reward/bootstrap handling, not a fresh measured-joint/PhysX limit experiment. Failure-media fixtures test capture/finalization, not browser playback or the new output-only exporter. Required live videos and full decoding remain separate root-owned work.

Initial harness attempt (`cpu_regression_20260914.xml`) failed collection with ModuleNotFoundError:wlr50_clean because direct pytest invocation did not put this src-layout worktree on sys.path. It ran no tests; the result is preserved. The successful rerun inserted `os.path.abspath('src')` at sys.path[0] before pytest. No package installation, environment upgrade or production fix was made.

Selected exact test functions (all within the five modules above):

- test_equal_inputs_and_history_share_one_N_independent_of_policy_role
- test_unfinished_predecessor_remains_concurrent_and_untouched_values_not_restored
- test_source_p03_logical_correction_is_reused_not_a_policy_cap
- test_counterfactual_retains_controller_drive_feedback
- test_counterfactual_uses_pre_dispatch_not_updated_final_drive
- test_audit_reuses_frozen_mapping_without_second_advance_apply_or_dispatch
- test_air_receiver_and_dynamic_two_contacts_allow_measured_rr_transfer
- test_receiver_motion_does_not_invent_com_translation
- test_static_candidate_angle_and_instant_lowload_do_not_complete_transfer
- test_roles_never_mask_real_task_failures
- test_actual_env_projector_reward_signature_and_eight_tick_action_hold
- test_phase_label_change_is_not_a_reward_event_or_potential_reset
- test_regularizer_can_be_disabled_and_transfer_attitude_is_less_restrictive
- test_failure_and_task_timeout_have_zero_potential_no_bootstrap
- test_env_valid_terminal_breaks_repeat_and_never_bootstraps
- test_current_A_reader_recipe_and_identical_BC_config_routing
- test_real_CPU_saved_HISTORY372_video_reload_retains_Adam_rng_budget_and_deterministic_kernel
- test_current_capture_keeps_real_P01_to_partial_terminal_without_PRE_or_POST
- test_done_only_local_timeout_retains_partial_terminal_and_is_never_success
- test_terminal_encoder_failure_preserves_physical_success_but_blocks_publication

Separately, output-only `rr_tracking_extract.py --validate-schema-only <historical video11 run>` passed on only the first physical/native rows plus the small manifest-bound observation calibration config. It rejects the known pre/post-step mismatch and exercises field mapping with an explicitly synthetic in-memory fixture; no B0/C0 data or evidence output was read/generated. Raw base quaternion Euler is labeled `base_chassis_rpy_derived_rad`; calibrated attitude uses the exact `q_world_body * fixed_chassis_to_body` convention only after SHA256 verifies the per-run recorded config. Missing calibration stays null, while a present mismatched config fails closed. Base origin height is explicitly named, not confused with CoM height. Tracking channel lookup validates exact servo names. No actual-extraction execution before finalized B0/C0 authorization.

## One added output-only regression closes the reward-only actor coverage gap

Root explicitly requested one focused fixture after the existing-test subset. New `test_reward_frozen_actor_invariant.py` is in this output directory, not production/tests. It uses the actual CPU RSL HISTORY372 actor through existing construction helpers, saves/reloads a **synthetic untrained fixture actor plus identity normalizer**, freezes parameters and supplies identical nonzero372 input (including the195:207 HISTORY slice). No project learned checkpoint is loaded, and there are zero physics ticks, PPO updates or optimizer steps.

Two separate RewardCalculator instances consume the same nonzero-attitude synthetic physical fixture. Only the in-memory `transfer_attitude_weight` changes .2→1.0 at fixed physical_transfer_fraction .75. Original production config remains unmodified. Signed total reward changes −0.00039666666666666057→−0.0005166666666666607; body-stability contribution changes −0.00008000000000000005→−0.00020000000000000015. Deterministic action is bitwise identical before/after; input tensors, actor state, identity normalizer, optimizer and torch RNG remain unchanged. This establishes the software invariant, not reward causality or learned policy improvement.

`reward_frozen_actor_regression.xml`:1 passed, no failures/errors/skips,3.651s, exit0, timestamp2026-09-14T00:46:22.934398−04:00. Exact scalar/hash receipt: `reward_frozen_actor_regression_receipt.json`. No additional framework or production changes. The earlier statement about lack of an exact existing selected test remains accurate for that earlier subset; this explicitly added fixture supplies the missing check.
