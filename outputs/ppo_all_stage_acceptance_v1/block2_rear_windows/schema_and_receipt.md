# Schema and bounded execution receipt

This is a report-only diagnostic, not an evaluation, checkpoint migration, or new simulation. The script imports only Python standard-library modules and writes only a newly created directory inside `outputs/ppo_all_stage_acceptance_v1`.

Source: finalized Block2 `20260909T0327591294869Z_ga8b148463115_1996f4a3ba204ea7b30a7bf6b1038046`. Two bounded first-line schema reads preceded the script's **one complete policy-audit pass**. No physical/raw trajectory, prefix stream, optimizer stream, tensor, checkpoint contents, or file hashes were read by the script.

The actual CPU command exited 0. Input size was 28,036,545 bytes. The full pass checked 384 sequential optimized rows, global 139521–139904, against both final manifests: 3071 physics ticks; P06/P07/P08/P09 = 363/2/2/17. Analysis took 0.386 seconds; this is parsing time, not simulator performance.

The output has 3 episode documents: 178-decision FALL, 117-decision FALL, and an 89-decision nonterminal P06 tail. Their 16/14/9 windows contain 66/42/33 unique selected rows. The default ceiling is 41 windows per episode, each at most 3 preceding + center + 3 following decisions; episode boundaries are never crossed. Each leg/event kind keeps its first and most recent occurrence; intermediate omissions are counted. Rear-front/clearance extrema and stage handoffs have separate slots.

Recorded PPO native summaries report 3071 verified/effect ticks, 3065 own-phase effects, and all four write counters zero on all 384 rows. These are preserved logger receipts, not an independent physical replay. Manifest subtraction excludes 1344 prefix decisions / 10752 prefix ticks; prefix-native validity is not inferred from that subtraction.

Policy-attributed rear events: RR initial ×3, RR Q ×2, RL initial ×1; no recorded C/P or Q-revocation in these episodes. Historical events at or before each recorded credit-start tick 3584 are excluded from the PPO event windows. The original event tick and first-observed global decision remain separate.

## Observation versus physical evidence

Across all 384 audit rows the checked row/audit/task locations contained no encoded observation, final observation, RPY, gravity vector, or raw base pose. The policy `old_distribution_mean_full12` and `old_distribution_std_full12` are action-distribution moments, **not** observation moments or body pose. Production observation construction has orientation/gravity fields, but their presence in memory does not mean the audit serialized them.

Available evidence includes physical-evaluator scalar body speeds / `body_forward_m`, actual world body AABB, per-leg geometry/contact/load, measured wheel speeds, selected tracking-reference measured joint radians, native readback, and original reward breakdown. AABB is not root position or center of mass; speed magnitude is not signed velocity; no RPY/gravity/FALL branch is reconstructed. Raw/nominal/projected values, same-dispatch native target records, reward intervals, and event timestamps retain their separate sampling meanings.

## Use after parent review

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -B 'C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/diagnose_completed_rear_windows.py' --run-dir '<actual finalized new-namespace TRAIN directory>' --output-dir '<new exclusive directory under outputs/ppo_all_stage_acceptance_v1>'
```

The script rejects live/wrong-namespace/non-TRAIN runs, mismatching final manifests, non-N1 inputs, ambiguous credit starts, prefix-in-storage evidence, policy/tick gaps, and source files exceeding the explicit byte budget. These are report-input integrity checks, not optimizer or task-success gates. No other run has been inspected or tested with this script.
