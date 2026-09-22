# P05 pre-edge v5 — data and read-only inspection only

**PASS, 16 targeted CPU tests.** Candidate438 rows; fixed train291 / validation147. CPU helpers exited. No model fitting, budget, Isaac, production/old-helper changes or checkpoint writes. This is not AUX execution authorization.

Current inspection source is actual CP216448, SHA `8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4`, runtime336/v2, counters216448/1656/33120. The source four-event mixed AUX ledger103/104 (front96/96 + RR7/8) and older separate7/8 remain unchanged.

## Exact local qualification and exclusions

Only the sealed block09 three learner episodes are used. All590 P05 inputs were considered; **152 assisted or already-initialized intervals were excluded**. Remaining438 are146 per episode. Input/start/end assist states are uninitialized WAIT, with empty owners and no assist correction. Because initialized latches until episode reset, this rules out hidden FL hip/knee ownership within the8-tick decision—not merely an empty endpoint mask. All12 original stochastic raw values match sealed actions, source μ/σ/logp, single draw and verified native dispatch; final transformed targets and source μ are not labels.

| Episode | All P05 | Eligible / excluded | FL lift | FL crossed | FL placed | P06 input |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 149 | 146 /3 | 2143 | 3240 | 3307 | 3312 |
| 1 | 197 | 146 /51 | 2135 | 3230 | 3696 | 3696 |
| 2 | 244 | 146 /98 | 2133 | 3223 | 4068 | 4072 |

Ticks are episode-local120Hz physics ticks. All three FL placement endpoints have **assist owners[0,1] and actual TOP**: these provide genuine later continuation evidence, not unassisted capture labels. Episodes0/1 subsequently fail at P09; episode2 remains partial at P12. Neither is relabeled whole-task success.

All eligible source-order rows are retained, including initial lower/ground-contact transition states before lift, rather than selecting only visually good actions. Each episode uses offsets0,2 modulo3 for97 training rows and offset1 for49 held-out rows. These are correlated rows of the same three trajectories, **not independent episode validation**.

Observations are directly saved389 under the same current-v2 codec/kernel and potential semantics. **No X17 remapping, reset reconstruction or live389 reconstruction occurred.** Collecting checkpoints, full boundary RNG digests and sealed rollout hashes are bound; no per-action stochastic replay is claimed.

## Coverage versus current deterministic stall

The fixed live28–38s window was read once (151 complete decision endpoints); no future polling. It is physical-state comparison only, never newly reconstructed actor inputs or fitting labels.

- Source eligible inputs span17.667–27.333s. Their last3s per episode comprise138 states: FL front−76.131…+59.760mm, gap20.211…73.484mm and actual FL hip28.096…52.479°. FR/RL/RR support is present in all138.
- Current t38: FL front−33.826mm, gap7.792mm, actual FL hip18.493°, knee−31.785°; FR/RL/RR support remains. **No source row simultaneously matches front and gap within±10mm**. Thus this candidate supports inspecting an earlier P05 approach correction, not claiming proven recovery from the already-stalled low-gap state.
- In the live window, nominal four-wheel values are zero in134/151 rows, but final wheel targets have a nonzero value in151/151. At38s final canonical FL/FR/RL/RR wheel targets are approximately−0.70320/+0.00599/+0.02891/−0.16976rad/s. These are targets, not measured wheel velocities or proof of traction. Assist never initializes in the window.
- The legitimate source stop is not changed. Nonzero final residual-driven targets do not establish a mask bug, a single-wheel cause, or successful propulsion. Existing joint/whole-body state differences matter.

## P05-column inspection scope

`data_v5.load_reviewed_data(metadata, contract)` supplies291/147 actual-raw labelled P05 rows and eight **unlabelled** current-v2 direct protection rows: one each P04/P06/P07/P08/P09/P10/P11/P12. There are no real P01/P02/P03/P13 protection rows here; synthetic onehot checks must stay explicitly synthetic.

This data supports a read-only test of existing first-layer column `W[:,4]`. With identical non-P05 inputs that column is multiplied by exactzero; that mathematical property is not future-trajectory preservation. Within P05, changing it can alter **both μ and logσ across all12 outputs**, not only FL or wheels. No fit, finite budget or expected physical improvement is authorized by this candidate.

Frozen hashes:

- data_v5.py: `db4629914ab2b0ee16149d4c443267a0d2484eae4b617a6ae0e1124baa8d5069`
- data_manifest.json: `1bb4096aede40b3ae1059011a59706b86668a1223f6da0f2a80558e5b1dd4718`
- candidate_data.npz: `4b8fcdac3d229dc5bcb87519bd2804e71361a051ed96a3e45f68ca2c8d985584`

The compact JSON contains every selected row's provenance, per-episode qualification/exclusion and fixed-window coverage. No old dataset/helper or successful zero baseline was altered.
