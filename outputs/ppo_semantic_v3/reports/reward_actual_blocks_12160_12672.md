# Actual reward audit: completed blocks ending 12160 and 12672

## Evidence boundary

Read-only PowerShell audit of these **completed** runs under runtime `5036e11622ac2e9b9eca86a3991f18781fe155e0`:

- P06 suffix: `runs/ppo_semantic_v3/train/20260906T0336310729329Z_g5036e11622ac_be84c27beb4744c781ff28ce0aa48fd9`.
- Fresh P01: `runs/ppo_semantic_v3/train/20260906T0344557730220Z_g5036e11622ac_474a218c14914a65b7d80eadcdbae170`.

Sources are `training_manifest.json`, `residual_and_projection_audit.jsonl`, `optimizer_updates.jsonl`, `completed_episodes.jsonl`, and each immutable checkpoint's manifest. No current/incomplete rollout, torch deserialization, Python, simulation, or new test was used. No weight or runtime change is proposed here.

Each block has exactly **512 credited decisions, four completed PPO updates, and 80 optimizer steps**. Credit indices are contiguous; logged raw actions match each applied-audit request. Checkpoint sidecars bind the correct source run. P06 teacher-prefix rows in PPO storage: **0**.

| Block | Credited global decisions | Completed PPO updates | Final lifetime optimizer steps | Credited physics time |
|---|---|---|---|---|
| P06→12160 | 11649–12160 | 57–60 | 1200 | 34.133333 s |
| P01→12672 | 12161–12672 | 61–64 | 1280 | 34.083333 s, spanning one terminal/reset |

Checkpoint recorded SHA256: 12160=`d2f4f9e4611278ecc5a8585c4ce023eaf108da4e570f27308c3cc825631e4b00`; 12672=`2c05f08e5d5910506646137517351ae6df5f3e65c34b6d4297635bb00c6b7134`.

## Actual signed family contributions

These are undiscounted sums of the already weighted, per-decision reward components, not advantages or returns. All costs use actual executed physics dt. Summation is from double-precision reward diagnostics; stored reward is float32 (maximum individual difference: 3.68e−9 / 7.77e−7).

| Family | P06→12160 sum | P01→12672 sum |
|---|---:|---:|
| task_progress | −6.480920 | −42.750384 |
| body_stability | −0.378914 | −0.399619 |
| contact_motion_quality | −0.001612 | −0.001609 |
| control_smoothness | −1.430811 | −1.210247 |
| control_regularization | 0 | 0 |
| **Total** | **−8.292258** | **−44.361860** |

Task-family positive and negative contributions are not hidden by the net sum:

- P06: **+3.467842** on 90 decisions, **−9.948762** on 422 decisions.
- P01: **+1.924402** on 115 decisions, **−44.674786** on 397 decisions.
- Every nonzero quality-family contribution is negative; regularization is disabled and has weight zero.

### Task versus quality scale

The logged formula is `5*(0.995*phi_after - phi_before) + terminal_event - 0.02*executed_seconds`. Per-decision reconstruction error was zero.

| Task component | P06→12160 | P01→12672 |
|---|---:|---:|
| Sum of `5*(phi_after - phi_before)` | +0.109973 | +0.454827 |
| Sum of discount component `−0.025*phi_after` | −5.908226 | −2.523544 |
| Total potential shaping | −5.798254 | −2.068718 |
| Time cost | −0.682667 | −0.681667 |
| Terminal events | 0 | −40.000000 |
| All quality families combined | −1.811337 | −1.611476 |

P06's mostly negative task total is dominated by the discounted potential term while physical potential makes little net progress; it is **not** an AIR penalty or a hidden reference-matching penalty. Its credited potential starts at 0.445505 and ends at 0.467500. The positive task contributions total more than the quality costs, but do not establish adequate task learning or ideal weights.

P01 has one actual `WHEEL_ONLY_CLIMB` terminal in P05, at global decision **12492**, episode time **22.083333 s**. That row has total reward **−41.910083**, event −40, and pre-terminal phi 0.381708, which is reset to zero by terminal semantics. PPO update 63 consequently encounters this terminal and reports value loss **108.334069**, versus 0.001–0.052 in the other audited updates; this is a scale observation, not proof of an implementation conflict. There is **no successful terminal** in either block. The P01 delta sum includes the real terminal and separate reset episodes; it is not a single-episode telescoping progress claim.

## Phase coverage is limited and explicit

Counts use the policy request's phase, not an invented full-task validation result. A decision may contain a phase handoff, so durations below are assigned by request phase, not exact within-decision phase occupancy.

| Block | Request-phase decisions |
|---|---|
| P06→12160 | P06:227; P07:1; P08:1; P09:283; P10–P13:0 |
| P01→12672 | P01:2; P02:366; P03:4; P04:1; P05:139; P06–P13:0 |

The P06 block's credited physical interval is 29.866667–64.0 s, excluding the teacher prefix. It stops at a rollout boundary in P09, not a task terminal: RR center remains 52.306 mm short of the front, with RR lift/placement false. The P01 block has one failed episode and continues a new one through episode time 12 s. Neither completed optimization block is evidence of full task success.

## No observed duplicated smoothness charge or automatic AIR/rebound tax

The reward source/config used by both runs match the current file bytes: source SHA `a77901cd5da8d49358e933ca34aef1c2698138a6b78a7df9c36e1c94976dbf3c`, config SHA `da574107eb0c0ba02ef02ff77cd00c7e9cfa4ef3208abae96435fafd1bc38bcb`. Relevant configuration is `smoothness_components: applied_only` and `contact_loss_semantics: recent_touchdown_without_active_command_change`.

- **Smoothness:** every row matches `−0.05*(actual_drive_first_difference + actual_drive_second_difference)` to at most **1.31e−18**. Nominal/residual first differences are logged diagnostics, not additional independently summed costs. Their integrated diagnostic values are 0.343529/22.707769 in P06 and 0.327478/16.826440 in P01; adding them again would contradict the actual totals. Final applied commands do include nominal and residual, so their physical changes affect the one applied-command penalty. This is not independent duplicate charging of the nominal and residual representations. Smoothness contributes about 79.0%/75.1% of the respective quality costs, not of all task reward.
- **Rebound:** `confirmed_post_touchdown_rebound` integrates to **0 in both blocks**, despite 287/186 touchdown events and decision-end AIR observations in 375/503 decisions (RR specifically 94/14). The code requires a recent touchdown within 0.25 s, sufficiently unchanged whole-body command, and excess upward speed after contact loss; AIR alone is not a penalty trigger. Current logs demonstrate no charged rebound here, not a universal classifier proof for all future landings.
- **Chatter:** integrated diagnostic values 5.379167/3.200000 are nonzero but are not directly added to the v3 contact cost. These aggregate per-wheel diagnostic integrals are not a count of task failures. Contact-family cost is only approximately −0.00161 in each block. Separate accumulated impact-versus-slip contributions are not exported, so this audit cannot apportion that small remainder honestly.

No conclusion relies on negative-advantage correlations. The exact contribution formula and actual sums provide the evidence.

## PPO clipping and action saturation

| Measurement | P06→12160 | P01→12672 |
|---|---:|---:|
| Mean observed PPO ratio clip fraction | 25.0391% | 25.7031% |
| Per-update clip fractions | 15.7813%, 35.3125%, 23.4375%, 25.6250% | 27.6563%, 23.2813%, 24.2188%, 27.6563% |
| Mean KL | 0.0184352 | 0.0173156 |
| Number of raw action scalars | 6144 | 6144 |
| Maximum absolute raw action | 0.619032 | 0.693742 |
| Maximum absolute `tanh(raw)` | 0.550454 | 0.600381 |
| Raw outside [−1,1] | 0 / 6144 | 0 / 6144 |
| `abs(tanh(raw)) >= 0.95` (also >=0.99) | 0 / 6144 | 0 / 6144 |

PPO clip fraction is the recorded mean over actual optimizer minibatch ratio comparisons; it is **not** actuator clipping. Raw Gaussian actions use tanh, not hard clipping to [−1,1]. The 0.95/0.99 values are explicitly declared post-hoc saturation diagnostics, not configured safety limits.

These compact training rows do **not** expose per-stage projector clipping flags or all per-tick rate-limit counters. Therefore native slew/hard-limit/safety saturation percentages are **unavailable**, not zero. Comparing raw directly with final native targets would conflate scaling, mapping, tracking, residual-rate projection, and physical safety; no such percentage is invented here.

## Bounded conclusion

In these two actual completed blocks, the implemented reward matches its declared signed decomposition. There is no observed extra nominal/residual smoothness charge, no charged active-AIR rebound, and no tanh saturation. Task potential and the real failure event dominate the negative totals; this is distinct from success coverage or evidence that the weights are optimal. Keep the ongoing block outside these totals and inspect its completed results separately; this audit adds no gate, weight change, or new framework.
