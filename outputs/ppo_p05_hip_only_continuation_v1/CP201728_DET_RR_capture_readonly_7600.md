# CP201728 natural-P01 deterministic evaluation: RR capture check through tick 7600

Read-only bounded inspection of `runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc/source`, runtime `0001c3138b0b`. No runtime changes, guard bypass, or extra Isaac process. This is an intermediate result through 63.333333 simulation seconds, not the final evaluation outcome.

## Finding

The new policy really lifts RR and crosses the front edge. The remaining task is real RR landing/capture. By the inspected endpoint, RR is still AIR, has zero bearing force and zero consecutive TOP samples. The P09 source has already passed its physical late-reconfiguration guard, executed its full-body event, and reached its authored endpoint. There is no remaining source dispatch/entry deadlock in this window.

The observed failure to capture is therefore a **current policy/full-body geometry outcome**, not evidence of a lost RR actuator command or false FL predecessor gate. These logs do not isolate a single causal joint or prove that removing one residual would produce a landing.

## Event and source chronology

| Episode tick | Time s | Actual event / dispatch condition |
|---:|---:|---|
| 5880 / 5888 / 5896 | 49.000 / 49.067 / 49.133 | P07 / P08 / P09 task labels enter; label changes do not replay source entries. |
| 6128 | 51.067 | P09 source actually starts after RL-response/unload readiness; FR and RL supply two real other supports. |
| 6143 / 6149 | 51.192 / 51.242 | RR initial free lift / qualified measured lift. |
| 6208 | 51.733 | Pending knee source waypoint 80 passes: current AIR gap +107.142 mm, vertical velocity +0.312752 m/s, two verified other supports. |
| 6769 | 56.408 | Completed native tick first carries authored four-wheel stop. |
| 6776–6857 | 56.467–57.142 | Source local tick 648 waits only for current RR over-top drop region. Initially front distance -15.573 mm, gap +34.609 mm. Extra physically eligible four-wheel approach starts in completed native tick 6777. |
| 6858 | 57.150 | RR front edge crossed; live late-group guard passes with front +0.733 mm, gap +34.176 mm, three real other supports. |
| 6859 | 57.158 | First completed native tick carries the late authored full-body event and FL wheel -1.07 rad/s. |
| 7075 | 58.958 | First completed native tick carries source endpoint four-wheel stop. |
| 7600 | 63.333 | Source status active/no wait, local source counter 1391, entry valid; RR lift/cross histories true, placement false. |

The source is `configs/recording_motion_contract.json`, P09: four-wheel +0.3 at source 1.933333 s, stop at 5.333333 s, late whole-body reconfiguration at 5.4 s, final stop at 7.2 s. The late event changes FL hip/knee, RL hip/knee and FL wheel; its explicit all-channel values retain RR hip/knee -6.9/-37.8 degrees. It is not an additional RR-only knee descent waypoint.

Production guards are in `semantic_supervisor.py`: `_sequence_permission` (1937), `_rr_pending_carry_readiness` (2049), `_rr_late_reconfiguration_ready` (2125), and `_continuous_advisory` (2205). At tick 7600 the task is EXECUTION, `entry_valid=true`, no entry reasons, no terminal reason, `placed_RR` progress 0.7 but **historical placed bit false**. The 0.7 partial value is not physical placement or success. P09 completion still requires actual placement.

## Same-tick control path at episode tick 7600

All joint entries below are canonical degrees; wheel entries are canonical rad/s. N is taken from the matching `native_tick_audit.jsonl`, not independently reconstructed or a next-observation nominal. Baseline includes the recorded same-tick mapper/controller contribution. The recorded effective PPO request equals its requested value here; final slew/clamp is empty. Final command and actual measurement come from the matching completed `capture_assist_ticks.jsonl` row. Its raw dispatch tick is 7779, retained separately from episode tick 7600.

| Channel | Source N | Post-mapper baseline | PPO requested = effective | Final command | Actual |
|---|---:|---:|---:|---:|---:|
| FL hip | -18.500 | -18.774 | -2.776 | -21.551 | -21.817 |
| FL knee | -31.400 | -31.590 | -20.780 | -52.370 | -52.626 |
| FR hip | 0.000 | 0.000 | +6.818 | +6.818 | +7.999 |
| FR knee | +31.100 | +30.133 | -33.589 | -3.456 | -2.942 |
| RL hip | +15.400 | +14.622 | -13.282 | +1.340 | +3.273 |
| RL knee | +19.400 | +18.848 | +3.484 | +22.332 | +22.816 |
| RR hip | -6.900 | -7.293 | +12.816 | +5.523 | +5.253 |
| RR knee | -37.800 | -38.627 | -13.860 | -52.487 | -52.253 |
| FL wheel | 0.000000 | 0.000000 | -0.373543 | -0.373543 | -0.379663 |
| FR wheel | 0.000000 | 0.000000 | +0.099999 | +0.099999 | -0.146216 |
| RL wheel | 0.000000 | 0.000000 | +0.008220 | +0.008220 | +0.178392 |
| RR wheel | 0.000000 | 0.000000 | -0.071715 | -0.071715 | -0.071931 |

The residual is changing the whole-body command, not just RR. This is not an imitation-angle requirement: different angles could legitimately succeed, but this observed set has not produced capture. Loaded FR/RL wheel measured rates differ from command; command delivery is verified, while contact/leg-motion coupling is not causally resolved by this audit. No claim of perfect wheel-speed tracking is made.

The physical wheel-axis mapping is also present, not guessed from canonical signs. Tick 7779 native wheel target order is `[+0.373542875, +0.099998929, -0.008220135, -0.071714856]`; FL/RL sign conversion is recorded by the verified actuator mapping.

## Tracking and joint margin

- All 1473 ticks from 6128 through 7600 have verified native audit, all-one policy permission mask, all 12 policy channels unmodified at actuator, matching actual dispatch mapping, and independently verified previous-ACK tracking reference. No in-episode root pose/velocity/force/gravity writes occur.
- No servo policy-headroom clipping occurs in this interval. Ordinary final slew acts during command transitions; at tick 7600 no final slew/clamp indices remain and capture assist owns no channels.
- Tick 7600 RR final-to-actual errors: hip +0.270157 degrees, knee -0.234123 degrees. RR is responding to the command, not frozen.
- RR knee actual -52.253371 is 7.746629 degrees above hard minimum -60; final command -52.487494 is 5.512506 above the reserved minimum -58. Actual RR hip +5.252900 has 140.252900/129.747100 degrees to hard lower/upper limits.
- Across this interval RR knee minimum actual is -53.197221 at 6880; minimum final command is -56.872253 at 6869, still inside the reserved range. Numerical angular margin is **not** proof that a particular remaining direction can reach the top without coordinated body motion.

## Measured whole-body context

| Tick | RR front mm | RR top gap mm | RR actual hip/knee deg | Body z mm | Body roll/pitch deg |
|---:|---:|---:|---|---:|---|
| 6858 | +0.733 | +34.176 | +12.902 / -44.309 | 74.720 | -8.721 / -1.728 |
| 6864 | +14.443 | +39.413 | +9.421 / -47.754 | 77.673 | -8.381 / -1.559 |
| 6880 | +42.227 | +63.467 | +3.991 / -53.197 | 90.336 | -7.870 / +0.074 |
| 7144 | +47.001 | +62.183 | +5.141 / -52.120 | 88.074 | -8.166 / +0.417 |
| 7600 | +49.978 | +61.492 | +5.253 / -52.253 | 87.851 | -8.023 / +0.421 |

After the live late event, the body rises while RR advances and its gap increases. Several joints, load distribution and wheel targets change together, so the data do not justify blaming knee motion or body height alone.

At tick 7600: FL is actual TOP support (2.284771 N, gap -1.031 mm), FR actual TOP support (13.888868 N, gap -1.908 mm), RL actual ground support (12.495871 N, front -130.828 mm), RR AIR/no support (0 N). Support polygon margin is +27.663 mm. The body is not wholly motionless: measured velocity is approximately [-0.003920, +0.005890, +0.016804] m/s. RR remains over valid top XY but has no contact to qualify as placed.

## Interpretation and next boundary

The first late-event wait was a real geometric wait and it resolved. The subsequent persistence is not that same gate: after source endpoint, the active policy and mapped held source command maintain a noncapturing configuration. This excludes the inspected missing-command/mask/entry-deadlock explanations; it does not uniquely identify a minimal physical correction.

Preserve this as improved RR lift/cross evidence, but do not call it a completed rear capture or full PPO success. Let the running full evaluation end naturally; retain its eventual outcome separately. No hot reward, action, or runtime change is recommended by this read-only inspection.
