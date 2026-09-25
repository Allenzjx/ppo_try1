# Completed PPO update 1726: physical-window credit

Only globals 225281–225408: 128 real samples / 1 PPO update / 20 optimizer steps. All 128 original raw actions appear exactly five times in 20 likelihood minibatches. No row ≥225409 read or credited.
Actual request phases: {'P07': 2, 'P08': 2, 'P09': 93, 'P10': 6, 'P11': 10, 'P12': 15}. First118 terminate HARD_JOINT_LIMIT; next10 are a new episode, nonterminal at the rollout boundary. Prefix has zero credit.
LR=1e-05; observed KL mean=0.018410, clip fraction=0.253125, value loss=283.528885. These are learning diagnostics, not physical success.

| Physical endpoint group | n | Reward mean [min,max] | Old V mean | Std advantage mean [min,max] | +/− advantage |
|---|---:|---|---:|---|---|
| RR_first_placement_decision | 1 | 0.165600 [0.165600,0.165600] | -12.377931 | -0.767893 [-0.767893,-0.767893] | 0/1 |
| RR_current_TOP_bearing | 20 | -2.152801 [-43.350361,0.191019] | -12.487132 | -1.502882 [-1.898875,-0.767893] | 0/20 |
| FR_window_motion_and_RL_unload | 15 | 0.009089 [-0.149331,0.191007] | -12.446248 | -1.468499 [-1.887504,-0.767893] | 0/15 |
| RL_initial_clearance_not_qualified | 1 | -43.350361 [-43.350361,-43.350361] | -12.935110 | -1.898875 [-1.898875,-1.898875] | 0/1 |
| FL_lower_target_headroom_clipped | 6 | -7.234531 [-43.350361,-0.007410] | -12.679120 | -1.821888 [-1.898875,-1.730197] | 0/6 |
| FL_actual_within_existing_2deg_lower_reserve | 3 | -14.456761 [-43.350361,-0.008131] | -12.754988 | -1.876993 [-1.898875,-1.844599] | 0/3 |

Raw GAE: directly recorded whole-rollout mean −14.902727, range [−30.415251,+1.699550]; P12 mean −28.397252. Terminal raw GAE −30.415251. Individual nonterminal raw GAE for these physical groups is **N/A** in JSON; only standardized advantages are bound per row. γ=.9985, λ=.99; terminal gets no bootstrap; the second-episode tail bootstraps from an unlogged official final value.

## Direction visible in existing update forwards

Δ below is last audited conditional μ minus original collection μ at the **same saved input**, before its last minibatch step (indices16–19), not final checkpoint μ or physical target change.

| Group | FL knee Δμ mean | FR knee Δμ mean | RR knee Δμ mean |
|---|---:|---:|---:|
| RR_first_placement_decision | -0.0006368 | -0.0010404 | -0.0010099 |
| RR_current_TOP_bearing | -0.0003655 | -0.0009723 | -0.0015817 |
| FR_window_motion_and_RL_unload | -0.0001855 | -0.0009548 | -0.0016852 |
| FL_lower_target_headroom_clipped | -0.0015301 | -0.0010934 | -0.0011763 |
| whole_update | -0.0002657 | -0.0008316 | -0.0009271 |

## Interpretation

RR first placement (225367) reward +0.165600, old V -12.377931, standardized advantage -0.767893. Contact makes a positive immediate contribution; subsequent safety failure affects its return, so this is not evidence that reward explicitly discourages touching the top.
Terminal225398 reward −43.350361 = approximately −40 safety event −3.349196 terminal potential removal −0.001167 elapsed-time cost. Weighted smoothness/contact/stability families are zero there; the logged nonzero *unweighted* smoothness diagnostic is not a competing actual penalty.
The observed mean directions/ratios do not establish that the policy has learned a safe knee trajectory. Multiple raw samples still shared clipped final targets in collection; only a subsequent physical evaluation can test changed behavior. This one update cannot isolate whether clipped exploration, source demand, dynamics, or data coverage dominates.

No model/checkpoint/Torch/Isaac load or production edits. Detailed event rows and per-phase directly logged raw-GAE statistics are in the companion JSON.
