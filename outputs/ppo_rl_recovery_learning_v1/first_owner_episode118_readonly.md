# First completed owner-recovery episode: bounded read-only audit

- Exact first 118 rows, global 225281–225398, 943 learner physics ticks; not a completed-update claim.
- P07 successful-nominal prefix ends tick 5160 / 43 s; request phases: {'P07': 1, 'P08': 1, 'P09': 85, 'P10': 6, 'P11': 10, 'P12': 15}. Not natural-P01 policy success.
- True safety failure: FL knee HARD_JOINT_LIMIT, tick 6103 / 50.858333 s, P12. Full task success=false.
- RR qualified/crossed/placed ticks: 5401 / 5618 / 5850. RL qualified/crossed/placed remain absent; initial clearance is not qualified swing.
- First sampled real RR TOP+bearing tick 5856; P09 late source starts at 6049. Sampled post-TOP loss transitions: 2.
- Both sampled losses ([5872, 5928]) precede late-owner acquisition; corresponding sampled winning/active owner bits are false. Reacquired TOP samples: [5920, 5976]. They do not test suspension of an already-issued late target.
- Terminal RR TOP 15.422771 N; FL TOP 17.405302 N. Terminal safety invalidates controller permission; that is not evidence RR physically lost bearing.

## Last FL knee control path (degrees)

| End tick | Actual mapped N | Gaussian raw | Requested residual | Effective residual | FINAL | Actual q | FINAL−q | Δq/Δt °/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6032 | -12.150 | -1.7063 | -33.530 | -33.530 | -45.680 | -45.536 | -0.144 | -1.885 |
| 6040 | -12.150 | -1.7648 | -33.949 | -33.949 | -46.099 | -46.525 | 0.425 | -14.825 |
| 6048 | -12.150 | -1.6711 | -33.541 | -33.541 | -45.691 | -46.872 | 1.181 | -5.213 |
| 6056 | -20.900 | -1.4686 | -32.375 | -32.375 | -53.275 | -48.672 | -4.603 | -26.999 |
| 6064 | -30.900 | -1.9906 | -34.681 | -27.100 | -58.000 | -52.820 | -5.180 | -62.225 |
| 6072 | -32.650 | -1.9066 | -34.445 | -25.350 | -58.000 | -55.698 | -2.302 | -43.162 |
| 6080 | -35.150 | -1.5383 | -32.826 | -22.850 | -58.000 | -57.244 | -0.756 | -23.189 |
| 6088 | -37.650 | -1.7707 | -33.973 | -20.350 | -58.000 | -58.198 | 0.198 | -14.313 |
| 6096 | -40.150 | -1.9977 | -34.699 | -17.850 | -58.000 | -59.020 | 1.020 | -12.331 |
| 6103 | -41.400 | -1.8630 | -34.306 | -16.600 | -58.000 | -60.012 | 2.012 | -16.998 |

Actual q is recovered from logged current joint hard-limit margin, cross-checked against both bounds. Δq/Δt is only endpoint finite difference, not measured velocity/force.

## Saturation and owner limits

| Joint | Lower headroom clipped | FINAL −58° | Clipped raw range | Mean range | Effective σ range |
|---|---:|---:|---|---|---|
| front_left_knee | 6/118 | 6/118 | [-1.9976741075515747, -1.53827965259552] | [-1.937086820602417, -1.473954439163208] | [0.1852319836616516, 0.37579089403152466] |
| rear_right_knee | 24/118 | 24/118 | [-0.6379841566085815, -0.5409307479858398] | [-0.6362688541412354, -0.514281153678894] | [0.01948435790836811, 0.02089722268283367] |

No sampled pre-action active owner or nonzero anchor in all 118 rows. No concrete active H-versus-actual dropout entrance can be measured in this episode. Per-physics owner receipt is validated internally but omitted from the returned JSON audit, so subdecision transient activations cannot be excluded.

With constant request after an active entrance the code keeps H=previous FINAL, not actual q. Thus it stops additional stale-N displacement but is not guaranteed to remove an existing servo position error; this is a conditional code property, not demonstrated active behavior in this episode.

The FL target is protected to −58°, but actual reaches −60.011567° (hard lower bound −60°). This proves target projection alone did not prevent this physical crossing; it does not isolate dynamics, contact coupling or inertia. Neither a fixed measured-q exit nor more noise is justified by this episode alone.

The actual N contribution changes −12.15→−41.4° after late source starts while RR has current TOP bearing; the residual remains about −33 to −34°. Thus this is combined source/residual demand, not evidence of an absent residual channel. From tick 6088 onward FINAL−q is positive while q continues negative: the final target is already restorative relative to measured position; past acceleration and physical load cannot be separated here.

Exact Gaussian probability of exiting the lower headroom region is not recoverable as a single raw threshold from these filtered, rate-limited, mapper-dependent endpoint receipts. Observed clipping rates and original mean/σ are reported without inventing an inverse.

Raw sample/μ/σ/logp bindings and all-1 mask/native verification pass for 118 rows and 943 physics ticks. No rear task teacher or prefix samples credited. Per-tick owner JSON evidence is missing despite internal verification; active public input count is zero, not proof of zero subdecision activity.

Physical-window counts (overlapping endpoint categories): {'RR_reachable_AIR_preparation': 14, 'FR_directed_body_CoM_motion_with_RL_unload': 15, 'RR_actual_bearing_front_preparation': 19}.

JSON contains selected exact control rows, frozen support threshold binding, event ticks, source timing and evidence limits. No model, simulator, production file or reward changed.
