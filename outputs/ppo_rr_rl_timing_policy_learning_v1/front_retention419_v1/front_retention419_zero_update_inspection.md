# Front-retention419 zero-update inspection

This is a read-only CPU sensitivity check. No fit, optimizer step, checkpoint, PPO/AUX credit, or physics was produced.

- Teacher actor: `7bd9db99a70ab4a9bdf6c251c48f638d8dd21bdc9a854766c8b0ebbc7bb75382`; student actor: `1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5`.
- Teacher targets are offline same-runtime conditional means on the student's actual 419-column observations (including HISTORY). They were not executed and are not success labels.
- Train uses one P01 plus 32 P02 rows from rollout1695; validation uses 32 P02 rows from rollout1696. Both partitions are from one continuous episode; P01 has no independent validation. P03-P06 learner data is absent and was not invented.
- Selected gradient L2: `9.31914656e-06`; phase-column L2 P01/P02: `[1.4737074138793105e-07, 9.317981493950356e-06]`.
- Actual same-input protection rows: P07=1, P09=8, P12=8. A hypothetical full unit step along the initial negative gradient leaves mean/log-sigma/sigma/request bitwise equal on those rows; this is not a closed-loop trajectory claim.
- Teacher/student actors are hash-identical before/after the inspection and CPU RNG was restored.

## Wheel request sensitivity

Values below are `cap*tanh(raw mean)` in rad/s. JVP is per unit learning rate along the frozen initial negative-gradient direction; it is not an applied update.

| set | student−teacher request signed mean FL/FR/RL/RR | initial request JVP signed mean FL/FR/RL/RR | JVP max abs FL/FR/RL/RR |
|---|---|---|---|
| train | -0.002037238, -0.001532893, -0.001034761, -0.001471279 | 4.987946e-08, 8.891988e-09, -1.597132e-08, 1.671283e-08 | 6.007314e-08, 9.707875e-09, 1.725168e-08, 1.894421e-08 |
| validation | -0.0023348, -0.001683599, -0.001269656, -0.001707167 | 4.892678e-08, 9.297309e-09, -1.654081e-08, 1.690853e-08 | 5.652152e-08, 9.64619e-09, 1.717149e-08, 1.752211e-08 |

Full row IDs, source hashes, conditional-mean deltas, JVP ranges, and invariance evidence are in the paired JSON.
