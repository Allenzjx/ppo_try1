# P13 actor-only CPU diagnosis: checkpoint 28032

This records an **executed read-only CPU diagnostic**, performed by the root agent after the P01 evaluation had exited and no Python/Isaac process remained. All strict checks passed. The report author did not repeat the forward pass. The subsequent P10 same-MDP 2,048-decision training was already running when this report was written; no Python, production edit or parameter change accompanied this report.

## Sources and verification

- Checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000028032.pt`.
- Real observation source: `runs/ppo_semantic_v3/train/20260906T0636173277212Z_g64abc5357d00_6614a15089014b2ebef51695b9b8cc22/rollouts/rollout_000172.pt`.
- Selected **128/128** saved observations, global decisions **26369–26496**. Each actual saved 324-dimensional policy observation has P13 one-hot and all four nominal wheel entries equal to zero. These are normalized saved policy inputs, not newly constructed task/goal templates.
- Checkpoint file SHA-256, strict actor state loading and actor parameter hash passed. Checkpoint/rollout runtime contracts match. The observation normalizer is identity; distribution type is `scalar`, so `std_param` is sigma directly, not log sigma.
- Official `MLPModel` ran in evaluation/inference mode with `stochastic_output=False`. No optimizer was instantiated or updated, no training RNG restored, no network reset, no checkpoint saved and no parameter changed.

## Measured learned sigma

Full12 order: FL hip/knee, FR hip/knee, RL hip/knee, RR hip/knee, then FL/FR/RL/RR wheels.

```text
[0.1458352506, 0.1502650827, 0.1468492150, 0.1507403553,
 0.1607256979, 0.1559512913, 0.1624415666, 0.1612131596,
 0.1535811126, 0.1551445574, 0.1463847458, 0.1448787302]
```

These are learned checkpoint values, not the initial 0.15 configuration.

## Wheel outputs on the same saved observations

Values are per-wheel statistics across the 128 observations, ordered FL, FR, RL, RR. Raw values are Gaussian latent units.

| Quantity | FL | FR | RL | RR |
| --- | ---: | ---: | ---: | ---: |
| Current cp28032 deterministic raw mean, signed average | 0.0025585182 | 0.0680774003 | 0.0446227267 | 0.0574083328 |
| Current deterministic raw mean, average absolute value | 0.0201876368 | 0.0680774003 | 0.0455507152 | 0.0574083328 |
| Current `0.6*tanh(mean)`, average absolute value, rad/s **before rate limits** | 0.0121083781 | 0.0407546572 | 0.0272944160 | 0.0343965776 |
| Saved historical behavior mean, signed average | −0.0517261811 | 0.0285836309 | 0.0285286922 | 0.0773623064 |
| Saved sampled action minus its **own historical mean**, RMS | 0.1400922984 | 0.1674246341 | 0.1383502185 | 0.13636521995 |
| Saved historical sigma | 0.1540244669 | 0.1558365673 | 0.1456689686 | 0.1437838227 |

## Interpretation and limits

There are **both** nonzero learned mean outputs and substantial sampled exploration residuals in this evidence. The FR/RL/RR average absolute pre-rate suggestions exceed the existing 0.02 rad/s wheel-command stop tolerance, even without adding fresh Gaussian noise. This does not mean every individual sample exceeds that tolerance: an average is not a threshold-pass count. FL's near-zero signed average also conceals nonzero magnitude, as its average absolute raw mean shows.

The historical noise RMS is computed from `saved action − saved behavior mean` in the same rollout. It is **not** `saved action − current cp28032 mean`: that latter subtraction would mix actual sampling noise with policy changes between behavior collection and checkpoint 28032. Current mean and historical sigma/noise are therefore labeled separately.

The `0.6*tanh(mean)` values are instantaneous scaled residual **suggestions**, not actual actuator commands, wheel velocities, or a newly simulated trajectory. The saved history, residual rate limits, mapper, physical response, contact and body-motion conditions have not been rerun under the new deterministic means. A zero nominal wheel target is not a zero actual wheel command. The diagnosis cannot establish which factor dominates P13 physical failure, and does not establish deterministic success or failure.

No cross-tick independent success probability is inferred: the observations, policy means, projected histories and physical dynamics are temporally coupled. This diagnostic motivates interpreting the next real P10/P13 evidence with separate mean/exploration labels; it changes no stop requirement, wheel mask, nominal command, learned sigma, policy weight or training parameter.
