# Completed-block learning-signal check

Prepared while Isaac was running. **Not executed with Torch yet.** This is an
output-side, read-only diagnostic. Production code, training data, models and
configuration are not changed. Nine standard-library tests passed.

After the parent explicitly confirms the training process has exited and the
four complete updates / 2048 activated decisions are sealed, run from the repo:

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' `
  outputs/ppo_rr_capture_first_cp225280_v1/analyze_rr_learning_signal.py `
  --isaac-stopped `
  --output outputs/ppo_rr_capture_first_cp225280_v1/RR_learning_signal_first2048.json
```

The script refuses missing/incomplete update blocks, non-PPO intervention rows,
existing output paths, mismatched raw samples / observations / stored Gaussian
mean or standard deviation / old log probability / reward / terminal values.
It checks the actual saved advantage normalization and independently recomputes
the collection Gaussian likelihood. It loads only immutable completed-update
checkpoints on CPU, excludes any simulator imports, optimizer or model writes,
and reconstructs the actual actor output on the original observations.

Physical buckets distinguish first TOP, reacquisition, real bearing hold,
bearing loss, AIR gap descent, AIR recovery and return to ground. They describe
decision endpoints; native hold elapsed time comes from the actual logged
120 Hz observer. Last-dispatch headroom clipping is not mislabelled as all
eight ticks of a decision. Negative advantages stay negative.

For each of four updates, the same saved observations are evaluated before and
after that update. This isolates parameter-induced mean changes from changes
in the visited state distribution. Both each update's own collection cohort
and a common 2048-state cohort are reported. RR hip-negative / knee-positive
signs are labelled candidate directions, not necessary or sufficient success.
The score-times-advantage diagnostic is not the actual clipped, multi-epoch
gradient. This review cannot replace a reloaded deterministic physical video.

Review the generated result for whether physically productive actions obtain
positive relative advantage, whether clipping collapses exploration, whether
conditional mean actually changes, and whether contact/hold states are sparse.
Any AUX suggestion requires those measurements and separate accounting; the
script does not perform AUX, recommend a hard-coded angle, or modify reward.
