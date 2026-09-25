# First2048: real saved-rollout / fixed-input learning signal

Executed on CPU **after** parent-confirmed normal Isaac exit. No optimizer,
checkpoint writes or simulator run. Production source was still ecf205/v1.
Detailed output: `RR_learning_signal_first2048.json`.

## Scope and integrity

2048 active on-policy RR decisions,4 PPO updates,80 Adam steps,4986 actual
frozen-prior prefix decisions with zero PPO credit,5 capture opportunities,
one stochastic local success,0 AUX. The fifth episode ends at the collection
budget rather than a fabricated task terminal. Latest immutable package:
`checkpoint_CP227328_local002048.pt`, SHA256
`afb7e0e2970c0e7b1c544d4b9f385cbb30e94aa5a8eeebe576c3323ce655d806`.

All stored observations/raw actions/old logp/mean/std/rewards/dones equal their
actual decision receipts exactly. Independent Gaussian logp error <=3.82e-6;
whole-rollout advantage-normalization error <=4.77e-7. Each collection actor
reloaded on CPU reproduces conditional means within2.39e-7. All five saved
packages have the same frozen-prior state hash. This verifies correspondence,
not deterministic physical capability.

## What the mean actually learned

For the same fixed281 observed AIR/legal-XY/gap-decreasing states, conditional
raw-mean changes across each update are:

| Update | RR hip delta | RR knee delta | Actual final LR |
|---|---:|---:|---:|
|1|-.000763|+.002428|.00045|
|2|+.001606|-.003274|.00030|
|3|+.001416|-.004145|.00001|
|4|-.000906|-.005247|.0000225|

After all four, local pre-HISTORY head means on this fixed cohort average
hip **+.013523**, knee **-.102374**. Their conditional contributions relative
to the initial zero local module are approximately **+.001352 / -.010237**
because the existing HISTORY factor is.1. On the same27 bearing-hold states,
final head means average hip+.023545 / knee-.061843. These are raw network
coordinates, not degrees or targets.

Thus the first update moved in the proposed hip-negative/knee-positive
direction, but later updates did not retain it. The final model has **not**
demonstrated that the useful descending action entered its deterministic mean.
Hip-negative is a candidate, not a necessary geometric success condition;
the successful first TOP actually occurred near hip+6.08° / knee-29.10°.

The collection-distribution `advantage * Gaussian mean-score` for the RR knee
averaged +.20351, -.08698, -.07581, -.25942 across updates1–4. This is not the
multi-epoch clipped optimizer gradient, but the sign reversal also appears in
the actual fixed-input mean changes. Merely increasing LR would not resolve
the direction of that later training signal.

## Productive physical samples versus failed continuation

| Endpoint bucket | Samples | Mean normalized advantage |
|---|---:|---:|
|AIR/legal/gap decreasing|281|+.2963|
|First TOP endpoint|5|+.7884|
|TOP bearing hold|27|+.6126|
|Bearing drop|12|-.0921|
|TOP reacquisition|8|-.0057|
|AIR after prior TOP|1385|-.1165|

The one successful initial stochastic episode's first TOP / hold-terminal
advantages are +3.048 / +3.079, raw GAE+34.58 / +34.87. It was collected before
the first optimizer update, so it is genuine exploration evidence, not proof
of learning improvement. Other short contacts eventually lost support; four
of five first-TOP raw GAEs are negative, all eight reacquisition raw GAEs are
negative, and20 of27 hold-state raw GAEs are negative. Normalized advantage
can still be positive within a worse rollout; this is not evidence that the
implementation secretly rewrote the reward or falsely labelled success.

1385/2048 (67.6%) samples are recapture AIR after earlier TOP, whereas only40
endpoint samples currently have TOP (first/reacquired/holding, about2%). This
supports the previously found handoff/retention problem and sparse sustained
contact, rather than “optimizer never saw RR” or “gradient zero.” Classification
uses decision endpoints; it does not reconstruct all120Hz contact events.

RR knee was headroom-clipped on584/2048 last-dispatch receipts (28.5%); hip on0.
Absolute tanh >=.95 occurred for58 knee and72 hip samples. These are different
constraints: most knee headroom loss is not explained by tanh saturation.
All12 channels remain in the Gaussian; no hidden residual mask was introduced.

## KL and immediate implication

Update3's RR hip/knee contribute5.30% of total KL; other10 contribute94.70%.
Update4's RR share rises to22.48%, but RR KL .002559 is still mostly std
change .002440, with mean-shift term only .000119. Final actual LR recovered
to2.25e-5; it is no longer1e-5. No claim that non-RR channels should be removed
or that sigma should be enlarged follows from this.

Recommendation: proceed with the narrow current-attempt qualification and
load-dependent source-handoff corrections, preserving actual weights, Adam,
LR and distribution; collect new compatible RR data. Do not add AUX merely
because one stochastic hold exists. If corrected-source fresh data still
fails to retain useful mean actions, a separately counted short, physically
successful capture/hold segment may be considered, but not failed long AIR
suffixes or angle suggestions as success labels. The pending reloaded
deterministic naturalP01 video remains the actual capability test.

## Migration helper tests

The five prepared synthetic447→448 tensor/Adam tests were also executed after
Isaac exit and passed: old columns/head/prior retained, new zero columns,
real optimizer ordering, Adam moments/steps/LR, no RNG consumption inside the
helper, and trainability of the new feature column. This is not a claim that
the real checkpoint has already been migrated or re-evaluated.
