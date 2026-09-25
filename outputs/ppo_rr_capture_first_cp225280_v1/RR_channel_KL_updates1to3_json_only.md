# KL attribution from actual completed minibatches

Read-only standard-library calculation; no Torch/model/simulator import.
Source: the first1536 credited decisions and actual `likelihood_0001..0003`
minibatch mean/std receipts. Every rollout identity was unique and matched.
Reconstructed KL(old Gaussian || current Gaussian) agrees with the official
update average within9.4e-10. Means/stds are actual conditional raw Gaussian
values, not FINAL joint targets.

| Update | Total KL | RR hip+knee | Other10 | RR share | Final actual LR |
|---|---:|---:|---:|---:|---:|
|1|.01411996|.00096186|.01315810|6.81%|.00045|
|2|.01600836|.00120076|.01480760|7.50%|.00030|
|3|.03160926|.00167626|.02993300|5.30%|.00001|

Update3's other10 contribution is six non-RR leg joints .01668439 plus four
wheels .01324861. Largest single channels: FL knee .00604069, RL wheel
.00541336, FR wheel .00535402. RR's mean-shift term is only .00013815; RR
standard-deviation-change term is .00153811.

Actual update3 minibatches5–19 exceed the global .02 decrease threshold.
The implemented adaptive schedule uses the **sum over all12 channels**, so
the measured LR decrease is predominantly driven by non-RR distribution
movement, not by a large RR conditional-mean shift. This does not imply those
other channels are dispensable or that a separate LR would physically succeed.
Small exploration sigmas make a given mean shift more expensive in KL; the
formula tests cover that fact, without recommending larger sigma.

Whether RR's physically helpful actions have usable advantages and whether
the saved conditional mean actually adopted them still requires the queued
post-Isaac tensor/checkpoint analysis. Do not claim learned RR placement from
gradient activity or these KL numbers. No reward, sigma, optimizer or source
schedule was changed by this diagnostic.

Full per-channel mean/scale decomposition and each of20 minibatches per update:
`RR_channel_KL_updates1to3_json_only.json`.
