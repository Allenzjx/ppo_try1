# Update7 sealed review — prepared, not executed

`analyze_rr_learning_signal_update7.py` reuses the common collector/storage checks and update6's pure HISTORY/raw/source-tracking/headroom functions. It does not import Torch until explicit Isaac-exit confirmation and completed-run/metadata/real-minibatch checks pass. Preparation did not read the active run, import Torch/PXR, or change production.

Run only after the sole Isaac process has naturally exited and `train_after_aux64_fresh512_0ff03ea` is sealed COMPLETE:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs/ppo_rr_capture_first_cp225280_v1/analyze_rr_learning_signal_update7.py --isaac-stopped --output outputs/ppo_rr_capture_first_cp225280_v1/RR_learning_signal_update7_after_aux64.json
```

Checks bind exact runtime `0ff03eafeeb75ba8505a99478cea2a6243cf93f5`, current tracking revision, 448 observations/512 rollout, sealed source/destination checkpoints and pointer hashes. Expected counts are CP228352 AUX64→CP228864 AUX64: 3,072→3,584 local decisions, PPO6→7, Adam120→140, AUX64 unchanged. The complete AUX event ledger must remain unchanged. Prefix remains PPO credit0.

The report checks all512 sampled raw actions/observations/old likelihoods against stored tensors and actual20 minibatches (five exposures per row). It reports saved phase counts, Gaussian logp, GAE normalization, checkpoint state hashes/frozen prior, same-input RR conditional means/local head/sigma, knee headroom and real angle/gap response. It uses two CPU forward passes and starts no optimizer, publication or physical process.

AIR/TOP/drop/reacquisition/success cohorts are derived afresh from this run; there is no first41 or new-success assumption. Contact counts refer to actual15Hz endpoints plus stored native-observer hold, not an independent120Hz rescan. An empty cohort is reported as n=0, not an invented successful example. Only current eligible TOP bearing plus real0.5s hold can support a recorded local success; this is not full-task or deterministic success. Fixed-input mean shifts and saved-GAE scores are not causal physical interventions or exact clipped gradients.

Tests: **7 new stdlib cases PASS**; combined common/update5/update6/update7 **35 PASS**. Actual tensor execution remains for the parent after exit; it was not performed during preparation.

Analyzer SHA-256: `765c4eb3fa4b6365659b3089082e9ca732870a244b003a2dec9b596cba1d7817`.
