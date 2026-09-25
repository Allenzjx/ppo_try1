# Update6 read-only analyzer — sealed tensor verification completed

New files: `analyze_rr_learning_signal_update6.py` and `test_rr_learning_signal_update6_stdlib.py`. Only outputs files changed. New stdlib tests: **13 PASS**; combined common/update5/update6 stdlib regression: **28 PASS**. No Torch/PXR/project-model import or Isaac action was performed during preparation.

Run only after the parent explicitly confirms the sole Isaac process has exited and `train_tracking_fixed512_1e10d39/run_manifest.json` is sealed COMPLETE:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs/ppo_rr_capture_first_cp225280_v1/analyze_rr_learning_signal_update6.py --isaac-stopped --output outputs/ppo_rr_capture_first_cp225280_v1/RR_learning_signal_update6_tracking_fixed.json
```

After the sole Isaac run exited normally, the parent executed the corrected analyzer successfully in about 5.5 seconds. The result is `RR_learning_signal_update6_tracking_fixed.json`: update6, 512 fresh samples, CP227840→CP228352, local totals 3,072 decisions / 6 PPO / 120 Adam / AUX0. This read-only command cannot launch another physical process; it did not perform training or establish physical success.

Correction record: the prepared `validate_loaded_source` mistakenly listed nonexistent `semantic_policy.py`. The parent replaced that entry with the actual model dependency chain `semantic_history_actor.py`, `semantic_p05_capture_actor.py`, and `semantic_rear_owner_actor.py` (retaining the local actor and training checks). These files are checked against the sealed runtime hashes; the guard was corrected, not removed or relaxed. No task/reward/model state changed. Preparation-only statements above refer to the earlier stdlib preparation, not the subsequently completed tensor verification.

Checks before tensor imports: COMPLETE training run; exact source `1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81`; `pending_source_tracking_inheritance_v1`; matching 448/512 runtime and sidecars; exact rebound CP227840 SHA `81aa94a31039730c7ad87a45b53de22a12361047d324d4d5700b835d7cddbdb5`; empty rollout and save/reload proof; update5→6, 2,560→3,072 decisions, 100→120 Adam, no AUX; actual 512 on-policy receipts; first successful episode remains the observed 41 rows ending8312. Missing or different data fail closed, rather than substituting old runs.

Analysis uses one CPU thread and only two checkpoint forwards on the same actual448 inputs. It reuses common storage/raw/Gaussian-logp/normalized-GAE/KL checks and update5's knee score calculation. Reports separate:

- first successful **episode41**, its precontact and contact/hold subsets, other471, and all512; these are cohorts, not 41 successful action labels;
- conditional RR mean, local head and sigma before/after; actual reward/return/value/raw GAE/normalized advantage and collection-distribution mean-score;
- knee headroom loss versus measured knee/gap response, without claiming single-axis causality;
- actual current source/carrier and previous-ACK tracking receipts. Legal P10 tracking is not confused with unintended tracking inherited from a deferred full12 event. No old5f tracking-defect report is read.

Exact source647 tracking cannot be independently reconstructed from learner-only rows that begin after the gate; this limitation is explicit. Carrier provenance and actual mapper tracking are reported from current receipts. Full temporal GAE recurrence/tail bootstrap is not independently reconstructed: the script verifies saved returns−values and whole-rollout normalization, keeping original signs.

No optimizer, auxiliary fitting, deployment, checkpoint save or fresh physical evaluation occurs. Fixed-input mean shifts are not deterministic task success. Output must be a new file within this isolated outputs directory.

Corrected, executed analyzer SHA-256: `91f20debbedcc928ae20fa9c11988a5d1ade1aa2864d1c37158c7a435c408024`.

Superseded preparation SHA-256 (before dependency-list correction): `16af2b9136b1438c4c58f7f1a549eef806f2c72fac9ac3642c4c864944edf238`.
