# Frozen v3 data composition — no fitting

Final bounded read-test: **19 real protection observations cover P03–P12**, including the six explicitly reviewed rear-phase rows below. P13 has no real protection sample. All 12 targeted tests passed; the CPU helper exited. This preparation performs no fit and adds no AUX/PPO credit.

`data_v3.load_reviewed_data(metadata, contract)` returns CPU float32 tensors and a complete provenance receipt. It selects no checkpoint or objective weights and authorizes no optimization.

| Role | Rows | Source and limits |
| --- | ---: | --- |
| Train P01 | 1 | CP201728 deterministic decision2/input8, field-reconstructed389; no reset row |
| Train P02 | 85 | Frozen CP201728 deterministic source indices `range(0,254,3)` |
| Validation P02 | 85 | Same episode, indices `range(1,254,3)`; temporally correlated |
| Deterministic P01 probe | 1 | Same P01 training row, not independent validation |
| P03–P06 protection | 13 | Block03 first episode, directly saved389; no new positive labels |
| P07–P12 protection | 6 | Same episode, first eligible input per phase; directly saved389, unchanged task-potential semantics; no action labels |
| Stochastic P01 probes | 2 | Block03 directly saved389; their random raw actions are not deterministic labels |

Training order is the single P01 row followed by 85 P02 rows. `train_phase_groups` provides explicit indices for the kernel's separately declared objective. The 84 unselected P02 rows remain retained in the original source. P01 has no independent validation, and P02 train/validation rows are correlated samples from one episode.

The new mean-head parameter subspace does **not** mathematically preserve P03+ means. The 19 observations are called `protection_observations`, not invariant states; they require explicit protection loss/trust checks chosen outside this loader. No coefficient or budget is selected here. Sparse historical coverage is not a guarantee for all states in those phases or for a future trajectory.

The additional six rows come only from the already reviewed block03 first episode. The bounded reader stopped after source row 1260 (1,261 rows read; cap 1,721), before its later failure. Each direct rollout input is aligned to the preceding synchronous endpoint, native verification, raw Gaussian provenance and the current physical evaluator. They are protection probes, not successful action labels or new PPO samples.

| Phase | Source index | Global decision | Input tick | Sealed rollout / offset | Reason task-potential remains unchanged |
| --- | ---: | ---: | ---: | --- | --- |
| P07 | 658 | 204435 | 5264 | 1563 / 18 | RR qualification/crossing not earned |
| P08 | 659 | 204436 | 5272 | 1563 / 19 | RR qualification/crossing not earned |
| P09 | 660 | 204437 | 5280 | 1563 / 20 | RR qualification/crossing not earned |
| P10 | 1092 | 204869 | 8736 | 1566 / 68 | RR already placed; workspace term bypassed |
| P11 | 1093 | 204870 | 8744 | 1566 / 69 | RR already placed; workspace term bypassed |
| P12 | 1260 | 205037 | 10080 | 1567 / 108 | RR already placed; workspace term bypassed |

For all six, old/current physical-potential values are identical in double precision and both equal saved X17 after float32 conversion. P10–P12 have a true RR-retirement predicate, but the placed-leg branch bypasses the workspace term: this is not mislabeled as an inactive predicate. Full row provenance and rollout hashes are in `CP212352_data_composition_rear19_readonly.json`.

Current compatibility requires exact reviewed5fd runtime/policy/normalizer semantics, immutable migrations and origins, old AUX7/8, and the complete current two-event64/64 front AUX ledger. Later ordinary PPO weights/counters are allowed; the returned receipt separately binds the actual caller's checkpoint SHA/counters/actor hash. The old CP211968 per-state admission keeps its original reference identity and is not presented as an inspection of newer weights. Neither semantic compatibility nor historical local front success establishes present actor or future-trajectory equivalence.

Validation: 12 targeted metadata/provenance/eligibility tests passed, including fail-closed runtime/origin/event1/old-AUX/normalizer/older-counter cases. An actual CPU-only load using explicitly named CP212352 succeeded with 86 train, 85 validation, 1 deterministic probe, 19 protection and 2 stochastic probe rows. This checkpoint was a read-test input, not a claim to be the latest checkpoint or an automatic source for a future fit. No actor was constructed, no optimizer run, no checkpoint written; all CPU helpers exited.

Frozen bindings:

- `data_v3.py`: `4e016816a06a3ff743883faab3309d1058a7c53bcc8495a9a36761908caddf59`
- `data_manifest.json`: `353e8f74e91dd657526e6961bf82aeab01858177d92bd0a47ea4a6eb4d01558e`
- `CP212352_data_composition_rear19_readonly.json`: `fb955ce6e208b766143ef978aaa6539dd7ebc5b6134a15cf0a642a524446d036`

The earlier 13-protection-row read-test remains intact as historical evidence: `CP212352_data_composition_readonly.json`, SHA256 `4a03604fa1f3d83c8d4906f91300c6436c74f77d8554fa8175f23f320bb2c1c5`. Its old loader/manifest hashes describe that earlier test, not this final 19-row composition.

All original candidate data, source logs, checkpoints and earlier hash-bound helpers remain unchanged. This composition adds0 AUX/PPO credit.
