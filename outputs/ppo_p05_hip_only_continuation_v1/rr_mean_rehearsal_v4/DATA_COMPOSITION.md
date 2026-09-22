# RR v4 fixed data — preparation only

Actual CPU build and read-only load passed. The CPU helpers exited; no actor was constructed, no fit/budget/optimizer/Isaac run occurred, and no checkpoint was written. All 17 targeted tests passed, including wrong runtime/old AUX lineage, mean-substituted labels, conflicting protection phases, split/window drift and bound-data tampering negatives.

The operation binds **actual CP214400**, SHA `8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9`, counters 214400/1640/32800 and frozen runtime5fd. It verifies the full actual source metadata against its sidecar, then stores only full-object digests and compact summaries of all three front AUX events96/96, old limited AUX7/8 and the three original counter origins. No historical ledger is rewritten. The compact manifest is 426,707 bytes; the read-test receipt is 1,323 bytes. Numeric NPZ stayed unchanged during this metadata-only compaction.

## Positive labels

Only block03 episode0 source indices747–1259, globals204524–205036, input ticks5976–10072 are included. The 513 direct saved389 rows and original all12 stochastic raw labels are matched to sealed rollouts1563–1567, μ/σ/logp, single sampling draw, native dispatch, all-one residual mask, valid physical state and no assist-owned action. The source collecting checkpoints and boundary RNG are bound. Per-decision RNG replay is not claimed.

| Phase | Source | Train | Validation |
| --- | ---: | ---: | ---: |
| P09 | 345 | 230 | 115 |
| P10 | 1 | 1 | 0 |
| P11 | 167 | 111 | 56 |
| Total | 513 | 342 | 171 |

Within P09/P11, fixed source-order offsets0,2 modulo3 are train and offset1 is validation. P10 is training only. All selected-window rows remain; no raw-action/outcome cherry-picking occurs. The two splits are correlated samples from one trajectory, not independent physical trials.

Only X17 is explicitly reencoded using the exactly aligned preceding physical endpoint and existing a802→5fd potential migration. Recomputed old potential matches directly stored float32 X17. Current X17 differs on237 rows, with float32 delta0…0.00983327627; every other input column and every raw label is unchanged. The old direct389 is retained separately. Source μ/σ/logp are provenance, never replacement labels or current-model PPO likelihood. Old rewards/GAE/returns are not training targets.

The window contains actual RR placement at tick8726 and continuation into P10/P11. It has169 placed-history inputs,54 current legal-TOP inputs,459 AIR inputs,427 current-qualified inputs and zero ground inputs. Thus placement history is explicitly **not** uninterrupted TOP contact or a guarantee that every AIR state is individually successful. P12 and the later RL failure are excluded. Earlier FL preparation is not relabeled pure-policy.

## Protection inputs, without action targets

305 inputs comprise frozen field-reconstructed deterministic P01 decision2 (1), frozen field-reconstructed P02 (all254, including84 formerly source-only rows), the existing19 direct probes filtered to P03–P08/P12 (16), and recent sealed block08 current389 P04/P05/P06 (2/16/16). Recent selection is uniform source-order linspace, at most16 per phase, not result-based. Frozen prefix actions are not learner protection rows.

Actual combined counts: P01=1, P02=254, P03=4, P04=3, P05=20, P06=20, P07=1, P08=1, P12=1. Separate original stochastic P01 probes=2, unlabeled. No P09/P10/P11 protection conflicts with the fit scope, and **P13 has no actual coverage**. P01 reset0 is neither reconstructed nor invented.

The kernel must generate protection targets from the actual current CP214400 conditional means. Historical action labels are not returned for protection. This finite, heavily front-weighted probe set is not a mathematical whole-phase mean-invariance or future closed-loop guarantee. Historical gap overlap is not whole-body current-state equivalence.

## Frozen interface and bindings

`data_v4.load_reviewed_data(metadata, contract)` returns train/validation observations and actual raw targets, exact per-phase groups, protection observations, two unlabeled P01 observations, and compact receipt. It rejects any source other than actual CP214400/frozen5fd; a future potential migration is outside this binding.

- data_v4.py SHA: `34df08f6281965efbdabe51d3c3cd35683c0a2158c63a0865c2f66d4a260edb1`
- candidate_data.npz SHA: `607c057d209aeb0278115632507f67042f10a59bc1ee142884d10a42b0265144`
- data_manifest.json SHA: `5136b93038e29c13c74820d75275c45c4a8471b7064beaa74ea6eff95d62af9f`
- loaded receipt digest: `c07be742b9e8f8854ed9f6755654e4057a87549b5e742a12c499742e16e64dea`

Preparation adds zero AUX/PPO credit and is not authorization to fit. Current-trajectory success and full-task success remain unproven by this historical dataset.
