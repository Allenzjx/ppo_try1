# RR workspace revision: first actual on-policy update

Bounded read-only CPU audit of real block05 update1590, run `20260922T0926100383683Z_g5fd88852bf20_344b6773d9854b41948d1632e1f0fc3f`. No synthetic checkpoints, GPU/Isaac, runtime/config edits or optimizer calls. CPU processes exited0. Only this first sealed batch was inspected.

- Actual migrated source `checkpoint_rr_postcross_workspace_v1_step_000207872_g5fd88852bf20.pt`, SHA256 `ac67a23edd97552cb5b2ff7f5e3484bc2e3ad47146cc7ed90fac45adfe764752`.
- Actual new checkpoint `checkpoint_step_000208000.pt`, SHA256 `73b5122ef3cc436bd04eab5d13fece48562d7f61fafee2bfcc82303eff2e6c2f`.
- **208000 policy decisions /1590 PPO updates /31800 optimizer steps**; new **128 /1 /20**.
- Runtime HEAD `5fd88852bf20c94cd74405c791a13a9fd9e0a3d8`, runtime SHA `992515c30a3ecb868198f38fb986b0a04d690557861e78d9595f8071a8fbc76e`.

| Preserved independent lineage | Origin: decisions /PPO /optimizer | Added through this real checkpoint |
| --- | --- | --- |
| Original P05 | 199680 /1525 /30500 | **8320 /65 /1300** |
| Feedback v2 | 203776 /1557 /31140 | **4224 /33 /660** |
| RR workspace revision | 207872 /1589 /31780 | **128 /1 /20** |

All branch objects and immutable migration records exactly match the actual migrated source. The complete historical task/AUX ledger—including events, reports and provenance bindings—is unchanged:7 accepted /8 attempted total,0 new AUX. Current RR migration plan SHA `2b0b4f03f2ce2a5bd98859ffa3064f40c213298b66df5e0c76dc93dcc6eb98a7` matches its actual file and embedded record. Its migration credit remains0; the128 decisions above are newly collected real PPO data, not publication credit.

## Natural start, raw storage and declared execution

This is full-episode natural P01 training: no prefix request or prefix-evidence stream, no teacher data, no inherited physics. Collection begins global207873 at tick8 /0.066667s; the128th endpoint is tick1024 /8.533333s. Stored389-dimensional input phases are **P01=2 /P02=126**; endpoint phases P01=1 /P02=127. All128 are nonterminal.

Sealed `rollout_001590.pt` has policy/critic shape128×1×389 and original actions128×1×12. Every raw sample, conditional mean/std, old logp/value, reward and done exactly matches collection evidence. Selected single-draw samples also match native dispatch audit; no extra random draws. Independent CPU Gaussian logp maximum error is **3.814697265625e-6**. All128 original samples are used exactly5 times in20 official optimizer minibatches.

All12 residual permissions are1; assist is WAIT128 with owner indices empty. Native mapping and all physics-tick receipts verify, no in-episode state writes, and no channel is replaced by capture assist in this batch. All17 appended feature values match policy records; for adjacent samples the12 assist features also exactly match the previous actual endpoint. This does not remove the declared assist from later phases or from the model lineage.

## Changed potential scalar is honestly bound, but not yet activated here

Saved observation index17 exactly equals float32 of the actual reward `potential_before` for all128 samples, including the real reset input. Each following input also equals the previous endpoint's `task_progress_potential`; reward `potential_after` equals the current task potential. This verifies the live numeric observation/reward path rather than just389-vector shape.

The immutable migration explicitly records index17's changed numerical meaning, `reward_changed=true`, `same_mdp_claimed=false`, unchanged physical dynamics, and preserved policy mapping only for identical numeric inputs. Current RR-retirement predicate is false at all128 inspected endpoints; RR qualified/crossed/placed input counts are0. **This batch does not demonstrate active RR retirement, RR improvement or task success.** It establishes actual post-migration learning and correct data/lineage continuity.

## Real optimizer and checkpoint evidence

CPU loading matches embedded metadata to the sidecar and independently reproduces actor, critic and Adam hashes. Actor and Adam actually changed from the migrated source; all12 Adam parameter step counters advanced20. Official update records finite nonzero gradients. Actual LR stays1e-5; Identity-normalizer hash, full runner config, policy contract and new runtime contract are preserved. The sidecar reports successful actual save/load roundtrip. Ordinary save metadata correctly says full_episode /natural P01, with no stale P06 suffix claim.
