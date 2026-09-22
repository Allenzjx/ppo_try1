# Block04: first actual P06-frontier PPO update

Bounded CPU-only audit of real run `20260922T0814225788964Z_ga802b24d78df_790387038d25495ea5f001f934d3ade2`, sealed update1574 only. No simulator, GPU, runtime/config/checkpoint writes, reward changes, or synthetic checkpoints. Reproducible output-only helper: `audit_block04_first_update_cpu.py`; CPU process exited0.

- Source `checkpoint_step_000205824.pt`: actual SHA256 `c1615e5d6987306d41dcdbbd4c3767bd0d28d30fda559d149e7c5b2ade886735`.
- New `checkpoint_step_000205952.pt`: actual SHA256 `b895d0ee429ebb8b58b451413c4a01d5898667b136b9a8e54b3442cc3f3bc364`.
- Counters: **205952 decisions /1574 PPO updates /31480 optimizer steps**, actual increment **128 /1 /20**.
- Runtime unchanged: `a802b24d78df5f1b8f0caf914ce7a8a68a98db8e`.

## Real prefix and credited collection

Frozen source-checkpoint policy performs natural P01→P06 initialization: **562 uncredited decisions /4496 physics ticks /37.466667s**, requested-phase counts P01=2, P02=335, P03=4, P04=1, P05=220. Every prefix record declares `policy_credit=false`; native verification and no in-episode state writes pass. Actual credit begins in P06 at tick4496, before RR qualification, crossing or placement. Source hash, frozen actor hash, independent frozen storage, runtime and policy contracts match actual CP205824.

Only new global205825–205952 populate `rollout_001574.pt`: **128 P06 inputs and128 P06 endpoints**, 1024 credited physics ticks, ending tick5520 /46s. Stored physical-core decision counts start563, while credited decision counts start1. Neither frozen-checkpoint nor teacher initialization enters PPO storage. No terminal samples or full-task success in this bounded batch. RR qualification/crossed/placed inputs and RR current-TOP endpoints are all0: this is genuine preparation coverage, not RR landing coverage.

## Raw policy versus declared FL capture assist

Policy/critic observations are identical389-vectors; original Gaussian raw actions are12-vectors. Every stored raw sample, conditional mean/std, old logp, old value, reward and done exactly matches collection evidence. The selected single policy draw also matches native dispatch audit; no extra random draws. Independent CPU Normal logp recomputation maximum error is **7.62939453125e-6**. All128 original samples appear exactly5 times in the20 official optimizer minibatches.

All12 residual permissions remain1, but **all128 endpoints explicitly assign FL hip/knee indices[0,1] to capture assist**. Other10 candidates are unchanged by this assist; no RR assistance. Final held FL knee verification passes. Input assist states: **HOLD44 /DESCEND71 /BLOCKED13**. All12 appended assist features exactly match the previous real endpoint, including the final prefix endpoint for the first learner input; all5 continuation features match recorded policy inputs. Every endpoint declares `hold_to_air_progress_window_v2`. FL pending and pending-advanced input flags are0. These are on-policy samples with an observable action transform, not a claim that all12 raw policy channels reach the actuator unmodified or that contact is pure-policy-generated.

## Real ordinary-save lineage and optimizer

| Lineage | Preserved origin: decisions /PPO /optimizer | Current added counts |
| --- | --- | --- |
| Original P05 | 199680 /1525 /30500 | **6272 /49 /980** |
| Feedback v2 | 203776 /1557 /31140 | **2176 /17 /340** |

Both lineage objects and migration bindings exactly match source CP205824. The inherited AUX ledger is unchanged at7 accepted /8 attempted, with0 new AUX updates. Metadata is actual P06 suffix sampling (`natural_P01_frozen_checkpoint_policy_prefix_then_semantic_suffix_N1.v1:P06:offset_0`), not stale P01/P12 metadata.

CPU loading matches embedded metadata to sidecar and independently reproduces actor, critic and Adam hashes. Actor and Adam changed; all12 Adam parameter step counters advanced20. Official update records finite nonzero gradients. Actual LR=1e-5, Identity-normalizer hash, full runner configuration and policy/runtime contracts are preserved. Sidecar records successful actual save/load roundtrip. No later batch was audited.
