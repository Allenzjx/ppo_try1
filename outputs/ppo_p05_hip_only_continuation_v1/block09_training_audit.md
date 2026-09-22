# Block09 sealed genuine PPO audit

**PASS: actual +2048 decisions /16 PPO /320 Adam**, cumulative **216448 /1656 /33120**. Planned2048, unconsumed0. Training process SUCCEEDED is not task success. Final checkpoint SHA `8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4`.

Credited phase inputs: **{'P04': 3, 'P05': 590, 'P06': 534, 'P07': 3, 'P08': 8, 'P09': 706, 'P10': 1, 'P11': 1, 'P12': 202}**; other P01–P13 phases0. Three frozen migrated-source prefixes total **792 actions/6336 ticks**, all credit0. Learner native-verified physics ticks **16378**, assist-owned endpoints **719**. Every original389/raw12/μ/σ/logp/value/reward/done matches sealed storage, and each sample is used5 times; CPU logp max error 5.7220459e-06.

| Episode | Learner decisions | Total seconds | Credited seconds | First unfinished/current stage | Result |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 484 | 49.850000 | 32.250000 | P09 | BODY_COLLISION |
| 1 | 904 | 77.833333 | 60.233333 | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 660 | 61.600000 | 44.000000 | P12 | sampling boundary; done=false |

Episode2 really earned RR placement at tick5759 and continued through P10/P11 to P12. It later returned RR to ground (first sampled ground endpoint tick6304); placement history is not current support. At61.6s it remains an unfinished P12 sampling-boundary partial, not a full PPO success. The other two actual results remain BODY_COLLISION and INCOMPLETE_CONTROLLER_BLOCKED.

Actual RR input counts: {'qualified_current': 629, 'crossed_history': 629, 'placed_history': 204, 'TOP': 9, 'ground': 1372}. New-v2 same-state gate counts: {'prefix_entry_gate_false_by_unearned_RR_bits': 3, 'v2_gate': 459, 'v1_gate_same_state': 456, 'v2_only_gate': 3, 'v2_unplaced_workspace_consumption': 425, 'v2_only_unplaced_consumption': 0, 'v2_vs_v1_effective_Phi_difference': 0}. Gate, unplaced workspace consumption and actual v2-v1 potential difference are distinct; these counts do not replace the prior same-state report or claim causal outcome improvement.

All16 actor updates have finite nonzero gradients; all12 Adam states advance320. **LR is adaptive, not constant**: min1e-05, max2.25e-05, final1e-5; per-update values are in JSON. Identity/full RNG schema and CUDA count persist, RNG advances normally. All four full event records and origins/migrations carry exactly: front96/96, RR7/8, mixed103/104, older separate7/8; no new AUX. Actual checkpoint hashes/embedded metadata and official reload verify.

Since P05 origin199680: **16768 policy decisions /131 PPO updates**; cumulative phase counts **{'P01': 52, 'P02': 6799, 'P03': 12, 'P04': 8, 'P05': 1602, 'P06': 2641, 'P07': 22, 'P08': 17, 'P09': 4645, 'P10': 3, 'P11': 304, 'P12': 663, 'P13': 0}** (prior115 + current16 updates). Prefix/AUX credit is excluded. CPU-only bounded audit; no simulation, fit, production/frozen-helper edit or checkpoint write. Helper exits after reports.
