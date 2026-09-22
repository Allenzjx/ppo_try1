# Block10A + 10B — actual sealed training

PASS: **2048 policy decisions /16 PPO updates /320 Adam steps**; cumulative **218496 /1672 /33440**. No new AUX.

Combined P01–P13 inputs: **{'P01': 4, 'P02': 508, 'P03': 0, 'P04': 2, 'P05': 582, 'P06': 426, 'P07': 1, 'P08': 1, 'P09': 524, 'P10': 0, 'P11': 0, 'P12': 0, 'P13': 0}**. Frozen prefixes: **644 decisions**, all zero PPO credit.

10A: natural P01, actual512/4/80; first episode P02 incomplete at15.683333s, second P02 nonterminal partial at18.4s. Its512 unconsumed decisions were not credited; the adjusted10B separately earned1536/12/240.

Actual preedge gate: **0 input states /0 endpoints**; FL-assist-owned **486 endpoints**. These are decision samples, not full per-tick activation counts.

10B episode0: 1145 learner decisions, 97.766667s, P09, INCOMPLETE_CONTROLLER_BLOCKED.
- FL actual placed event tick4911; first sampled placed endpoint TOP=True, verified bearing=True, assist owner=[0, 1]. This is not labeled pure-policy capture.
- Afterwards: placed-history 854 endpoints, current TOP 494, verified TOP support 494, non-TOP 360, ground 0, AIR 360. Final TOP=True, support=True, ground=False, AIR=False.

10B episode1: 391 learner decisions, 47.533333s, P06, nonterminal sampling partial.
- FL actual placed event tick4908; first sampled placed endpoint TOP=True, verified bearing=True, assist owner=[0, 1]. This is not labeled pure-policy capture.
- Afterwards: placed-history 100 endpoints, current TOP 31, verified TOP support 31, non-TOP 69, ground 0, AIR 69. Final TOP=False, support=False, ground=False, AIR=True.

Both individual audits verify sealed raw389/raw12/μ/σ/logp/reward/done and five PPO uses, actual actor/Adam changes, Identity/full RNG progression, official save/reload, all five origins and the unchanged complete AUX ledgers. LR is taken from each real update, not assumed constant.

Final checkpoint SHA256 `6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227`. Training completion is not full-task success. No success claim is made for the concurrently running natural-P01 evaluation. CPU-only report generation; helper exits afterward.
