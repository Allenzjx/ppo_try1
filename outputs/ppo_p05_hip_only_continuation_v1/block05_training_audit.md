# Block05 sealed natural-P01 training audit

**Training completed correctly, but task coverage stopped at P02. The new RR workspace rule received zero active samples.** This is not a traversal success or evidence that the RR reward fix improved rear-leg behavior.

Run `20260922T0926100383683Z_g5fd88852bf20_344b6773d9854b41948d1632e1f0fc3f`, runtime `5fd88852bf20c94cd74405c791a13a9fd9e0a3d8`. CPU audit exited0 in5.787s; no GPU/Isaac, production/config/checkpoint changes or additional optimizer calls. An initial reader-only class-name import error was corrected to existing `TaskStageSupervisor`; it was not a training/runtime failure.

## Actual learning and checkpoint

- All **2048 planned decisions consumed**,0 unconsumed; **16 genuine PPO updates1590–1605 /320 Adam steps**.
- CP209920 lifetime **209920 decisions /1605 PPO updates /32100 Adam steps**.
- Checkpoint `checkpoints/history/checkpoint_step_000209920.pt`, independently hashed SHA256 `11476243212e12e5e6aa0591de4b00f41f75b1f718e142a862112d66736255ac`; actual save/load roundtrip recorded true.
- Actual source was the zero-update migrated CP207872, SHA256 `ac67a23edd97552cb5b2ff7f5e3484bc2e3ad47146cc7ed90fac45adfe764752`. Migration is not learning credit.

| Preserved origin | Original decisions /PPO /Adam | Cumulative added through CP209920 |
| --- | --- | --- |
| P05 |199680 /1525 /30500 | **10240 /80 /1600** |
| Feedback v2 |203776 /1557 /31140 | **6144 /48 /960** |
| RR workspace revision |207872 /1589 /31780 | **2048 /16 /320** |

All origins and immutable migration records remain intact. Complete historical AUX ledger/events/data/helper/source bindings are unchanged:7 accepted /8 attempted total, **0 new AUX**. Identity, full runner config and policy contract preserved; effective LR1e-5. Actor and Adam hashes changed from the actual source; all12 parameter-state Adam counters advanced320.

## Exact phase and physical coverage

Natural P01 starts, **0 prefix/teacher decisions**. Stored input counts: **P01=14, P02=2034; every phase P03–P13=0**. Endpoint counts: P01=7, P02=2041; P03–P13=0. All2048 stored phase one-hots match collector phases.

Actual task physics **16360 verified ticks /136.333333s across7 episodes**, excluding reset/settling. All12 residual permits open; no in-episode state writes. Capture assist owns0 endpoints because no episode reaches its FL window.

RR input current qualification, crossing history, placement history and known current TOP counts are all0; corresponding endpoint counts all0.2041 input TOP states are bound to adjacent recorded endpoints;7 initial-reset TOP classifiers remain explicitly unknown rather than fabricated false. Their RR-retirement necessary Q/cross bits are false.

The new rule has **0 predicate-active inputs,0 actual workspace-share-consumed inputs,0 nonzero new-versus-old potential differences**; endpoint counts and summed potential difference are also0. This is verified by pure potential recomputation on each recorded state with/without the opt-in and matching the actual task potential—not inferred merely from phase labels or configuration presence. Branch2048/16 counts mean training under the new version, **not2048 samples exercising the RR change**.

FR active-lift history appears in1593 endpoints, but FR crossing/placement/TOP remain0. No leg crosses or places in this block. The first unmet task is the P02 FR approach/front-edge crossing, before FL or either rear-leg task.

## Episode outcomes

| Episode | Decisions | Physical ticks /seconds | Actual result |
| --- | ---: | --- | --- |
|0|337|2694 /22.450000|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|1|331|2646 /22.050000|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|2|235|1876 /15.633333|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|3|324|2585 /21.541667|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|4|234|1866 /15.550000|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|5|334|2669 /22.241667|P02 INCOMPLETE_CONTROLLER_BLOCKED|
|6|253|2024 /16.866667|P02 partial, nonterminal at saved update boundary|

All6 terminal outcomes remain incomplete, not success or relabeled physical safety failures. The partial episode is not counted as another completed failure. Terminal rows and original done values are retained in training.

## Storage and comparison limits

All16 finite sealed rollouts have actor/critic389 observations and raw12 Gaussian actions. Original raw actions, conditional mean/std, old logp/value, rewards, done, potential index17 and appended features match recorded collection. One sample per decision, no extra random draw; each sample is used5 times in20 official minibatches. Independent CPU logp maximum error **5.7220458984375e-6**. All16 updates record finite nonzero gradients, changed actor hashes and a continuous before/after hash chain.

Earlier natural-P01 block03 reached P12 in one episode and supplied real RR placement/TOP samples. Earlier block04 supplied P09–P11 and RR placement after an **uncredited frozen-P01→P06 prefix**; it did not test the updated learner's natural-P01 front passage. Current block05 supplies only front-stage learning and no RR-rule activation. These different checkpoints/curricula show lost task coverage, but do not establish its unique cause. In particular the RR-specific potential rule did not alter potential on any state sampled here, so these data do not support blaming its active shaping for P02 failure or claiming it has been trained successfully. Root's separate saved/reloaded deterministic run remains the actual evaluation of CP209920.

Machine-readable counts: `block05_training_audit.json`. Reproducible stdout-only reader: `audit_block05_cpu.py`. No further rollout, video or runtime work was performed by this audit.
