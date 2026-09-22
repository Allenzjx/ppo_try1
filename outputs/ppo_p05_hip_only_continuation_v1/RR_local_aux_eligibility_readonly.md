# RR local auxiliary eligibility — read-only decision support

**Conclusion:** There is a defensible, very short *RR contact plus immediate handoff* positive in block03. There is no sustained P09→P12 success demonstration. The existing finite-AUX helper cannot consume this data/current389 actor unchanged. No auxiliary update is execution-ready from this review alone; current formal PPO is unaffected.

Scope: real run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`, only the previously identified first RR-placement/handoff window and its negative tail, ticks8712–10640. No new checkpoint, dataset file, implementation, optimizer step, simulator or GPU use. CPU processes exited. The enclosing episode later remains P12 INCOMPLETE, not a success demonstration.

## Physical positives and mandatory negative context

| Actual recorded window | Evidence | Eligible interpretation |
| --- | --- | --- |
| Placement event tick8726; decision204867 ends8728 | RR placed history first true, but endpoint AIR, force0 | Event/approach context only, not sustained-support positive |
| **Decisions204868–204877**, actions from tick8728 through8808 | 10 TOP/support endpoints; consecutive TOP counter8→80, confirming80 consecutive physics samples (0.667s). P09→P10 at8736, P10→P11 at8744, then8 P11 requests | **Local contact plus immediate handoff positive**, not stable entire P11 or full traversal |
| Decisions204878–204899, endpoints8816–8984 | 22 endpoints without RR TOP/support; force0; gap reaches39.426mm | Keep as immediate negative follow-through; do not relabel or omit when describing the positive's durability |
| **Decisions205036–205042**, endpoints10080–10128 | P11→P12 at10080, then6 P12 requests;7 TOP/support endpoints, consecutive TOP7→55, force13.234–14.377N | Separate local reacquisition/P12-entry positive; not a continuous extension of the earlier contact |
| Decisions205043–205082, endpoints10136–10448 | 40 endpoints without RR TOP/support; force0; gap reaches56.819mm | Negative retention context. No reliable RL completion claim |
| Decisions205083–205106, endpoints10456–10640 |23 TOP endpoints before leaving legal top XY; front margin128.413→−6.997mm. At10640, top-surface force2.009N still exists but classified legal TOP=false | **Not a positive handoff/forward-progress demonstration**: contact while retreating toward/off the edge is not sufficient |

First10 positive endpoints have measured RR force1.981–14.300N, legal TOP XY, finite/valid physics, no terminal, and one to three actual other supporting contacts. RR front margin decreases149.866→120.378mm during this short handoff; therefore its label cannot claim forward locomotion improvement. These measured values are evidence, not proposed fixed-force/support-count gates. The later7-state positive has front margin144.821→142.250mm. A transient RL active-lift history at the end of that short segment is not durable RL crossing/placement.

No assist owns any channel in the selected positive/adjacent-boundary records; all12 residual permissions are1, native mapping is verified and in-episode state writes are absent. They are actual stochastic-policy actions from natural-P01 learning, not manual diagnostics or frozen-prefix/teacher actions. The whole episode still used the declared FL assist earlier.

Sealed rollouts1566 and1567 retain exact389 observations and original12-dimensional raw samples. Selected positives/boundaries match those raw samples exactly and each was already used5 times in formal PPO. Any future reuse is explicitly *additional off-policy auxiliary supervision*, with0 new PPO samples/updates credited. Labels must use original raw latent actions, never final actuator targets, mapper deltas, or transformed targets assigned fabricated Gaussian log probabilities.

## Existing finite-AUX: reusable safeguards, not a drop-in RR implementation

Inspected `outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary/{finite_auxiliary_mean.py,reviewed_data.py,auxiliary_cli.py}`. Hard restrictions include exact old372 input, old task-conditioned actor class/kernel, P05 crossed AIR state, FL row0 only, old FL diagnostic schema and `FL_AIR_gap_approach_not_capture` label. Its CLI and checkpoint name are also FL-specific. Old conditional-sigma reconstruction is not the current389/receiving-wheel profile. Do not bypass these checks or slice away the17 current features.

Reusable design: independent temporary SGD leaves for selected mean-head parameters; real actor functional forward and genuine HISTORY rho=.9 derivative; no changes to trunk/critic/log-sigma/normalizers/PPO Adam/LR/RNG; empty rollout at a legal update boundary; finite nonincreasing loss; rollback-and-stop on the first trust-bound violation; explicit auxiliary ledger and unique saved/reloaded checkpoint; ordinary PPO then collects a fresh rollout. Existing ceilings are16 attempts, auxiliary LR≤.05, same-state training REQUEST shift≤1degree, holdout shift≤.25degree, joint Gaussian KL≤.1. These are historical ceilings, not evidence that the same settings or old FL-only tests suffice for RR.

An eventual small adaptation would need the actual389 actor/distribution, explicit selected RR mean rows and channel-wise physical units, full Gaussian KL for all changed rows, a new truthful local-contact label, sealed raw-data binding and current-checkpoint inspection. It must not deploy a teacher/controller, change nominal or FL assist, silently supervise other channels, reset the network/Adam, or insert this old trajectory into PPO storage. No such adaptation was made here.

The ordinary trainer carries the complete `task_conditioned_hip_wheel_branch`, so the auxiliary ledger can persist without a new MDP or observation layout. Preserve historical7 accepted/8 attempted and append a separately named RR event/counter contribution; retain P05 and feedback-v2 origins/counters and current migration records. The older receiving-wheel migration's exact7/8 validator concerns that historical migration boundary—do not rewrite its immutable factor or rerun it merely to permit a later auxiliary event. Existing FL-specific event/checkpoint names must not be reused to mislabel RR learning.

## Why this is not yet an AUX execution recommendation

- One correlated trace and0.667s/0.458s local contact windows do not isolate RR hip/knee causality from wheels, other legs or nominal changes. The recorded raw innovations have mixed signs; there is no evidence for a universal hip/knee offset.
- The first positive is followed immediately by contact loss; the second is followed by longer loss. Supervision of all TOP states would even reward the later retreat. Retain all boundary/negative context; do not concatenate isolated positives into a success trajectory.
- Shared RR head rows affect P01/P02 and whole-body support on natural starts, even if other same-input mean channels and sigma stay numerically unchanged. Stored HISTORY contains previous stochastic samples; fitting those states does not prove deterministic replay can create them.
- No independent389 natural-P01/early-phase holdout was selected in this bounded review, and no current latest-checkpoint same-state error/gradient/trust inspection was performed. The old24-state372 holdout is incompatible. Those checks and a saved/reloaded natural-P01 deterministic run would be necessary before claiming benefit; they are not new gates on the currently running PPO.

**Decision support:** keep the first10 actions (plus acquisition and immediate-loss context) as a narrowly described candidate, and the later7 actions as separate corroborating reacquisition context. Do not execute the old helper or call either segment sustained rear-task success. Root may decide on a small reviewed adaptation after the active block's legal boundary; all current learning remains genuine PPO with0 new AUX from this audit.
