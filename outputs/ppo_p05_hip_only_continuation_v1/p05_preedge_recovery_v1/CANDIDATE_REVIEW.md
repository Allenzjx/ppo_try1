# Unapplied P05 pre-edge recovery candidate

Production remains unchanged at HEAD 336b7c56d2f048d44c866b2357d33e1eb1f21dce. No Isaac, optimizer, checkpoint migration or video encoding was run.

Apply-ready text: `production.patch` (SHA256 175aab7a2f7be33682c1e53e57e9b90ed79f652d9a9975d73a591033ed79c48f). It changes only:

- `src/wlr50_clean/ppo/semantic_supervisor.py`: explicit default-off nominal mode, current predicate, diagnostics and wheel-only interception.
- `configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml`: one opt-in key, `nominal.p05_preedge_approach_recovery: p05_preedge_approach_recovery_v1`.
- `tests/unit/test_semantic_p05_preedge_recovery.py`: portable CPU regressions.

The finite window is absolute P05 stage age [30,40) seconds. Current FL AIR must have strictly positive conservative top gap, valid lateral span, front distance [-0.10,+0.005] m, prior genuine lift, FR placement history, no FL crossing/placement and at least two verified other supports. Validity/safety and fresh authored wheel events take precedence. Only the existing four-wheel .3 rad/s nominal suggestion is reused. Eligibility can return after a temporary support loss only inside the original absolute window; it cannot renew the deadline.

Endpoint evidence is the already-issued P05 source endpoint plus the subsequent consecutive physics tick under the existing one-write/one-step ordering. It explicitly is **not independent ACK verification**. A future real run still needs its existing native dispatch evidence. Missing or nonconsecutive tick evidence cannot enable the new advice; upstream malformed sensor contracts may already raise.

No phase handoff, capture/pending flags, five scheduler bits, servo source targets, source clock, evaluator, physical potential/reward, action caps, sigma, residual masks, mapper, HISTORY or capture assist was changed. The existing source stop is legitimate; this is separately versioned finite recovery advice, not a stop-bug fix or learned PPO gain. No new P06 layer is started by this predicate.

Observability is limited: the current nominal recovery advice is encoded, but source internals and full contact classes are not individually encoded. The diagnostic deliberately does not claim the 389-vector is fully Markov. Official checkpoint-policy/successful-nominal prefixes preserve their existing controller clocks; no legacy late-offset handoff equivalence is asserted.

Validation:

- 61 targeted synthetic CPU tests PASS, including expiry, positive-gap constraint, support loss, history negatives, source endpoint/tick evidence, fresh stop precedence and old post-cross branch separation.
- Output-only `candidate/verify_readonly.py`: production baseline versus candidate with mode absent has identical Full12, tracking, normal bias, diagnostics and source samples for 5,000 P05 ticks.
- AST comparison proves TaskEvaluator, TaskStageSupervisor, physical potential and continuous scheduler unchanged; single config-key delta verified.
- Generated apply_patch text was applied **in memory only** and matched all candidate lines; no production patch applied.
- Independent static review found no blocking implementation issue.

Re-run candidate tests with CPU Python, CUDA_VISIBLE_DEVICES=-1:

`python -m pytest --confcutdir=outputs/ppo_p05_hip_only_continuation_v1/p05_preedge_recovery_v1/candidate outputs/ppo_p05_hip_only_continuation_v1/p05_preedge_recovery_v1/candidate/tests/unit/test_semantic_p05_preedge_recovery.py -q`

Only the three listed files belong in the production patch. The overlay conftest and read-only verification helper are output-only. Root owns the separate reviewed same-389 control-MDP migration. These results are wiring evidence, not physical recovery or task success.
