# DRAFT runner — NOTEXECUTED, JSONL only

`diagnostic_runner.py` is an outputs-only proposed invocation path. It has not imported Torch, loaded a checkpoint, installed a real-process wrapper or run Isaac. Current formal PPO has priority; there is no scheduled/background probe.

## Scope and evidence

- Requires explicit immutable checkpoint and manifest SHA256, exact current422 policy/layout/runtime contract and source HEAD. No implicit migration, checkpoint-last pointer, mean-head reset or model alternation.
- Official existing `checkpoint_loader` loads one saved actor/critic/optimizer/normalizer. The official RSL constructor may allocate **empty** storage; no `alg.act`, transition insertion, rollout collection, returns, optimizer update or checkpoint write is called. Root explicitly accepted empty allocation. Original frozen-state hashes are checked on completion and error cleanup. All learning counters remain zero.
- Uses validated `CheckpointPolicyPrefixRequest(source='successful_nominal', target_phase='P07')` and existing provenance schema. Ordinary N+0 starts at natural P01, same source/mapper/sensors/HISTORY. At the first observed P07 decision boundary it selects the same frozen policy for the remainder. It intentionally does not invoke the training prefix wrapper's miss→freshP01 fallback; a missed prefix cannot silently become a second episode.
- Headless JSONL evidence only; **no video is produced**. Do not relabel this as the requested formal-C/video delivery. Each120Hz tick includes original policy request/density, original projected residual, complete actual ACK/headroom/explicit injected diagnostic receipt, measured physical evaluator and real joint/CoM/contact observations in the companion physical-observation stream. No scene/physical properties change.
- The unique real headroom call supplies current mapped nominal/controller. Context is captured from the adjacent committed ACK and current evaluator/source frame. Previous effective correction is read from the **actual preceding headroom ACK**; request HISTORY remains the ordinary original raw/filtered policy history, while actual-applied HISTORY remains the genuine final command. No second mapper or actuator write.
- RR owners are reconstructed read-only from the same ordered live source layers, touched/retired channels and latest authored channel event already reached by each layer. IDs exclude moving timestamps. New owner/event, contact, permission loss, no response or active saturation releases once. Unexpressible−20 endpoint/held knee before activation waits without spending the probe. Formal source clocks, late FL/RL dependency and all12 masks are unchanged.
- Original deterministic proposal log density is calculated from the official **one-forward** recorded conditional mean/effective sigma; it is not an injected-action likelihood or an additional stochastic draw. The raw proposal is not falsely credited as executed unchanged.
- Probe release/first TOP does **not** end the episode. The loop continues with the original frozen policy until the ordinary real task terminal or global200s. RR failure or an independent safety/physical-audit error is preserved. No artificial successful termination or early video cutoff.

## Audit attribution, not an audit bypass

The wrapped pure function is used by both the actual adapter and unchanged native auditor. Auditor replay keeps the same exogenous RR target in its zero-current-policy counterfactual, while independently reconstructing clamp/slew/native tensors. Actual readback, one-write and mapper-clock checks remain mandatory. An audit failure raises; there is no bypass/retry.

The original auditor's legacy `all12_policy_channels_unmodified_at_actuator` / `policy_request_execution_semantics` summaries were designed for production paths and are **not valid policy-provenance claims for this diagnostic**. Every diagnostic tick/decision and run manifest carries `ATTRIBUTION`, explicitly superseding those claims and retaining physical-readback requirements. The companion `native_tick_audit.jsonl` is the original raw recorder evidence and must be interpreted with this run manifest; its nested legacy attribution is not a claim of formal-C or pure-policy success. The explicit headroom receipt identifies selected6/7, original policy request/log density, injected requests and zero PPO credit.

No production hash changes, but the manifest truthfully declares `in_memory_runtime_override=true` and hashes runner/wrapper/planner. Identical checkpoint/source bytes do not establish identical runtime behavior.

## Checks actually executed

**25 pure tests passed in0.023s**:13 target planner,9 pure wrapper,3 runner bookkeeping/syntax. They do not load Torch/Isaac or collect the deferred native test module. The runner and native-test syntax parsed successfully. This is not an execution/physical proof.

## Native checks prepared but NOT executed

`test_native_audit_draft.py` contains4 CPU-Torch integration cases using the repository's real mapper/dispatch/tracking/native float32-buffer fixtures:

1. actual adapter dispatch + independent native audit, one mapper/write, same-probe zero-policy RR counterfactual;
2. deliberately corrupt simulator target buffer → genuine auditor rejects;
3. deliberately forge headroom receipt → genuine auditor rejects and wrapper restores;
4. pre-edge WAIT → exact original dispatch/tensors.

The module skips before importing Torch unless **root first confirms idle Isaac** and sets its explicit process-local test flag. Proposed later command, not executed:

```powershell
$env:RR_DIAG_NATIVE_TESTS_IDLE = '1'
& $rrProbePython -m pytest -q test_native_audit_draft.py
Remove-Item Env:\RR_DIAG_NATIVE_TESTS_IDLE
```

`$rrProbePython` must first be assigned root's verified current envIsaac Python executable; its location is deliberately not guessed here. No environment is spawned by this document. Do not run blanket discovery while physical sampling is active. A production-audit incompatibility blocks invocation and must be reported, not patched around.

## Remaining root review before any run

Confirm the native4 results; inspect actual current owner/event and adjacent-tick context; choose a sealed exact-runtime422 checkpoint; verify single-process resource lock and error-tail artifact requirements. The command additionally requires `--execute-real-diagnostic`, explicit checkpoint/manifest hashes and expected HEAD. No ready-to-run checkpoint path is guessed or pinned to an active update. Camera capture is deliberately omitted to keep this bounded diagnostic isolated.
