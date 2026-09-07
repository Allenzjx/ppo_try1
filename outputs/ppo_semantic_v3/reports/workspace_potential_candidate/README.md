# Workspace reciprocal potential — independent CPU candidate

UNWIRED. Only this output directory was created; no production/config/test-suite file changed. No physical task was run and no new task success is claimed. C103168 evaluation was not touched.

`workspace_potential_candidate.py` exports exactly:

```python
interval_distance_progress(x, lower, upper, within_lateral_span, scale=.25)
```

For lateral-valid measurements it returns1 within the existing interval and `scale/(scale+distance_to_interval)` outside; lateral-invalid returns0. It validates all inputs first: finite real non-boolean numeric inputs, actual boolean lateral flag, strict lower<upper, positive scale. It reads no phase/history/pose/clock and changes no hard task predicate. Production would retain the existing .25m scale, not expose a new tuning gate.

## Numeric implementation

Avoids `scale+distance` overflow by dividing the smaller positive magnitude by the larger. If two finite opposite-sign coordinates have an overflowing difference, it normalizes the coordinates before subtracting. The bounded result remains finite even at ±`sys.float_info.max`.

IEEE limitations remain explicit: extremely distant progress can correctly underflow to0; an extremely small positive outside distance can round the soft value to1. In particular, at `nextafter(-.22, -inf)`, the candidate is1 while the current original workspace predicate is below1. **Never feed this soft value into entry/completion `>=1` logic.** The intended integration changes only the two workspace contributions within physical_potential. Public predicates, hard geometry, Q/C/P, nominal scheduling and evaluator success remain unchanged.

Stable arithmetic can differ from the naive ratio by an ulp at ordinary inputs. This is explicitly a new soft-reward version, not claimed bitwise identity with an old formula.

## Executed result

Actual bounded CPU run: **17 tests passed, 0 failures/errors/skips**, test harness wall0.004212s, process exit0. See `cpu_trial_01.json`. Loop/subTest input variants are not counted as extra tests. No Torch/PT/GPU/Isaac module or production runtime module was imported.

Coverage includes all six C101376 first/nearest/terminal rear distances; continuity across old−.47m truncation; both interval boundaries; symmetric left/right distance; approach/retreat ordering; lateral false; bool/string/NaN/Inf/invalid interval/scale; finite subtraction and denominator overflow; positive subnormals and legitimate underflow; near-boundary rounding; high-precision Decimal comparisons; and stationary-Phi PBRS not paying positive reward.

The harness **reads and compiles only** the unchanged production `TaskStageSupervisor.predicate` and `_clip` AST into a local namespace. It exercises the workspace branch to demonstrate preserved original completion values. It does not import or monkeypatch the live module, construct a complete supervisor, or claim end-to-end integration.

Exact command (PowerShell, repository working directory):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONPATH='C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\workspace_potential_candidate'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -P outputs/ppo_semantic_v3/reports/workspace_potential_candidate/test_workspace_potential_candidate.py
```

The script prints its receipt and does not write files. The saved trial receipt was copied from the successful stdout using apply_patch.

## Remaining production integration coverage

Root's future integration still needs to establish:

- v3-only mode validation and exact old-mode fallback; only the two workspace/preparation potential terms call this helper;
- fixed original `.1` leg share, `.85/4` global share, all-leg symmetry, and unchanged predecessor gating for unload/lift/carry/capture;
- unchanged placed retention, public predicate/entry/completion, nominal inputs, evaluator history and terminal success on the same actual snapshots;
- unchanged324-column layout, explicitly changed Phi scalar/reward contract, new-MDP checkpoint provenance and fresh rollout; no mixing old rewards/storage;
- real logs exercising the newly responsive far-distance branch, without equating arithmetic response to physical progress or success.

These are not claimed complete by the17 candidate tests. Current live tasks do not depend on this candidate. Work stopped after the bounded CPU result and these output files.
