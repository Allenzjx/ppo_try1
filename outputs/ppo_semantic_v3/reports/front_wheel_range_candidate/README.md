# Front-wheel range candidate — CPU-only, not published

Status: **77 passed, 0 failed/errors/skipped**, one pytest invocation, exit0. JUnit `cpu_results.xml` reports **5.782s** (console5.78s). The Python process has exited. No candidate was connected to the running environment; production source, real configuration, existing unit tests, checkpoints, run evidence and training budgets were not edited.

## Exact candidate and artifacts

- `test_front_wheel_range_cpu.py` imports the real `build_semantic_projector`, `PhaseTransitionBridge`, semantic residual dispatch, frozen RobotAdapter/mapper and native-target auditor. Its only robot replacement is the existing CPU target-buffer fixture, with reversed articulation joint IDs; it does not simulate dynamics/contact.
- `pytest_tmp/profiles0/original_v3.yaml` is a byte copy of the current real v3 profile. `candidate.yaml` differs semantically only in16 entries: P06–P13 FL/FR `.6 → 1.2`. Rear wheels remain.6; P01–P05 remain.3; servo caps, rate limits and all other profile fields remain identical. Neither profile is used by production.
- All temporary good/bad YAML fixtures and the JUnit output are below this directory. Source profile and324 schema bytes were checked unchanged before/after the test session. No production `.pyc` or pytest cache was requested.
- CUDA was hidden; Torch intra-op/inter-op and OMP/MKL/OpenBLAS were restricted to one CPU thread. A fail-fast CUDA-lazy-initialization guard remained active, `torch.cuda.is_initialized()` stayed false, and no Isaac app/native module was loaded. No optimizer, checkpoint load, training, ffmpeg, simulator or second Isaac process was started.

## Executed coverage (77 cases)

| Group | Cases | Actual assertions |
|---|---:|---|
| Exact candidate diff | 1 | Only16 forward-wheel cap entries change;1.8rad/s² rate preserved |
|13 phases × both raw signs | 26 | Exact per-channel scale; early-phase/servo/rear values unchanged; first slew-limited tick remains equal |
| Four frozen strong suggestions × desired net −.02/0/+.02 | 12 | P07FR−.63/P09FL−1.07/P13FR−.72/P13FL+1.09: finite candidate raw and true float32 native effect; strongest opposing old.6 request still cannot reach the target |
| Same-direction hard intersection × controller0/.05 | 4 |1.2 expression cannot bypass±2.0943951023931953; original controller bias separately preserved; actual native agrees with final clamp/sign/cast |
| Positive/negative residual slew | 2 | Every tick≤.015; front reaches1.2/rear.6; zero request decays without forced zero jump |
| P05→P06 then all contiguous transitions throughP13 × signs | 16 | Feasible nonzero history retained onhold; no scale clipping/no forbidden drop; next tick decays at unchanged rate |
| Original+.3 four-wheel cancellation and candidate per-wheel cancellation | 2 | Old exact`−atanh(.5)` condition preserved; candidate front`−atanh(.25)`/rear`−atanh(.5)` yields residual−.3 and native0 |
| Same old uniform cancellation raw under candidate | 1 | New front net−.3 vs old0 is explicitly detected, not declared equivalent |
| Same raw old/new after slew | 4 | Candidate front expression/drive doubles for fixed zero nominal; rear unchanged; actual native targets differ as predicted |
| Existing324 observation encoding × signs | 2 | Previous residual old wheel scale.12 unchanged: new front±1.2 encodes±10, rear±.6 encodes±5; no clipping/float32 overflow; applied hardlimit/2.1<1 |
| Malformed candidate profiles | 7 | Short row/zero/NaN/Inf/later cap shrink/over-span/wrong phase order remain rejected |

Native dispatch checks use the genuine production composition and original mapping, not a reimplementation. Each exercised dispatch makes exactly one mapper advance, position setter, velocity setter and write. The audit does not make a second advance/write; target buffers, final-servo history and event list remain unchanged during its read. All relevant float32 selected targets match the predicted final clamp/sign/cast **exactly**; cancellation arithmetic additionally uses tight5e−14 logical tolerances and2e−9 native small-target tolerances, not widened old test expectations.

## Limits of this evidence

This proves the tested **CPU numerical/interface contract**, not world motion, contact, collision avoidance, dynamic stopping, improved stability, shared-policy learning or GPU/PhysX timing. The fixed nominal vectors are declared CPU counterexamples taken from frozen source extrema, not injected into a live rollout. The fixture retains real mapper/setter logic but does not update real measured joints or support forces.

The324 history scale test establishes no numerical overflow or clip at the new requested cap; it does **not** establish that a learned actor is insensitive to an input magnitude increasing from5 to10. Same raw producing larger front physical requests is a verified intentional difference. P13 stopping can become more sensitive even though its physical thresholds and continuous progress formula remain unchanged.

This bounded test did not rerun the proposed heteroscedastic new-MDP/checkpoint migration integration, use actual checkpoint weights, or validate real control-response windows. Existing live evidence and failed outcomes are unchanged. The root thread still decides whether to publish any single-factor cap version only after the current fixed training block and its complete evaluation; no new task-success gate follows from these tests.

Reproduction used the existing locked Python with `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, plugin autoload disabled, then:

```powershell
python -m pytest -p no:cacheprovider outputs/ppo_semantic_v3/reports/front_wheel_range_candidate/test_front_wheel_range_cpu.py --basetemp=outputs/ppo_semantic_v3/reports/front_wheel_range_candidate/pytest_tmp --junitxml=outputs/ppo_semantic_v3/reports/front_wheel_range_candidate/cpu_results.xml
```

This is a receipt of the single completed invocation, not an instruction to rerun while Isaac is active. Preserve the first result and choose fresh output/temp paths for any separately authorized later run.
