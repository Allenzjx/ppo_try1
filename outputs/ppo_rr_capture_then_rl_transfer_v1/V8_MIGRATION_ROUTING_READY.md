# V8 signed-gap wheel continuity — strict preparation

No real publication, PPO/AUX optimization, simulator run, or pointer promotion was performed during this preparation.

Source HEAD: `60abc00957c0c988da6689e2fb3cfd5c8da22a47`.

| Source | Checkpoint SHA256 | Manifest SHA256 | Decisions / PPO / Adam |
| --- | --- | --- | --- |
| ancestor | `47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430` | `a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1` | 220544 / 1688 / 33760 |
| latest | `48091ffdb1b0324ae2941f4f9a376d1b4cf9702da4c10b844f7c185ca7e1bb07` | `118f4d25072db174ca6e69ddf6a32835a413fe987a33fe748db8349f56c0080d` | 221184 / 1693 / 33860 |

Strict runtime delta: wheel transform, execution profile revision, new migration module, migration hook, training hook/routing, CLI routing. Six paths only; assist, context, task spec, physics, rewards, sigma, caps, numerical codec and feature meanings remain unchanged. The already-observed signed gap/contact now select `forward_floor` instead of `release_slew` within the existing signed band. The armed bit was already true and is not a new or changed feature. Negative-gap gain remains clipped to zero, retaining a nonnegative support-wheel floor; actual TOP/support/phase/source-stop release rules remain unchanged.

Schema `wlr50_clean.rr_signed_wheel_same410.v8`, factor `rr_signed_wheel_v8_factor`, receipt `rr_signed_wheel_v8_migration`. Exact actor/critic, complete Adam/effective LR, Identity, full RNG, all origins/AUX and complete v7 ancestry are preserved. Fresh rollout, zero migration learning credit. Neither source borrows the other's counters.

After final reviewed HEAD, root may run the prepared source-device publisher:

```powershell
$env:PYTHONPATH='src'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_rr_capture_then_rl_transfer_v1/publish_rr_signed_wheel_v8.py --expected-head <EXACT_FINAL_HEAD> --source ancestor --publish
```

Use `--source latest` independently. It creates unique source/head names, officially saves and independently reloads, and does not promote the main pointer. Actual ancestor training requires `-CheckpointOutputBranch ancestor220544_signed_wheel_v8`; full-episode natural P01 and checkpoint-policy suffix continuation both retain same-branch history/pointer routing. V8 receipt validation cannot fall back to historical v7/v6. Ancestor training without a branch remains rejected at CLI and direct API.

Targeted CPU results: v8 plus branch tests **78 passed, 1 skipped in 13.89s**. Both source roundtrips and actual synthetic128-row normal-save carry for v6/v7/v8 passed; those synthetic updates carry no robot PPO credit. The sole skip was Windows symlink creation privilege, with separate alias-escape coverage passing. Final new-module rerun covers the corrected unchanged-observation metadata assertion. All test processes exited and `git diff --check` passed.

Frozen preparation hashes:

- New module: `e5e7942caa00011aa124fb6fc10f06804fae8d354f8d11925a14bbc81540308f`
- New module tests: `3fe66029c78ee39226b30329f9f081348a1dc822f8113055067668ea45b48c33`
- Branch tests: `6cc5e1937faaf032a5c3f22ba3543f9eeb9a5a408b2dc13eeb5d7a4f529673c4`
- Publisher: `4cb0f244b00d533603cbcab02a6ab9321b10de0843a19f1f1549a6f23b4627aa`
