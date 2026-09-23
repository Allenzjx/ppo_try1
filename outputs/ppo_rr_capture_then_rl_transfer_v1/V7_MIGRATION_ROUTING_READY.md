# V7 signed AIR capture migration — prepared, not published

The two immutable sources are the officially saved/reloaded v6 checkpoints at HEAD `97c4367ee293cb4dd191944ef663bbda63a5b235`:

| Source | Checkpoint SHA256 | Manifest SHA256 | Decisions / PPO / Adam |
| --- | --- | --- | --- |
| ancestor | `0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01` | `5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c` | 220544 / 1688 / 33760 |
| latest | `297d68b5fab18270903bce708ad1dc0062145da3fe876fa88316c998128e879f` | `8631c3587d2b7c33b087a7bf8093bc4b0f3967e102f970cc5b1772b943a345d1` | 221184 / 1693 / 33860 |

Strict migration scope is seven runtime paths: assist, context, execution profile, new `semantic_rr_signed_contact_migration.py`, semantic migration, semantic training, and semantic CLI. Other configuration bytes and old migration modules are unchanged. AST checks bind unchanged 14 fields/scales/modes/window semantics, the explicit new search/feedback strings, and the context's [-0.015, 0.025] band to unchanged task geometry. AIR permission is not TOP/contact/bearing credit; search budget stays 53deg/45s.

Schema `wlr50_clean.rr_signed_contact_same410.v7`; factor `rr_signed_contact_v7_factor`; persisted receipt `rr_signed_contact_v7_migration`. Exact actor/critic, full Adam/LR, Identity, full RNG, all origins/AUX, and complete v6 ancestry are retained. No rollout or physical state is inherited; migration adds zero learning credit. Latest and ancestor counters stay independent.

Publication helper, only after final reviewed HEAD and root authorization:

```powershell
$env:PYTHONPATH='src'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_rr_capture_then_rl_transfer_v1/publish_rr_signed_contact_v7.py --expected-head <EXACT_FINAL_HEAD> --source ancestor --publish
```

Use `--source latest` separately. It retains the source device/CUDA RNG visibility, saves unique source/head names, performs official independent reload, and never promotes the main pointer. It has not been executed during preparation.

Ancestor training must use explicit `-CheckpointOutputBranch ancestor220544_signed_contact_v7` (Python equivalent `--checkpoint-output-branch`). Natural P01, checkpoint-policy suffix training, and evaluation resolve the isolated same-branch pointer. V7 metadata must validate its v7 receipt/current runtime; a malformed v7 receipt cannot fall back to v6. Historical v6 exact routing remains available only against its matching runtime contract. CLI and direct training API reject ancestor training without an output branch. Main historical checkpoints, latest pointer, and latest learned counters are not replaced or borrowed.

CPU targeted tests: **111 passed, 1 skipped in 19.09s**, covering new v7 and historical v6 strict dual-source full-state roundtrips; source/target/config/AST negative cases; both v6/v7 actual synthetic128-row updates and normal receipt carry; no-flag and malformed receipt rejection; same-branch P01/P04 resume and fresh-P01 eval; main five history files/pointer protection. Windows actual symlink creation was skipped for privilege; resolved-alias negative test passed. Earlier fixture-only failures were corrected without weakening production guards. No synthetic test counts are robot PPO credit.

`git diff --check` passed. All CPU helpers exited. Prepared final hashes:

- New module: `a9f709f06b4f16530b862c97d47a51e170fd683fa97b1c35d702327673e043e6`
- New migration tests: `68ee691354b17b2b2068a5e96f62840556dd59d2fa0988e22e9672d9401e172a`
- Branch tests: `8a1e44ced6f1cbcecd8709db55a996aa7df5b9a47639733dcccb6f893fef58c7`
- Publisher: `276b03a55246471c46093312c8e39cf0972982f9889d55d8b147c81f98f8011a`
