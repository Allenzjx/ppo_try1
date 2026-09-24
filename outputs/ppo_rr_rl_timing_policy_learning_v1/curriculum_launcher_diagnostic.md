# Initial curriculum launch diagnostic

The first orchestrator attempt under `curriculum/first2048_gfa4b98ed506e` is FAILED with zero completed blocks/zero learning credit. Its original PowerShell stderr was captured by `subprocess.run` but not persisted by the exception handler; it cannot now be reconstructed honestly.

Confirmed static failure mechanism: `run_semantic_ppo.ps1` matches `isaaclab` anywhere in each Python process command line. The parent executable path `C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe` itself matches, even when executing only `wlr50_clean.ppo.semantic_curriculum` (stdlib JSON/subprocess orchestration). A read-only PowerShell regex check returned `MatchedText=isaaclab`. Direct PowerShell launch has no such parent and subsequently started the normal Isaac training entry successfully. This is consistent with the first attempt's two-second failure before any train directory, but is not a recovered original stderr transcript.

Current training is unchanged. No additional Python or physics process was launched for this diagnosis, and no production file was edited.

Smallest operational option for future whole courses: launch the stdlib coordinator with the existing `C:\Users\kskzz\miniconda3\python.exe`, whose path does not contain `isaaclab`. The child wrapper still selects the locked env_isaaclab interpreter for all actual training. Do not restart the current P01 block or double-count its future samples.

Smallest later code repair: preserve the busy guard and single-process lock, exempt only the direct parent PID when its command line explicitly identifies the exact stdlib curriculum module; do not exempt arbitrary Python helpers. Persist a per-block launch record containing command, return code, stdout and stderr on both normal return and CalledProcessError, before re-raising. Add a focused parent-classification test and a mocked failed-child stderr-persistence test. Apply only at a legal saved boundary with explicit runtime binding, not during the active run.

The first manual actual run is `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260923T1908118345669Z_gfa4b98ed506e_1e77c2dd353745ebb86708c5095a34c9`. At the bounded diagnostic read, optimizer_updates.jsonl had no completed row yet; no PPO update was credited from planned counts.
