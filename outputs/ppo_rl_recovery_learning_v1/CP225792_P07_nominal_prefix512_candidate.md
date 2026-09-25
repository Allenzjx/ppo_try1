# CP225792: existing P07 nominal-prefix continuation candidate

Status: planning only; not executed. Wait for the sole natural-P01 video (session 34489) to finish and seal normally. No source/config/reward change is required. This targets the rear continuous chain while retaining the existing front KL protection; it does not retrain or replace the accepted front controller.

Run from the existing project directory, with the pinned clean runtime unchanged:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 892385cba8a7089b52567bb7f558c018fac82a77 -SemanticVersion v3 -ExperimentId rr_rl_timing_policy_learning_v1 -Stage phase_suffix -FromPhase P07 -TeacherOffsetDecisions 0 -PrefixSource successful_nominal -Decisions 512 -MaxDecisions 3000 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -CheckpointOutputBranch cp225280_front_preserved_v1 -Checkpoint 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000225792.pt'
```

This differs from the actual preceding 512-decision P12 run only in source checkpoint (latest CP225792), start phase (P07 instead of P12), and teacher offset (0 instead of 8). Do not pass warm-start, distribution-migration, resume-migration, or a new LR option. The wrapper supplies `--headless`, the locked Isaac Python/PYTHONPATH, a unique run directory, and a single-process lock. It refuses an active Isaac process, a HEAD mismatch, or dirty runtime.

## Verified source and inherited state

- CP225792 SHA256: `17a7e849afc242856f035350d814349be1e101b49385c42ab9c90966a84fc5a7`; sidecar SHA256: `d42847f6c90612366bdb16ca574e11403430ac368ed487a69a0d904b73d8cc6d`.
- Manifest: 225792 decisions / 1726 PPO updates / 34520 optimizer steps; `optimizer_learning_rate=1.0000000000000004e-05`.
- Explicit `front_preservation439_branch_identity` resolves the existing 439-observation / 12-action actor and `n1_rear_owner439_collection512_v1`, with 512 steps per environment. Official load preserves weights, Adam, normalizer, RNG, and recorded actual LR; this is continuation, not another publication or migration.
- Existing replay spec remains coefficient 1.0 / minibatch 32; dataset SHA256 `d78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38`. `train_semantic` constructs the regularizer directly from this identity. Replay is front protection inside the same 20 official PPO optimizer minibatches, not additional on-policy samples or separate AUX steps.
- Existing execution profile keeps P05 FL assist on, RR capture assist null, RR capture wheel mode off. No diagnostic intervention or rear helper is introduced by this command.

## Existing implementation support and evidence scope

- `scripts/run_semantic_ppo.ps1`: exposes P07, successful_nominal, offset 0, seed, branch and checkpoint parameters and forwards them unchanged.
- `semantic_cli.py:309–327,1338–1389`: supports this N1/v3/phase_suffix combination, official-loads the supplied checkpoint, then installs an initialization-only zero-residual nominal prefix using the same current N/controller/mapper/history.
- `semantic_checkpoint_prefix.py:267–345`: advances real physics from P01 until P07 is observed, stops prefix immediately at offset 0, and hands over that same core state/history; no teleport or history reset at handoff. Every prefix decision has `policy_credit=false`; only subsequent current-policy steps enter training. If P07 cannot be legally reached, it records the exact miss and falls back once to fresh P01, rather than claiming rear coverage.
- `semantic_front_preservation.py:231–244`: resolves and validates the explicit 439/512 identity; `semantic_training.py:2132–2135` restores front replay. Existing phase-suffix budget is 131072; recorded use is 98432, leaving 32640, so this request fits without budget changes.

If all 512 learner decisions and the update complete, expected totals are CP226304 / 1727 PPO / 34540 optimizer steps, with 2 cumulative replay-protected updates, 40 replay minibatches, 1280 row exposures, and zero extra AUX/on-policy credit. These are prospective, not completed counts.

At 15 Hz, 512 decisions cover 34.133 seconds of student interaction. P07 prefix exposes RR preparation/lift/carry/capture and potentially RL preparation continuously; it does not guarantee the policy reaches RR contact or RL within this block. Verify `prefix_evidence.jsonl` accepted P07/current history plus credited phase counts and measured RR contact/bearing/RL events afterward. If the policy remains in P09, report that actual coverage rather than labelling the block P12/RL training. Normal phase changes remain nonterminal. Suffix progress is not natural-P01 PPO success, and the nominal prefix never establishes that the policy learned the front prefix.
