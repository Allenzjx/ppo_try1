# Bounded continuation and receipt

The official entry is `scripts/run_semantic_ppo.ps1`. Run sequentially after the fixed B2/C pair and reviewed reward migration. No successful baseline gate is required by these train arguments.

```powershell
# Replace placeholders with the committed reward runtime and actually validated plan.
& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead <reward-head> `
  -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 `
  -Stage full_episode -FromPhase P01 -Decisions 512 -NumEnvs 1 `
  -Seed 1001 -Device 'cuda:0' -CheckpointIntervalUpdates 1 `
  -Checkpoint <actual-source-checkpoint> -ResumeMigration <reviewed-body-reward-plan>

# Use the first block's actual latest verified checkpoint. Same runtime: no migration flag.
& ./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead <same-reward-head> `
  -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 `
  -Stage phase_suffix -FromPhase P06 -PrefixSource frozen_fsm -TeacherOffsetDecisions 0 `
  -Decisions 512 -NumEnvs 1 -Seed 1001 -Device 'cuda:0' `
  -CheckpointIntervalUpdates 1 -Checkpoint <first-block-actual-latest-checkpoint>
```

Retain the proposed 512 + 512 minimum allocation. Each is four complete 128-row N1 rollouts and, if completed, four PPO updates / 80 optimizer steps (5 epochs x 4 minibatches). Total planned addition is 1024 / 8 / 160. Use actual saved increments, not planned requests; graceful update-boundary stops can consume fewer. Non-multiples of 128 round up actual samples, while stage request credit consumes only the request. `-MaxDecisions` is not the training allocation. Cadence 1 offers a verified checkpoint every update; cadence 4 is also supported and a requested update-boundary stop still saves the boundary. No `-NewMdpWarmStart`, architecture migration or LR override is needed.

The current checkpoint effective Adam LR is 1e-5; the runner's 3e-5 YAML/default is not the restored effective LR. The loader preserves complete Adam state/LR, identity normalizers and HISTORY372/12. Seed 1001 must match the checkpoint restore contract. New rollout storage is required by semantic migration. A legal physical reset is not bitwise continuation of simulator state.

P01 full episodes learn early FR preparation/carry and all later phases actually reached. P06 suffix starts with the existing *fresh physical P01 frozen-FSM prefix*, not an airborne RR snapshot. P06 covers rear approach, FL receiving space and bridge shaping; PPO then controls P06 -> P07 -> P08 -> P09 and later reached phases. The teacher prefix has no PPO sample/reward credit. P06 sampling targets an early rear precursor but cannot guarantee later phase coverage within 512 samples. Do not rename suffix success full-P01 policy success or count the teacher as training. A checkpoint-policy prefix would currently risk spending the focused block on early policy failure; the existing frozen-FSM prefix is the smaller targeted choice.

If the P06 block spends its entire first update in P06, that is evidence for later allocation, not grounds to skip preparation, pre-lift RR or require full teacher success before optimizer startup. The prior completed P06 block actually provided P06=115, P07=1, P08=1, P09=11 across 128 credited samples, with 896 extra teacher decisions excluded. Its long prefix cost supports retaining a single focused block instead of many short fresh-prefix runs. The next candidate can have different counts.

## Output-only receipt

`summarize_completed_training.mjs` uses only explicitly named **completed** run manifests, checkpoint sidecars and credited audit/update/episode JSONL. It refuses active/incomplete runs, inconsistent counters and duplicate or non-chain runs. It does not load/hash checkpoint tensors, read rollout tensors, read active logs, modify production or update checkpoint pointers.

```powershell
& 'C:/Users/kskzz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' `
  outputs/ppo_timing_task_priority_v1/summarize_completed_training.mjs `
  --run <completed-P01-run> --run <completed-P06-run> `
  --output outputs/ppo_timing_task_priority_v1/<new-summary>.json
```

Each receipt distinguishes requested phase/stage budget from actual decision-start phase counts; includes teacher exclusions, terminal vs nonterminal tails, effective LR trajectory, normalizer/hash continuity, seed, actor/critic dimensions, rollout size, exact counter increments and recorded save/reload proof. It makes no new full-task success claim and does not treat metadata checking as an independent reload. Validation against the two earlier immutable runs reproduced 640 / 5 / 100, all P01-P13 sample counts and 896 excluded teacher decisions.

Code references: `semantic_cli.py:69`, `semantic_cli.py:166` (phase/stage constraints); `semantic_training.py:1151` (budget/batch accounting), `semantic_training.py:1275` (update counters), `semantic_training.py:1285` (checkpoint cadence), `semantic_training.py:1333` (actual vs requested results); `stage_task_spec.yaml:182` (P06 precursor).
