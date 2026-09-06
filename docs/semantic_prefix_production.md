# Production N1 teacher-prefix continuation

`semantic_prefix.py` implements a reset-only, unmodified A teacher with a live
semantic observer from physical tick zero. It supports fixed P06–P13 targets;
P06/P07 are the preparation starts, not only already-airborne P09/P10 states.
One run fixes a target and optional teacher offset. This is not a sampled mixture.

```python
request = PrefixRequest(target_phase="P06", teacher_offset_decisions=0)
backend = PrefixSemanticIsaacBackend(
    app, prefix_request=request, audit_actuator_target_effect=True,
    execution_profile=config_root / "execution_profile.yaml",
    task_spec_path=config_root / "stage_task_spec.yaml",
)
core = SemanticEpisodeEnv(backend, action_config=config_root / "execution_profile.yaml",
    reward_config_path=config_root / "reward_config.yaml",
    observation_schema_path=config_root / "observation_schema.json", collect_trace=False)
env = PrefixRslAdapter(core, seed=1001, device="cuda:0", evidence_sink=persistent_jsonl)
```

The total prefix limit is 1800 decisions, including the optional offset and
bounded takeover (at most 30 decisions). No historical angle, velocity, exact
sensor tick or recorded physical state is restored. A target is attempted from
its live semantic stage and physical validity, not a requirement that its task
already be complete. An offset missed because the stage already ended is a
recorded initialization miss, not a forced phase assignment.

The handoff uses the last actual atomic dispatch nominal, tracking and bias.
It keeps the same RobotAdapter/mapper, sensing readers, evaluator, contact
history, bridge/action history and 200-second task clock. An already-zero
teacher bias permits immediate credit; otherwise its bounded smooth retirement
is reset-only work. If a short stage naturally completes during retirement,
the actual credit-start phase is recorded; it is not counted as sampled PPO
preparation. Failure to initialize produces one fresh P01 fallback, explicitly
labelled and counted separately. Interface errors and evidence I/O errors raise.

Only calls after initialization enter official RSL storage and global policy
decision counts. Ordinary semantic stage transitions are not episode endings.
Prefix core physical counts are retained under `physical_core_including_prefix`;
credited decisions, ticks and per-phase counts are separate. Full entry evidence
is written once at `policy_credit_start`; per-decision `curriculum_start` contains
only a compact reference and attempt index. `SUFFIX_SUCCESS` and
`FULL_TASK_SUCCESS` remain distinct. Full P01 evaluation/video never uses this
teacher initialization.

## Short physical workspace response

After committing the reviewed runtime and confirming no other Isaac/Python task
is active, the root agent can run the module below (use a new run directory):

```powershell
$env:PYTHONPATH = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -P -m wlr50_clean.ppo.semantic_workspace_probe `
  --expected-head <full-reviewed-commit> `
  --run-dir 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_semantic_v3\interface_checks\<new-run-id>' `
  --from-phase P06 --seed 1001 --decisions 64 --raw-magnitude 0.5
```

This performs three independent physical resets in the same backend:
`nominal_zero`, `old_range`, and `new_range`. All use the same v3 nominal/task
implementation; the last two use the same declared raw diagnostic vector and
old/new action scales. Each response begins only after reset-only A roll-in.
The final 200-second task deadline still includes that roll-in.

Each segment retains prefix evidence, native actual/counterfactual targets and
measured 120 Hz response observations, initial/final body/CoM/joints/contact
states, actual stage counts, timing, artifact SHA-256 and physical termination.
Physical failure or timeout ends that segment but is not an interface error.
No actor/optimizer is constructed and no probe sample receives optimization
credit. These responses diagnose physical influence; they are not success,
stability-improvement or automatic training-gate certifications. Compare actual
starting states before attributing differences to action scale.

For an interface-repair rerun, `--segments new_range --decisions 32` executes
only that new segment in a fresh run directory. It does not overwrite or repeat
the previous two responses. The manifest reports measured reset count (including
any prefix fallback) and does not claim `same_raw_old_new` when old_range is absent.
Inspect `run_manifest.json.lifecycle` and the final workspace manifest, not only
the process exit code: Kit's immediate shutdown can mask a Python exception.

The v3 execution profile explicitly enables `independent_post_mapper_residual.v1`.
The frozen controller correction is still bounded by its original +/-10-degree
envelope; the separate PPO residual is composed after the single nominal mapper
advance and before the unchanged final hard limits and 1.25-degree/tick slew.
ACKs expose both inputs separately. The legacy combined-offset field is retained
and explicitly labelled for existing native-audit readers. v2 without this
profile opt-in retains its original execution path, and all frozen A code remains
unchanged. Exact projected-zero dispatch calls the original adapter directly.

Offline coverage includes shared live supervisor history for P06–P13, early
P06/P07 before RR lift, exact receipt binding, real backend/reset and native
float32 RobotAdapter with actual semantic core/RSL, two reset generations, one
official 128-decision/20-optimizer-step update excluding prefix credit, and
three-reset probe plumbing. Scene/sensors and teacher commands in the cross-layer
fixture are synthetic; no test claims physical teacher-prefix reachability.
