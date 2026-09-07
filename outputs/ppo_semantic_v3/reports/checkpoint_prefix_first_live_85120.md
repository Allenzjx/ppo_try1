# Frozen-checkpoint prefix: first real P06 credit and saved update

Status: bounded live evidence through **85,120 policy decisions / 630 PPO updates / 12,600 optimizer steps** only. This is the first +128 / +1 / +20 boundary of a running requested-2,048 block, not block completion or task success. No later episode or update is included.

## Sources and scope

- Runtime: `a580fc2add8137ea599df558f74d34ec9c50ef97`.
- Run: `runs/ppo_semantic_v3/train/20260906T2211507063967Z_ga580fc2add81_e539358648084c70bf6b28037515480c`.
- Actual started arguments: v3, N1, seed1001, `phase_suffix`, P06, offset0, `prefix_source=checkpoint_policy`, `NewMdpWarmStart=true`, checkpoint cadence4. Source is immutable `checkpoint_step_000084992.pt`; no policy-distribution migration.
- Read-only PowerShell: one parsed prefix through its first credit-start record; one parsed 128-row policy ledger; fixed endpoint selections and small source/initial/85,120 sidecars. No checkpoint tensor loading, new hashing, Python, GPU/Isaac invocation, production edit, or master-ledger update.
- Source checkpoint SHA recorded in the verified source receipt: `ebabc03d1d3611bbe4ab1c05feddbbe3102eedb71ae9d5924cd8f77f2826ea01`. This report compares existing receipts; it does not claim an independently recomputed file hash.

## Real prefix and credit seam

The first prefix was accepted without fallback after **352 deterministic decisions / 2,816 physical ticks**. Its phase counts were P01=1, P02=197, P03=3, P04=1, P05=150. Every row was explicitly `policy_credit=false`, scope `checkpoint_policy_initialization_excluded`, and not full-task PPO success. No legacy-FSM teacher was used in this prefix.

The log begins with one fresh P01 bootstrap at episode tick0. The final prefix decision starts in P05 and ends in P06 at tick2816 / **23.4666666667s**. Credit starts there, with **176.5333333333s** of the original task horizon remaining, `from_P01_current_policy=false`, and `requested_phase_still_active_at_credit=true`. The phase transition is not a new physical reset or a restart of the task clock.

The first credited decision is global84,993: policy decision count1, underlying core decision count353, endpoint tick2824 / 23.5333333333s. It is scoped `checkpoint_policy_initialized_suffix`. The prior prefix's complete 12-channel residual vector equals the first P05→P06 bridge record's previous and carried vectors; no channels are dropped or phase-scale clipped. Reported maximum logical handoff action change is 4.4408921e-16deg for servos and 6.9388939e-18rad/s for wheels, floating-point arithmetic scale rather than a reset-to-zero action. The ordinary first bridge hold tick2817 is correctly excluded from own-policy effect (`handoff_hold_used=true`, `own_phase_request_effect=false`). Native command tick2996 is distinct from episode tick2817; it is not substituted for the task clock.

| Fixed evidence | Decisions | Physics ticks | Native verified | Actual-effect ticks | Own-phase effect ticks | Terminal decisions |
|---|---:|---:|---:|---:|---:|---:|
| Frozen checkpoint initialization | 352 | 2,816 | 2,816 | 2,816 | 2,812 | 0 |
| First credited rollout, global84,993–85,120 | 128 | 1,024 | 1,024 | 1,024 | 1,023 | 0 |
| Combined physical work in this fixed window | 480 | 3,840 | 3,840 | 3,840 | 3,835 | 0 |

All 128 credited rows are P06, continuous in global count and episode tick, and explicitly exclude checkpoint-prefix data from PPO storage. All prefix/credited rows have the four in-episode pose/velocity/force/gravity write counters equal to zero and `no_in_episode_state_writes_verified=true`. Their compact native summaries are fully verified; zero effect would also have been legitimate data, so these counts are observations, not a new success gate.

## History belongs to the physical process, not to new PPO credit

At the first credited endpoint, history retains the real prefix events:

| Leg | Qualified lift tick | Front-cross tick | Placement tick | Credit ownership |
|---|---:|---:|---:|---|
| FR | 45 | 1594 | 1608 | Frozen source policy prefix, excluded |
| FL | 1693 | 2655 | 2816 | Frozen source policy prefix, excluded |
| RR | none | none | none | Not yet achieved |
| RL | none | none | none | Not yet achieved |

The first credited endpoint tick2824 has both front legs genuinely TOP/obstacle-pair active, with FL/FR load fractions .187235699/.297102114. RR/RL are genuinely GROUND, not qualified: RR front−535.172578mm, clearance−50.704832mm, load.205899682; RL front−541.658290mm, clearance−51.133888mm, load.309762505. A historical RL `whole_body_initial_clearance` at tick13 is not hard qualification and is not new PPO credit.

At the saved-window endpoint tick3840 / 32s, P06 is still nonterminal. RR remains GROUND (front−411.917333mm, clearance−50.348989mm, load.474154210). RL is AIR but still below the top (front−415.338759mm, clearance−43.948987mm, load0). The recorded post-credit RL initial-clearance events are ticks2901,2971,2990,3104,3150,3420,3572,3637,3707,3776; none is a hard Q/C/P event. Both front legs are currently TOP at this endpoint, with FL/FR load.446720598/.079125193. There is no rear-leg success, suffix success, full-task success, or stability-improvement claim in this window.

## Initial migration and immutable saved boundary

Initial sidecar:

`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000084992_sebabc03d1d36_ga580fc2add81_a998142b8b2057410513df6a2ec1b172c16ea56b027dde9274e7fa7547cd2def_manifest.json`

It records initial checkpoint SHA `751e02bcbee46a52d45f970eb0f6a6a526d57f64f725b7ca07b3db392b2e0074` and `save_load_round_trip=true`. Source→initial comparisons:

- Entire heteroscedastic actor, including learned std: unchanged parameter SHA `8eb3d28a16b8da10a7660d3ae2cbe2b8642bb7b14b043909183663297263d56c`.
- Critic unchanged: `e0ed040af0a99c852616344460ef1a2d696c6bee29c26bf25d1e5888837e0cab`.
- Identity normalizer unchanged: `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`; complete serialized training RNG state equal.
- Adam is intentionally fresh: state SHA changes from `8f90fc81dbc16a65eb5978e11c4f8eb5bb914d0757dbe7ce4c75b960d709559c` to `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`; actual recorded LR changes from 1e-5 to 3e-5. The warm-start record declares no inherited rollout/physical state; source-loading code requires empty `(128,1,12)` storage and no pending transition before the new collection.
- Initial global84,992 / PPO629 / optimizer12,580, original origin10,112, and full33,280 / suffix41,600 / smoke0 budgets are preserved, with no prefix decisions added to them.
- Initial curriculum/provenance names the immutable source84,992 and independent frozen actor, while current sampling is `natural_P01_frozen_checkpoint_policy_prefix_then_semantic_suffix_N1.v1:P06:offset_0`. Top-level suffix flag is correctly true.

The actual same-state comparison occurs at the unchanged credit state P06/t2816. All six recorded old/new logical projection vector differences are zero, and `actual_environment_or_bridge_history_modified=false`. This is a same-state diagnostic, not a claim that the new reset distribution produces a paired physical improvement. The six selected v3 configuration hashes are unchanged; the versioned implementation change is the checkpoint-prefix/orchestration code.

First update evidence (`optimizer_updates.jsonl`, first row): update630/global85,120, **20 optimizer operations**, actor changed from the source hash above to `7db7f4c0607f93f5b02f16eb45cf02c54975cc7836549070ff44bbf195965211`, finite nonzero gradients true, gradient norm range1.002910900–1.414212941. Update-end LR is 1e-5; this does not assert a constant LR for every minibatch.

Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000085120_manifest.json` records SHA `a4352847dddc0fa237a7bbc4159fcaf8d0c00480aba2ba77e73d3730be90da0a`, round-trip true, counters **85,120 / 630 / 12,600**, actor matching the update's after hash, unchanged normalizer, and budgets **full33,280 / suffix41,728 / smoke0**. Its curriculum still binds the frozen source84,992 actor rather than the newly updated actor.

The requested2,048 block continues outside this report. No later count, terminal outcome, full P01 evaluation, or video result is precredited. This fixed first window is now closed to further scanning.
