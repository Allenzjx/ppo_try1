# P06 frozen-FSM prefix — first actual 107392 update

Fixed scope: run `train/20260907T0253258293143Z_gf1a9bbf650b1_a6d3b0a9bb214f398e3579e3f350e7ba`, runtime `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`, N1/seed1001/P06/frozen_fsm/offset0. Read the first complete prefix through its first `policy_credit_start`, source/first-save JSON sidecars, and exactly 128 credited rows **107265–107392**. No later tail, Python/PT/tensor load/GPU/Isaac, checkpoint hash recomputation, production/test/config edit or master-ledger change. Requested2048 is not a completed count.

## Actual ordinary resume and first update

| Item | Source107264 | First saved107392 | New credit |
|---|---:|---:|---:|
| Global policy decisions | 107264 | 107392 | 128 |
| Lifetime PPO updates | 803 | 804 | 1 |
| Lifetime optimizer steps | 16060 | 16080 | 20 |
| Full-episode budget | 49408 | 49408 | 0 |
| Phase-suffix budget | 47744 | 47872 | 128 |
| Smoke budget | 0 | 0 | 0 |

Original v3 origin remains10112; `10112+49408+47872=107392`. Both checkpoints report round-trip=true. `checkpoint_step_000107392_manifest.json` binds this exact run via `source_run`. Source/current complete runtime contracts and policy contracts compare exactly equal. Current started arguments explicitly show `new_mdp_warm_start=false`, `resume_migration=null`, `policy_distribution_migration=false`; no new-MDP record file exists in this run. This is a fixed reset-course change, not another physical-MDP/optimizer reinitialization.

Source actor fingerprint `78064cd4f0207a9c06c805f79f1630cd78981bd66ee05a74bc3a1356289f9a26` exactly equals first-update actor-before; actor-after becomes `c62e0afceea848dcc01e10fa23142554f47387dc5c146ca15364f32d5609c453`. The fingerprint covers all named actor parameters including the learned heteroscedastic log-std head, not only its deterministic mean. Identity normalizer fingerprint is unchanged: `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

Source Adam fingerprint is `9015f50b4a1268861b5392f01cdfb4deaca7f84deedc0cf30f328548da37c20d`, source critic `378e31acc2df6c1f67acde068f68395ca40cdb035be75f6ff5ef5041717688fb`. Both effective source and first-update LR are1e-5. Their post-update fingerprints differ as expected after learning. This audit did **not** independently load the source tensors, capture pre-update Adam/critic tensors or compare restored live RNG. Preservation is supported by actual receipts plus the fail-closed ordinary load path (`semantic_training.py:350–419`): official restore, loaded actor/critic/Adam/normalizer fingerprint checks, verified RNG restoration, fresh empty rollout/pending-transition checks, and no new Adam construction. A historical `new_mdp_warm_start` field retained in ancestry is not a new warm start in this run.

First save reports20 optimizer steps, actor changed and finite nonzero gradients; recorded gradient norm range1.0029086267–1.4142133437, KL mean0.01566273345, clip fraction0.24375, value loss0.00340694524, surrogate loss−0.00923300460. These are actual update804 values, not the planned block totals.

## First real teacher initialization, excluded from PPO

Exactly451 prefix-stream records were read:448 `reset_only_prefix_decision`, one start, one result, one credit-start. All451 have `policy_credit=false`; all448 raw requests are exactly zero and all3584 physics ticks are verified, with complete no-in-episode-state-write evidence. The first decision starts at P01; the last is source P05/end P06. Prefix result: accepted=true, miss=null, offset0, no fallback.

Credit opens at episode tick3584/time29.8666666667s, actual/requested P06, remaining170.1333333333s. Handoff records3584 teacher calls, target first observed tick3584 and the same actual handoff tick. Its measured completed-stage prefix is P01–P05. The handoff nominal is `[22.8,-13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]`; inherited tracking-name list is empty.

| Leg at credit start | Current contact/support | Load fraction | Front distance (m) | Top clearance (m) | Hard Q / C / P |
|---|---|---:|---:|---:|---|
| FL | TOP / support=true | .224897046 | +.085491959 | +.000605491 | true / true / true |
| FR | TOP / support=true | .248550916 | +.181731704 | −.001061373 | true / true / true |
| RR | GROUND / support=true | .228028268 | −.494883957 | −.050949587 | false / false / false |
| RL | GROUND / support=true | .298523770 | −.510246924 | −.049513364 | false / false / false |

These are the live handoff evaluator fields, not a pose template. Teacher event ticks: FR Q71/C1665/P1695; FL Q2461/C3115/P3583. Neither rear leg has initial-clearance or hard Q/C/P at this boundary. FR/FL achievements and teacher physical work are **not** counted as current PPO achievements.

## Continuous suffix credit and native execution

First PPO row107265 is credited decision1/physical-core decision449, ending tick3592/time29.9333333333s. Row107392 is credited128/core576, ending tick4608/time38.4s. Therefore448 excluded teacher decisions +128 PPO =576 core decisions;3584 prefix ticks +1024 credited ticks =4608. Prefix time remains inside the original200s task horizon. All128 rows bind `teacher_initialized_suffix`, `from_P01_current_policy=false`, prefix_attempt_index0 and `prefix_teacher_data_in_ppo_storage=false`.

Actual phases are **P06=128**; no termination, success or later-phase coverage is claimed. All128 raw requests exactly match their applied audit requests; raw/old mean/std/log-prob/value/reward values are finite, with observed std range.08977381885–.40079218149. The production runner checks exact equality of sampled raw and stored raw/old-distribution tensors after every `process_env_step` (`semantic_training.py:710–746`). The completed update supports passage of that check; this audit did not reopen the binary rollout.

All1024 credited physics ticks are verified, actual-native-effect=true and own-phase-request-effect=true. There are0 handoff-hold ticks in this credited window; this does not assert that the preceding teacher-to-semantic transition had no hold. Every credited row reports0 root-pose, root-velocity, force/impulse and gravity writes. All128 detailed endpoint tracking-reference receipts independently verify, with dispatch=previousACK+1 and prior-feedback-sample+1=current mapper feedback=previous ACK write count.

First credited native command ticks3764–3771 correspond to episode3585–3592; last command ticks4780–4787 correspond to episode4601–4608. This is the expected continuous command clock with settling offset, not a reset to tick0 at credit onset. Endpoint tracking-name lists are empty and endpoint reference-use count is0. Per-tick compact records do not contain complete channel reference evidence; no whole-window reference-use count is inferred from endpoints.

Source ownership further supports continuity: `semantic_prefix.py:174–202` reuses the same supervisor and seeds semantic nominal/tracking from the actual prior dispatch; `315–388` calls core.reset only at initialization (or one miss fallback), not at successful credit opening. Thus reader/contact/mapper, reward, residual/bridge and measured-history state continue across accepted handoff. This is code-backed object/state continuity plus measured clock continuity, not a claim that compact logs independently serialize every internal filter value.

## Brief actual control and physical coverage

All128 endpoint nominal wheels remain `[.3,.3,.3,.3]`. Residuals are active and can oppose those suggestions; observed endpoint ranges below are canonical wheel rad/s, not world translation or guaranteed motion direction.

| Wheel | Projected residual range | Actual canonical target range |
|---|---|---|
| FL | −.795752730 to −.041042452 | −.495752730 to +.258957548 |
| FR | −.947975296 to −.120000000 | −.647975296 to +.180000000 |
| RL | −.378286184 to −.015694335 | −.078286184 to +.284305665 |
| RR | +.057824671 to +.525672486 | +.357824671 to +.825672486 |

Both front-wheel residual ranges include values beyond the old.6 bound. This is observed implementation authority, not proof that opposing/assisting a suggestion caused a task outcome. At row107392, canonical targets are `[.016094039,−.321375098,.152981939,.661720662]`, while measured wheel velocities are `[.016825549,−.533916175,.195431337,.669414163]`. Native float32 targets are `[−.016094038,−.321375102,−.152981937,.661720634]`; the FL/RL sign conversion must not be mistaken for a canonical target reversal.

In this fixed window RR front distance spans−.495142227 to−.448985290m and ends−.480272271m; RL spans−.509686851 to−.433516052m and ends at the latter. Both remain behind the existing workspace lower bound−.22m; no RR/RL hard Q/C/P appears. First unfinished task remains rear approach/P06, not a newly failed hard guard.

FL is current support at8 decision endpoints and AIR at120. At the last endpoint its historical placed=true remains, but current AIR/support=false/load0, clearance+.083354806m and front+.058537546m. FR remains TOP/load.410724466; RR/RL remain GROUND/load.102338029/.486937505. This preserves the necessary distinction between historical placement and current support. Potential is.465460250 at handoff and.468122416 at the last endpoint; no causal improvement or task-success conclusion follows from this one sampled window.

Conclusion: first accepted frozen-teacher prefix and first128 actual PPO transitions are correctly separated, with same-runtime ordinary resume, completed update/save and continuous verified actuation. The task is still an unfinished P06 suffix at this boundary. No later tail or planned2048 credit is included.
