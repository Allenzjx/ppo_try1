# Sealed front divergence: CP221184 v4 vs CP220544 v2

Read-only CPU audit; no actor forward, physics, optimization or source modifications. Streams stop at first P06; the current run ends in P05. Detailed selected per-channel native/target/actual evidence is in the adjacent JSON.

| Candidate | Whole source duration | Prefix analysis last stage | Recorded local terminal |
|---|---:|---|---|
| CP220544_v2 | 104.033333s | P06 | None / None |
| CP221184_v4 | 60.733333s | P05 | INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED |

## Stage entry samples

| Candidate | Stage | Tick | Time(s) |
|---|---|---:|---:|
| CP220544_v2 | P01 | 1 | 0.008333 |
| CP220544_v2 | P02 | 16 | 0.133333 |
| CP220544_v2 | P03 | 2368 | 19.733333 |
| CP220544_v2 | P04 | 2400 | 20.000000 |
| CP220544_v2 | P05 | 2408 | 20.066667 |
| CP220544_v2 | P06 | 6120 | 51.000000 |
| CP221184_v4 | P01 | 1 | 0.008333 |
| CP221184_v4 | P02 | 16 | 0.133333 |
| CP221184_v4 | P03 | 2448 | 20.400000 |
| CP221184_v4 | P04 | 2480 | 20.666667 |
| CP221184_v4 | P05 | 2488 | 20.733333 |

## FL earned qualification and first ground recontact

### CP220544_v2
- FL_history.active_lift: False -> True at tick 2409 / 20.075000s / P05
- FL_history.front_edge_crossed: False -> True at tick 3544 / 29.533333s / P05
- No ground recontact after FL qualification in analyzed prefix.

### CP221184_v4
- FL_history.active_lift: False -> True at tick 2500 / 20.833333s / P05
- FL_history.active_lift: True -> False at tick 3680 / 30.666667s / P05
- First ground after earned lift: tick 3680 / 30.666667s; front -64.515 mm; gap -51.276 mm; target FL hip/knee [18.701696222246323, -31.8795279160949], actual [19.008573421094262, -32.015341714358804]; wheels target [-0.7094958123029445, 0.0283684437773019, -0.03397015095268095, -0.15203346505733378], actual [-0.6771556735038757, -0.11488095670938492, 0.038917019963264465, -0.1434268355369568].

## First recorded same-tick differences

- source_N: tick 2368 / 19.733333s (P03 vs P02); maximum absolute delta 22.9.
- final: tick 1 / 0.008333s (P01 vs P01); maximum absolute delta 0.0315334962.
- q: tick 1 / 0.008333s (P01 vs P01); maximum absolute delta 0.0014777888.
- wheel_velocity: tick 1 / 0.008333s (P01 vs P01); maximum absolute delta 0.00258922577.
- body_position: tick 1 / 0.008333s (P01 vs P01); maximum absolute delta 1.95205212e-06.
- raw_policy: tick 8 / 0.066667s (P01 vs P01); maximum absolute delta 0.00353693805.

Both masks and native write verification are reported separately from residual/controller effects. Canonical wheel rotation is not wheel-end displacement or proof of traction. No historical placement is substituted for current contact.

Different deterministic policies and evolving physical trajectories. Full actor input vectors were not persisted here; no same-input actor replay or matched-policy physical counterfactual was run.

640 rear-only real decisions / 5 PPO updates / 100 Adam steps are preserved as learned lineage; excluded deterministic prefixes are not front rehearsal samples. No task success or stability-improvement claim.
