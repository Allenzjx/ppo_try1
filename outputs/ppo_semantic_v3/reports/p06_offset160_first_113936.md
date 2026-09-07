# P06 offset160 — first actual handoff and 16 policy decisions

Fixed scope: `train/20260907T0531193843603Z_gf4bfe2560bfd_4b60e01ba2614a4a90ad4c6a7c06a5e1`, HEAD `f4bfe2560bfd228541f7829d830fa441054eab1d`, N1/seed1001. Read only the **first** prefix through `policy_credit_start` and policy rows **g113921–113936**. Later prefix/episode/policy/update rows were not inspected. This is collected transition evidence, not a new saved/optimized boundary or a full-block result. Master is unchanged.

## Actual prefix and credit boundary

The real first prefix is accepted, with no miss/fallback: **608 teacher decisions /4864 physics ticks**, not a number inferred from an older run. Its phase counts are P01=1, P02=207, P03=4, P04=1, P05=235, P06=160. All608 prefix actions record zero raw policy request and zero projected residual; all4864 native ticks verify, all no-state-write checks pass, and every prefix decision is `policy_credit=false`. Residual-effect counts are0 for this zero-residual teacher, not absence of nominal physical actuation.

P06 was first observed at tick3584; the requested offset contributes160×8=1280 further real ticks. The source teacher made4864 control calls. Handoff snapshot and `credit_start_observation` compare exactly equal: **tick4864 /40.533333333s**, requested and actual phaseP06, P06 stage age10.666666667s, remaining whole-task time159.466666667s. No episode clock or consumed horizon was reset at credit start. `from_P01_current_policy=false`, scope=`teacher_initialized_suffix`; the credit-start evidence row itself is not a PPO sample.

The first PPO action covers **ticks4865–4872**, ends at40.6s/g113921, and is policy decision1 but physical-core decision609. The16th ends attick4992/41.6s/g113936, policy decision16 /physical-core decision624. Thus initial clock4864 and teacher count608 are directly established here.

## Handoff geometry, exact contact and task history

Front distance below is live wheel-center X minus obstacle-front X; gap is live collider-bottom Z minus obstacle-top Z. These values independently recompute from the saved raw observation and exactly match the semantic task snapshot.

| Leg | Front distance mm | Bottom gap mm | Current contact | Load fraction | Hard Q/C/P history |
| --- | ---: | ---: | --- | ---: | --- |
| FL | +237.753514 | +0.728292 | TOP, obstacle pair active; not AIR/ground | .192344755 | true/true/true |
| FR | +336.640481 | −0.319415 | TOP, obstacle pair active; not AIR/ground | .260441160 | true/true/true |
| RL | −352.447670 | −50.001742 | GROUND; not AIR/TOP | .310457748 | false/false/false |
| RR | −338.263508 | −50.828266 | GROUND; not AIR/TOP | .236756337 | false/false/false |

**FL is genuinely supporting at handoff**, not merely historically placed. Its verified exact wheel/obstacle pair is active, normal force5.527402878N; saved contact point is `[.755473793,.083679669,.050428439]`m with three active history samples. FR obstacle normal force is7.484286308N. RL/RR verified ground normal forces are8.921610832/6.803656578N; both obstacle pairs are inactive/zero force. The small signed collider gap alone is not substituted for the measured contact.

Teacher histories are retained without re-crediting them: FR Q71/C1665/P1695 and FL Q2461/C3115/P3583. Both rear legs have **no hard qualification, crossing or placement**, and both current `initial_clearance` and `soft_air_actuation_earned` are false at handoff. The event history contains no RR initial-clearance event through this boundary. The partial P06 completion `rear_approach=.470209319` is a workspace-progress signal, not rear lift qualification or task completion.

The real mass-weighted CoM is `[.468085516,−.123645612,.164296415]`m, velocity `[+.016226265,+.000020274,+.003672858]`m/s, valid=true. All four actual wheel contacts supply the support polygon; the CoM projection is inside with signed margin **+.203865633m**. Base velocity is `[+.015919210,+.001048132,+.009093842]`m/s. Raw all-finite is true and exact body-collision detected/active/persistent are false. This measured CoM/support evidence is distinct from a soft initial-lift hint and does not imply that the rear leg is airborne, has earned above-top clearance, or will succeed.

## First16 PPO rows: no rear qualification or phase completion yet

All16 source/end phases are **P06→P06**. There are128/128 verified/effect/own-request-effect native ticks, no terminal, all bootstrap=true, time_outs=false, physical evaluator valid=true and zero four-type state writes. The explicit `prefix_teacher_data_in_ppo_storage` flag is false in all16. Global and episode-tick sequences match the handoff clock. All16 preserve the teacher's six exact front Q/C/P timestamps; RR/RL Q/C/P remain false.

RR stays GROUND at every sampled decision endpoint, with `initial_clearance=false`; no RR initial event appears in the retained event history. It moves from front−337.848258mm/gap−50.833857mm atg113921 to−319.073823mm/−50.088859mm atg113936; load changes.232007874→.084281673. RL is also GROUND at the first and last endpoints, ending front−329.032507mm/gap−50.395490mm/load.499348366. These finite changes are not qualification or a causal decomposition of actuation.

FL's true handoff support is **not maintained continuously at decision endpoints**:11 areTOP, while5 areAIR/load0—g113927,113931,113932,113935,113936. Historical placed remains true in all of them. This bounded observation does not establish any later failure or assign a unique cause.

The inherited teacher nominal at handoff is `[22.8,−13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]` in canonical Full12 units, controller-bias vector0. The first PPO request is nonzero and receives real native effect: first projected residual is `[2.617246267,1.977553041,−4,−.666970279,−1.771861539,−4,−1.598812062,−4,−.12,−.12,−.12,.017667240]`; actual canonical wheels are `[.18,.18,.18,.317667240]`rad/s. This is live policy influence after takeover, not teacher data stored as PPO.

## Same-MDP source binding; no fresh-Adam claim

The started record names the actual immutable source `checkpoint_step_000113920.pt`, matching its sidecar path, and the complete source runtime contract equals the started runtime. All explicit runtime, new-MDP and policy-distribution migration records are null; `new_mdp_warm_start=false`. This is ordinary same-MDP continuation with a different reset course, not another return-profile or optimizer reset.

The named source sidecar records113920/855/17100, roundtrip=true, checkpoint SHA `54bbc7de91a4d7a7461c52b2165abf6286127918522cc52606a96bb140499a6a`, actor `fb377ac16b8ec08d5592bbd7e70b3f766ca251abc9f1453aa72db0bf54cd1253`, critic `978759480351566c3b130bfade90484994e4e7a91ab55acff7b319ac90194cd4`, identity normalizer `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`, and Adam state `04a97c7f1ee912b128ec05b58f041088d812a931acc29f49324468fee0fc27d0` at recorded LR1e−5. The existing return profile remains `v3_gamma_09985_lambda_099_v1`; source spent full53504/suffix50304/smoke0 is unchanged by initialization.

These JSON/path/runtime receipts establish the requested source binding and ordinary-resume semantics. This audit deliberately did not load PT, recompute checkpoint hashes, inspect an optimizer update, or independently prove the in-memory Adam tensors. It does not claim a fresh3e−5 Adam, an optimized16-decision checkpoint, a successful suffix/full episode, or a better policy. No new gate, parameter change or later-run prediction is proposed. Reading and report writing stopped at this fixed window.
