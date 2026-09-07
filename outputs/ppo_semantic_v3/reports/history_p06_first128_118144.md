# History-conditioned P06 offset160 — first verified128, checkpoint118144

This is a **fixed completed-update window**, not completion of the planned1024-decision block. Run `train/20260907T0736098630371Z_ge8462f066908_a07ce6b30c64439e8e8735a0926c4cc6`, HEADe8462f066908, N1/seed1001/frozen-FSM P06/offset160. Read only the first prefix through its credit-start record, policy rows **g118017–118144**, first update888 and its immutable initial/first-save receipts. No later policy, episode, checkpoint or evaluation result is included.

The128 rows contain **P06=126, P07=1, P08=1** source-phase samples and end in **P09**, nonterminal. They execute1024 real physics ticks. This is observable predecessor-course progression, **not RR qualification, placement, suffix success or full-P01 success**.

## Initial boundary and real prefix

The explicit NewMdpWarmStart loader publishes the new conditional-policy boundary from118016/887/17740. Its record says physical MDP/reward unchanged, parameter layout unchanged, policy and fixed-mean control changed. Source and initial sidecar fields match for actor parameters (including learned logstd), critic, identity-normalizer state, complete recorded training RNG, lifetime counters, spent budgets and original10112 origin. Adam state differs deliberately: source recorded LR1e-5 → fresh Adam3e-5, with old rollout inheritance=false. Initial roundtrip=true; implemented sampling explicitly names `P06:offset_160`. These are verified-loader/save receipts and a read-only JSON comparison, not a new PT/tensor reload or repeated checkpoint hash.

The first actual teacher prefix is accepted, no fallback: **608 excluded decisions /4864 ticks**, source phases P01/P02/P03/P04/P05/P06 =1/207/4/1/235/160. All608 records are policy_credit=false, raw0, and no-state-write verified;4864 native ticks verified. No policy data is credited for the prefix or credit-start row.

P06 first appears at tick3584; offset160 contributes1280 more ticks. Credit starts at **tick4864 /40.533333333s**, with159.466666667s still available in the same whole-task clock. The handoff carries source_control_tick4863, command/native tick5043, teacher_calls4864, zero controller-bias vector, and nominal canonical Full12 `[22.8,-13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]`. This is an actual roll-in/handoff, not a history-pose reset.

| Leg at credit start | Front mm | Bottom gap mm | Current contact | Load fraction | Hard Q/C/P |
| --- | ---: | ---: | --- | ---: | --- |
| FL | +237.753514 | +0.728292 | TOP, not AIR | .192344755 | teacher true/true/true |
| FR | +336.640481 | −0.319415 | TOP, not AIR | .260441160 | teacher true/true/true |
| RL | −352.447670 | −50.001742 | GROUND, not AIR | .310457748 | false/false/false |
| RR | −338.263508 | −50.828266 | GROUND, not AIR | .236756337 | false/false/false |

Teacher FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 are retained, not re-credited to PPO.

## Conditional distribution: what is directly established

The immutable initial contract records `history_conditioned_heteroscedastic_log_v1`, rho=.9, existing normalized previous_raw slice195:207/clip20, m=.1μ+.9h, and **unchanged learned conditional innovation σ**. No square-root stationary-variance scaling is declared.

Actual same-observation initial comparison has h=0 across all12. For FL/FR wheels, source μ is−.256861299/−.189003423, target m is−.025686130/−.018900342, and shared σ is.280396223/.172814324. The first real PPO transition's **entire12-dimensional recorded old mean matches the initial target mean exactly**; its σ vector also matches the recorded shared σ vector. Initial source→target conditional KL is**4.623683929**, not zero. Preserved weights therefore do not imply preserved action distribution or fixed-mean behavior.

The first sampled wheel raw values are FL−.227845490/FR+.131762624. In the next row, these are the previous raw history implied by the existing environment update/encoder path; recorded conditional means are FL−.230282277/FR+.117797658. All sampled raw magnitudes in this128-row window are below2.832, so the ±20 history clip is **not exercised** here.

Subsequent means/stds are the actual RSL transition distribution parameters recorded by production instrumentation. The compact policy stream does not save each complete input observation or its separate base-μ head; algebraically reversing m−.9h is not an independent network-forward proof. This review reconstructs history from the issued raw sequence and verified reset semantics, but does not reload the network. A direct double-precision12D Normal log-density recomputation from the recorded m/σ/raw agrees with logged float32 oldLP to maximum absolute error **1.00585e-6**. This is numerical consistency, not bitwise torch/GAE replay or a test of marginal stationarity.

Across the entire128-decision window, all raw channels take both signs. Longest runs below count consecutive decision samples, not a claim that commanded joints or body motion preserve that sign.

| Channel | Actual conditional mean range | Actual σ range | Sampled raw range | Raw positive/negative counts | Longest positive/negative run |
| --- | --- | --- | --- | ---: | ---: |
| FLhip | -1.405822…1.603474 | 0.142492…0.335347 | -1.570157…1.724094 | 93/35 | 75/25 |
| FLknee | -1.951212…0.551755 | 0.141633…0.222216 | -2.086746…0.598190 | 25/103 | 6/60 |
| FRhip | -2.424734…0.780743 | 0.093150…0.528034 | -2.613855…0.904141 | 39/89 | 15/40 |
| FRknee | -1.113790…0.377688 | 0.113049…0.182654 | -1.180514…0.389145 | 28/100 | 12/86 |
| RLhip | -1.177292…0.293166 | 0.076423…0.188047 | -1.261155…0.331834 | 18/110 | 10/103 |
| RLknee | -1.046695…0.227451 | 0.120809…0.188639 | -1.116164…0.292572 | 22/106 | 4/36 |
| RRhip | -1.207805…0.755387 | 0.165620…0.201920 | -1.327135…0.821428 | 57/71 | 53/29 |
| RRknee | -1.194790…0.356897 | 0.145816…0.201361 | -1.307633…0.414364 | 33/95 | 11/46 |
| FLwheel | -2.589454…1.030537 | 0.147642…0.659497 | -2.831569…1.166731 | 41/87 | 11/24 |
| FRwheel | -1.583571…0.775425 | 0.117339…0.357116 | -1.713866…0.822039 | 52/76 | 28/24 |
| RLwheel | -1.271310…0.515518 | 0.121458…0.169945 | -1.343477…0.545944 | 29/99 | 23/88 |
| RRwheel | -1.306201…1.537214 | 0.094789…0.399849 | -1.504559…1.620252 | 76/52 | 36/21 |

## Filtered request, final drive and native target are different quantities

The following values are decision-end observations. Requested residual means the existing filtered/projected request, **not raw latent or the unfiltered cap×tanh suggestion**. Effective headroom residual matches that request on all128×12 checked fields, maximum difference0; no endpoint headroom clipping occurs. This does not prove that every intermediate tick had no clipping: full effect/mapping validity is logged per tick, while the detailed headroom vectors here are endpoint audits.

| Wheel | Nominal canonical range rad/s | Filtered request range rad/s | Request +/− count | Actual canonical drive range rad/s | Drive +/− count | Native float32 axis-target range rad/s |
| --- | --- | --- | ---: | --- | ---: | --- |
| FLwheel | 0.000000…0.300000 | -1.186953…0.602728 | 40/88 | -0.886953…0.902728 | 73/55 | -0.902728…0.886953 |
| FRwheel | 0.000000…0.300000 | -0.929518…0.811414 | 47/81 | -0.629518…1.111414 | 74/54 | -0.629518…1.111414 |
| RLwheel | 0.000000…0.300000 | -0.523503…0.298485 | 29/99 | -0.223503…0.598485 | 69/59 | -0.598485…0.223503 |
| RRwheel | 0.000000…0.300000 | -0.543581…0.554796 | 79/49 | -0.243581…0.854796 | 108/20 | -0.243581…0.854796 |

Actual canonical wheels equal nominal+filtered request at all checked endpoints (maximum numerical difference0); native FL/RL axis signs are reversed relative to canonical, FR/RR are not. The native column is the independently audited float32 target buffer, not the canonical double vector mislabeled as readback.

Thus this window includes **both positive and negative filtered front-wheel requests**, not only larger negative requests. FL's positive request reaches+.602728 and negative−1.186953; FR reaches+.811414/−.929518. Their actual canonical commands also take both signs. Initial front nominal is+.3; it retires toward0 at the phase handoffs. This window does not contain a negative front nominal needing cancellation, so no such cancellation capability is claimed from these samples.

Rear servo requests also take both signs. Canonical degrees and native radians must not be compared as though they were one space:

| Servo | Nominal canonical ° | Filtered/effective residual ° | Actual final canonical drive ° | Actual native position rad |
| --- | --- | --- | --- | --- |
| RLhip | 6.900000…14.300000 | -20.433176…7.684028 | -14.783176…13.334028 | -0.235854…0.254883 |
| RLknee | 0.000000…0.000000 | -29.024297…7.399917 | -28.382137…8.042077 | -0.137516…0.498207 |
| RRhip | 0.000000…0.000000 | -20.845139…16.193039 | -22.195042…14.843136 | -0.264699…0.381739 |
| RRknee | 0.000000…0.000000 | -31.092308…12.066711 | -30.137958…13.021061 | -0.223504…0.529763 |

Using the recorded same-tick geometry-corrected native target, effective combined bias, previous final servo target and original hard limits, the scalar final hard/slew formula reconstructs all1024 endpoint servo channels with max error0. The largest endpoint final-servo step is1.25°;3 of those endpoint-channel candidates are limited by that final slew. This does not count all intra-decision slew events or imply target tracking by the real joint. Mapper/controller effects, mechanical sign/standing transforms and contact dynamics still separate requested residual from measured motion.

All1024 compact native ticks verify and show actual effect: **1022 own-phase request-effect +2 handoff holds**, command clock=episode clock+179 throughout. All128 endpoint audits report setter/dispatch equality, reconstructed mapping match and same-tick counterfactual=true. Four state-write counts remain0; all128 no-write checks and teacher-exclusion flags pass. No independent GPU buffer replay was performed.

## Real support, rear events and phase progression

At decision endpoints FL is **TOP/supporting29 times and AIR99 times**, with load0 on AIR rows; maximum recorded load.554206454. It ends AIR/load0/gap+43.740054mm despite its persistent teacher placed history. FR has120 TOP and8 AIR endpoints. These are sampled current contacts, not claims of continuous support between endpoints.

RR has121 GROUND/7 AIR endpoints, **no new initial-clearance event and no hard Q/C/P**. Its sampled gaps stay below the top, −51.943290…−48.273495mm. RL has125 GROUND/3 AIR endpoints and one new **initial** event at tick4881/40.675s (measured upward excursion3.585506mm), but **no hard Q/C/P**. That early event is not above-top qualification.

RR sampled front range is−348.247182…−182.178285mm, ending−212.873247mm/GROUND/load.117146549. RL ranges−347.976160…−174.374726mm, ending−174.374726mm/GROUND/load.513487205. Relative body-forward position ranges−34.737256…+93.403283mm. These values establish observed preparation motion, not an isolated action-to-displacement causal model.

| Credited transition | Global /tick /time | Recorded completion | Terminal/bootstrap |
| --- | --- | --- | --- |
| P06→P07 |118142 /5872 /48.933333333s | rear_approach=1 | false/true |
| P07→P08 |118143 /5880 /49.0s | RR/RL workspace=1; support_RR=1 | false/true |
| P08→P09 |118144 /5888 /49.066666667s | workspace_RR=1; load_ready_RR=1 | false/true |

No reset or artificial done occurs at these transitions. Residuals remain nonzero across the handoffs: front-wheel requests atg118142 are+.389486/+.418714, then+.284486/+.313714, then+.179486/+.289329. The last row's four-wheel nominal is0, but its nonzero actual wheel command remains; no unconditional policy-history clearing is observed. The next unfinished rear task is P09 RR qualification/crossing/placement. This fixed window has **zero P09 source-phase policy samples**, only its newly entered endpoint.

## Descriptive comparison with old#34, same offset, first128 only

Comparator: `train/20260907T0531193843603Z_gf4bfe2560bfd_4b60e01ba2614a4a90ad4c6a7c06a5e1`, globals113921–114048, old heteroscedastic policy, source113920. Its first teacher handoff geometry/clock matches the values above. Both windows contain1024 credited ticks, but actor checkpoints and conditional-policy initialization differ; this is **not a paired causal comparison**.

| Front wheel | Raw +/− counts, old→new | Longest raw +/− runs, old→new | Filtered request +/− counts, old→new | Actual canonical drive +/− counts, old→new |
| --- | --- | --- | --- | --- |
| FLwheel | 23/105 → 41/87 | 2/13 → 11/24 | 0/128 → 40/88 | 27/101 → 73/55 |
| FRwheel | 3/125 → 52/76 | 1/91 → 28/24 | 0/128 → 47/81 | 3/125 → 74/54 |

Old first128 remain entirely P06; new first128 contribute126P06+1P07+1P08. Old endpoint RR/RL front is−339.125436/−330.617618mm, versus new−212.873247/−174.374726mm. Old FL isTOP11/AIR117 endpoints, new29/99; both end FL AIR/load0. Neither window records rear hard Q/C/P. The different command signs, runs and physical endpoints are observations; they do not establish that rho, sigma convention or this policy version caused a control improvement or will yield success.

## First real update/save, not planned-block completion

Actual update888 covers exactly these128 transitions and20 optimizer steps. Actor hash changes, gradient norm range1.077243702…1.414213497, KL mean.017200292, clip fraction.265625; recorded update-end LR1e-5 is not a per-minibatch constant-LR claim. Source/initial→update-before and update-after→saved actor fields match.

The immutable first save records **118144 /888 PPO updates /17760 optimizer steps**, roundtrip=true and the new policy contract. Spent full_episode53504/suffix54528/smoke0/origin10112 account for exactly+128 suffix decisions from source118016. No prefix decision or CPU test is added to these counters. The initial and first-save files were read as receipts, without another checkpoint hash or PT load.

Only this fixed prefix/window, its initial/update/save receipts, and the old#34 first128 comparator were used. Planned1024 completion, later P09 outcomes, task success, full-P01 success and performance superiority are not asserted. Production/config/tests/master remain untouched; report writing stops here.

