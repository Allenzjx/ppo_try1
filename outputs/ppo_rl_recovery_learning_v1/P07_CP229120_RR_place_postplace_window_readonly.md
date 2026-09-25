# CP229120 P07 student window: RR placement and post-placement support loss

Status: read-only diagnostic, not a training or release gate.

## Scope and clock

- Source run: `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1046432416199Z_g65a9255be6d9_bff8c2f6d5574a42925e6b9d8f382a94`.
- The analysis is capped at episode physics tick `6536` (`54.466667 s`). Only complete JSONL records already written at that boundary were selected; later rows from the continuing run were ignored.
- The natural/N prefix ended at tick `5160` (`43.000000 s`, P06). The first student decision ended at tick `5168`; therefore RR qualification, crossing, placement, and the later support loss are all in the student-controlled suffix. The prefix is not credited with RR placement.
- Exact event ticks use the recorded 120 Hz physical-event evidence: RR qualified `5432`, crossed `6185`, and placed `6270`. Each audit row aggregates eight 120 Hz ticks and is keyed by its 15 Hz endpoint. The command fields are the **last-substep** values, not a reconstruction of all eight internal commands; post-step contact/geometry is measured at the endpoint.
- Canonical wheel order is `[FL, FR, RL, RR]`. `N` is `applied_audit.nominal_action_full12[8:12]`; mapped N is `actuator_target_effect_audit.native_drive_target_full12[8:12]`; raw policy is the dimensionless `policy_request.selected_raw_full12[8:12]`; `P` is the filtered/projected physical residual `projected_residual_full12[8:12]`; canonical FINAL ACK is `actual_drive_target_full12[8:12]`. `applied_action_full12` is only the pre-backend request, not FINAL authority. Actual native setter targets are independently recorded in `actual_native_targets.wheel_velocity_rad_s` with axis signs `[-,+,-,+]`; measured canonical response is `physical_evaluator.measured_wheel_velocity_rad_s`.

Within all 172 selected decision rows, mapped N equals source N exactly on the four wheels, all three copies of the raw policy sample match, and the wheel part of the pre-backend request equals the canonical FINAL ACK. Thus any N-to-FINAL difference shown below is the recorded projected policy contribution, not a nominal-mapper discrepancy. The native setter receipt independently verifies the same FINAL after documented axis conversion.

## Event sequence

- The first N-versus-policy/FINAL divergence is already the first student endpoint `5168`: N=mapped N `[+.300,+.300,+.300,+.300]`, raw policy wheels `[-.038,-.075,-.063,-.025]`, projected physical P `[-.046,-.089,-.063,-.015]`, and FINAL ACK `[+.254,+.211,+.237,+.285]`. It is not a post-placement-only effect.
- At endpoint `6184` (one physics tick before exact RR crossing), last-substep source/mapped N was `+0.3` on all wheels, but the policy contribution made FL FINAL negative.
- At endpoint `6264`, RR already reported TOP contact and force, but remained `1.940 mm` outside legal top XY and was not placed.
- Exact placement occurred at tick `6270`, inside the eight-tick aggregation ending at `6272`. The row's last-substep source N is finite FL-only reverse `[-1.07, 0, 0, 0]`, while policy also requests negative FL. Because the log does not retain N for every internal substep, it cannot order the source-N transition against placement tick `6270` more finely than this shared interval.
- The first recorded endpoint with RR current contact lost was `6328` (`52.733333 s`): RR was AIR/`0 N` but still within legal top XY. The preceding endpoint `6320` still had RR TOP/`3.952 N`. Therefore loss occurred in `(6320, 6328]` on this sampled clock.
- FL-only reverse is present in every last-substep snapshot from endpoints `6272` through `6352`. At endpoint `6360`, the observed source snapshot is all-wheel reverse `[-0.3, -0.3, -0.3, -0.3]`.
- RR first appeared outside legal top XY at endpoint `6416` (`53.466667 s`), `0.733333 s` after the first recorded RR contact-loss endpoint. Contact loss therefore preceded recorded XY exit.
- FL transiently lost contact at `6384` and regained TOP at `6392`; its first loss persisting through the cutoff began at endpoint `6480`.

## Recorded last-substep command and endpoint response

All values are canonical rad/s. `P` is the applied physical policy request, not the normalized network sample. FINAL is the canonical actuator ACK; measured qd is the post-step physical response. The listed command vectors are not asserted to be constant over all eight internal ticks.

| endpoint tick / phase | source N = mapped N | policy P | FINAL ACK | measured qd |
|---|---|---|---|---|
| `6184` / P09 | `[+.300, +.300, +.300, +.300]` | `[-.531, +.139, -.404, -.063]` | `[-.231, +.439, -.104, +.237]` | `[-.262, +.276, +.051, +.238]` |
| `6264` / P09 | `[+.300, +.300, +.300, +.300]` | `[-.508, -.034, -.122, +.001]` | `[-.208, +.266, +.178, +.301]` | `[-.246, +.220, +.214, +.345]` |
| `6272` / P10 | `[-1.070, 0, 0, 0]` | `[-.388, -.066, -.128, +.000]` | `[-1.458, -.066, -.128, +.000]` | `[-1.520, -.087, -.047, +.082]` |
| `6320` / P12 | `[-1.070, 0, 0, 0]` | `[-.851, -.238, -.339, +.019]` | `[-1.921, -.238, -.339, +.019]` | `[-1.862, -.122, -.350, -.475]` |
| `6328` / P12 | `[-1.070, 0, 0, 0]` | `[-.758, -.358, -.415, +.051]` | `[-1.828, -.358, -.415, +.051]` | `[-1.971, -.219, -.919, +.051]` |
| `6360` / P12 | `[-.300, -.300, -.300, -.300]` | `[-.758, -.358, -.195, -.248]` | `[-1.058, -.658, -.495, -.548]` | `[-1.110, -.556, -.379, -.548]` |
| `6416` / P12 | `[-.300, -.300, -.300, -.300]` | `[-.297, +.482, -.070, -.029]` | `[-.597, +.182, -.370, -.329]` | `[-.569, +.226, +.266, -.318]` |
| `6536` / P12 | `[-.300, -.300, -.300, -.300]` | `[-.614, +.069, -.139, +.002]` | `[-.914, -.231, -.439, -.298]` | `[-.914, -.359, -.248, -.297]` |

## Current support and geometry at key endpoints

| endpoint | RR current state | RR geometry | FL current state | interpretation |
|---|---|---|---|---|
| `6264` | TOP, `3.645 N`, placed false | outside legal XY `1.940 mm`; gap `-1.192 mm`; front `-6.940 mm` | TOP, `3.451 N` | contact existed, but placement legality was not yet satisfied |
| `6272` | TOP, `4.288 N`, placed true | legal XY; gap `-0.947 mm`; front `-3.323 mm` | TOP, `3.090 N` | first 15 Hz endpoint after exact placement tick `6270` |
| `6320` | TOP, `3.952 N`, placed true | legal XY; gap `-0.287 mm`; front `+83.632 mm` | TOP, `12.011 N` | last selected endpoint before recorded RR support loss |
| `6328` | AIR, `0 N`, historical placed true | still legal XY; gap `-0.688 mm`; front `+98.860 mm` | TOP, `7.577 N` | first recorded loss; not yet an XY-exit event |
| `6360` | AIR, `0 N` | still legal XY; gap `+17.904 mm`; front `+86.124 mm` | TOP, `6.269 N` | source has transitioned to all-wheel reverse |
| `6416` | AIR, `0 N` | outside legal XY `10.523 mm`; gap `+60.519 mm`; front `-15.523 mm` | TOP, `1.110 N` | first recorded post-place legal-XY exit |
| `6480` | AIR, `0 N` | outside `41.389 mm`; gap `+56.003 mm`; front `-46.389 mm` | NONE, `0 N` | first FL loss persisting through cutoff |
| `6536` | AIR, `0 N` | outside `76.743 mm`; gap `-24.175 mm`; front `-81.743 mm` | NONE, `0 N` | cutoff; historical placement does not mean current support |

Cutoff sanity check: in the unique endpoint-`6536` row, RR is AIR/`0 N`. The leg that is GROUND near a `-50 mm` clearance is RL (`14.660 N`, `-50.401 mm`), not RR; these legs must not be interchanged.

## Four-leg configuration and load-path addendum

The eight joint values below are not a separately sampled post-step evaluator vector. The audit records, per servo, a tracking-reference nominal and `current_actual_canonical_error_deg`; the values are the exact logged reconstruction `nominal_deg - current_actual_canonical_error_deg`, corroborated by the sibling `actual_measured_physical_rad`. Order is `[FLh, FLk, FRh, FRk, RLh, RLk, RRh, RRk]`, in degrees. CoM deltas are direct world-frame `x/y` differences from the fixed endpoint `6272`; no interpolation or moving receiver frame is used.

| endpoint | reconstructed measured h/k (deg) | current loads `[FL,FR,RL,RR]` N | RR / FL legal top XY | CoM delta from `6272` (mm x, y) |
|---|---|---|---|---|
| `6272` | `[+44.57,-39.87,+36.27,-20.10,+19.33,+0.98,+4.04,-58.09]` | `[3.090,11.550,10.362,4.288]` | `true / true` | `[0,0]` |
| `6320` | `[-2.69,-57.34,+17.20,-42.61,+5.42,+26.08,-1.71,-47.06]` | `[12.011,11.944,8.466,3.952]` | `true / true` | `[+80.245,+6.230]` |
| `6328` | `[-10.40,-57.73,+14.90,-43.93,+5.24,+25.98,-5.99,-46.45]` | `[7.577,14.717,11.732,0]` | `true / true` | `[+88.506,+6.960]` |
| `6360` | `[-26.96,-58.12,+18.30,-39.82,-1.09,+25.06,-15.61,-43.50]` | `[6.269,14.466,10.290,0]` | `true / true` | `[+69.168,+11.551]` |

Current contact states are FL/FR TOP and RL GROUND at all four endpoints. RR is TOP at `6272` and `6320`, then AIR at `6328` and `6360`. Thus the `+80.245 mm` world-x CoM motion by `6320` coexists with retained RR support, while the first sampled RR loss is at `6328`; the table does not resolve their intra-decision ordering or assign causality.

For descriptive context only, the sealed successful-N response table (`6152..6233`) shows a different load path: FR knee stayed near `+30 deg` and FR unloaded from OBSTACLE to AIR, while FL knee moved about `-12.0 -> -33.1 deg` and FL/RR remained loaded. Here, FR knee is already negative (`-20.10 -> -43.93 deg` through first RR loss), FR remains heavily loaded TOP (`11.55..14.72 N`), FL is more folded (`-39.87`, then about `-57.7 deg`) while TOP, and RR changes from TOP `3.952 N` to AIR `0 N`. These are configuration/load correlations across different trajectories, not evidence that the N angles are necessary, not an angle target, and not a control-fix recommendation.

## Source stop, transition hold, and attribution limits

- Effective source-N zero is directly visible at endpoints `5328..5640` and once at `6056`. Source N is `+0.3` on all wheels at endpoints `6064..6264`; no selected endpoint from placement through cutoff has all-zero source N. Because only the last substep is stored, this does not exclude an unlogged intra-decision zero.
- `final_stop_owner` is the P13/post-window stop owner and is not stop authority for this P07--P12 window. It is therefore not used to infer whether the finite recording source stopped.
- Five `handoff_hold_used` events occurred, each for one 120 Hz tick only (`5169`, `5177`, `6273`, `6281`, `6297`). This flag holds the projected residual at a phase handoff; it is not a source-N hold and not a wheel-stop flag. There is consequently no evidence that these holds delayed a recorded source stop.
- After RR support loss, the P12 joint lane correctly held for `current_RR_support_before_new_RL_unload`, while the P09 source wheel layer remained active. That distinction matters: the joint dependency wait did not stop the wheel source.
- Source reverse is temporally associated with loss at the recorded endpoints: FL-only reverse is the last-substep N at the first post-placement endpoint and at the first recorded RR contact-loss endpoint; all-wheel reverse is observed before recorded XY exit. However, policy requests are nonzero and materially alter FINAL throughout. Even before the reverse source snapshots, policy overcomes `+0.3` source N and makes FL FINAL negative (for example endpoints `5648`, `6184`, and `6264`). At endpoint `6272` and at first recorded loss `6328`, source and policy both reinforce negative FL.
- Consequently this window does **not** isolate source N as the cause of the loss or prove that source reverse alone pushed RR out. It shows a concrete continuation-compatibility concern: the sampled finite source sequence and learned policy combine into reverse FINAL commands around placement/support loss, current support loss is recorded before the later legal-XY exit, and no all-zero source-N endpoint intervenes after placement. The 15 Hz audit cannot establish the unrecorded intra-decision ordering required for a stronger claim.

The result is evidence for reviewing nominal continuation/stop ownership at the placed-RR transition; it is not a counterfactual, an A/B causal result, a new control proposal, or additional PPO/AUX credit.
