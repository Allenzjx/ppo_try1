# CP201728 RR post-cross landing learning signal: bounded read-only result

Runtime `0001c3138b0b`; current `p05_hip_only_continuation_v1` task/reward configuration. Inspected the already-recorded CP201728 natural-P01 deterministic run `20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc`, matched decision endpoints 6848–7600, and the preceding completed 2,048-decision update block `20260922T0415552437349Z_g0001c3138b0b_7b34b44b4b9d49ddb087585dec7acef6`. No code/configuration changes or physics launches; the next training block is untouched.

## Answer

1. **RR descent and real contact already have a learning signal.** Lowering a positive top gap increases the active task potential. Consecutive real TOP observations add capture credit; actual placement changes to the placed-leg retention branch.
2. **Earned unload/initial-lift progress is not automatically lost on TOP landing.** Post-cross unload credit is explicitly retired to 1; valid same-attempt RR TOP retains lift credit. Ground recontact or loss of required live control evidence is different.
3. **A separate preparatory shaping term remains after crossing and can locally compete with descent:** RR workspace still depends on the current 0.5-second FL radial-contraction response and FL joint-margin proxy. This is not an actuator bug or a demonstrated sole cause of hovering.
4. **The preceding new training block had zero RR-crossed/TOP/placed samples.** Its improvement to crossing was not trained on successful post-cross RR landing samples from that block. Historical checkpoint training coverage is not inferred here.

## Existing production reward

`semantic_supervisor.py:1228–1276` uses the same global physical potential independent of the P09 label. Before RR placement its weighted contribution is:

`Phi_RR = 0.2125 * (0.1*workspace + 0.1*unload + 0.25*(0.25*initial + 0.75*lift_credit) + 0.35*carry + 0.2*capture)`

For a qualified, crossed RR over legal top XY, `semantic_supervisor.py:1333–1364` takes the generic RR branch, **not** the FL-only multiscale branch:

`capture = 0.5 * xy * 0.025/(0.025 + abs(gap)) + 0.5 * min(consecutive_TOP/2, 1)`

Thus while AIR and with positive gap/valid XY, smaller gap strictly increases the capture share. The dependence on absolute gap does not reward deeper penetration beyond the surface. This term is in the active `task_progress` family, not the zero-weight contact-quality family.

`semantic_reward.py:436–453` consumes `5*(0.9985*Phi_after - Phi_before)`, with task family weight 1 and time cost 0.02/s. At 15 Hz, a stationary potential still incurs discount and time cost; a small physical improvement can make reward less negative without making total reward positive. That is not a missing signal.

### Landing does not itself erase earned lift/unload

- Qualified crossing forces `unload=1`, independently of the currently measured transfer/load score (`semantic_supervisor.py:1254–1272`). Loading RR is therefore not automatically punished as lost unload progress.
- `_current_lift_credit` (`1394–1408`) returns 1 for a crossed RR with current valid lift; it does not require current AIR.
- `_observe_functional_rr` (`896–989`) retains established same-attempt lift across TOP contact. The free-AIR diagnostic counter resets on contact, but `_rr_established` is not cleared by TOP. Current validity requires no ground contact, current ground-relative height/control evidence, and two verified other supports—not fresh upward motion during landing.
- Initial-clearance history is cleared by ground recontact, not ordinary TOP contact (`638–642`). Actual TOP samples can therefore add capture credit while the initial and established-lift shares remain.
- Loss of two-other-support corroboration, real ground recontact, invalid geometry or independent safety failure can reduce current lift credit. This is distinct from treating any landing load as an error. The inspected CP201728 window never actually reaches RR TOP; the landing conclusion above is code-path verification, not an invented physical landing trial.

## Matched real decision evidence

All values were recomputed from each recorded evaluator state using the production supervisor helpers; reconstructed total potential equals the recorded potential exactly at the selected endpoints. `capture PBRS` below isolates only the RR capture share, not total policy effect or total reward.

| Decision interval | RR gap before→after mm | RR capture PBRS | Total PBRS | Actual reward | Interpretation |
|---|---:|---:|---:|---:|---|
| 6856→6864 | 33.523→39.413 | +0.041176 | +0.068033 | +0.066699 | Real crossing at 6858 enables capture shaping; this is not reward for increasing gap. |
| 6864→6872 | 39.413→52.647 | -0.007080 | +0.006959 | +0.005625 | RR gap worsens; other physical contributions rise enough to offset its negative local contribution. |
| 6904→6912 | 71.947→67.014 | +0.001425 | -0.000309 | -0.001642 | Actual descent improves capture signal despite slightly negative total reward. |
| 6992→7000 | 63.639→62.224 | +0.000440 | -0.003853 | -0.005186 | Same positive descent signal; discount/other potential terms remain. |
| 7592→7600 | 61.295→61.492 | -0.000116 | -0.004932 | -0.006265 | Hover oscillation slightly increases gap. |

At 7600: initial=1, lift credit=1, carry=1, used unload=1, workspace=0.685920, capture=0.144523; RR still has no TOP observations/force. Global Phi=0.610738729; RR Phi=0.169468012. All 95 examined decision intervals have zero weighted collider-geometry cost and zero weighted contact-quality cost. No active impact, stability or contact-load penalty explains this hover window. The live body/obstacle separation at 7600 is 82.591 mm, already above the 20 mm quality margin.

## Concrete competing shaping: preparation response persists after crossing

`_workspace_potential_progress` (`semantic_supervisor.py:1306–1331`) still uses:

`workspace_RR = 0.5*RR_edge_interval_progress + 0.25*FL_joint_margin_proxy + 0.25*clip(FL_radial_contraction_over_last_0.5s / 0.001)`

The components come from `semantic_transfer_roles.py:170–185,258`. This is a live receiver-space proxy, not exact reachability, support-force evidence, or a historical pose. Unlike unload, this preparatory share is not retired after qualified crossing. Created space that is merely held stops contributing positive short-window contraction once its motion leaves the window.

Observed competing pair 7152→7160:

- RR remains crossed/qualified/AIR with three real other supports; gap improves 62.283067→62.196051 mm.
- FL remains real TOP support. Its 0.5-second contraction measure decreases 0.084039→0.035714 mm; margin proxy changes 0.747396→0.745601.
- RR workspace decreases 0.707859→0.695329.
- Undiscounted RR capture-potential change is **+0.000006074**, while workspace-potential change is **-0.000266264**. Their sum is -0.000260190 even before discounting. The preparatory proxy therefore demonstrably outweighs this small useful descent increment.

Over 6880→7600, RR gap improves 63.467→61.492 mm, yet workspace drops 1→0.685920 as the short FL motion response expires and FL knee margin stays below 10 degrees. This is a genuine local competing shaping influence to keep in view. It does **not** establish that the observed policy tried a feasible full landing and was penalized, or that this term is the unique cause; no such landing sample exists in the inspected run. Potential shaping/local PPO optimization should not be confused with proof that the global task optimum favors hovering.

## Actual preceding update-block coverage

Read-only counts from all 2,048 `residual_and_projection_audit.jsonl` decision endpoints used in the preceding new block:

| Observed endpoint condition | Count |
|---|---:|
| P09 stage | 451 |
| RR current qualified lift | 12 |
| RR crossed | 0 |
| RR actual TOP | 0 |
| RR placed | 0 |

Therefore, post-cross landing coverage is genuinely absent in this **new block**, even though P09 stage labels are common. The CP201728 full deterministic crossing is evaluation evidence, not an optimizer training sample. No conclusion is drawn about all earlier historical checkpoints.

## Boundary

Continue the currently running authorized block. The existing signal is not absent and no earned-lift/unload-on-landing bug was found. The specific post-cross workspace competition is documented for interpretation of subsequent real capture samples; no reward edit, new gate, fixed pose, or mandatory intervention is implemented or recommended mid-run by this check.
