# 57ae41e sealed headless: actual RR capture response

The CPU-only diagnostic completed successfully; no simulator/model/optimizer was run. All four evidence streams were read once. Parsed counts: 13,156 physical rows (tick 0 included), 13,155 contiguous native rows, 1,645 decision rows, 20 stage/history rows. Source and hashes are retained in `fixed_COM_RR_response.json`.

**Real result:** 109.625 s / tick 13,155, P09 `INCOMPLETE_CONTROLLER_BLOCKED`, no RR TOP/placed and no P10/RL transfer. This was headless: no new video. These are evaluation decisions, not learning credit.

## Timing, response, and finite stop

| Episode tick / time | Actual milestone | RR bottom gap / center front distance |
|---|---|---|
| 8,280 / 69.000 s | First P07; fixed CoM→FL reference | −51.013 / −224.662 mm |
| 9,516 / 79.300 s | RR assist takes hip/knee ownership, BLOCKED pending actual crossing; zero search travel | +27.866 / −4.807 mm |
| 9,555 / 79.625 s | Real RR cross history | +27.510 / +0.089 mm |
| 9,556 / 79.633 s | First actual DESCEND | +27.656 / +0.234 mm |
| 9,773 / 81.442 s | First committed-source-end wheel envelope and nonzero projection | +62.281 / +28.215 mm |
| 10,756 / 89.633 s | First positive knee-search step after 20° hip travel | +47.479 / +73.969 mm |
| 12,915 / 107.625 s | Terminal minus 2 s | +16.067 / +53.468 mm |
| 13,155 / 109.625 s | BLOCKED `finite_search_travel_or_margin` | +13.429 / +50.089 mm |

Last 2 s **net** change, not a monotonicity claim: RR gap −2.6382 mm, front −3.3790 mm, actual hip −2.08310→−2.08786° (−0.00476°), actual knee −27.26226→−25.27003° (+1.99223°). Body z increased 0.1437 mm. Thus there was continuing measured lowering while the overall wheel/leg/body system moved; this does not isolate a knee-only physical derivative.

At termination: public combined search travel = 40° (20° hip + 20° knee), descent elapsed = 30 s. The remaining 32 s time ceiling is not what stopped it. RR final hip/knee = −1.78591/−25.36118°, actual = −2.08786/−25.27003°; actual-minus-target = −0.30195/+0.09115°. Both are well inside unchanged command limits (hip [−135,135]°, knee [−60,210]°): hip downward headroom 132.912°, knee positive headroom 235.270°. These physical limits are **not authorization to extend the bounded search**. FL actual knee −58.26094° is only 1.73906° above its −60° lower limit (final −58°).

## Four-wheel command versus response (canonical order FL / FR / RL / RR)

First activation used pre-observation tick 9,772, generated post-step tick 9,773. All FL/FR/RL current supports were verified; source finite pulse and explicit final stop were retained and the committed-end proof passed. Gain=.578016. Candidate [−.923088,+.069484,−.015701,−.079777] became FINAL [−.908088,+.084484,−.000701,−.079777] rad/s: first three changed at exactly +1.8 rad/s²; RR was untouched. The initial negative FL target decays through the existing slew envelope rather than jumping instantly positive.

Terminal dispatch used pre-observation tick 13,154; native global dispatch tick 13,334 generated episode post-step tick 13,155. Source N still holds all four zeros. Final source sample=4247, endpoint/last-authored-stop=864 (source time 7.2 s), no fresh/finite wheel owners, committed source proof true. This is a scoped controller projection after a legitimate source stop, not loss of nominal, not a residual-mask bug, and not new learning.

| Wheel | Raw policy | Effective pre-projection residual/candidate rad/s | FINAL canonical rad/s | Native target rad/s | Actual canonical rad/s |
|---|---:|---:|---:|---:|---:|
| FL | −1.044075 | −.935391 | +.048356 | −.048356 | +.016952 |
| FR | +.053128 | +.063693 | +.063693 | +.063693 | +.050659 |
| RL | −.001511 | −.001511 | +.048356 | −.048356 | +.164394 |
| RR | −.147422 | −.087818 | −.087818 | −.087818 | −.087593 |

All twelve residual permissions are 1; the scoped wheel layer is explicitly a subsequent projection, not a mask. Terminal gain=.161186/floor=.048356; selected channels8/9/10, wheel FINAL change rates [+0.004787,0,+0.004787,0] rad/s². Native targets are independently read from `robot._joint_*_target_sim` after the existing write; native IDs were not persisted and remain null. Measured RL speed differs substantially from its positive target, so target agreement is not being presented as tracking or traction proof. RR is AIR: its negative spin is not ground traction.

## Actual current bearing and ownership at termination

FL TOP 3.2906 N, FR TOP 13.4250 N, RL GROUND 12.0377 N are verified current supports; RR AIR has 0 N and no obstacle pair. Historical FL/FR placement does not replace these current readings. RR remains current-Q true, legally over TOP XY, but has no contact/place.

RR hip/knee owner remains `rr_capture_assist` [6,7] even in final finite-budget BLOCKED. Source N RR [−6.9,−37.8]°, actual mapper N [−16.9,−47.8]°, effective policy [+16.57652,−10.2]° would produce pre-assist [−.32348,−58]°. The explicit capture owner instead commands [−1.78591,−25.36118]°. These are controller actions, not a claimed learned mean change. Other servo FINAL/actual pairs (hip,knee): FL [−24.0185,−58]/[−24.0659,−58.2609], FR [8.7954,−22.0435]/[8.9517,−22.5653], RL [2.1183,20.3103]/[3.4201,20.4145].

## Fixed mass-CoM direction, without moving-foot credit

First P07 tick8,280 uses actual mass COM0=[.550942,−.128444,.182059] m, FL wheel center0=[.782014,.084238,.167926] m, fixed planar d0=[.735779,.677222]. At terminal the true mass COM is [.774283,−.091149,.162300] m. CoM displacement projects +189.587 mm on that unchanged d0; receiver displacement is separately +105.471 mm and does not enter CoM progress. This diagonal projection contains substantial forward traversal; it is not proof of sufficient lateral unloading or stable support. Last 2 s fixed CoM progress increased only .7462 mm. Body terminal roll/pitch from its recorded unit quaternion: −9.368°/+2.556°.

P10 was never reached, so no COM→FR anchor or measured RL-direction progress exists. Actual RR hip mount world position is unavailable in this headless schema (no USD joint localPos0 transform); the separately recorded rear-right-upper link origin z=.232838 m is **not** substituted as hip height. No causal claim or success is inferred from the fixed projection.
