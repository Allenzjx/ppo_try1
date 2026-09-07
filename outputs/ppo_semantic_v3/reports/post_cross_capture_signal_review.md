# Post-cross capture signal: bounded read-only review

## Finding

**The proposed `within_top_xy=false => all soft capture/approach credit is zero` premise is not true in the inspected implementation or either evaluated runtime.** `_current_capture_progress` in the current source and in both historical sources `f1a9bbf650b1` (C107264) and `42b91e857a0a` (C103168) uses continuous outside-distance attenuation, not the `within_top_xy` Boolean as an eligibility gate. C107264's actual outside-XY samples retain nonzero capture credit and measurable changes in that credit. This review therefore does not establish an outside-XY plateau in the observed failure window or support repairing such a gate.

Scope: the two existing final diagnosis reports; the relevant historical/current supervisor source and reward/config formulas; nine fixed post-cross C107264 decision endpoints and three C103168 endpoints. Values below were calculated with PowerShell from recorded current geometry/history and compared with recorded potential. No new rollout, raw/native rescan, tensor, Python, GPU/Isaac, hash, candidate implementation, production/config/test/master edit or active-P07 inspection. These are decision-end samples, not 120 Hz extrema or a controlled causal comparison.

Sources:

- C107264: `runs/ppo_semantic_v3/validation/20260907T0242264425838Z_gf1a9bbf650b1_be40bfe9223b49a4bec2468f632992d7/residual_and_projection_audit.jsonl`; existing `eval_107264_diagnosis.md`.
- C103168: `runs/ppo_semantic_v3/validation/20260907T0147389603120Z_g42b91e857a0a_a502f3cc872645a496b2fa1d6968b077/residual_and_projection_audit.jsonl`; existing `eval_103168_diagnosis.md`.
- `src/wlr50_clean/ppo/semantic_supervisor.py`: `TaskEvaluator.observe`, `TaskStageSupervisor.predicate`, `physical_potential`, `_current_capture_progress`, `_current_capture_retention`.
- `src/wlr50_clean/ppo/semantic_workspace_potential.py`, `semantic_reward.py`, and `configs/ppo_semantic_v3/{stage_task_spec,reward_config}.yaml`.

## Exact component and its actual limits

For an unplaced leg with hard active-lift qualification **and** historical front crossing, let `d` be the recorded nonnegative `top_xy_outside_distance_m`, `h` the signed current top clearance, and `T=clip(consecutive_top_samples/2)`:

```text
X = clip(1 - d / 0.25)
Z = 0.025 / (0.025 + abs(h))
capture = 0.5 * X * Z + 0.5 * T
capture contribution to global Phi = (0.85 / 4) * 0.2 * capture
                                   = 0.0425 * capture
```

`d` is the Euclidean distance to the same live obstacle rectangle expanded by the existing 5 mm measurement tolerance. Thus a small negative frontdistance may still be inside XY with `d=0`; a sample outside the rectangle has `d>0`, not an automatic zero. The helper validates finite clearance/distance and rejects negative distance.

There are real, narrower plateaus: without hard Q+C this helper returns zero; with `d>=0.25 m`, the **proximity half** is clipped to zero regardless of clearance. In a real no-contact outside sample, `T` is also zero. Those are not the situation at C107264's terminal `d=0.023900113 m`. Inside XY, the XY factor is intentionally constant at 1, but the vertical factor still changes with the absolute gap. For `0<d<0.25 m`, both reducing `d` and reducing positive `h` increase this component in the arithmetic model. This statement is not evidence that any chosen actuator action achieves those physical changes.

The public `placed_FL` completion-progress scalar is a different quantity: before genuine placement it combines `.35*lift + .35*cross + .15*top_geometry + .15*top_contact_fraction`. It can be `.7` or `.85` over ranges of geometry. **A plateau of that diagnostic/completion scalar is not a plateau of the global potential.** The global capture helper does not simply reuse `.7`/`.85`.

## Fixed C107264 post-cross samples

FL is hard Q=true, crossed=true, placed=false, AIR with load0, no obstacle contact and `T=0` in every selected row. Q/C occurred at1808/2854. In the following table, gap/front/outside are millimetres; `capture` includes its half-share but not the outer `.0425` factor.

| Tick | Front | Gap | Within XY | Outside d | X | Z | Capture | Capture in Phi | Recorded Phi |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 2856 | +2.871457 | +94.551628 | true | 0 | 1 | .209114677 | .104557338 | .004443687 | .405004276 |
| 2880 | +17.998222 | +83.767285 | true | 0 | 1 | .229848525 | .114924262 | .004884281 | .405418157 |
| 2912 | −1.305815 | +27.496492 | true | 0 | 1 | .476222297 | .238111149 | .010119724 | .410641558 |
| 2920 | −3.040310 | +24.940958 | true | 0 | 1 | .500591121 | .250295561 | .010637561 | .411154798 |
| 2928 | −2.703127 | +27.111118 | true | 0 | 1 | .479744072 | .239872036 | .010194562 | .410682069 |
| 3000 | −3.470656 | +26.679431 | true | 0 | 1 | .483751452 | .241875726 | .010279718 | .410720100 |
| 4000 | −15.333768 | +27.550404 | **false** | 10.333768 | .958664927 | .475733732 | **.228034622** | **.009691471** | .409781114 |
| 5320 | −28.794315 | +27.938729 | **false** | 23.794315 | .904822740 | .472244058 | **.213648581** | **.009080065** | .408781775 |
| 5328 | −28.900113 | +27.966564 | **false** | 23.900113 | .904399547 | .471995881 | **.213436430** | **.009071048** | .408769993 |

At2920, `top_geometry=true` and public `placed_FL=.85`; adjacent2912/2928 have `top_geometry=false` and `.7`. The continuously calculated capture value and global Phi are not the public scalar's Boolean geometry jump. At4000→5320, both endpoints are outside XY, yet capture falls from .228034622 to .213648581. This directly falsifies a zero/flat outside-XY capture component in those observed states.

Other Phi components also matter. For these nine C107264 rows, FL workspace=1, preserved post-cross unload=1, initial=1, lift credit=1 and carry=1; FR is already placed and its current retention=1. Thus the sampled fixed base is `.3825`, finish=0, and:

```text
Phi = .3825 + .0425*capture_FL + .02125*(workspace_RL + workspace_RR)
```

C107264's new reciprocal **soft workspace** terms for the still-unplaced rear legs remain active despite their predecessors not all being placed: RL/RR are .421988702/.427921361 at2856, .420193009/.427676972 at2920, .410621869/.417126017 at4000 and .401572590/.407789521 at5328. These change as the body/legs move. They do not grant hard workspace completion, lift, crossing or placement. Reconstructing complete Phi from the recorded geometry/history at all nine selected rows matches its recorded value within `5.56e-17` in this PowerShell arithmetic check.

## C103168 comparison: same capture formula, different workspace version

The earlier runtime already has the same distance-based capture helper. FL's selected rows are all inside XY, hard Q+C, unplaced, AIR/load0/no contact:

| Tick | Front mm | Gap mm | X | Z | Capture | Capture in Phi | Recorded Phi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2792 | +2.082292 | +87.498312 | 1 | .222225556 | .111112778 | .004722293 | .387222293 |
| 3016 | +5.836252 | +16.587553 | 1 | .601141404 | .300570702 | .012774255 | .395274255 |
| 5273 | +5.441311 | +16.635617 | 1 | .600447453 | .300223727 | .012759508 | .395259508 |

C103168 still uses the older clipped workspace potential; both rear workspace credits are zero in these three selected states. Its sampled Phi is therefore `.3825 + .0425*capture_FL`, reconstructed within `5.56e-17`. Do not compare the two runs' total Phi values as if their workspace definitions and actual states were identical. In both runs FL's true contact half remains zero, so near-platform AIR is not placement or success.

## Reward is not zero, and signal does not establish cause

Recorded reward uses `5*(.995*Phi_next-Phi_before)` once per policy decision, actual elapsed-time cost, body-stability and applied-drive smoothness costs. Contact-motion cost is zero in these selected no-contact rows; disabled residual regularization contributes zero. At a task terminal, reward uses next-Phi=0 and failure event−40, even though the diagnostic physical snapshot still exposes its nonzero current Phi.

- C107264 tick2920: diagnostic Phi rises .410641558→.411154798, but potential shaping is −.007712670 and total reward −.009518632; the discounted formula and other costs prevent interpreting an increase in Phi as necessarily positive reward.
- Tick4000 (outside XY): shaping −.010299862, body −.000083800, smoothness −.000625000, time cost −.001333333, total −.012341996. At5320: shaping −.010272877 and total −.012313579. Neither total reward nor its potential is zero/constant merely because XY is false.
- Terminal5328: snapshot Phi=.408769993, reward next-Phi=0, shaping−2.043908877, event−40, total−42.045951104. This finite-horizon termination convention is not an outside-XY component becoming zero.

The two actual P05 timeouts and the lack of FL contact remain as recorded. This analysis establishes the available arithmetic signals, not policy sensitivity, an attainable control gradient, stationary preference, the unique cause of missing contact, or improvement from any revision. No candidate change is selected or implemented.

## Boundaries to preserve in any later review

The helper is leg-parameterized and applied with the same formula to all four legs; predecessor/order checks remain in `physical_potential`. Already-placed legs bypass first-capture and retain the existing `.8+.2*current_retention` branch, including legitimate later AIR. Actual placement still needs historical crossing and consecutive measured loaded TOP samples (verified obstacle contact, current top geometry and nonnegative frontdistance); no amount of soft proximity replaces that evidence. This report neither alters hard contact/history, adds a gate, nor proposes a leg-specific exemption. Review completed; stopped without reading the active course.
