# C118016 P05 capture control — bounded read-only review

Status: completed-log diagnosis only; no production/config/test change or new simulation. Runtime source is `git show f4bfe2560bfd:<path>`, not the subsequently edited history-policy kernel. No Python, Torch, checkpoint loading, or repeated hashing was used. The active #37 run was not read.

## Scope and outcome

Source: `runs/ppo_semantic_v3/validation/20260907T0707251670686Z_gf4bfe2560bfd_b8a1a4c5cb244de4951774a4483803e5`. Existing [final diagnosis](eval_118016_diagnosis.md) supplies the full event/native ledger. This review made one selected-field decision-ledger pass, summarized the final 120 decision ends (ticks 4400–5352), selected a few P05 landmarks, and read the terminal physical observation. It did not repeat a complete 120 Hz raw/native audit.

The fixed-mean natural-P01 evaluation ended after 669 decisions / 5352 physics ticks / 44.6 s: P05 age 30 s, `INCOMPLETE_CONTROLLER_BLOCKED`, task success false, physical validity true, no hard physical failure, and zero optimizer updates. Software execution succeeded; physical task completion did not.

FL really qualified at tick 1843 and crossed at 2869, but never placed. P05 decision ends contain no FL obstacle contact, TOP contact, or TOP geometry. Terminal FL is AIR, front distance +6.726686 mm, bottom clearance +31.968420 mm, load 0, consecutive TOP samples 0. The first missing task is **real FL platform capture**, not a historical entry condition or an uncredited crossing. Terminal `placed_FL=0.7` is partial progress, not a placement event.

## Actual command and measured response

Canonical servo values below are degrees; wheel commands are rad/s. “Native baseline” is the mapper output before the separate policy bias; “actual” is the canonical drive target, not a measured joint angle or raw native radian tensor.

| Tick / P05 age | FL front / gap (mm) | FL nominal hip,knee | Policy residual hip,knee | Actual drive hip,knee |
|---|---:|---:|---:|---:|
| 1848 / 0.800 s | −124.497 / +5.224 | 45.0, −36.7 | +4.567, −3.405 | 49.567, −41.355 |
| 2872 / 9.333 s, after crossing | +5.732 / +86.253 | 48.2, −18.7 | +5.250, −4.081 | 54.700, −21.531 |
| 2904 / 9.600 s | +32.235 / +85.134 | 35.0, −13.4 | +5.020, −3.398 | 40.020, −15.548 |
| 2944 / 9.933 s, closest sampled post-cross gap | +11.823 / +25.385 | 22.8, −13.4 | +4.843, −3.872 | 26.393, −16.022 |
| 5352 / 30.000 s | +6.727 / +31.968 | 22.8, −13.4 | +5.037, −3.790 | 28.512, −15.940 |

All these post-cross samples are AIR/load 0. The closest decision-end clearance is still above the configured +25 mm TOP-geometry ceiling; more importantly, there is no verified loaded obstacle contact. Being inside top XY alone is insufficient.

The final 120 decision ends show a near-stationary command pattern:

- FL nominal is exactly `[22.8, -13.4]`; mapped baseline is exactly `[23.475257861, -12.15]`; separate controller bias is zero at the terminal receipt.
- FL raw output ranges are hip `[0.287324, 0.287822]`, knee `[−0.159741, −0.158480]`. Residual ranges are `[+5.034061,+5.042319]` and `[−3.801509,−3.771996]`. Actual drive ranges are `[28.509318,28.517577]` and `[−15.951509,−15.921996]`.
- Nominal wheel commands are all zero. Actual wheel means in FL/FR/RL/RR order are `[-0.07559973,-0.05087923,-0.03540554,+0.06880669]`. They are policy residuals, not hidden continued nominal advance. This deterministic evaluation does not establish the policy's stochastic exploration coverage.
- FL residual equals `tanh(raw) * [18,24]` within `2.665e−15` at every selected tail boundary. Requested and effective post-mapper FL residuals agree exactly; candidate before final slew and final actual FL target agree exactly. All tail receipts report empty servo-clipped indices; FL geometry-adjustment cells are zero. Thus no sampled evidence supports “the tail requested a different FL target but cap/headroom/final slew removed it.” This boundary check is not a claim that no transient correction exists at any intervening 120 Hz tick.
- FL front distance stays +6.671…+6.770 mm and clearance +30.968…+32.342 mm. There are three current supports throughout the tail, but FL is not one of them.

At terminal, measured FL hip/knee are `27.938122 / -15.943152` degrees, compared with actual targets `28.512273 / -15.939635`; errors are `+0.574151 / +0.003517` degrees. The wheel body vertical velocity is **+0.054913 m/s** at this instant, while base vertical velocity is +0.029699 m/s. It is not a measured downward approach at that endpoint. Joint signs alone do not establish Cartesian down/up authority, and no FK perturbation or controlled physical counterfactual was performed.

The terminal supports are FR TOP (load 0.418400), RL GROUND (0.507382), and RR GROUND (0.074217); FL is AIR/load 0. CoM is inside the measured support polygon with +19.473 mm margin. Base velocity is approximately `[-0.006357,-0.002918,+0.029699] m/s`. These facts explain why the physical evaluator can remain valid while FL capture is absent; they do not prove that this support arrangement caused hovering.

## Finite nominal schedule versus semantic completion

The frozen P05 contract's `physical_purpose` explicitly states that it commits the placement posture and that top load latches during the immediately continuous P06 wheel advance. Its source wheel suggestion is four `+0.3` commands from relative 0.866667 to 9.066667 s, then four zeros. The last source servo waypoint at 9.733333 s is FL `[22.8,-13.4]`.

Runtime `NominalMotionProvider._continuous_advisory` preserves unfinished older channel owners and lets newer touched channels take ownership. It does not dispatch a future P06 layer while the semantic stage is P05. The v3 P05 task requires `placed_FL` before transition, and P06 also lists `placed_FL` as a valid-start condition. No P05-specific geometry-feedback descent or wheel-tail assist is supplied; the special measured wheel assists concern P02 and rear P09/P12, and the P06 tail requires an actual P06 layer.

This is a concrete **advisory/completion sequencing gap**: the source description expects part of capture during the next rolling suggestion, whereas semantic execution waits for capture before that suggestion can start. In this run, after the finite P05 endpoint the residual must provide any further adjustment. The actual tail instead holds nearly fixed FL targets and issues the small wheel values above. All 12 residual channels remain open; P05 wheel residual range is ±0.3 rad/s. This does not prove that continued P06 motion is necessary/safe, that current residual authority is inadequate, or that a fixed replay/looser contact gate is warranted.

Static exploration-domain caveat for a *hypothetical*, not implemented, four-wheel +0.3 suggestion while still in P05: the unchanged ±0.3 wheel residual gives the closed interval `[0,+0.6]`, not the current nominal-zero `[-0.3,+0.3]`. It can cancel to zero only at the negative residual boundary and cannot command reverse. With ideal nonsaturated `tanh(raw)` there is no finite raw margin at that boundary; actual floating-point saturation/quantization would need its own exact-target check and is not evidence of useful reverse authority. The ±2.1 physical wheel hard limit does not restore the missing negative side. Thus adding that nominal suggestion alone would change the expressed action domain; it would not preserve symmetric policy freedom or the previous physical output for the same raw action. A later selected nominal change would require the same explicit cancellation/headroom/rate/finite-raw review, without this report choosing a new cap or asking for a success gate. Nothing was changed or physically tested here.

## Reward: not a flat capture signal or a demonstrated contradiction

Runtime `semantic_supervisor.py:physical_potential` and `_current_capture_progress` use the opted-in post-cross surface-proximity rule. For this qualified, crossed, unplaced FL, with current top XY inside and zero TOP samples:

`capture = 0.5 * 0.025 / (0.025 + abs(bottom_clearance_m))`.

This component is continuous above +25 mm; the hard TOP-geometry boundary does not switch it off. At the terminal clearance it is **0.219419813**, contributing **0.009325342** to global Phi. With the same other measured quantities and clearance lower by 1 mm, global Phi would increase by **0.000166618**; the one-step PBRS difference would be **+0.000831840** at gamma 0.9985 and potential weight 5. These are formula comparisons, not simulated reachable actions or a net-return guarantee. At +25 mm capture is 0.25, at zero gap without TOP samples 0.5; the remaining share requires genuine contact.

After qualified crossing, unload progress is explicitly fixed to 1, hard lift credit and carry are 1, and the initial-clearance latch is not reset by post-cross loading. Therefore ordinary capture loading does not necessarily surrender the earlier unload/lift/carry credit. Already-placed FR uses the separate current-retention formula; unplaced FL does not obtain placement credit merely by hovering. No phase-label or exact-joint reward was found.

The other terms can impose real tradeoffs, but do not prove a conflicting objective:

- Contact quality can penalize a fast touchdown, loaded slip, or a narrowly defined passive rebound. Load alone is not a cost: it scales an impact term. Downward touchdown speed ≤0.05 m/s has no impact penalty; merely acquiring a slow contact is not charged as generic contact loss. In this tail the **actual contact-quality reward is zero**. The aggregate `touchdown_events=1` per decision is not an FL touchdown—FL has no contact—and must not be misread as one.
- Body stability penalizes measured attitude/rate/acceleration. Applied-command smoothness penalizes actual first/second command differences, so an additional adjustment can incur cost. These are whole-robot terms; this review cannot assign their observed costs to the FL action alone.
- Residual-magnitude regularization is disabled. Consequently a constant nonzero servo/wheel residual is not directly penalized for its magnitude. There is no P05 absolute wheel-stop target reward; the controlled-stop branch is not the missing P05 task.

Across the 119 nonterminal decisions in the tail, mean weighted rewards are task progress `−0.004412823` (including PBRS `−0.003079489` and time cost), body `−0.000074033`, contact `0`, smoothness `−0.000625004`, regularization `0`. Phi ranges `0.409844686…0.410062348`. A nearly unchanged positive Phi generates negative `gamma*Phi_next−Phi_previous`; this is the documented discounted PBRS convention, not evidence that hovering earns a bonus or that the whole reward is zero.

At the actual deadline the absorbing next Phi is zero, PBRS is `−2.049500012`, failure event is `−40`, and total terminal reward is `−42.051533986`. Earlier displayed physical Phi is not bootstrapped through this terminal. Thus the episode is not being reported/rewarded as a successful capture. Whether finite training nevertheless favors insufficiently changing actions requires controlled learning evidence, not this one failed rollout.

## Finding and attribution limit

The measured action did not achieve FL capture. The event classification is consistent with verified AIR, zero obstacle load/contact, and the positive gap; no false placement, missing legal crossing credit, or sampled tail dispatch/clipping error was demonstrated. There is a specific finite-prior/semantic-transition coordination issue worth distinguishing from policy behavior, plus ordinary smoothness/stability/contact tradeoffs. Neither is established as the unique cause. The capture shaping itself is nonflat and does not contain the suspected unavoidable post-cross unloading penalty.

A later explicitly selected experiment could compare matched physical states around the finite endpoint with separately controlled FL/wheel adjustments, retaining the same contact/safety evaluator and measuring actual vertical/forward response and weighted reward changes. That is only a discriminating measurement option, not an implementation recommendation, success prerequisite, new entry gate, or authorization to modify the running batch.

Sources inspected at runtime revision: `configs/recording_motion_contract.json`, `configs/ppo_semantic_v3/stage_task_spec.yaml`, `configs/ppo_semantic_v3/reward_config.yaml`, `configs/ppo_semantic_v3/execution_profile.yaml`, `src/wlr50_clean/ppo/semantic_supervisor.py` (physical potential/capture, stage update, continuous nominal scheduling), `src/wlr50_clean/fsm/motion_executor.py`, and `src/wlr50_clean/ppo/semantic_reward.py`. All historical results and the original contact requirement remain unchanged.
