# P10 block: real RL suffix events; P13 not yet controlled

Read-only report, 2026-09-10. Production HEAD `64c03243ac05`; no code, parameters, tests, or simulator changes. Run `train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa` under `runs/ppo_fsm_reference_p09_stable_v2`. Evidence: completed audit bytes 0–12,410,152 (128 rows), prefix credit-start line 951, optimizer update, and snapshot `20260910T085154904868Z.json`. No active P04 data used.

## Credit and actual events

The teacher physically reached P10 at tick 7584 / 63.2 s with RR already Q/C/P (teacher ticks 6912/7109/7579); RR was currently TOP, supporting, with valid load. RL was grounded and had no Q/C/P. Teacher RL initial events at ticks 7/1722 are not new policy credit. Entry was moving, not a stationary pose: measured body linear velocity [0.13063, −0.07137, 0.13480] m/s.

Actual policy samples: P10=1, P11=1, P12=71, P13=55. The same suffix produced these fresh RL events:

| Event | Physics tick / time (s) | Audit row | Measured evidence |
|---|---:|---:|---|
| P10→P11 | 7592 / 63.266667 | 1 | RL proximity + role preparation; FR is AIR, not fictitiously bearing |
| P11→P12 | 7600 / 63.333333 | 2 | RL proximity + transfer; RL valid load falls .06808→.03683, CoM motion toward FR remains positive |
| RL I | 7746 / 64.550000 | 21 | Lift gain 3.634 mm; whole-body actual joint response 21.565° |
| RL Q | 7750 / 64.583333 | 21 | Lift gain 8.643 mm, RL own response 6.762°; at row end AIR, above-top gap −39.093 mm |
| RL C | 8089 / 67.408333 | 64 | By 8096 front gap +6.092 mm, above-top gap +26.297 mm, AIR; not yet placed |
| RL P | 8164 / 68.033333 | 73 | By 8168 TOP, six consecutive TOP samples, valid load .07268, support true; P12→P13 |

During preparation, FL/RL/RR were measured supports and receiving FR was AIR. At Q/C, FL/RR supported. At Q, RL nominal hip/knee [.5,35.3]°, residual [−19.942,16.574]°, final target [−20.692,53.124]°, actual [−18.727,49.892]°; other leg angles also changed substantially. This is real whole-body and RL response, not commanded-motion proof alone. Q, later C, and later P are distinct observations.

## Continuous control, not a new fixed entrance

All 1,024 executed physics ticks have verified native target effect; no forbidden in-episode state writes. Ordinary handoffs retain residuals to numerical precision (max quoted servo step <4e−15°); P10→P11/P11→P12 nominal owner requests change 5.3°/3.2°. P12→P13 introduces a real +0.3 rad/s wheel-owner request while retaining residual/filter handoff. This is not a residual or wheel-zero reset, nor a promise of interpolated logical targets. At inspected event/boundary rows all 12 channels are enabled and target acknowledgements valid. No explicit software dispatch defect was identified by this bounded audit.

## First unfinished task: P13 controlled finish

The block ends normally at tick 8608 / 71.733333 s, not at a task terminal (`terminal=false`, reason null; completed-episodes file empty). P13 received only 440 ticks / 3.666667 s. All 55 P13 decision ends have valid final region, current support, verified physical evidence, and all historical P; none has `final_controlled` or a started post-completion observation.

Current production predicate (`semantic_supervisor.py:842–899`) needs body speed ≤.05 m/s, angular speed ≤.30 rad/s, and every actual wheel speed ≤.25 rad/s, together with valid region/history and at least two current TOP supports. Counts failing these measured rate limits at P13 decision ends are 46/55, 28/55, and 55/55 respectively. These counts are sampled decision ends, not every physics tick.

At 8608, body speed **.115041 > .05**; angular speed **.284159 ≤ .30**. Actual wheel speeds FL/FR/RL/RR are **[.892218, −.037253, .383814, .504495] rad/s**: FL/RL/RR still exceed .25. Nominal wheels remain [.3,.3,.3,.3] in all 55 P13 rows; final residual [.527667,−.296748,.083550,.238865] makes applied targets [.827667,.003252,.383550,.538865]. Thus FR nearly cancels the rolling suggestion while three wheels retain positive requests and response. Neither nominal alone nor tracking alone explains all motion.

The source P13 remains in its finite rolling segment: frozen source wheel launch spans 0–16.733333 s, finite phase 17.8 s, source scale 1.0 (`fsm_states.yaml:3102–3503`, recording contract). `NominalMotionProvider` uses these source requests and supplies home/zero after its endpoint (`semantic_supervisor.py:1478–1487,1790`). At only 3.667 s, this block has not reached that nominal stop tail; region validity does not itself retire P13 rolling. This is an observed advisory/finish-goal tension, not evidence of broken downlink or a basis to claim a future stop will succeed.

Whole body and all leg geometries are within the platform target, outside distance 0; body-forward coordinate moved net +65.018 mm after RL placement (not monotonically). Final FL/FR/RR are TOP supports. RL is AIR, +25.369 mm above top with valid zero load, after historical P. That current AIR is not itself a failing finish condition: the region and three other TOP supports pass. Home error 35.319° and command-wheel tolerance .02 are strict-recovery diagnostics/soft progress, **not additional all-stage hard finish gates**. Existing 1 s P13 post-completion observation has not started; it is unrelated to prohibited RR fixed hovering.

## Accounting and limits

New credit is exactly 128 policy decisions / 1 PPO update / 20 optimizer steps: cumulative **141,824 / 1,073 / 21,460**. Update log verifies changed actor hash and finite nonzero gradients. Checkpoint: `outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141824.pt`. All events above occurred before this block's optimizer update; they do not demonstrate improvement from update 1073. Teacher RR success is not policy RR success. RL suffix placement is not natural-P01 full-task success. No collision/fall occurred in this bounded suffix, but P13 remains incomplete at collection end; subsequent full natural-P01 evaluation is still required.
