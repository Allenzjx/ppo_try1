# Block 03: RR capture → P10/P11/P12 handoff, read-only

Run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`, first natural-P01 learner episode, runtime `a802b24d78df`. Bounded evidence: `residual_and_projection_audit.jsonl`, global decisions **204864–205189**, endpoints **8704–11304** (72.533333–94.200 s). The 326 decisions include 2608 actual ticks (first interval starts at 8696). No production/config/reward patch, model update, extra physics or whole-history audit was performed for this diagnosis. The earlier reward-draft request was superseded; no reward patch was prepared.

## Finding

**The captured RR nominal is preserved until an explicit newer P10 source owner changes its knee.** There is no reset to an old entry, zeroed residual history, RR mask, lost actuator write or P11/P12 replay of the P09 entry vector in the inspected handoffs. P10 deliberately changes RR knee −37.8→−34.6→−29.3→−27.2°; P11 changes FR hip and P12 changes RL/wheels while retaining RR nominal hip/knee −6.9/−27.2°. This is an authored cooperative action, not an accidental channel overwrite. It is still a real change away from the capture-producing final target, and these data do **not** establish that its timing is physically ideal for the learned configuration.

RR does not permanently lose capture at one handoff: there is an immediate pre-P10 contact interruption, later TOP reacquisition, P11 AIR/TOP alternation, and another good TOP state at P12 entry. The eventual retreat from the platform happens much later during P12, while RR nominal is unchanged and the body, RL motion and policy wheel commands evolve together. No single-joint or wheel counterfactual has been tested.

Also correct the statement “RL never lifted”: the real evaluator records **RL qualified lift at tick 10090**. It subsequently loses clearance and does not cross/place; absence of lift at 94 s is not absence of an earlier lift event.

## Chronology and current support truth

| Tick / s | Event and physical state |
|---|---|
| **8726 / 72.716667** | Actual RR placement event. Nominal capture owner holds **−6.9/−37.8°**, retires only preexisting RR servo owners. This is not a final-command or actual-angle lock. |
| 8728 / 72.733333 | First sampled endpoint after placement is already AIR/0 N, gap −0.984 mm; still P09, no new P10 source yet. `placed_RR=true`, but `rr_placed_currently_usable=false`. |
| **8736 / 72.800** | P09→P10: RR TOP 4.262 N, front +149.866 mm. The just-completed action is still P09 with old capture nominal. |
| **8744 / 72.866667** | P10→P11: RR TOP 9.814 N. First P10 action has actually changed RR knee; P10's unfinished endpoint continues across the label change. |
| 8752 / 72.933333 | P10 endpoint −27.2° is present while P11 starts FR hip. RR TOP 14.300 N; temporary current-lift invalidity is not automatically loss of usable TOP. |
| **8816 / 73.466667** | First sampled post-handoff AIR/0 N and entry-invalid state (`placed_RR`), gap −0.994 mm. FL is also AIR/0 N. This is current support/geometry loss, not forgotten placement history. |
| 8832 / 73.600 | RR is AIR +1.601 mm with qualified current lift, legal XY; usable/entry becomes true again without inventing bearing force. Subsequent real TOP recontacts occur, e.g. tick 8992. |
| **10080 / 84.000** | P11→P12 with RR TOP **14.377 N**, front +144.821 mm. RL is unloading/AIR near ground; first P12 native action follows this handoff. |
| **10090 / 84.083333** | Actual RL qualified lift; sampled gap −33.281 mm at 10096. Initial lift is not above-top carry or crossing. |
| 10136–10400 / 84.466667–86.666667 | P12 reaches authored four-wheel reverse interval (local 56–320). RR AIR above top; RL has returned to ground. |
| 10408 / 86.733333 | Authored wheel stop is present at the sampled endpoint. Residual wheel commands remain nonzero. |
| 10456–10464 / 87.133333–87.200 | RR real TOP again (15.853→10.839 N); RL briefly rises again, but still behind edge. |
| **10632 / 88.600** | RR still TOP 5.406 N, front +7.224 mm; body has retreated substantially during RL source return. |
| **10640 / 88.666667** | RR remains corner TOP 2.009 N but front −6.997 mm, outside current top XY; usable/entry correctly becomes false. |
| **10672 / 88.933333** | RR on ground, front −52.730 mm / gap −49.998 mm. RL also ground. Historical RR crossing/placement remains recorded, not claimed as current support. |
| 11304 / 94.200 | P12, entry false due current `placed_RR` usability. RR ground+obstacle, front −51.058 mm; RL unplaced/uncrossed. No terminal occurs within this inspected window. |

RR placement event is independently retained in the current history. Its earlier crossing/lift ticks in this same episode are 5976/5496. The report uses authoritative event ticks, not the task label or an approximate progress update.

## First target difference, without attributing mapper history to PPO

All values are canonical hip/knee degrees. N is read from the **same dispatch tracking evidence's requested command**, not independently recomputed. Baseline is the same-tick geometry-corrected mapper plus finite source controller contribution. PPO requested/effective are equal for RR in these samples; final is the actual delivered command. “Feedback actual” is the measured state entering that final substep, reconstructed as recorded nominal minus recorded canonical tracking error; it is **not** mislabeled as a separate post-step joint readback. Raw dispatch tick equals episode endpoint +179.

| Endpoint | N RR hip/knee | Pre-PPO baseline | PPO requested = effective | Final | Feedback actual |
|---:|---:|---:|---:|---:|---:|
| 8728 | −6.900/−37.800 | −6.178685/−36.290527 | +19.036804/−14.684279 | +12.858119/−50.974806 | +10.268627/−50.717043 |
| 8736 | −6.900/−37.800 | −6.178685/−36.290527 | +18.859322/−13.527997 | +12.680638/−49.818524 | +11.189123/−50.598276 |
| 8744 | −6.900/−29.300 | −6.178685/−31.350000 | +16.630467/−13.792824 | +10.451782/−45.142824 | +10.827420/−49.677509 |
| 8752 | −6.900/−27.200 | −6.178685/−27.200000 | +13.377436/−13.371299 | +7.198751/−40.571299 | +9.271406/−47.116365 |
| 8816 | −6.900/−27.200 | −6.178685/−27.200000 | +12.255114/−14.190332 | +6.076430/−41.390332 | +5.395340/−41.235234 |
| 10080 | −6.900/−27.200 | −6.178685/−27.200000 | +6.623585/−14.928011 | +0.444900/−42.128011 | −0.323304/−43.630278 |
| 10088 | −6.900/−27.200 | −6.178685/−27.200000 | +6.201090/−14.229931 | +0.022406/−41.429931 | −0.506237/−43.661034 |
| 10640 | −6.900/−27.200 | −6.178685/−27.200000 | +15.781118/−14.337389 | +9.602433/−41.537389 | +10.571452/−40.808505 |

At 8744 raw mapper RR knee is −32.100°, with finite P10 **+0.750° source controller bias** making baseline −31.350°. At 8752 that finite bias is zero and mapper knee is −27.200°. This is a concrete source/controller contribution to the first change; attributing the entire final-angle difference to PPO would be wrong. RR's final knee moves +9.247° from 8736→8752 and actual joint response follows with ordinary transient tracking error. The logs do not show actuator omission.

At the five boundary-adjacent decisions (8736, 8744, 8752, 10080, 10088), stored previous raw policy action exactly equals the preceding selected sample; encoded previous filtered request matches its predecessor within **1.59e−6** float32 precision. All cap-transition gates are false, RR caps stay [24°,36°], and previous-ACK reference verification passes. Thus there is no observed history clear, cap reset, or stage-generated zero residual.

## Four-wheel final command versus measured angular velocity

Canonical rad/s, order **FL / FR / RL / RR**, from recorded wheel order. Signs remain canonical joint-axis mapped signs; they are not assumed identical to world-x translation. Loaded wheel rates may differ from targets because leg/body/contact motion is coupled.

| Endpoint | Same-tick nominal wheels | Final wheels | Measured wheels |
|---:|---|---|---|
| 8736 | [0,0,0,0] | [−.387221,+.038580,+.052814,−.132651] | [−.341342,−.052402,−.010082,−.118837] |
| 8744 | [0,0,0,0] | [−.302904,+.017345,+.132402,−.072678] | [−.307333,+.020581,−.045430,−.068325] |
| 10080 | [0,0,0,0] | [+.304818,+.501270,+.305174,+.047803] | [+.700455,+.498506,+.304832,−.293262] |
| 10144 | [−.3,−.3,−.3,−.3] | [−.269553,+.057676,−.083068,−.431900] | [−.270160,−.008042,+.003898,−.431857] |
| 10400 | [−.3,−.3,−.3,−.3] | [−.287414,−.037411,−.269770,−.461947] | [−.222279,−.153404,−.197318,−.461320] |
| 10408 | [0,0,0,0] | [+.132586,+.268152,+.100284,−.151598] | [+.176789,+.190245,+.180710,−.151131] |
| 10640 | [0,0,0,0] | [−.890469,−.329644,+.169209,−.070296] | [−.890601,−.493013,+.203650,−.143273] |
| 10672 | [0,0,0,0] | [−.410469,−.086419,+.075618,−.323075] | [−.410366,−.224138,−.001330,−.455532] |

The negative source interval is genuinely authored, not a sign-mapping accident. PPO partially cancels/reverses some channels (FR at 10144), while no wheel mask removes nominal. The sharp final retreat at 10632→10672 occurs **after** the authored source stop; those final wheel commands are then policy residuals over zero nominal. Body-forward progress falls from 0.303328 m at 10544 to 0.138933 m at 10672 (−164.394 mm), while RL nominal is completing its hip return. This documents whole-body retreat, not merely RR wheel-center motion or proof of one wheel's causality.

## Ownership, source clocks and usability scope

`_continuous_advisory` (`semantic_supervisor.py:2205`) retains existing held targets and grants newer ownership only to changed channels. Capture at 8726 retires preexisting RR servo owners; P10 is an explicit later knee owner. P11 does not own RR joints; P12 owns RL joints and its authored wheel groups. P10 endpoint continues after P11's task label starts. P12's wheel start/stop and later RL sequence appear in order without an expired-event burst or phase-entry restoration.

`_current_rr_placement_usable` (line 165) deliberately accepts either actual TOP or legal above-top qualified AIR after real placement. It does **not** label AIR as support: bearing remains zero. At 10640 it rejects the now-outside XY even though a finite-radius TOP corner still bears. This explains the entry reason `placed_RR`; the history has not been deleted.

Current `entry_valid` is a task/continuation diagnostic, not a universal source-clock freeze: `_sequence_permission` applies the special physical waits to P07–P09 (and pending P06), not all P12 source events. Hence the last P12 RL return events can still run after RR loses usability; the source then holds its endpoint while residuals remain available for recovery. That is current explicit scheduling behavior, not evidence of fake RR support. Whether more physical receiving-side coordination is needed is a task-design question to evaluate after the block, not grounds to hot-stop or rewrite this run.

All 326 recorded decision summaries verify all 2608 native task ticks. All 326 endpoint dispatches have matching setter/native mapping and all-one masks; RR is never capture-assist-owned. No in-episode state writes or terminal occurs in this window. RR has no headroom clipping; **FL knee index 1 is headroom-clipped at 31 endpoints**, including 10400/10408, so whole-body headroom limitations are not ignored. These counts describe existing recorded audit evidence, not an independent replay of every tick.

## Conclusion and limit

Real RR capture and subsequent real RL lift are new task evidence, not full success. The first post-capture target alteration is an explicit P10 cooperative knee/source-controller change; the eventual RR retreat is later and occurs with continuous policy history, functioning actuators, changing whole-body geometry and significant residual wheel action. This rules out the inspected reset/mask/old-entry/last-write explanations but does not isolate source timing versus learned coordination as a sole cause. No reward/sigma/teacher/fixed-angle change is proposed here, and this report is not a gate for the next saved/reloaded full-P01 evaluation.
