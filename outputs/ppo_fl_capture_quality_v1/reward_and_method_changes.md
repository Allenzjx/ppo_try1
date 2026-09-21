# FL capture + front body quality — implementation and initial sizing

## Real collection status, 2026-09-18

The initial-sizing sections below were written before collection; they are not
the current training status. Five sealed real Isaac blocks have now added
2,048 learner decisions, 16 PPO updates and 320 optimizer steps, ending at
CP180480 / update1375. Actual coverage is P01=6, P02=488, P03=9, P04=18,
P05=988, P06=226, P07=6, P08=13, P09=294, P10–P13=0. Real reset prefixes
and output-only CPU optimizer tests are excluded. P03–P06 contributes
1,241/2,048 samples. See first2048_training_summary.md/json and block01–05 receipts.

The first optimized 128 samples contained 1,024 physics samples with positive
beta (.015–.03/s), integrated body cost .013710158 and task contribution
.403582552. Thus front quality is actually trained, not just configured.
Several stochastic training episodes physically placed FL and entered P06;
they subsequently collided or fell. These are not fixed-checkpoint evaluation
or full-task success. Both saved-checkpoint videos are now sealed and linked
in DELIVERY.md: deterministic C remained in P05; same-CP stochastic C truly
placed FL then failed at P09 on low body height. Same-version B is running;
its paired quality comparison remains necessary for benefit claims.

An execution audit also found a separate P05→P06 history/scale interaction:
FR knee cap 24→112deg magnifies the same latent residual fraction. The first
P06 raw sample decreased, yet its requested physical residual increased.
The transition hold and 60deg/s slew operate as configured; this is not a
mask failure or an instantaneous actuator jump. A physical-REQUEST history
re-expression exists only as an isolated CPU candidate and is not active in
this collection or its completed C/current B evaluations. No CAPS/noise/filter claim is
made from this observation.

Implemented in an isolated `ppo_fl_capture_quality_v1` profile. This report
includes code/CPU tests, initial offline sizing and the real collection status
above; it does not prove an improved robot. The old task-first epsilon-zero profiles and successful N_ref remain
unchanged. No mapper, nominal capture primitive, action capacity, HISTORY,
exploration temperature, safety threshold or hard task predicate is changed.

## Capture signal: geometry is not placement

Stage marker: `post_cross_FL_multiscale_positive_gap_plus_real_contact_v1`.
Only the old FL geometric half of the existing capture contribution is replaced.
It requires earned lift/crossing history and **current** legal top XY/lateral
region. With `g=max(0, wheel_bottom_z-platform_top_z)` in metres:

`proximity = .5/(1+g/.025) + .5/(1+g/.003)`.

Outside that current region its contribution is zero; existing workspace/carry
terms handle horizontal approach. The remaining half still requires real top
contact samples. AIR at zero gap remains unplaced; two real top samples remain
required. Negative gap is a plateau, never extra penetration credit. Once placed,
the existing current-retention branch bypasses this approach term. No joint
angle, direction, force magnitude or imitation target is added. The potential
remains global and history-aware; normal phase changes do not reset it or end
the episode. Existing terminal-zero potential and rollout-tail bootstrap stay.

At the measured failed gap 3.470128 mm, geometric proximity is .670891 (old
.878113); the new scale deliberately leaves more useful slope near contact.
The local **undiscounted** `5*ΔPhi` for 3.470128 mm → 0 is .0349678 versus
.0129505 (2.70012×); for one millimetre closer it is 1.82584× the old increment.
These local numbers are not a whole-episode reward claim: properly discounted
potential differences telescope and do not manufacture a new success event.

The existing 372-vector contains continuous, metre-valued gap in both goal and
wheel-geometry groups with scale 1, clipping ±20 and no quantization. A 3.47 mm
value is .00347, not clipped or numerically zero; float32 spacing there is about
2.33e-10 m. This does not prove network sensitivity, but there is no observed
precision loss justifying a new observation dimension in this first block.

## Small positive front-body cost

Objective/revision: `fl_capture_front_body_quality_v1`.
For physical samples in P01/P02 only:

`beta = .03 * (1 - .5*physical_transfer_fraction)` per simulated second;
`cost = beta * (.5*tilt_cost + .5*rate_cost) * dt`.

Each raw cost is the mean of the two clipped squared normalized roll/pitch
components, with scales .5 rad and .5 rad/s. Thus beta is always .015–.03 during
these phases, including necessary transfer, lift loss, preparation and carry.
No capture/stop/quality-success gate can turn it off. Zero measured tilt/rate
can naturally yield zero cost; that is not a zero coefficient. Other phases
remain task-only. Acceleration, contact, smoothness and residual-size family
weights stay zero. The old carry exemption does not control the new objective.

`front_quality_sample_audit` records actual phase, physical-history substate,
transfer fraction, beta, raw tilt/rate, pose/rates and integrated cost for each
eligible physics sample (normally eight per policy decision). Reward summaries
record the separate weighted tilt/rate sums; their sum must equal the negative
body family contribution. Historical CAPTURED does not assert current bearing.
New training must aggregate those fields by actual phase/substate and correlate
with genuine task events, rewards and advantages; config values alone are not
evidence that quality samples reached the optimizer.

### Initial cost magnitude on sealed old physical trajectories

The [budget JSON](front_quality_initial_budget.json) and
[reproducible helper](audit_front_quality_budget.py) use actual 120 Hz poses,
wrapped Euler derivatives and the exact phase-transition timestamps, stopping
at first real FR placement. Old transfer fraction was logged only at decision
endpoints, so prospective **exact-geometry bounds**, not a fake exact reward,
are reported. These trajectories used quality weight zero and no new policy
was run by this calculation.

| Trajectory | FR event duration | P01/P02 raw quality integral | Proposed cost bounds | Old positive potential sum |
|---|---:|---:|---:|---:|
| N_ref | 12.5167 s | 1.200361 s | .018005–.036011 | 1.898297 |
| CP178432 | 11.8750 s | 1.198997 s | .017985–.035970 | .767450 |

For CP178432 the upper bound is 4.69% of accumulated positive potential and
10.72% of net task reward in P01/P02 requested decisions (different accounting
denominators, not a return ranking). The endpoint transfer proxy is .017986.
Tilt accounts for about 89–90% of this raw integral, rate about 10–11%; this is
reported rather than hidden by equal coefficients. The fixed initial .03 block
is small but nonzero; change it only at a later saved update boundary based on
actual contributions and FR task preservation. No lexicographic task guarantee
or successful-policy ranking follows from the bounded failure-cost check.

Predeclared primary metric is the combined Euler-rate RMS
`sqrt(mean((roll_rate²+pitch_rate²)/2))`, with peak roll/pitch norm as guard,
over P01→first FR placement; task events and duration accompany it. Parent
`QUALITY_PROTOCOL.md` and `event_quality.py` are the single comparison protocol.
Old CP178432 RMS .11899695 rad/s / peak .31695654 rad; N_ref .11788940 / .30946430.
They are descriptive, not a new-policy benefit or same-version full-task pair.
The missing FL capture window is null, not a zero-quality score; P05 stall is
not added to the FR window to dilute RMS.

## Author methods actually used / not used

[IndustReal author paper](https://arxiv.org/abs/2305.17110) and
[official algorithm tools](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/industreal/industreal_algo_utils.py):
`get_sdf_reward` calculates geometric proximity, while engagement/insertion
helpers separately establish task completion. The reusable principle is a
dense, task-space geometry signal separated from a real success predicate.
Here an existing planar wheel-bottom gap is sufficient; the .025/.003 mixture
is our engineering adaptation, **not the paper's formula**. No SDF dependency,
SAPU sample/reward filtering, penetration rule, policy integrator or simulator
asset change is imported. Near-contact real-prefix sampling is handled by the
existing training curriculum, not by fake contact or historical-state injection.

[CAPS author site and public code](https://ai.bu.edu/caps/) and
[original paper](https://arxiv.org/abs/2012.06644): the public
`CAPS_code.zip` was read in memory, not installed. The PPO implementation stores
real `obs,next_obs` pairs and compares current-policy means at these inputs for
its temporal loss; a separate nearby-input mean difference is the spatial loss.
This clarifies what a future task-compatible regularizer would require, but
**no CAPS loss is added in this first block**: periodic mapper compensation is
not evidence of policy-generated high-frequency noise. No shuffled-row temporal
pairing, old-action imitation, Boolean/contact perturbation or added filter is
introduced. No external source code was copied. Existing Gaussian/HISTORY
rho=.9 and temperature=.25 remain unchanged; gSDE/colored noise/Grad-CAPS are
deferred alternatives, not claims of implementation.

## Verification and remaining physical work

- 34 new focused tests pass: positive beta through preparation/carry/capture,
  actual semantic-task envelope labels, other-phase zero quality, exact audited
  cost accounting, invalid config rejection, FL mm-gap slope, AIR not placed,
  real-contact capture, negative-gap plateau, outside-region suppression,
  post-capture legacy retention and non-FL formula preservation.
- 268 existing reward/capture/retention/carry/transfer/observation tests pass.
- An additional old workspace integration test expects twice the current
  transfer-era workspace share. Its first failure was reproduced with the
  original HEAD supervisor loaded only in memory; no production fix or test
  weakening was made for this pre-existing fixture mismatch.

CPU tests establish the new arithmetic and unchanged hard-event path, not
physical controllability, optimizer participation or task success. Root owns
the separate signed FL controllability probes, saved-state migration, genuine
P05/front on-policy updates, same-version B/C runs and normal-speed videos.
