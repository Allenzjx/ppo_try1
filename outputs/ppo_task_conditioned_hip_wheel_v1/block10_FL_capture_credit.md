# Two real FL captures: local credit and same-input means

Exactly12 original372 inputs were examined, six per episode, with real HISTORY.
Capture history ticks are3057 and3512. The containing updates are1472 and1481;
the last two episode0 follow-up samples cross into1473. No optimization, physics,
reward/production/aux change or extra training gate was performed.

## Actual rewards and advantages

`touch` means current top support; `placed` means the placement event was earned;
subsequent AIR does not erase the historical event. All12 samples have done=false.
Raw GAE is saved `returns-old_values`. A is the actual whole-rollout standardized
advantage, checked against all five appearances in official minibatch likelihood.

| Episode / end tick | Current status | Reward | Old V | Raw GAE | Actual A |
|---|---|---:|---:|---:|---:|
| 0 /3048 | AIR | -.009445 | -21.867456 | -.624023 | -.050375 |
| 0 /3056 | touch, not yet placed | +.093879 | -21.558043 | -.967440 | -1.642411 |
| 0 /3064 | placed3057; P05→P06 | +.179392 | -21.818619 | -.843155 | -1.066242 |
| 0 /3072 | AIR | -.114380 | -22.941755 | +.066946 | +3.152875 |
| 0 /3080 | AIR; update1473 | -.043008 | -23.084198 | +.617899 | -.184763 |
| 0 /3088 | AIR; update1473 | -.025908 | -23.193626 | +.744089 | +.240602 |
| 1 /3496 | AIR | -.002205 | -21.447182 | -.530319 | -2.513820 |
| 1 /3504 | AIR | +.002302 | -21.406397 | -.607992 | -2.730081 |
| 1 /3512 | placed3512; P05→P06 | +.203674 | -21.372635 | -.683971 | -2.941627 |
| 1 /3520 | touch | +.012575 | -22.647491 | +.357344 | -.042341 |
| 1 /3528 | AIR | -.133857 | -22.509167 | +.174686 | -.550905 |
| 1 /3536 | AIR | -.055252 | -22.558777 | +.328083 | -.123810 |

Reward decomposition is explicit: throughout these windows terminal event=0,
weighted stability/contact-motion/smoothness/regularization families=0, and
reward = potential shaping minus .0013333333 time cost. Capture0 Phi rises
.42945003→.46629458 (shaping+.18072550); capture1 .42824666→.46995315
(shaping+.20500782). First lost-support rows reduce Phi to.44435185 and.44761156,
with shaping-.11304628 and-.13252414. Thus contact/load benefit and retention
loss are present in reward; this is not proof of a missing contact reward.

Positive capture reward does not imply positive advantage. Using recorded next
old values, the capture-row TD residuals are-.909332 and-1.037211 respectively:
the old critic evaluates the following P06 state lower. Later rewards/values
and GAE also contribute. Episode0's first AIR row is the **nonterminal rollout
tail** (index127/update1472), so its +.066946 raw GAE includes normal tail-value
bootstrap; it is not an episode terminal. Episode1 lost-support raw GAE is also
positive, yet its normalized A is negative. These are real local credit/label
differences, not by themselves a bug or a reason to increase reward coefficients.

## Direct hip/knee evidence, not attribution from joint log probability

Both vector entries below are raw Gaussian units `(FL hip, FL knee)`.
Update Δμ uses identical saved inputs/H on CPU with the true before/after weights.

| Global / event | Actual action minus old μ | Postupdate conditional Δμ |
|---|---|---|
| 192894 / first touch | (-.042116,+.127114) | (+.000691,-.000419) |
| 192895 / capture0 | (+.056317,+.166024) | (+.000591,-.000316) |
| 192896 / AIR0 | (+.029607,+.012706) | (+.000511,-.000228) |
| 193990 / capture1 | (-.137945,+.076820) | (+.001373,+.001276) |
| 193992 / AIR1 | (+.032725,+.010210) | (+.001256,+.001166) |

Direct recorded loss derivatives also exist per channel. Capture0's hip head
derivative is positive in all five appearances (+.02818..+.02885), while the
**aggregate update** still moves this state's hip μ positive. Capture1's hip
derivative is negative in all five (-.02691..-.02141), and its aggregate μ also
moves positive. Knee derivatives and every appearance/clip flag are in the JSON.
Shared parameters, other samples, Adam and clipping prevent assigning the final
mean change to one sample's derivative or its joint probability ratio.

Before/after CP pairs:1472 =192768→192896;1473 =192896→193024;
1481 =193920→194048. CPU preupdate means agree with original saved GPU old means
to maximum2.98e-8 across these inputs. Full original hip/knee actions, old μ,
all12-observation hashes, complete reward breakdown, oldV/GAE/A, per-appearance
head derivatives and source checkpoint hashes are in `block10_FL_capture_credit.json`.

Interpretation boundary: both captures quickly lose current support, so this
window cannot label either whole sequence a successful retention example. The
observed negative capture A and phase/value transition justify examining credit
calibration with additional genuine preparation→capture→retention experience,
such as an authorized P04 continuation. They do not establish causality, prove
that a fixed lowering action should always be reinforced, or authorize an
automatic reward/critic/HISTORY change. No broad scan or new physical probe was run.
