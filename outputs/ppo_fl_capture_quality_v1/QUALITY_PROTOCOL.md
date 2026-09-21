# FL capture / front quality v1 — declared before new physical trials

Protected reference: N_ref 20260917T0424208857504Z, 73.808333 s SUCCESS.
Recovery origin: CP178432, 178432 decisions / 1359 PPO updates / 27180 optimizer steps.
No partial rollout is resumed. Preserve actor, critic, Adam, Identity normalizers and RNG.

Primary front-quality metric: combined chassis roll/pitch Euler-rate RMS,
sqrt(mean((roll_rate^2 + pitch_rate^2)/2)), rad/s, using adjacent actual 120 Hz
observations, wrapped angle differences and the existing chassis transform.
Secondary guard: peak norm of chassis roll/pitch, rad. FR lift, crossing,
placement, duration and body clearance must accompany either quality claim.

Windows are fixed by task events, not the result: natural P01 tick 0 to first
real FR placement; FL first active-lift to first real FL placement, and first
FL crossing to placement. Missing placement makes that complete-task window
unavailable, never zero. An incomplete prefix may be reported only as such.
No long P05 idle tail is added to the FR metric. Window durations are reported;
slower is not automatically better and one pair establishes no robustness.

B_control and C use the same candidate configuration and nominal/controller;
only learned residual differs. Original N_ref has older RR semantics and is
preserved, not mislabeled as this B_control. Probe interventions are separate
from formal C and excluded from optimizer credit and task-success claims.

First collection target: 2048–4096 new actual learner decisions, majority
P03–P06, with genuine P05 transitions in complete optimized rollouts, new
P01/P02 nonzero-quality samples and rear maintenance. Curricula use real
prefixes; no state injection. Fixed configuration within each update/block.

Do not add CAPS, gSDE, colored noise or another filter solely because the
mapper has a periodic component. Preserve the current conditional Gaussian,
rho=.9, temperature=.25 initially and distinguish requested policy, actual
dispatch and physical response. No hidden FL nominal capture primitive.
