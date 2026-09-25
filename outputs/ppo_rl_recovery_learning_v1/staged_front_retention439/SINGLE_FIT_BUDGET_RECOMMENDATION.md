# front-retention439 single-fit budget recommendation

Status: **historical pre-fit recommendation, now executed once exactly as
specified**. The separate authorized fit completed32/32 and passed official
save/reload. It remains a conservative engineering budget, not a
physical-success guarantee. The AUX publication is separately accounted and
adds zero PPO credit.

## One fixed `Budget`

```python
Budget(
    max_attempts=32,
    learning_rate=1000.0,
    maximum_train_request_shift_full12=(
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.005, 0.005, 0.005, 0.005,
    ),
    maximum_validation_request_shift_full12=(
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.005, 0.005, 0.005, 0.005,
    ),
    maximum_per_state_full_gaussian_kl=0.10,
    maximum_abs_log_sigma_change=0.125,
)
```

Use this once, with no LR scan, retry at another LR, or trust-bound widening.
The existing first-rejected-proposal restore-and-stop rule remains decisive.
The 32 attempts were a maximum, not a requested number of accepted updates.

## Actual 439 inspection signal and single-fit outcome

Before execution, the bound CP229120 inspection measured train/holdout raw
half-MSE `4.773714e-5 / 5.055324e-5` and sparse-gradient L2
`1.089121997e-5`. At that scale, LR `1e-3` would imply a leaf-step L2 of only
about `1.09e-8`, with a real float-ULP/no-effective-change risk. LR `1000`
instead implied roughly `0.01089` leaf-weight L2 per unclipped first proposal;
the predeclared REQUEST/Gaussian guards, not an LR search, remained the actual
accept/reject authority. Several source-to-target action mismatches exceeded
the trust envelope, so the objective was deliberately partial retention rather
than full imitation of the older requests.

The one authorized run accepted and attempted32 steps. Its observed maxima were
joint REQUEST `0.035654°`, wheel REQUEST `0.000918925 rad/s`, per-state full
Gaussian KL `0.0122388`, and absolute log-sigma change `0.0561200`. The
published AUX checkpoint/sidecar SHAs are respectively
`3b55427ae443accbae34413498189036f907f36832e769350435a95b41cc0148` and
`7af9749c7e810f70a230b561727bbfb2efcba9b9f7abebd2386f167634890a52`.
PPO counters remain `229120/1755/35100`; this does not establish physical
success or future-trajectory equivalence.

## Numeric source and why the envelope is tighter

The closest actually executed finite retention run is
`outputs/ppo_rr_capture_then_rl_transfer_v1/front_retention_410_v1/CP222592_AUXFR410_01_execution.json`
and its bound fit report
`CP222592_AUXFR410_01_execution_fit.json`. It used the same independent
zero-momentum/zero-weight-decay SGD pattern with fixed LR `1000.0`, at most 32
attempts, cumulative full-Gaussian checks, and accepted 32/32. Its *reviewed*
limits were joint REQUEST `2.0`, wheel REQUEST `0.05`, bidirectional KL `0.5`,
and absolute log-sigma `0.25`.

Those older limits are not copied as a physical guarantee. At its accepted
step 32, the observed cumulative maxima were:

- train: joint REQUEST `0.122506`, wheel REQUEST `0.002728`, KL `0.079064`,
  absolute log-sigma `0.099902`;
- validation: joint REQUEST `0.121894`, wheel REQUEST `0.002762`, KL `0.077805`,
  absolute log-sigma `0.099857`.

The proposed 439 envelope therefore keeps the actually used LR/finite count
but rounds roughly twice those observed REQUEST changes upward (`0.25` joint,
`0.005` wheel), while tightening KL to `0.10` and log-sigma to `0.125`.
This is deliberately a small retention move. The old run had a different
network/input/data scope and made no physical-success claim, so these numbers
remain conservative engineering choices. The root zero-update inspection must
report the actual 439 gradient/JVP and original-to-target REQUEST mismatch; it
must not tune this proposal by searching multiple budgets.

## Data boundary for this one fit

- Optimize only the sealed probe-v2 zero-based decision rows `0..997` for
  P02/P05/P06/P09, using the deterministic balanced 32-train plus 32-contiguous-
  holdout rows per phase already specified by the data contract.
- Rows `998+` include the intervention and subsequent failed trajectory and are
  excluded. No RR failure tail, delayed post-release gap minimum, teacher
  action, or inferred success target may enter the objective.
- Targets remain the exact executed pre-intervention student conditional raw
  requests. The intent is to retain already demonstrated predecessor behavior,
  not teach RR placement or RL completion.
- Real compatible P10–P12 inputs must verify same-input full-Gaussian
  invariance. If no real P13 row exists, record `P13 actual_rows=0` and use only
  the separately reviewed zero-selected-phase-column algebra/synthetic unit
  proof; do not fabricate a physical P13 observation or block the entire
  candidate solely because none exists.

The authorized execution turned only its accepted temporary SGD steps into one
explicit AUX event: accepted `32`, attempted `32`, PPO added `0`. It did not
deploy a teacher or attach a physical-success label; ordinary PPO must start
from a fresh rollout and carry the ledger separately.
