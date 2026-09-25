# Narrow review: fixed sigma-reference coordinates for the local mean

Status: mathematical/read-only implementation review. **Not implemented.** No
Torch execution, checkpoint migration, reward, sigma or production edit.

## Boundary function and probability

For channel j, let the current local mean be `delta_j = W_j h + b_j` and a
strictly positive, finite, fixed versioned scale be `s_j`. Set
`W'_j = W_j / s_j`, `b'_j = b_j / s_j`, then execute
`delta_j = s_j (W'_j h + b'_j)`.

This preserves the mathematical local output on every existing observation.
The actual raw Gaussian mean remains
`mu = (1-rho) * (prior_raw_mean + applied_local_delta) + rho * history_center`,
with the existing `rho=.9` applied once. Keep the conditional standard
deviation, history-center units, raw sample, tanh, caps and physical mapping
unchanged. Therefore the same input/history has the same mean, std and raw
Gaussian probability mathematically. No extra latent-variable density or
Jacobian is required: this changes network parameter coordinates, not the
sampled action variable.

Do not use the current learned sigma as an implicit changing scale; that
would couple the mean to the variance head and require a different design.
The reference scale is a non-trainable serialized buffer/config constant,
not a hidden time-varying controller. Inactive-prefix handling must retain its
literal zero local contribution. Float32 division/multiplication can change
ULPs, so migration needs numerical same-input mean/std/logp checks; bitwise
physical trajectory identity is not guaranteed by the algebra.

## Adam mapping is not optimizer equivalence

Write new parameter `phi = theta / s`. For the unchanged loss function at the
same state, `g_phi = s * g_theta`. Thus transform the stored mean-row Adam
statistics as `m_phi = s*m_theta`, `v_phi = s²*v_theta` (and AMSGrad max-v by
`s²`), preserving step/bias correction. Apply this to the first12 rows of the
LOCAL final Linear's weight and bias only. Leave log-std rows, trunk, critic,
prior and their optimizer states unchanged. Use actual named-parameter and
optimizer mappings; do not guess Adam IDs. Existing weight decay is zero;
nonzero coupled/decoupled decay would need a separate derivation.

However, with unchanged scalar LR `alpha` and epsilon, the effective old
coordinate update becomes approximately

`delta_theta = -alpha * s * m_hat / (sqrt(v_hat) + epsilon/s)`.

It is **not** the old Adam update. Away from epsilon domination, physical
mean-row changes are approximately scaled by `s`. With absolute scales .3
for RR and .035–.06 elsewhere, RR mean-row motion also becomes about3.3 times
slower, while other channels become about17–29 times slower. This can reduce
their relative KL use but does not restore RR learning speed at the actual
current LR1e-5. Treat it as an intentional optimizer/parameterization change,
not “exactly the same continuation with renamed parameters.” A relative
normalization of scales would define another materially different step-size
choice; it should not be adopted silently.

For exact Adam dynamics equivalence in an ideal uncoupled coordinate, one
would also need per-coordinate LR `alpha/s` and epsilon `s*epsilon`. This is
not supplied by the proposed unchanged-LR scheme and would remove its intended
preconditioning effect. Ordinary Adam parameter groups do not supply separate
LRs for rows of one tensor without a further implementation change.

## What this cannot resolve alone

At the equivalent function, local-trunk backpropagation through the mean is
unchanged because `s*W' = W`. The shared trunk can still move many channels,
including the std head. Existing RSL separately clips the actor and critic
global parameter-gradient norms; rescaling only some actor rows changes that
clipping factor and therefore can change other actor updates too. Mapped
historical moments are a coordinate transform of the saved statistics, not
proof that all past updates would have produced identical moments under the
new clipping geometry.

The measured update3 KL is .03160926: RR contributes .00167626 and other10
.02993300. But scale/std changes alone contribute .01392563, about44.1% of
the total. A mean-row scaling does not directly regulate that component.
It also cannot repair RR eligibility after GROUND, nominal ownership, or
premature late-group release, and cannot make an unrewarded good action gain
positive advantage.

## Recommendation for this run

First complete the fixed-observation saved-model and GAE analysis. Apply the
already identified predicate/source correctness fixes at a cold boundary,
with the minimal448 migration and fresh compatible rollout. Those are
specific semantic/physical defects; this coordinate change is a plausible
optional preconditioner, not a demonstrated cure.

If the fixed-state analysis confirms RR's useful action directions receive
usable advantages but barely enter its mean while non-RR changes dominate
global KL, consider a separate version. Verify exact old-column/state
migration, unchanged initial raw probability, frozen prior, meaningful
post-update RR mean change, channel KL and real deterministic placement.
Do not simultaneously enlarge sigma, reset the mean, or claim that reduced
KL itself means improved RR capability. No AUX follows automatically from
this proposal.
