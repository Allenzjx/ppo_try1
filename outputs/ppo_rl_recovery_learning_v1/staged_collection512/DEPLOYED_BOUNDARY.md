# Collection512 implementation boundary

Frozen production revision: `59e868f3e223c589e7645a0f5d63f91fa6119fb6`.
The staged five-file patch is the initial proposal, not the final runtime.
Real checkpoint reload revealed one additional common policy resolver that
reconstructed128. Final implementation changes exactly six runtime paths,
including `semantic_policy_distribution.py`; explicit512 receipts now select
the same runner configuration in that resolver. No control, reward, physical
asset, observation or action-distribution change was made.

The first zero-credit `g912358692dfa` publication was saved but failed the
independent ordinary loader. Its checkpoint/sidecar remain in history as
**FAILED_PUBLICATION_NOT_DEPLOYED**; no training, pointer promotion, or new
PPO/AUX credit used it. Do not treat it as a valid evaluation model.

The revised publication starts again from the unchanged latest complete
CP229632 /1759 PPO /35180 Adam (65a), retaining all learned parameters,
Adam/LR, Identity normalizers, RNG and the independent AUX32 ledger. Its
actual publication/reload result is recorded separately in
`../collection512_CP229632_g59e868f3e223_publication.json` when completed.
Existence of this note alone does not claim that verification completed.

Boundary tests: staged-only19 stdlib tests;76 existing return/GAE/reward
tests;2 final CPU tests for common policy resolver and one actual official
synthetic512 collector/update. Synthetic update has zero robot/PPO credit.
Old front-retention regression additionally ran30 pass/18 skip because its
selected72e-source tests correctly reject the now65a latest pointer.

New collection:512 consecutive learner actions,5 epochs/4 minibatches,
128 samples per minibatch,20 Adam steps per PPO update. Historical128
updates are not relabeled. Ordinary phases do not end episodes; real done
cuts GAE; nonterminal tail bootstraps. Gamma=.9985/lambda=.99 are unchanged.
Rear task/geometry/forced-forward assists stay OFF, original FL assist ON.
Longer collection is a training experiment, not a claimed task solution.
