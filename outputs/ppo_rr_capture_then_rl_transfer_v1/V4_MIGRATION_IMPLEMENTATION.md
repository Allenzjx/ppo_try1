# Strict same410 v4 migration implementation

The new factor is `wlr50_clean.rr_carry_handoff_same410.v4`; its receipt is
`rr_carry_handoff_v4_migration`. It permits exactly the reviewed 12 runtime
paths and no deletion. Only the wheel module and this migration module may
be added. The four protected configurations remain byte-identical; the
execution profile may change only its revision and wheel mode, and the task
spec may add only `rr_contact_handoff_semantics` with the reviewed value.

The sealed source is CP221184 / 1693 PPO updates / 33860 optimizer steps,
SHA256 `e721e9b52d6f05e8024a397363ca65e1b151a5a19efd2c5c805b59d53e1b0d6b`,
with sidecar SHA256 `849bd97eed1ac1c23fba0fe86ec6624a9ea11427f33a2063268607663179fcf1`
and source HEAD `e24a3c2630b0b95439a7a71fe6a8a3f610a9a385`.
Target HEAD must be provided explicitly and must be the clean committed
runtime; no target hash is presumed here.

The factor preserves all actor/critic parameters, complete Adam state and
effective LR, Identity normalization, full RNG, existing branch origins,
all auxiliary ledgers and v2/v3 receipts. It creates no branch or learning
credit and requires fresh rollout storage. The 410 numeric positions and
codec remain unchanged, while the changed X408 contact-handoff permission
and X409 armed carry-wheel envelope are declared control/observation
semantics changes, not a same-MDP or same-trajectory claim.

CPU verification completed: 21 new tests passed, including the generic
validator and official source load / target save / independent fresh load
against temporary Git and synthetic checkpoints. Same-input Gaussian
mean/sigma and critic outputs remain exact. The combined new migration,
old v2/v3 migration and RR410 training-entry suite passed 83 tests in 15.80 s.
CUDA was hidden and OMP/MKL used one thread. All helpers exited.

No real checkpoint was written, no optimizer was run on real data, no
Isaac process was launched, and no commit or actual migration publication
was performed by this subtask. Normal-save receipt carry is implemented;
its next real-PPO propagation still requires actual run evidence.

APIs: `build_rr_carry_handoff_migration`,
`validate_rr_carry_handoff_migration`, and `preserved_keys` in the new
module. Actual publication should use the existing source-device official
load/save/fresh-reload protocol and a new immutable filename.
