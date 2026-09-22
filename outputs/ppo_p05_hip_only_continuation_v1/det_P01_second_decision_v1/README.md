# One-row historical P01 decision2 candidate

**PASS for bounded candidate feasibility only.** Source CP201728 actual deterministic decision2 uses input tick8 (0.066667 s); it runs8 ticks and hands off to P02 at tick16. Reset/decision1 input was not reconstructed. Its actual recorded action is used only as the previous-raw history required by decision2.

X389 was not directly stored by the original video. This single row is reconstructed from explicitly aligned task/physics, two-level HISTORY, next-dispatch mapper prestate/current ACK and previous physical-sample derivatives. No missing state was guessed. Original CPU replay errors: μ 4.65661287308e-10, base μ 2.98023223877e-08, history 0, σ 3.72529029846e-09; fixed tolerance1e-6 accommodates original CUDA versus CPU float arithmetic and is not bitwise-input proof.

Current5fd compatibility holds for this recorded state: assist WAIT is inactive, RR qualification/cross/placed false, receiver-retirement0. Old/current potential both 0.0689728523956, exactly equal to X17 after float32 encoding. Existing two-migration structural proof and unchanged relevant codec/kernel/config bytes are bound in the manifest. This is not current actor or future-trajectory equivalence.

Source-local evidence: real P01 task handoff at16; FR qualification tick23 (later in P02, not prematurely credited to P01); FR actual placed tick2072 with current legal TOP/bearing/support evidence at decision259. These are historical source outcomes, not current learner success or a full-task success claim. All8 action ticks have verified actual raw, all12 permission and no assist ownership.

`candidate.npz` contains exactly1×389 reconstructed input and1×12 unchanged actually executed deterministic raw action (the recorded conditional mean), plus source mean/sigma and IDs. A single state provides no representative P01 coverage and no independent train/validation split. No fitting, budget, AUX/PPO credit, teacher or checkpoint was created; old bound candidate/v1 helpers remain untouched. The CPU helper exits after this preparation.
