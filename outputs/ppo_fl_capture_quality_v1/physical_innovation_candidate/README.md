# Isolated FR-knee physical-innovation-scale candidate

**Prepared and CPU-tested only. Not adopted, not registered with production, not a valid production checkpoint or migration.** The currently running block12 remains on its original kernel. This directory changes no source, configuration, current checkpoint pointer, old checkpoint, DELIVERY or RECOVERY file.

## Single intended variable

`PhysicalInnovationCandidate` inherits the existing REQUEST-history quarter-temperature actor. It retains the same 372 observations, MLP weights, normalizer, rho=.9, learned mean and sigma parameters, all 12 action channels, action ranges/caps, and stateless history semantics. No new parameter or buffer is introduced.

| Observation phase | Effective sigma change |
| --- | --- |
| P01–P05 | All 12 channels unchanged |
| P06–P13 | Only FR knee, index3, multiplied by 24/112 = 3/14 |

The deterministic path delegates directly to the production parent. The stochastic path makes one MLP evaluation and one official Gaussian draw; it adds `log(24/112)` to the selected effective log-sigma **before** the official distribution cache is updated. Sampling, log-probability, entropy and KL therefore consume that same distribution. It does not scale a sampled action, fix a knee target, mask a channel, change HISTORY, add noise, or use gSDE. Phase comes from the validated current observation; the change persists through P06–P13, not only stage entry.

Motivation is the observed 24→112 degree FR-knee request-cap increase. At a fixed conditional mean, the first-order request-space innovation is approximately `cap * sech(mean)^2 * sigma`; 24/112 compensates that cap factor, not all nonlinear effects. It does **not** ensure a physical displacement or prove the previous failures were noise-caused. Gaussian raw support remains unbounded and the existing tanh/caps remain untouched; reaching a large request is less probable, not forbidden. No claim of restored task ability or improved stability is made.

Important counterevidence retained: CP182528 stochastic tick5167 FR request was about−22.307 degrees while conditional-mean request was already−20.323 degrees (innovation only−1.984 degrees); terminal6699 innovation was positive+3.684 degrees, not universally negative. This candidate does not correct an accumulated mean/HISTORY offset, body/load tracking, or FL hip sigma (block11 terminal raw sigma about.5341). These are separate observed effects.

## CPU evidence

Latest `cpu_tests_LR_reload.xml`: **20 passed, 0 failed, 0 skipped**, 3.580s test duration. The previous `cpu_tests_final.xml` also passed all20; the final replay additionally checks restoration of the adaptive optimizer's scalar learning-rate attribute as well as Adam groups. Tests use the full CP183552 actor/critic/Adam state read-only and synthetic valid observations, not a new physical rollout. All parameters/Adam are checked equal immediately after load; deterministic outputs, cache behavior and CPU RNG are checked exactly.

- P01–P05: exact mean, sigma, sampled action, logprob and RNG equality with the parent.
- P06–P13: exact conditional mean, other11 sigmas/samples and RNG equality; FR-knee sigma differs only by the specified factor. Log-space float32 addition/exp is compared with direct multiplication at relative tolerance1e−6; the first test run's two failures were overly-tight3e−7 tolerance (absolute difference2.794e−9), not a production or candidate change. `cpu_tests.xml` preserves that initial result; final test only corrected tolerance and a test's range-label index.
- Mixed phase batches, positive stage age, repeated observations, shuffled minibatches, padded observations, invalid encoding rejection; likelihood and entropy match the official Normal cache; KL uses the same cached parameters.
- Positive sigma permits finite log density for both large positive and negative raw actions; this is support evidence, not an observed physical motion.
- One official RSL PPO synthetic-update test: 16 samples, 5 epochs ×4 minibatches =20 CPU optimizer steps. Every stored sample's observation/action/logprob/distribution parameters stay aligned, each is used5 times, and the first minibatch ratio maximum error is **0.0**. Save/load preserves all actor/critic/Adam tensors; a restored CPU RNG reproduces sampled actions exactly. These are **0 real policy decisions, 0 real PPO updates, 0 physics ticks**. This test ran once per pytest invocation, including the initial tolerance-failing invocation; no CPU result is merged into training credit.

The files named `pytest_tmp_*/.../NOT_FOR_PRODUCTION.synthetic.pt` are generated test artifacts with explicit candidate-only/zero-real-credit metadata, not proposed training checkpoints. Original CP183552 and its metadata remain unchanged. The official production loader would not accept this unregistered kernel; that is intentional. Any future adoption requires the parent's explicit safe boundary, reviewed version/contract/migration and fresh rollout collection. This directory does not implement or bypass those production checks.

Independent read-only review found no blocking forward-path issue. It also confirmed an important integration boundary: current production request auditing has an exact actor-type allowlist and reconstructs sigma using only the original .25 temperature. Future adoption must explicitly update/version the shared sigma auditing path as well as actor/contract/loader migration; simply registering this actor is insufficient. These tests exercise official RSL `PPO.load/save`, not production `load_semantic_checkpoint`; test RNG replay uses explicit `torch.set_rng_state`, not an already-implemented new production resume path.

Re-run in a fresh CPU process, using a new basetemp/output receipt name to preserve evidence:

```powershell
$env:PYTHONPATH='C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -m pytest outputs/ppo_fl_capture_quality_v1/physical_innovation_candidate/test_candidate_cpu.py -q --tb=short -o junit_family=xunit1 --basetemp=outputs/ppo_fl_capture_quality_v1/physical_innovation_candidate/pytest_tmp_04 --junitxml=outputs/ppo_fl_capture_quality_v1/physical_innovation_candidate/cpu_tests_04.xml
```

No Isaac, CUDA, actor clearing, real-training branch, reward edit, cap edit or production migration was performed. All CPU helper processes exited after validation.
