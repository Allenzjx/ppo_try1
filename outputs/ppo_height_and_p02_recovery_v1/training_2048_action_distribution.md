# Completed 2048 training: conditional action distribution

One completed audit streaming pass; 2048 decisions, source168192 → saved170240. No model load, simulation or production change.

| Phase | Decisions | Conditional sigma min–max (all channels) | Pooled z mean / population std | abs(raw)≥2 / ≥3 |
|---|---:|---:|---:|---:|
| P01 | 12 | 0.0671–0.3627 | -0.0715 / 0.9849 | 0.00% / 0.00% |
| P02 | 516 | 0.0267–1.6124 | 0.0168 / 0.9873 | 1.41% / 0.19% |
| P03 | 20 | 0.0294–0.8682 | 0.0528 / 0.9668 | 2.50% / 0.00% |

## Per-channel observed statistics

| Phase/channel | sigma min / mean / max | conditional mean average | raw min / mean / max | abs(raw)≥2 / ≥3 | z mean / std / min / max |
|---|---|---:|---|---|---|
| P01/front_left_hip | 0.2145 / 0.2686 / 0.3567 | 0.1678 | -0.1019 / 0.2603 / 0.7272 | 0.00% / 0.00% | 0.3566 / 0.9413 / -1.4479 / 1.5370 |
| P01/front_left_knee | 0.1896 / 0.2186 / 0.2507 | 0.0375 | -0.2264 / -0.0530 / 0.2062 | 0.00% / 0.00% | -0.4296 / 0.6846 / -1.2452 / 0.9538 |
| P01/front_right_hip | 0.0725 / 0.1076 / 0.1264 | 0.0381 | -0.1312 / 0.0441 / 0.3859 | 0.00% / 0.00% | 0.0143 / 0.9206 / -1.3673 / 1.3137 |
| P01/front_right_knee | 0.2358 / 0.2760 / 0.3627 | -0.1111 | -0.6130 / -0.1152 / 0.5331 | 0.00% / 0.00% | 0.0075 / 0.9267 / -1.1872 / 2.0327 |
| P01/rear_left_hip | 0.1775 / 0.1862 / 0.1960 | -0.3030 | -0.8427 / -0.3013 / 0.2949 | 0.00% / 0.00% | -0.0111 / 1.3138 / -2.9777 / 1.8841 |
| P01/rear_left_knee | 0.0671 / 0.0888 / 0.1030 | 0.0171 | -0.1458 / 0.0374 / 0.2002 | 0.00% / 0.00% | 0.2095 / 1.0068 / -1.4725 / 1.3960 |
| P01/rear_right_hip | 0.2604 / 0.2858 / 0.3422 | 0.0635 | -0.2368 / 0.1628 / 0.5485 | 0.00% / 0.00% | 0.3458 / 0.5828 / -0.7548 / 1.1562 |
| P01/rear_right_knee | 0.1119 / 0.1297 / 0.1421 | -0.1604 | -0.6559 / -0.2717 / 0.0441 | 0.00% / 0.00% | -0.8552 / 0.8732 / -2.3077 / 0.6178 |
| P01/front_left_ankle | 0.1933 / 0.2692 / 0.3325 | -0.1722 | -0.7822 / -0.1424 / 0.6769 | 0.00% / 0.00% | 0.1143 / 1.0924 / -1.7415 / 2.2620 |
| P01/front_right_ankle | 0.0711 / 0.1201 / 0.1562 | -0.1096 | -0.3227 / -0.1304 / 0.0376 | 0.00% / 0.00% | -0.1506 / 0.5786 / -1.1338 / 0.6354 |
| P01/rear_left_ankle | 0.1704 / 0.1988 / 0.2267 | -0.1675 | -0.6221 / -0.2680 / 0.1892 | 0.00% / 0.00% | -0.5109 / 1.0046 / -2.1299 / 1.3578 |
| P01/rear_right_ankle | 0.0817 / 0.1108 / 0.1210 | 0.2525 | 0.0010 / 0.2551 / 0.5154 | 0.00% / 0.00% | 0.0511 / 0.8688 / -1.4421 / 1.2598 |
| P02/front_left_hip | 0.0825 / 0.2101 / 1.4012 | 0.5160 | -2.7354 / 0.5363 / 2.2154 | 0.78% / 0.00% | 0.0568 / 0.9859 / -3.1631 / 3.0548 |
| P02/front_left_knee | 0.1116 / 0.1938 / 0.5837 | 0.9493 | -0.5724 / 0.9396 / 2.0355 | 0.19% / 0.00% | -0.0483 / 0.9843 / -2.7883 / 2.7607 |
| P02/front_right_hip | 0.0367 / 0.1396 / 1.6124 | 0.6307 | -3.0302 / 0.6358 / 2.3643 | 1.74% / 0.19% | 0.0943 / 1.0135 / -2.5659 / 2.9218 |
| P02/front_right_knee | 0.1028 / 0.2238 / 0.8020 | -0.1006 | -1.5982 / -0.0960 / 4.0518 | 0.58% / 0.19% | -0.0142 / 0.9875 / -2.6256 / 3.7838 |
| P02/rear_left_hip | 0.1652 / 0.2311 / 0.3472 | -1.3355 | -2.4940 / -1.3222 / 0.4015 | 7.17% / 0.00% | 0.0529 / 0.9724 / -2.5387 / 3.0405 |
| P02/rear_left_knee | 0.0501 / 0.1110 / 0.1877 | -0.0185 | -0.8347 / -0.0114 / 0.6666 | 0.00% / 0.00% | 0.0635 / 1.0002 / -2.7161 / 2.8649 |
| P02/rear_right_hip | 0.1554 / 0.2426 / 0.5014 | -0.3572 | -2.6449 / -0.3634 / 0.9355 | 0.97% / 0.00% | -0.0181 / 0.9650 / -2.7136 / 2.6966 |
| P02/rear_right_knee | 0.0833 / 0.1360 / 0.4786 | -0.2985 | -1.2820 / -0.3025 / 0.5409 | 0.00% / 0.00% | -0.0507 / 0.9816 / -3.4751 / 2.4175 |
| P02/front_left_ankle | 0.0598 / 0.2172 / 1.5892 | -0.5700 | -4.2773 / -0.5574 / 2.5168 | 5.23% / 1.94% | 0.0642 / 0.9625 / -3.0253 / 2.3955 |
| P02/front_right_ankle | 0.0267 / 0.1003 / 1.3265 | -0.4354 | -2.2882 / -0.4317 / 0.8105 | 0.19% / 0.00% | -0.0007 / 1.0400 / -3.3145 / 2.9082 |
| P02/rear_left_ankle | 0.1105 / 0.1744 / 0.5347 | -0.4389 | -1.8872 / -0.4416 / 0.6515 | 0.00% / 0.00% | -0.0146 / 0.9712 / -3.1742 / 3.1230 |
| P02/rear_right_ankle | 0.0377 / 0.0659 / 0.2970 | 1.1074 | 0.1080 / 1.1083 / 1.7820 | 0.00% / 0.00% | 0.0162 / 0.9679 / -2.6085 / 3.0455 |
| P03/front_left_hip | 0.0758 / 0.2174 / 0.6159 | 0.5111 | -0.1496 / 0.5817 / 1.3229 | 0.00% / 0.00% | 0.2667 / 0.8494 / -1.1716 / 1.8967 |
| P03/front_left_knee | 0.1098 / 0.1983 / 0.4281 | 1.1564 | 0.3644 / 1.1730 / 1.7527 | 0.00% / 0.00% | 0.1056 / 0.9888 / -1.7186 / 1.8815 |
| P03/front_right_hip | 0.0341 / 0.1266 / 0.4692 | 0.3393 | -1.2572 / 0.3239 / 0.9659 | 0.00% / 0.00% | -0.0518 / 0.7495 / -1.9396 / 1.7206 |
| P03/front_right_knee | 0.1017 / 0.2571 / 0.6375 | 0.5015 | -0.4782 / 0.4761 / 2.9586 | 15.00% / 0.00% | 0.0567 / 1.1685 / -2.3710 / 2.6456 |
| P03/rear_left_hip | 0.1902 / 0.2394 / 0.3115 | -1.4356 | -2.0013 / -1.4665 / -0.7354 | 5.00% / 0.00% | -0.1484 / 0.8856 / -1.7095 / 1.9138 |
| P03/rear_left_knee | 0.0562 / 0.1046 / 0.1608 | 0.0681 | -0.6276 / 0.1058 / 0.8020 | 0.00% / 0.00% | 0.3262 / 0.7226 / -0.8566 / 1.6577 |
| P03/rear_right_hip | 0.1550 / 0.2480 / 0.4023 | -0.5442 | -2.3545 / -0.5679 / 0.7791 | 10.00% / 0.00% | -0.0766 / 1.0905 / -2.4691 / 2.2722 |
| P03/rear_right_knee | 0.0838 / 0.1304 / 0.2128 | -0.2492 | -0.6483 / -0.2457 / 0.0781 | 0.00% / 0.00% | -0.0716 / 0.9331 / -2.1408 / 1.2861 |
| P03/front_left_ankle | 0.0533 / 0.2377 / 0.8682 | -0.5377 | -1.0774 / -0.4250 / -0.0707 | 0.00% / 0.00% | 0.4635 / 0.8628 / -1.0744 / 2.1504 |
| P03/front_right_ankle | 0.0294 / 0.0931 / 0.3334 | -0.3625 | -0.6909 / -0.3856 / 0.7164 | 0.00% / 0.00% | -0.2692 / 0.9794 / -2.2294 / 1.7895 |
| P03/rear_left_ankle | 0.1093 / 0.1825 / 0.3498 | -0.0731 | -0.8796 / -0.0119 / 0.4838 | 0.00% / 0.00% | 0.3215 / 1.0561 / -1.6569 / 1.7709 |
| P03/rear_right_ankle | 0.0420 / 0.0656 / 0.1129 | 1.2637 | 0.9321 / 1.2462 / 1.8380 | 0.00% / 0.00% | -0.2887 / 0.8588 / -1.8846 / 1.2219 |

## Interpretation and limits

- P02 contains localized tails, not blanket saturation: 87/6192 channel-samples (1.405%) have abs(raw)≥2 and12/6192 (0.194%) have abs(raw)≥3. FL wheel contributes10 of those12 extreme samples (10/516=1.938% of that channel); its maximum abs(raw)=4.2773 at global170215 has conditional mean−1.9914, sigma1.1106 and z−2.0583. FR knee reaches raw4.0518 at global169084, mean1.6006/sigma0.6478/z3.7838. These are real local exploration/mean-tail events, not evidence that sigma is always1 or all channels are noise-dominated.
- P02 pooled z has mean0.01677, std0.98732, abs(z)≥2 frequency4.231% and abs(z)≥3 frequency0.258%. Channel innovation-RMS/conditional-mean-RMS ranges0.0614–0.5473 in P02; P01 has several ratios above1 but only12 decisions. That describes coverage relative to the HISTORY-conditioned mean, not underlying base-network command strength or beneficial physical exploration.
- Large raw values can also follow a large conditional mean: P03 FR-knee raw2.9586 at global169085 has mean3.6196/sigma0.6375/z−1.0369, immediately after the P02 FR-knee extreme above. This does not justify resetting continuous HISTORY or assigning a policy motive; ordinary phase transitions remain nonterminal.
- Sigma is the recorded conditional innovation sigma, not marginal raw variability or a stationary variance. The actor leaves learned log-sigma unchanged when forming the HISTORY-conditioned mean (semantic_history_actor.py:29–63).
- The tanh thresholds describe latent compression, not native saturation. Same-decision raw/native identity and all12 masks were verified; the JSON includes actual endpoint projected residual and native-target-effect statistics.
- C170240 ended at tick752/HARD_JOINT_LIMIT; all 94 issued P01/P02 conditional-mean actions have |raw|<2 (largest 1.4764). Its deterministic failure therefore does not require an extreme sampled innovation or tanh-tail saturation explanation.
- The existing C extract has raw ranges but no means; no C source rescan or invented mean was used. Its trajectory/checkpoint differs from individual training states and cannot isolate a learned motive or sigma effect.
- First three phase sample counts are12/516/20; much of the block was P05 (1258). Prefer more natural-P01/early-task credit before a distribution change, while preserving Adam/normalizer/HISTORY and verifying real endpoints. Do not alter rho/sigma from this one failure.
- Full per-channel quantiles, innovation ratios, exact tail counts and extreme-event decision bindings are in training_2048_action_distribution.json. Finite-run values are descriptive, not a formal normality or safety test.

## Minimal next-course assessment (recommendation only)

The proposed4×256 independent natural-P01 collection blocks are consistent with these data: retain all production/distribution/reward/HISTORY settings and chain each actual saved checkpoint/Adam/normalizer/RNG; each block contains2 complete128-decision PPO updates. Bootstrap its nonterminal collection tail using the existing value path, then start the next block from a legal natural reset. Do not synthesize done at P02→P03, FR capture or the collection boundary, and do not treat256 requested decisions as256 P02 samples. This increases opportunities for fresh early-task entries; it does not guarantee deterministic recovery or an exact phase allocation. Record actual resets and request-phase counts. Retain the subsequent256-decision P06 maintenance with real P01 prefix and teacher exclusion because early-focused blocks need not reach rear phases. Re-evaluate the final actual checkpoint naturally fromP01. Additional launcher/reset overhead is the tradeoff; no new physics or source change is required.

There is not sufficient causal evidence here for a single sigma/rho version change. Local high-sigma and tanh-tail events deserve retention in diagnostics, but the formal deterministic failure occurs with abs(raw)<2 and no sampling noise, while early entry/settle coverage is sparse. Sampling-only is the narrower next candidate; these recommendations have not been implemented or run.

Source audit: C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_fsm_reference_p09_stable_v2\train\20260915T0246086426747Z_ga9c52261ba87_56a6fc5cd8bd42bd9d91b7a75f5ce321\residual_and_projection_audit.jsonl
C source receipt: C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_height_and_p02_recovery_v1\p02_policy_recovery_checkpoint170240.json
