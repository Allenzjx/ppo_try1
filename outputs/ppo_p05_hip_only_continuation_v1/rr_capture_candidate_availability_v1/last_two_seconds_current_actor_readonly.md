# Last 2 seconds before RR capture + first TOP interval: 41-state read-only check

Exact input scope **8488–8808**, indices1061–1101/global204838–204878, entirely sealed rollout1566. The 2-second cutoff is8726−240=8486;8488 is the first existing8-tick input. No reset input was invented. Current actor is explicit CP214400 `8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9` on in-memory derived X389 with only Φ/index17 reencoded; raw targets remain the actual stored stochastic samples. No dataset, gradient, budget or fit is produced.

- Pre-contact30 inputs8488–8720: RR gap first→last **22.3694→0.360297mm**; min/max [0.36029665631236163, 22.533150354576676]mm. Qualified rows 4/30, legalXY 30/30; currentTOP0. 4 need changed X17. Per-row gap/support is retained in JSON, not assumed monotonic.
- Actual placement8726; first input after it8728 is AIR, not continued contact. Exact currentTOP input8736–8808 has10 rows and hands off throughP10/P11. After8808, previously audited8816 is AIR again; no longer contact run is claimed.
- Pre-contact verified support counts FL/FR/RL: {'FL': 4, 'FR': 30, 'RL': 1}; TOP10 counts {'FL': 10, 'FR': 9, 'RL': 5}. The JSON preserves classifications/forces, so body support is not inferred from limb position.

The following is a per-channel **fixed historical-state** overview. Raw/μ/σ are latent units. REQUEST error is current conditional mean's capped request minus the actual sampled action's capped request; first8 are degrees, wheels rad/s. Source μ is provenance only, never a replacement label.

| Channel | Actual raw mean | Stored source μ mean | Current μ mean | Current σ mean | REQUEST MAE | REQUEST signed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FL_hip | -0.366816 | -0.352023 | -0.345608 | 0.107757 | 2.68832 | 0.557339 |
| FL_knee | -0.744806 | -0.749255 | -0.786701 | 0.100088 | 1.88079 | -0.941938 |
| FR_hip | 0.273819 | 0.273365 | 0.28792 | 0.0106676 | 0.315083 | 0.312675 |
| FR_knee | -0.309866 | -0.305914 | -0.319195 | 0.0164576 | 1.93342 | -0.950854 |
| RL_hip | -0.601473 | -0.605113 | -0.591375 | 0.0710296 | 0.862008 | 0.153248 |
| RL_knee | 0.101833 | 0.0991016 | 0.097324 | 0.00841752 | 0.297325 | -0.160427 |
| RR_hip | 0.776826 | 0.760822 | 0.764866 | 0.128758 | 0.934899 | -0.119119 |
| RR_knee | -0.406301 | -0.405256 | -0.397884 | 0.0149834 | 0.468481 | 0.256971 |
| FL_wheel | -0.327273 | -0.335614 | -0.383497 | 0.195052 | 0.160229 | -0.0596503 |
| FR_wheel | 0.147308 | 0.144839 | 0.137069 | 0.0567742 | 0.0458148 | -0.0114879 |
| RL_wheel | 0.108419 | 0.10588 | 0.1074 | 0.0665542 | 0.045339 | -0.000903469 |
| RR_wheel | -0.168615 | -0.173718 | -0.170923 | 0.0446873 | 0.0200486 | -0.00143921 |

This short sequence contains real contact-and-continuation information and measurable current-versus-actual action differences. It is **not** blanket eligibility for every preceding AIR row, proof old actions would succeed from current states, or proof finite auxiliary learning will help. The immediate contact interruption and later interval boundary remain negative qualifications. Current natural evaluation takes priority; a real current RR capture could make this candidate unnecessary. CPU-only process exits after report creation.
