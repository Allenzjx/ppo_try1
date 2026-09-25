# CP230144 FR-knee direction diagnostic through tick2560

**Every cell is formal baseline / intervened probe.** Angles degrees, position mm, force N. Decision endpoints only; no claim of physical success or AUX/PPO data.

| Tick / time s | FRk N; mapped N | FRk filtered REQUEST | FRk FINAL; actual | FRh FINAL; actual−FINAL | FL gap; front | CoM z | FR; RL force |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2096 / 17.4667 | +45.900 / +45.900; +45.318 / +45.318 | -4.370 / -4.370 | +40.948 / +40.948; +41.795 / +41.795 | +4.552 / +4.552; +8.163 / +8.163 | -6.402 / -6.402; -207.572 / -207.572 | +144.881 / +144.881 | +12.045 / +12.045; +13.567 / +13.567 |
| 2216 / 18.4667 | +45.900 / +45.900; +45.318 / +45.318 | -5.463 / -8.319 | +39.855 / +36.999; +40.561 / +37.888 | +5.509 / +5.492; +9.976 / +13.058 | +16.229 / +9.680; -170.428 / -169.007 | +147.126 / +144.375 | +12.374 / +12.356; +13.894 / +13.920 |
| 2336 / 19.4667 | +45.900 / +45.900; +45.318 / +45.318 | -5.763 / -8.370 | +39.555 / +36.948; +40.277 / +37.670 | +5.755 / +5.727; +10.943 / +14.156 | +13.155 / +6.181; -157.940 / -156.751 | +145.782 / +142.890 | +12.204 / +12.266; +14.468 / +14.358 |
| 2456 / 20.4667 | +45.900 / +45.900; +45.318 / +45.318 | -5.845 / -7.438 | +39.472 / +37.880; +40.118 / +38.374 | +5.824 / +5.800; +11.374 / +13.633 | +12.675 / +7.696; -145.136 / -144.422 | +145.105 / +142.983 | +12.462 / +12.582; +14.493 / +14.471 |
| 2560 / 21.3333 | +45.900 / +45.900; +45.318 / +45.318 | -5.867 / -6.259 | +39.451 / +39.058; +40.199 / +39.759 | +5.844 / +5.836; +12.245 / +12.979 | +8.725 / +6.922; -134.290 / -134.311 | +143.784 / +143.006 | +12.102 / +12.105; +13.878 / +13.835 |

The intended change reached the actuator: entry REQUEST/FINAL was -4.369525/40.947977 degrees; at hold tick2336 it is -8.369525/36.947977, exactly -4 degrees relative to that entry. The formal student itself changed after entry, so the same-tick FINAL difference is only -2.606811 degrees, not -4. The measured knee difference is -2.606830 degrees. Source N and mapped N are identical in all five pairs.

At tick2336 the intervention makes FL clearance 6.974 mm lower, CoM 2.892 mm lower, and FR hip tracking error 3.213 degrees larger than the simultaneous formal baseline. A roughly1.189 mm forward advantage does not compensate for the lost clearance or establish capture. By tick2560, after release, FINAL knee difference has decayed to -0.392555 degrees, FL gap remains1.803 mm lower, and front distance is effectively equal (probe0.021 mm farther behind). This bounded window does not support this particular FR-knee-minus4 candidate as a clearance repair; it does not prove every FR adjustment unhelpful.

There are exactly45 modified decisions (start ticks2096..2448), only raw channel3 changed; the other11 equal the original student computed on the actual intervened HISTORY, not the no-intervention baseline. At all59 decision endpoints2096..2560 FR has verified TOP bearing (minimum12.023 N), and RL has verified bearing (minimum13.567 N).

**Correction:** the earlier claim that the120Hz files were empty was wrong. Windows directory `Length` was stale while the writer was active. Using `open → seek(END) → tell` fixed readable snapshots returned552,408,868 bytes of diagnostic physics,76,046,423 bytes of native audit and110,837,646 bytes of physical observations. Reading only complete lines through tick2560 confirms2560 contiguous observer/native rows (ticks1–2560), plus2561 physical observations (ticks0–2560). No run or production repair was needed.

In the465 physical observations at ticks2096–2560 inclusive, **FR has current legal TOP bearing and RL has current bearing at every120Hz tick**. Minimum forces are11.814 N and12.488 N, respectively; every observer row has the ordinary single-write ACK and verified native audit. The probe observer records WAIT at2096, ACTIVE for all360 ticks2097–2456, and RELEASED for104 ticks2457–2560. Eligibility remained true throughout; the release occurred at the normal three-second decision boundary, not because support was lost.

Actual FR joints in the five-row table were reconstructed from current joint-margin evidence, not targets. Intervention raw does not equal the no-intervention baseline; no physical success/PPO/AUX claim. Episode continues unchanged; the readout stopped at tick2560 without waiting for or modifying the writer.
