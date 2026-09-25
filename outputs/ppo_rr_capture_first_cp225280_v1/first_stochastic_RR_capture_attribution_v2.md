# First stochastic RR capture — episode 1

189 on-policy samples; 998 uncredited frozen-prefix decisions. Zero optimizer updates before this success; local mean stayed zero. This is headless stochastic exploration, not a video, learned gain, deterministic result or full-task success.

RR pairs are hip/knee; angles deg, raw Gaussian samples unitless. REQUEST is the real filtered/headroom input, not a re-created nominal subtraction.

This v2 supersedes the earlier Markdown's incorrect all189-controller-bias-zero sentence; the earlier JSON already reported that boolean as false. The nonzero inherited P10 correction is now explicit.

|Tick / phase|Raw sample|N source / mapped|Controller bias|REQUEST|FINAL|Actual|gap mm / force N / hold s|
|---|---|---|---|---|---|---|---|
|9424 P09→P09|0.745/0.212|-6.900/-37.800 / -8.150/-39.050|0.000/0.000|15.166/7.526|7.016/-31.524|7.002/-34.479|11.663 / 0.000 / 0.000|
|9432 P09→P09|0.757/0.736|-6.900/-37.800 / -8.150/-39.050|0.000/0.000|15.337/11.526|7.187/-27.524|6.939/-31.765|4.937 / 0.000 / 0.000|
|9440 P09→P10|0.600/0.407|-6.900/-37.800 / -7.963/-37.800|0.000/0.000|12.887/13.898|4.924/-23.902|6.081/-29.098|-0.049 / 12.072 / 0.033|
|9448 P10→P11|0.140/0.919|-6.900/-29.300 / -7.963/-32.100|0.000/0.750|9.387/17.398|1.424/-14.452|4.429/-24.417|-0.004 / 15.311 / 0.100|
|9464 P11→P11|-0.720/1.042|-6.900/-27.200 / -7.963/-27.200|0.000/0.000|1.887/24.898|-6.076/-2.302|-0.877/-11.447|-0.409 / 13.600 / 0.233|
|9496 P11→P11|-0.519/0.292|-6.900/-27.200 / -7.963/-27.200|0.000/0.000|-11.452/16.898|-19.415/-10.302|-8.475/-5.782|-0.611 / 13.261 / 0.500|

- First TOP is reached while source RR target remains P09 [-6.9,-37.8] degrees; positive knee residual unfolds RR before P10.
- Before first TOP, hip actual stays positive near+6 degrees; this sample does not prove negative hip was necessary for touchdown.
- After real placed, P10/P11 source knee unfolds toward-27.2 degrees while sampled residual and HISTORY also contribute; not all post-touch motion is policy-only.
- At endpoint9448 only, RR controller bias is[0,+0.75deg], the preserved P10 FSM post-mapper normal correction; other188 sampled endpoints are zero. No RR completion helper is inferred from this generic bias.
- P09-to-P10 and P10-to-P11 remain nonterminal and bootstrappable; only verified local hold is terminal.
- Success occurred before any optimizer update with local mean still zero: genuine stochastic exploration success, not learned or deterministic success.
- Final load_fraction_valid is false; bearing force/current bearing are separately reported, not a reliable normalized load fraction.

Native-observer TOP counters consistently imply first contact tick9436; placed event9437; first endpoint9440; terminal9496 with61 TOP samples,0.5s hold and13.2609N verified bearing. All recorded dispatch ticks9433–9496 are contiguous/verified. No separate per-tick contact file was inspected.

Across all189 samples: original raw equals issued raw; no diagnostic field or RR capture-assist owner/correction; all12 actuator audit unchanged and no episode state writes. Generic source controller bias is NOT entirely zero: at9448, RR knee+0.75deg equals0.5×10.6×0.14150943396226415 from preserved FSM P10 normal correction. Other188 sampled endpoints have zero RR controller bias. Phase transitions9440/9448 are nonterminal;9496 is genuine local terminal with no bootstrap.

Clock distinction is retained in JSON: endpoint9496, source observation9495, native dispatch9675. No camera footage was produced by this analysis.
