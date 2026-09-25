# Episode 2: first RR support loss (read-only)

P11 is not success. RR TOP/placed appeared at endpoint9696; next endpoint9704 is AIR and hold reset. Local gate stays active, history placed remains true, no local terminal.

|Tick / phase|N RR h/k|REQUEST RR h/k|FINAL RR h/k|Actual RR h/k|gap mm|TOP / hold s|
|---|---|---|---|---|---|---|
|9688 P09→P09|-6.900/-37.800|19.356/-7.137|11.206/-46.187|9.050/-50.740|8.061|False / 0.000|
|9696 P09→P10|-6.900/-37.800|19.543/-3.137|11.393/-42.187|9.924/-47.488|-0.249|True / 0.008|
|9704 P10→P11|-6.900/-29.300|20.580/0.363|12.430/-32.187|10.797/-42.001|0.126|False / 0.000|
|9712 P11→P11|-6.900/-27.200|21.704/3.706|13.554/-23.494|11.894/-34.279|7.385|False / 0.000|
|9736 P11→P11|-6.900/-27.200|17.971/-7.317|9.821/-34.517|10.087/-31.101|27.120|False / 0.000|
|9872 P11→P11|-6.900/-27.200|17.743/-30.087|9.593/-57.287|11.228/-51.484|56.601|False / 0.000|

Initial loss is not RR knee re-folding: actual knee moves+5.487deg and FINAL+10deg; mapped N+existing P10 bias contribute+7.70deg and filtered policy+3.50deg before final slew. Full-body late source begins9695. Later P11 AIR knee re-folding coincides with larger gap but follows initial loss; this is not isolated causal proof.

- Source change coincides with loss; without a same-entry counterfactual, it is not isolated causal proof.
- P11 source roles continue while local gate remains active; placed is historical, TOP/hold are current and false/zero after loss.
- Only logged endpoints and native-observer counters inspected; exact first loss tick unavailable.
- Existing P10+0.75deg normal correction is not an RR capture completer; no control change made.

First update independently reconciles512 new samples (P09=397/P10=2/P11=113),20 minibatches/2560 old-likelihood exposures, each saved sample seen5 times, no mismatch. Checkpoint file SHA matches sidecar;512/1/20,1995 uncredited prefix,2 opportunities,1 pre-update local success,AUX0. Frozen-source byte hashes match; no tensor deserialization or new ability claim.
