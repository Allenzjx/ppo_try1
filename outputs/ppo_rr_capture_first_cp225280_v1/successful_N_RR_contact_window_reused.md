# Successful N: reused narrow RR contact window

This note only preserves the already verified sub-agent result; no source log was rescanned and no new simulation was run. N used the earlier `ee5a` control version. The present local experiment uses accepted `49eb23163a6e20bc56301dbafb59b137ecebce66` configuration with frozen `ecf205e` runtime, `rr_live_swing_evidence_v3` and cooperative-preparation v4. These are **not identical control versions or a same-state causal pair**.

| Native episode tick | Verified event / contact evidence | Canonical wheel FINAL [FL,FR,RL,RR], rad/s | Measured canonical wheel velocity, rad/s |
|---|---|---|---|
|Pre-cross forward window (exact tick not retained here)|Four-wheel source forward target|[+0.3,+0.3,+0.3,+0.3]|N/A in this reused excerpt|
|6112|First physical RR TOP contact|N/A at this exact tick|N/A|
|6155|RR legal front crossing and placed event|N/A at this exact tick|N/A|
|6160|RR remains TOP; consecutive TOP samples49|[-1.07,0,0,0]|[-1.137,-0.148,+0.228,+0.104] (three-decimal report precision)|
|6224|RR remains TOP; consecutive TOP samples113|N/A at this exact tick|N/A|

Physical TOP contact began43 native ticks (approximately0.358s) before the legal crossing/placement event. Thus first contact, legal crossing and placement are different events; legal-crossing history cannot be used to date first contact. Continuous TOP counts at6160 and6224 are consistent with a6112 start.

At6160, zero FINAL targets for FR/RL/RR did **not** mean zero measured angular velocity. The FL negative target was a source command in this N window, not evidence by itself of a PPO error. Native-axis velocity arrays were not retained in the excerpt and remain N/A; they were not reconstructed from canonical signs or replaced with targets. No load fraction or CoM conclusion is inferred from wheel rotation.

Only preserved exact event counts and the supplied rounded wheel readings are tabulated. Other joint/force/gap numbers from the original analysis are not restated from incomplete memory. This note does not certify that the same late action timing is suitable for the present policy's changed whole-body state.
