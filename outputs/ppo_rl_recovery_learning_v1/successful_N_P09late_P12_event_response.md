# Successful N: P09 late → P12 exact-tick event/command/response

This is a bounded, read-only mechanism reference from the sealed successful N+0 episode. Every row is an exact 120 Hz episode tick; no interpolation or new physics was used. Angles are absolute canonical degrees, wheels are canonical rad/s, geometry is mm, and forces are measured normal force. The source-command column is the authored nominal; the JSON separately retains native drive target and physical commanded target at that tick. N is a mechanism reference, not a required pose.

## Compact event table

|event|tick / t|source command (selected)|measured response|contact / geometry context|
|---|---:|---|---|---|
|PRE_LATE_RR_LOADED_CONTEXT|6152 / 51.267s|FL k=-13.4, FR h/k=0.0/31.1, RL h/k=28.2/0.0, RR h/k=-6.9/-37.8, w=0.30,0.30,0.30,0.30|FL k=-12.0, FR h/k=0.5/30.3, RL k=0.2, RR k=-39.4; actual w=0.27,0.20,0.36,0.24; vxy=0.017/-0.003|FR OBSTACLE 7.37N; FL OBSTACLE 5.93N; RR OBSTACLE 6.48N front/gap=-0.27/-0.30; RL GROUND 9.59N front/gap=-159.10/-50.42|
|P09_LATE_ATOMIC_FL_RL_LAUNCH|6156 / 51.300s|FL k=-31.4, FR h/k=0.0/31.1, RL h/k=15.4/19.4, RR h/k=-6.9/-37.8, w=-1.07,0.00,0.00,0.00|FL k=-12.1, FR h/k=0.5/30.2, RL k=0.3, RR k=-39.4; actual w=-1.10,-0.11,0.08,-0.18; vxy=0.016/0.010|FR OBSTACLE 7.12N; FL OBSTACLE 8.19N; RR OBSTACLE 8.01N front/gap=0.14/-0.27; RL GROUND 9.99N front/gap=-158.73/-50.50|
|P09_LATE_EARLY_RESPONSE|6160 / 51.333s|FL k=-31.4, FR h/k=0.0/31.1, RL h/k=15.4/19.4, RR h/k=-6.9/-37.8, w=-1.07,0.00,0.00,0.00|FL k=-13.0, FR h/k=0.3/30.1, RL k=1.5, RR k=-39.4; actual w=-1.14,-0.15,0.23,0.10; vxy=0.049/-0.004|FR OBSTACLE 3.19N; FL OBSTACLE 10.14N; RR OBSTACLE 9.69N front/gap=0.75/-0.27; RL GROUND 7.86N front/gap=-158.83/-50.47|
|P10_RR_KNEE_POSITIVE_ENDPOINT|6168 / 51.400s|FL k=-31.4, FR h/k=0.0/31.1, RL h/k=15.4/19.4, RR h/k=-6.9/-29.3, w=-1.07,0.00,0.00,0.00|FL k=-17.3, FR h/k=-0.1/30.0, RL k=7.1, RR k=-37.2; actual w=-0.52,0.00,0.03,-0.07; vxy=0.176/-0.007|FR AIR 0.00N; FL OBSTACLE 11.02N; RR OBSTACLE 14.35N front/gap=3.07/-0.10; RL GROUND 2.99N front/gap=-157.13/-50.67|
|P11_RESPONSE_FR_AIR_RR_LOADED|6176 / 51.467s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=15.4/19.4, RR h/k=-6.9/-27.2, w=-1.07,0.00,0.00,0.00|FL k=-23.6, FR h/k=1.7/29.9, RL k=13.6, RR k=-33.0; actual w=-1.21,-0.00,0.10,0.05; vxy=0.230/0.008|FR AIR 0.00N; FL OBSTACLE 11.85N; RR OBSTACLE 11.14N front/gap=7.98/-0.02; RL GROUND 2.33N front/gap=-150.66/-50.95|
|P12_RL_KNEE_POSITIVE_AND_FR_HIP_RESPONSE|6201 / 51.675s|FL k=-31.4, FR h/k=9.0/31.1, RL h/k=15.4/35.3, RR h/k=-6.9/-27.2, w=-1.07,0.00,0.00,0.00|FL k=-33.1, FR h/k=8.9/29.9, RL k=29.4, RR k=-27.6; actual w=-1.05,0.00,-0.00,-0.02; vxy=0.157/0.059|FR AIR 0.00N; FL OBSTACLE 14.45N; RR OBSTACLE 14.00N front/gap=28.12/-0.65; RL GROUND 0.78N front/gap=-139.75/-50.00|
|P12_FOUR_WHEEL_REVERSE_BEGIN|6233 / 51.942s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=0.5/35.3, RR h/k=-6.9/-27.2, w=-0.30,-0.30,-0.30,-0.30|FL k=-31.7, FR h/k=4.7/29.9, RL k=36.4, RR k=-27.2; actual w=-0.40,-0.30,-0.30,-0.26; vxy=0.180/-0.120|FR AIR 0.00N; FL OBSTACLE 15.69N; RR OBSTACLE 14.72N front/gap=50.21/-1.07; RL AIR 0.00N front/gap=-94.71/-50.02|
|REVERSE_AND_LOAD_RESPONSE|6256 / 52.133s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=0.5/35.3, RR h/k=-6.9/-27.2, w=-0.30,-0.30,-0.30,-0.30|FL k=-31.5, FR h/k=3.7/29.9, RL k=36.0, RR k=-27.4; actual w=-0.36,-0.30,-0.30,-0.09; vxy=0.034/-0.035|FR AIR 0.00N; FL OBSTACLE 14.03N; RR OBSTACLE 14.40N front/gap=58.36/-0.89; RL AIR 0.00N front/gap=-54.40/-33.43|
|FIRST_REVERSE_STOP|6497 / 54.142s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=0.5/35.3, RR h/k=-6.9/-27.2, w=0.00,0.00,0.00,0.00|FL k=-31.1, FR h/k=3.6/30.0, RL k=36.6, RR k=-28.4; actual w=0.08,-0.05,0.00,-0.10; vxy=-0.010/-0.004|FR AIR 0.01N; FL OBSTACLE 14.54N; RR OBSTACLE 14.15N front/gap=27.28/-0.21; RL AIR 0.00N front/gap=-90.38/-0.01|
|SECOND_STOP_BEFORE_RL_CROSS|6652 / 55.433s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=31.2/-18.7, RR h/k=-6.9/-27.2, w=0.00,0.00,0.00,0.00|FL k=-31.2, FR h/k=3.6/30.0, RL k=-15.9, RR k=-28.4; actual w=0.12,0.11,0.00,-0.09; vxy=0.001/-0.005|FR OBSTACLE 1.44N; FL OBSTACLE 14.84N; RR OBSTACLE 12.37N front/gap=42.15/-0.10; RL AIR 0.00N front/gap=-3.89/65.23|
|RL_FRONT_EDGE_CROSSED_EVENT|6658 / 55.483s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=27.0/-18.7, RR h/k=-6.9/-27.2, w=0.00,0.00,0.00,0.00|FL k=-31.2, FR h/k=3.6/30.0, RL k=-17.4, RR k=-28.3; actual w=0.11,0.01,0.00,-0.08; vxy=-0.005/-0.004|FR OBSTACLE 1.57N; FL OBSTACLE 13.91N; RR OBSTACLE 12.19N front/gap=42.12/-0.10; RL AIR 0.00N front/gap=0.75/63.83|
|RL_CAPTURE_RESPONSE|6728 / 56.067s|FL k=-31.4, FR h/k=3.7/31.1, RL h/k=-6.9/-18.7, RR h/k=-6.9/-27.2, w=0.00,0.00,0.00,0.00|FL k=-30.9, FR h/k=3.8/30.1, RL k=-19.6, RR k=-28.2; actual w=0.06,-0.15,-0.04,-0.12; vxy=-0.046/-0.013|FR OBSTACLE 8.30N; FL OBSTACLE 3.86N; RR OBSTACLE 5.40N front/gap=41.15/-0.12; RL OBSTACLE 11.56N front/gap=69.72/-0.23|

## What the successful source actually establishes

- **FL action is not a single phase label.** At tick 6156 the P09-late atomic source changes FL hip/knee and the FL wheel together while also changing RL hip/knee. The FL knee and wheel remain active through the P10 and P11 RR/FR changes and into P12; the four-wheel reverse begins only at tick 6233. This is overlapping command evidence, not proof that one fixed FL pulse caused the body motion.
- **The strong late group has prerequisites in this run.** Immediately before launch, RR is already crossed/placed and carrying TOP load. P10 then unfolds RR knee; P11 opens the FR hip while FR knee remains near the positive reference configuration. FR is allowed to go AIR during preparation; RR and FL retain real TOP support. The source therefore does not support treating an AIR RR as load-bearing.
- **The selected response has measurable overlap, not just matching phase names.** From tick 6152 to 6201 the actual FL knee travels from about -12.0° to -33.1° while FL obstacle load grows from 5.93 N to 14.45 N; FR hip moves from about 0.5° to 8.9° while FR knee stays near +30° and FR unloads to AIR; RR remains obstacle-loaded near 14 N. At tick 6233, when the four-wheel reverse begins and RL is AIR, measured body vxy is +0.180/-0.120 m/s (negative y is the FR side in this run). This is the clearest selected concurrent front/FR-side motion signature, but it does not identify FL alone as causal.
- **RL preparation overlaps body/wheel action.** P12 first drives RL knee positive while the prior FL/RR/FR configuration is still evolving. Four-wheel reverse starts at tick 6233 with FL and RR loaded; it is not an isolated RL-joint-only maneuver. The first stop at 6497, forward resume at 6522, and second stop at 6652 are explicit source commands. The second stop precedes RL edge crossing at 6658; capture follows with all four TOP at 6728.
- **Response, not command sign, is the transferable fact.** Positive body-x and negative body-y (the FR side in this source) occur in parts of the window together with changing loads and RL unloading, but the logs do not identify one channel as independently causal. A new policy may reach the same physical result with different absolute angles.

## CP225280 entrance contrast (descriptive, not causal)

|event|tick / t|FL knee|FR knee|RR front/gap|RR TOP/load|late source|
|---|---:|---:|---:|---:|---|---|
|P09_ENTRY|6752 / 56.267s|-34.09|-5.14|-219.60/-50.24 mm|False/1.41N|p09=0.000s, wait=False|
|RR_QUALIFIED|6995 / 58.292s|-45.02|-21.73|-177.01/-41.30 mm|False/0.00N|p09=0.100s, wait=False|
|RR_CROSSED_AIR|7979 / 66.492s|-45.31|-19.70|0.01/59.03 mm|False/0.00N|p09=5.400s, wait=True|
|TERMINAL|10940 / 91.167s|-45.18|-18.93|99.89/56.12 mm|False/0.00N|p09=5.400s, wait=False|

CP225280 never reaches the successful source's loaded P09-late entry: RR crosses while still AIR and remains 0 N; its P09 source time stalls at 5.4s and the loaded late group is not observed. FL knee is much more negative, and FR knee is negative rather than the successful reference's positive receiving geometry. That is a coupled entry/configuration difference—not evidence that copying +30° FR knee or any exact N angle is sufficient.

## Evidence boundary

- N manifest: `03549f2590759fe22c09ec83ec12282f17f06de722a15ece547873140a083a20`; native 120 Hz audit: `63dc84aa942e1fcf8855e8282a5bbea2144c1181a95b00b7edc02a69e09bdfed`; physical observations: `64dfbede89104dc223022a25eefcea8c09dc5f1e34d461b97a1fb840661d464d`.
- Motion contract: `f3c930aaf44ff3599c6df87478e3a33a594ea0fb706183e6924e3f025f537a0b`; CP225280 audit: `e321cf020a44bc3a9ad6887d1452d1a9d2067c18a3d19d3dc7c8d53e5de4f076`.
- Contact class/force and geometry are copied from sealed logs. Missing values would be N/A; nothing is interpolated. User mechanism priorities remain hypotheses until a same-runtime physical diagnostic or policy evaluation produces the corresponding response.
