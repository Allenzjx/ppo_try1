# C76,416 P05 final120 decisions — actual command and clipping audit

Scope: completed run `runs/ppo_semantic_v3/validation/20260906T1908330051233Z_gc34262abffc1_fd1eba3dea714afca139357945d47440`, unchanged c34262a. Only decision529–648, policy ticks4,225–5,184 (960ticks/8s), plus the preceding decision/physical sample at4,224 for differences. PowerShell parsed the JSON tails; no new physics, Python, configuration change, checkpoint hash, training credit or master-ledger change.

**Finding:** this window does not show a sustained FL descent request being rejected by logical headroom or geometry. All requested FL residuals reach the pre-final-mapper-bias path essentially unchanged. The policy holds an almost fixed joint-space bias around static nominal advice. A real periodic mapper/final-target variation exists inside each decision and is hidden by reading only15Hz endpoints. Its small final-slew interaction must not be mislabeled as the P09 logical preclip defect or assumed to cause the missing touchdown.

## Request → actual target

The v3 P05 caps are FL hip18°/knee24°, wheel0.3rad/s; residual slew60°/s and1.8rad/s². Hard drive limits remain hip±135°, knee−60…210°, with the original logical2° reserve. Source c34262a `action_projection.py` and `semantic_residual_adapter.py` are the applicable execution contract, not the separately edited future headroom branch.

| FL channel | Nominal advice, all960ticks | Raw policy range | Requested = projected residual range | Mapper native | Actual drive |
|---|---:|---:|---:|---:|---:|
|Hip, degrees |22.8 |0.207616866…0.209431976 |+3.684317867…+3.715609380 |18.929718476 or20.179718476 |22.614036342…23.895327855 at120Hz |
|Knee, degrees |−13.4 |−0.033354182…−0.032722306 |−0.800203643…−0.785055153 |−12.15 throughout |−12.950203643…−12.935055153 |

FL controller bias is0, as is every other controller-bias channel. No nominal-geometry evidence/adjustment is present in any of these960ticks; this rear-only branch is not active in P05. The largest difference between `tanh(raw)*cap` and projected FL residual is2.22045e−15°/9.99201e−16°. Across all12 channels and all120 decision endpoints it is at most4.44090e−15. Thus the logical safety interval, phase caps and residual slew do not reduce these actual requests. Maximum successive decision residual changes are FLhip0.016278383° and knee0.013978871°, far below4° per decision and0.5° per physics tick.

Minimum FL distance to the logical2°-reserved boundary is hip106.484391°/knee43.799796°; the corresponding same-tick native+policy desired-target margins are hip109.104672°/knee45.049796°. Neither final physical hard clamp is near activation. This is not the rear knee's nominal−37.8/native−27.8 case.

### Intradecision target behavior matters

In **every** decision, mapper FLhip native is20.179718476 for its first4 physics ticks and18.929718476 for its last4,480ticks each. For example4,225–4,228 use the higher target and4,229–4,232 the lower. All120 decision-end records therefore see only the lower native value. Their actual hip range22.614036342…22.645327855° is not the full120Hz range above. Reconstructing the unchanged frozen hard/slew formula from each recorded native,bias and previous-final value gives120Hz mean hip target23.253971114°; measured hip mean23.203499133°. This is a measured periodic target pattern, not evidence of an extra mapper advance.

Final servo slew is1.25°/tick. On59 positive hip transitions the desired step is slightly greater than1.25°, so the existing limiter withholds **0.000000257…0.016278383°** for that tick; zero negative hip transitions and zero knee ticks are clipped. Example t5,177: previous22.617433272°, native20.179718476° + residual3.695379300° = desired23.875097776°, difference1.257664504°, so0.007664504° is delayed. This is small ordinary positive-direction final slew, not a large blocked downward joint request. Actual float32 target audits are verified. A joint-space sign alone does not prove wheel vertical direction.

## Wheels, support and actual vertical motion

All wheel nominal/native values are0. All120 decision commands equal the corresponding requested residual without wheel clipping:

| Wheel | Actual logical drive range, rad/s | Mean, rad/s |
|---|---:|---:|
|FL |−0.015115795…−0.014842840 |−0.014976897 |
|FR |−0.049546235…−0.049185830 |−0.049354039 |
|RL |−0.016035816…−0.015383985 |−0.015783541 |
|RR |+0.050275414…+0.050646137 |+0.050528914 |

These are logical drive signs; the native physical wheel signs remain the frozen FL−/FR+/RL−/RR+ mapping. They show small persistent policy wheel commands, not cancellation of a moving wheel nominal. They do not establish their effect on capture.

The last8s are actually approximately **6.35–7.68mm** above the top, rather than uniformly5–6mm: clearance min6.350314/max7.680011/mean6.914582mm. FL has0 ground or obstacle contacts, load0 and no actual support. Support count is3 on840ticks and2 on120ticks; terminal support is FR/RL/RR, not FL.

Finite differences of the measured collider bottom give vertical speed−24.802387…+24.481416mm/s,480 downward and480 non-downward ticks. Net bottom motion is only **−0.624254mm over8s** (mean−0.078032mm/s), from gap7.021551 to6.397297mm. Thus there is real downward motion within oscillation, but no sustained net touchdown approach in this fixed tail. Wheel-center forward displacement is only+0.079691mm, from front+5.575720 to+5.655411mm. Measured hip range23.089306…23.314514° and knee−12.948485…−12.940718° likewise show a nearly held configuration. Last bottom finite-difference speed is−19.530952mm/s, still no contact at termination.

Base net displacement over the same samples is X+0.056654mm/Z−0.164516mm. Recorded instantaneous base-Z velocity fields are+16.188…+32.189mm/s, not equal to the displacement-derived average; these readback fields are reported separately and are not integrated into a fictitious upward trajectory. This bounded audit does not resolve their sampling/integration-time relationship.

## Interpretation and bounded next observation

There is **no ongoing nominal descent trajectory** in this tail: nominal hip/knee are fixed22.8/−13.4. Policy residuals remain positive hip/negative knee with a very narrow range, so the data support an almost stationary biased pose plus internal target oscillation—not “the policy issued a continuing foot-down trajectory and logical/geometry clipping erased it.” Relative to nominal, the mapper and residual partly offset in hip;120Hz mean actual hip is nominal+0.453971°, while mean knee is nominal+0.459688°. Endpoint-only hip comparison would incorrectly suggest nominal−0.170907°. No P05 Jacobian or executed alternative is stored here, so neither pair of joint offsets proves upward/downward Cartesian causality or counterfactual touchdown.

For subsequent already-selected natural-P01 training, retain the existing960-tick-style distinctions when a P05 capture window appears: raw request versus projected residual, native periodic extrema versus15Hz endpoints, actual bottom descent/contact and current load. Compare a genuinely changing request with its actual target before deciding whether execution or the learned stationary policy is responsible. No new threshold, reward choice, training gate or intervention is proposed by this report. The completed P05 incomplete classification and all historic evidence remain unchanged.
