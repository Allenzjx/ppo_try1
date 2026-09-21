# P05 controllability response: three bounded trials sealed

These are external controllability interventions, not PPO learning or full-task success. The two hip±1.5deg cases had **no FL obstacle contact, no placed event and no P06 entry**. The third hip−2deg case **did physically contact, earn placed and enter P06**, then lost current FL support. All three runs were read only after sealing; no active training/runtime files were touched.

The probe started at tick2608 with22.235mm gap and actual hip29.196deg: the source had reached its endpoint, but the physical leg was still descending. The22mm→submillimetre transient is therefore **not** wholly credited to the−1.5deg intervention. The cleaner response window is the32-decision hold, ticks2673–2928. It was compared with the same ticks in the preserved, unmodified CP178432 evaluation. The17 logged pre-intervention ticks have exactly matching FL joint positions and base positions; this checks selected measured entry evidence, not every latent/contact solver state.

| Hold-window quantity | Original CP178432 | External hip−1.5° | External hip+1.5° |
|---|---:|---:|---:|
| FL gap mean mm |4.147073|.933323|7.208808|
| FL hip mapped nominal mean deg |22.845245|22.834918|22.852059|
| FL hip requested residual mean deg |−.025789|−1.580887|1.419113|
| FL hip final target mean deg |22.819391|21.254031|24.271172|
| FL hip actual mean deg |22.768327|21.205581|24.218574|

The intervention anchor was−.080887deg, so anchor−1.5=−1.580887deg exactly. Its requested residual and headroom-effective residual agree throughout the hold; all12 masks remained1. Mapped N differs by only−.010327deg, while real hip angle changes−1.562746deg and measured gap drops3.213750mm. This is evidence **against nominal cancellation of this sustained offset**. It does not claim arbitrary large or transient commands are always followed.

The same-measured-base, hash-bound USD wheel-center FK decomposition predicts−4.115503mm from changed FL hip/knee, then+.878536mm from substituting the actual probe base pose, for−3.236967mm total. The measured wheel-bottom gap change is−3.213750mm. This decomposition is order-dependent kinematic bookkeeping, not force-level causal identification: the other11 policy channels react to real observations, the knee also changes, and collider-bottom orientation is not the same derivative as wheel center. It nevertheless explains why a fixed-base estimate overstated the final gap reduction.

FR/RL/RR retained actual verified contact throughout the hold. Their minimum/mean/maximum summed pair normal forces were11.072/12.207/13.466N,13.394/14.488/15.817N and1.098/2.057/3.098N. FL force remained zero. No imaginary FL bearing is used. Hold gap range was.394–1.631mm; after release, follow-window mean gap returned to3.105mm.

Tracking is not perfectly static: mean signed target−actual is only.04845deg, but mean absolute error is.59470deg and maximum.73050deg. The short hold has a15Hz target/actual/gap peak (256 samples,120Hz, Welch detrend-linear), target peak-to-peak1.25deg and actual.213974deg. The earlier long-hover7.5Hz loop changed frequency under this offset;30Hz remains the feedback update rate. This is not a formal stability improvement claim.

## Sealed positive response and paired interpretation

The positive case triggered at the same tick2608, gap22.235492mm and exact anchor−.080887deg. The17 pre-intervention FL q/base samples match the negative case exactly. Its hold was the same2673–2928 window. Requested and effective hip residual both remained+1.419113deg, with no headroom clipping and all12 masks1. Compared with the original same-clock trajectory, hip q rose1.450248deg and gap rose3.061735mm; the mapped N changed only+.006815deg. Positive hold gap was5.867–8.944mm. After release its follow-window mean gap returned to3.259526mm.

The positive FK bookkeeping predicts+3.905167mm from changed FL q,−.812418mm from the actual base response, total+3.092749mm; the true bottom-gap change is+3.061735mm. As in the negative case, body/contact coupling opposes part of the fixed-base displacement. This is not a claim that only the hip caused every component. FR/RL/RR maintained contact throughout; force min/mean/maxN were10.985/12.269/13.405,13.313/14.491/15.684 and1.020/1.990/3.088. FL remained unweighted AIR.

Positive hip mean absolute tracking error was.595754deg, maximum.761901deg, despite only.05260deg signed mean error. Its target and q had15Hz peaks, q peak-to-peak.219987deg. The positive gap spectrum instead had its largest short-window component at.46875Hz, so the negative gap's15Hz dominance must not be copied onto the positive result.

Together,±1.5deg give approximately1.0043deg of actual hip change and2.0918mm of gap change per degree of intervention across the two hold means. The response is directionally consistent and roughly symmetric around the original; neither sustained offset was erased by nominal feedback. This supports real policy-channel authority and argues against a mapper rewrite as the immediate response. Both interventions still failed to touch, and neither demonstrates learned capture or improved full-task stability.

## At most one further diagnostic

After the positive trial sealed, root ran the one additional **hip−2.0deg test with the same entry/ramp/hold**. The estimate was another1.05–1.07mm downward, comparable to the remaining.933mm average gap. Requested residual was−2.080887deg around the same entry anchor, far inside the P05±18deg cap and reserved actuator bounds. This last probe is now sealed; its actual result follows. No more probes or probe-success gate were added; formal PPO training proceeds separately.

Keeping the same entry makes this a one-factor amplitude comparison with the same planned two-second hold, shortened only if real capture occurs. Delaying entry would additionally change controller phase and physical state; it is not needed to answer the current authority question. A hip/knee combination is not justified by these results: forward distance during the first negative hold remained about+.104→+.110m in legal top XY, and adding positive knee would reduce the requested downward displacement. Do not scan further amplitudes or count this external intervention as a learned policy improvement.

## Third hip−2deg: true contact and transition, not retained support

`P05_hip_minus2_01` sealed at tick3008/P06, probe complete, no task terminal. The same natural-P01 policy prefix led to the same entry2608. Its sustained request reached−2.080887deg through the unchanged mapper/projection chain; all12 masks remained1 and requested versus headroom-effective FL offsets differed by0.

| Actual event | Tick | Evidence |
|---|---:|---|
| First real FL obstacle contact |2843|pair active, normal force1.041N; gap−.340mm|
| Physical evaluator placed event |2844|second qualifying contact; normal force1.295N; gap−.321mm|
| First returned P06 decision |2848|current TOP/support true; native commands through this tick still fromP05|
| First P06 command / release starts |2849|early release responds to capture; no hard-history or mapper reset|
| First contact loss |2850|first active-contact interval ended2849|
| Contact regained |2872|current TOP/support true; bearing11.200N at returned decision|
| Last contact of the second interval |2934|second interval2872–2934;70 active physics samples total|
| Final diagnostic sample |3008|FL AIR74 consecutive samples, gap10.347mm, bearing0N, no current support|

Capture caused release early: hold2673–2848 lasted1.4667s rather than the planned2.1333s. Its gap min/mean/max was−.530/−.043/.583mm, with6 active-contact samples in that hold. Negative geometric gap before2843 was **not** counted as contact. Release2849–2912 had42 contact samples, and follow2913–3008 had22. The true placement/history event remains recorded, while the final AIR state is separately reported; history is not evidence of continued bearing.

The first loss cannot simply be labelled “the hip offset was withdrawn upward”: at2856 the physical hip request was−2.10006deg, slightly more negative than the hold request, because P06 caps and the continuing policy participate in the release blend. Other joints, wheels and body state were also changing. Later the request relaxed to−.78975deg; final FL hip target/actual was21.4572/21.3696deg, knee−12.2417/−12.2693deg. Base height moved from94.367mm at2848 to97.691mm at3008, while FL front distance reached+.152572m. Final other-support flags wereFR/RL=true, RR=false. These observations do not isolate a single cause of loss.

The narrow conclusion is that the existing residual channel and physical execution chain **can produce real FL touchdown and advance P05→P06 with a small legal perturbation**, without altering N, hard limits, mapper or sensors. Sustained contact/appropriate later support transfer, learned capture, complete crossing and quality improvement remain separate objectives. The3 external probes add0 training decisions and0 PPO updates. No new probe or production correction is proposed from this result; the real PPO block is independent of this diagnostic's label.

Evidence: `hip_minus_01_response.*`, `hip_plus_01_response.*`, `hip_minus2_01_response.*`, `sealed_FL_frequency_and_probe_design.json`, and the immutable formal CP178432 failure chain. Only output reports were updated.
