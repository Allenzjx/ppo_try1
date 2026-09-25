# RR-first continuation — live status, not success certification

Latest sealed composite: CP230400_gain10_aux64, local5120 decisions /10 PPO updates /200
PPO Adam steps. Frozen prior: CP225280 (original historical count225280), original
Identity normalization, frozen actor excluded from local Adam. Actual LR1e-5.
Original front FL assist ON; rear capture/geometry/wheel helpers OFF. Separate
finite AUX:64 independent SGD steps,11 real terminal-approach/contact/hold rows;
only local mean-final RR6/7 rows/bias changed. Not on-policy/PPO credit.

Current source:0d0f8948999222f9b8710b0eb913a9419537be83. The tracking ownership
repair occurred AFTER update5. Its cold rebind preserves actor, critic, Adam,
LR, RNG and prior migration hashes; zero new learning credit. Previous versions,
N reference, original FSM, models, videos and all failed records are retained.

Physical task sequence:

Frozen live P01 → FR/FL/P06 → RR unload/lift/cross → qualified AIR over legal
top XY → local all12 PPO active → RR descent/contact/bearing → measured0.5s hold.
Ordinary phase transitions do not terminate learning. Real local hold success
is not full obstacle success. RR success is required before new RL curriculum.

The capture gate remains active through contact/drop. It never counts AIR as
bearing; old placed/cross history cannot qualify a new unlifted landing after
GROUND. Pending source events cannot acquire tracking responsibility before
consumption. Original explicit wheel stop and all12 policy freedom remain.

Historical update5 (superseded below):512 samples, all P09,1995 uncredited prefix decisions,
1 update. No new TOP/hold endpoint. First episode min sampled gap1.73449mm,
then knee refolded to−58° while FL lost support and RL contacted front wall;
ended INCOMPLETE_CONTROLLER_BLOCKED at89.725s. Second episode is a nonterminal
budget tail. Joint direction/clearance changes alone are not placement success.

Deterministic natural P01 evaluation completed:
`runs/ppo_rr_capture_first_cp225280_v1/eval_CP227840_DET_tracking_1e10d39`.
The one reloaded package contains the frozen prior and local learned module.
No teacher switch, automatic rear descent, new reward, sigma reset or diagnostic
override.91.166667s,RR incomplete, final gap58.1095mm/0N/no TOP. Source and all
three videos are sealed under video_review/CP227840_DET_tracking_fixed_review.
998 prefix raw/FINAL requests match CP225280 exactly; FR/FL placed and RR
qualified lift/cross all match reference event ticks. Repaired source tracking
is correct, but all370 active decisions still have FINAL RR knee−58°.

Historical update6 block under repaired tracking, now COMPLETE:
`runs/ppo_rr_capture_first_cp225280_v1/train_tracking_fixed512_1e10d39`.
Its result and checkpoint are recorded below. The old v1 success was not
automatically a compatible AUX demonstration. Do not
increase sigma solely because73.24% old-v2 samples hit knee headroom: most such
requests were not tanh-saturated and many collapsed to the same final−58°.

See RECOVERY.md for immutable checkpoint paths and exact source provenance.

Historical update6 evidence: the block is
sealed (512 new samples/20 Adam), including first41 stochastic success at
69.266667s/0.5s/12.605516N. Second471 reached TOP but dropped; nonterminal tail
bootstrapped. Source owner repair plus stochastic policy/control produced that
trajectory; not deterministic success or RL progress. Physical phase counts
104 P09,2 P10,2 P11,404 P12 remain RR-only objective, not RL training credit.

One separate AUX64 trial was strictly saved/reloaded after this update. It only
slightly improved knee fitting and did not improve AIR hip fitting; do not call
it a physical fix. Fresh512 on-policy run, now COMPLETE:
`runs/ppo_rr_capture_first_cp225280_v1/train_after_aux64_fresh512_0ff03ea`,
former session36141. Actual update7 produced CP228864_aux64 (3584/7/140).
New512 rows/20 Adam,2992 prefix credit0;423 incomplete +46 stochastic local
success (0.55s/10.0536N) +43 nonterminal budget tail. No new AUX. Tensor audit
passed; all12 issued actions and single HISTORY agree, frozen prior unchanged.
See UPDATE7_LEARNING_ATTRIBUTION.md for small same-input mean changes and limits.

Completed sole Isaac session54703 was the saved/reloaded naturalP01 DET:
`runs/ppo_rr_capture_first_cp225280_v1/eval_CP228864_aux64_DET_0ff03ea`.
OriginalFL assistON,rear helpersOFF,finiteAUX64 training lineage disclosed.
91.166667s, INCOMPLETE_CONTROLLER_BLOCKED; RR TOP false across all10940 native
ticks, final gap57.408820mm/0N.998 pre-gate raw/FINAL requests and accepted lift/
cross/place events match CP225280 exactly. Only4/370 active knee targets remain
at-58deg; actual knee unfolding max1.290069deg, minimum gap55.230760mm. This is
real but insufficient mean-action change, not deterministic RR placement.
Full/detail/comparison videos are sealed, visually inspected and delivered under
video_review/CP228864_aux64_DET_0ff03ea_review. They evaluate0ff, NOT the newer
coordinate-migrated checkpoint. See CP228864_aux64_DET_result_readonly.md/json.

At the subsequent cold boundary,0d0f894 changes only RR local-mean parameter
coordinates (gain10 for6/7), with final-head rows divided10 and corresponding
actual Adam moments multiplied10/100. This preserves initial physical action,
std, other10 means, prior/critic/RNG/LR and AUX64 lineage; future optimization
dynamics deliberately differ. No action-range/reward/task/physics change or new
PPO/AUX credit. Actual CUDA migration/save/fresh strict reload passed; mean error
1.19e-7, same-action raw logp error0. Old source/checkpoints/videos retained.

Cold-migrated predecessor, now superseded by the update8 checkpoint below:
checkpoints/history/checkpoint_CP228864_local003584_aux000064_lineage448_v2_g0d0f89489992.pt
SHA ea6a93db6342a5eaf09cf119c914ebc6e3ddba730e384b7751914cfdfb443101;
sidecar6c02b9d5c099fc3a8a7332e2df726a174b72acbf87359d61441b597aacbadc48.
Counts remain3584/7/140,AUX64. Receipt:
runs/ppo_rr_capture_first_cp225280_v1/migrate_rr_mean10_CP228864_0d0f894/coordinate_receipt.json.

The fresh gain10 block is COMPLETE, former session32484 exited0 naturally:
runs/ppo_rr_capture_first_cp225280_v1/train_rr_mean10_fresh512_0d0f894.
New512/1PPO/20Adam,1995 prefix credit0,0 newAUX.143 first-opportunity rows end
in stochastic RR capture+hold success0.541667s/11.394116N;369 second-opportunity
rows end at nonterminal budget tail91.066667s/gap25.277119mm, not failure/done.
Actual optimizer phases504P09/1P10/1P11/6P12, all RR-only objective.14 cumulative
opportunities/9 completed episodes/4 stochastic local successes (one old v1),
13963 uncredited prefix rows. No new RL curriculum.

Latest CP checkpoint_CP229376_local004096_aux000064_lineage448_v2_g0d0f89489992.pt
SHA cd6d724af6901d03ee235f3b5d3160b4f9db86195d0cf423b3e1e159625368da;
sidecar da5917b6a2affeb9fb0d1d52597d89e0aa15928221de060d5acdc157646f989c.
Cold Tensor verification completed5.31s/PASS before newIsaac launch.512 rows
exact;20 minibatches/five exposures; prior unchanged, actualLR1e-5.485 legalAIR
fixed inputs mean shifts hip-0.017138386/knee+0.008015373 raw; these are not
degrees or physical success. GlobalKL0.014084862,RR fraction10.4974%.

Completed Isaac session59631: saved/reloaded naturalP01 DET
runs/ppo_rr_capture_first_cp225280_v1/eval_CP229376_gain10_aux64_DET_0d0f894.
Same packaged prior/local policy all episode, all12 active capture freedom,
originalFL ON/rearhelpers OFF, finiteAUX64 lineage disclosed. Natural exit0,
91.166667s/10940ticks/1368frames/errornull; RR/fullfalse. Finalgap51.159650mm/0N,
actualRR[2.855070,-54.383540],FINAL[3.119494,-54.635472]deg.998 frozenprefix
raw/FINAL exact to accepted CP225280; RRqualified6995/cross7979 preserved. No
new deterministic TOP/hold; first unfinished task remains RR true placement.
New source sealed, full/detail/CP225280 videos exported, fully decoded, visually
checked and delivered under video_review/CP229376_gain10_aux64_DET_0d0f894_review.
Former export session79455 exited0. Full1368 frames/91.2 encoded seconds, normal
speed, no black frames, complete failure tail. Detail371 same-run frames.

Current sole Isaac PID74420/session86454, unchanged production0d0f894:
runs/ppo_rr_capture_first_cp225280_v1/train_gain10_continuous2048_from_CP229376_0d0f894.
Explicit CP229376 above,2048 activeRR samples/four complete512 updates requested.
Each full update saves; surviving physical episode/history continues across
updates, only true terminal restarts a real frozenprefix. No reset of network,
optimizer, sigma or reward, no new AUX/rearassist and no newRL curriculum.
Do not count updates9–12 before their actual complete saves. Do not hot-edit or
start concurrentTorch/PXR. CPU-only video export is not a second simulator.

Update9 is now ACTUALLY complete (2026-09-25 11:18 UTC). This block has512 new
RR decisions/1PPO/20Adam,3989 credit0 prefix decisions; phase490P09/3P10/8P11/
11P12 allRR objective. First3 real opportunities succeeded stochastically
(87+294+124 rows); next7 rows from fourth realprefix complete512. Fourth episode
continues without reset across the optimizer boundary. Current later pending
rows are not part of this checkpoint. No newAUX; taskv2 total2560/5PPO.
CP checkpoint_CP229888_local004608_aux000064_lineage448_v2_g0d0f89489992.pt,
SHA e60c82d31267049a8f00ebb64c0e117e250c82485d4013355334ca9d20a518bf,
manifest1266ed34ca1a2f8c7d44bdb1c06a41a1b32c7d29940c2c1d41daadc9ff4f1234.
Save/load roundtrip and emptyrollout receipttrue; independent bytehashes match.
Actual512 JSON samples each appear5 times in20 officialminibatches,2560 matching
oldlogp exposures,selected=issued/nativeverification512/512. Tensor review waits
for soleIsaac exit. KL0.01169645,clip0.1625,actualLR1.5e-5 from preserved adaptive
schedule, no manual LR change. CP229888 UNEVALUATED; CP229376 remains latestvideo.
The2048 block is still running; updates10–12 not yet credited.

Update10 is now complete (2026-09-25 12:18UTC). Block totals1024/2PPO/40Adam,
6980 prefixcredit0; newest512 phase496P09/2P10/2P11/12P12, allRR. Cumulative
local5120/10/200,AUX64,taskv23072/6. CP230400_local005120_aux000064_lineage448_v2_g0d0f89489992.pt,
SHA804d990df6c7aa21e0acfe427066b068aace35495924eb10cfa97f6f3b2e777c,
sidecar87098425655b8855414827fb69a28da9107bbba30ecea5685e169e6040668710.
ActualLR1e-5/adaptive,KL0.015652224,clip0.210546875; roundtrip/emptyrollouttrue,
bytehashes verified. Seventh episode continues across update10 at8296/69.133333s.
FirstTOP8312,maxhold0.425s then dropped; no local success in this snapshot.
Subsequent phaseP12/AIR is not success: actualbearing/hold remains required.
NewCP230400 UNEVALUATED; latestvideo stillCP229376. SoleIsaac continues.
