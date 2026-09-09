# C3 / checkpoint 133248 — P05 capture incomplete

Run `20260908T0348320962319Z_gdb991aa68103_ef640fd945134f319e9affad5e47f0fe`; frozen HEAD `db991aa68103`, explicit372 role layout, deterministic natural P01, seed2001, no prefix.

**685 decisions / 5,480 ticks / 45.666667s; optimizer0.** P05 reached its30s deadline: `INCOMPLETE_CONTROLLER_BLOCKED`, task success false, no external-window truncation. Physical valid=true, hard failure=null; final raw finite=true and body collision=false.

## FL: why .85 is still not placement

FL qualified at tick1971 and crossed at2102, but never recorded placement. At5480 it is **AIR**, load0, support=false, obstacle pair inactive, consecutive TOP samples0; current AIR count3546. Its front distance is+116.066mm and bottom clearance+14.055mm. XY and top_geometry are true: the existing geometric tolerance is−15…+25mm, not a declaration of contact. Both exact ground/obstacle pairs are verified, inactive and zero-force; points are null.

The source predicate gives **.35 qualification + .35 crossing + .15 current top geometry + 0 contact persistence = .85**. Hard placement still requires two consecutive samples with active obstacle contact, top geometry and nonnegative front distance. Those contact samples are missing. No additional load-ratio or fixed-pose threshold is invented here. C2's .7 had the same Q/C and no contact persistence, but top_geometry=false at+43.824mm clearance. The.15 difference is exactly geometric partial credit, **not** a new placement or predicate contradiction. This decomposes `completion_values`, not the actual PBRS potential; it does not establish a reward plateau. `CAPTURE`/`pending_capture=true` describe pending work, not success.

FR Q/C/P ticks55/1753/1784; current TOP/load.430575. RL/RR have no hard Q/C/P and remain GROUND, loads.497885/.071540. Thus three current supports do not mean four placements or whole-task final support.

## Coverage and recorded quality

P01–P05 decision counts: **2 / 216 / 5 / 12 / 450**; P06–P13:0. Valid nonterminal handoffs occur at16/1744/1784/1880; no P05→P06 transition. Manifest phase ticks sum5,480 and final raw time matches5480/120.

Global roll/pitch RMS .130458/.096691rad; angular-acceleration RMS5.238936rad/s². P05 values .030261/.023906rad and4.660277rad/s². Its stored CAPTURE-labelled window now has2,434 ticks/20.283333s (EXECUTION1,166), but P05 aggregate FL obstacle force and touchdown count remain zero. Touchdown-speed quality is unavailable, not zero-impact success. Unsampled later phases/windows and the fixed all-phase score remain null.

Final wheel nominal is[0,0,0,0]; residual equals final canonical drive[−.209371,−.031021,−.045477,+.327977]rad/s. Measured body linear/angular speeds .030045m/s/.116162rad/s. Final decision has8/8 verified native/effect ticks, four state-write counters0 and no finite fallback/bootstrap; the entire native stream was not re-audited.

**Formal natural-P01 C success remains0/3.** Historical A remains incomplete/unpaired. Scope: final manifest, final raw/audit rows, small transition ledger and relevant predicates only; no full trajectory scan or new physics. No unique cause or stability superiority is inferred.
