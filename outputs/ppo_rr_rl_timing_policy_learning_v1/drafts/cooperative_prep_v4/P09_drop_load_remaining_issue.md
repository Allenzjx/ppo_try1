# P09 late drop-load: bounded remaining issue, not an implemented guard

Read-only source inspection; no Recording rescan, physical run or supervisor change. The reported sealed P10 suffix had real RR TOP samples followed by lost load/RL remaining on ground. That observation is compatible with the source path below but does not prove the whole trajectory was caused by this one owner.

## Concrete behavior

Current `semantic_supervisor.py` `_sequence_permission` gates P09 late **only when its next source tick equals the late group tick** (source 5.4s/648). After that, it does not continuously check RR bearing for the late owner. `_continuous_advisory` keeps applying the last touched owner values, including in later semantic phases unless a newer owner replaces that channel.

The source contract has only three events from source5.33s onward: four-wheel stop5.3333s, the full12 late group5.4s, and four-wheel stop7.2s. At the late group the changed channels are FL hip0/knee1, RL hip4/knee5 and FL wheel8. FL hip/knee become -18.5/-31.4°, RL hip/knee become15.4/19.4° and FL wheel becomes -1.07 canonical rad/s. These are logical source values, not measured movement or final post-residual targets.

Critically, `fsm/motion_executor.py:226` emits the **complete absolute nominal target at the event**, not an interpolated servo trajectory. Its own comment states the adapter creates the smooth120Hz drive path. There are no subsequent P09 late servo waypoints to pause. Thus:

- Freezing a source cursor after the group does **not** stop the adapter moving toward those already-issued far-away servo targets.
- Reusing the last `sample`, skipping a source re-assignment while `proposed` begins at `self.nominal_full12`, or holding “previous nominal” all retain that same demand. They are not an executed-target hold.
- Freezing the whole P09 clock additionally delays its explicit7.2s stop and can prolong the FL negative source pulse. Current code correctly keeps that source clock moving once started, so the source pulse is finite; later policy-negative velocity is a separate cause.
- P12 already pauses its independent RL joint cursor on lost RR support unless RL has a currently qualified AIR swing. Its wheel clock continues to complete stops. This prevents new RL source events, but also does not withdraw an already-issued RL remote target. It must not be reported as a complete physical unload stop.

## Smallest honest design, and why no quick source-only patch is supplied

A genuine source-pull suspension must be enforced where the **last actually executed target and the separately attributed current nominal/policy contributions are known**, still before the existing final hard clamp/slew and single atomic write. It would affect only live P09 late-owned indices0,1,4,5, not newer owners, FR/RR capture, explicit wheel stop or any residual permission. On lost required RR bearing it must withdraw that source pull while retaining the executed target; policy proposals remain Full12/open. On restored bearing it resumes with the existing final slew, without replaying the5.4s group or negative pulse.

The supervisor currently receives measured joints plus its nominal/source histories, not an exact execution-layer decomposition suitable for that hold. `previous_final - raw_residual` is not valid because mapper/controller/headroom/slew histories intervene. Holding measured q is a new pose controller; returning to pre-late source poses is a reversal, not suspension. Neither is an acceptable disguised fix.

A second source cursor alone cannot solve this source's single-step servo event. An execution-boundary hold requires a small explicit, audited interface plus any hold/owner state to be observable; if new state is needed it cannot be hidden just to retain422 dimensions. That is a separate reviewed control change, not the sigma patch and not a rear capture assist. Until designed and measured, no claim that RR drop-load fully stops old late pulls is warranted.

For the current iteration, keep the existing first-start bearing gate, P12 live-AIR exception and timely stop behavior; proceed with the reviewed precontact cooperative exploration. Record the unresolved issued-target persistence. Do not delay that training for a broader guard refactor or label the current software as having solved continuous load-loss protection.

## Narrow regression candidate

`test_p09_drop_load_known_behavior.py` is an **unexecuted output-only CPU source-scheduler regression candidate**. It documents the current counterexample and protects timely once-only wheel stops. Passing it would confirm the remaining issue, not certify a new guard. A future actual guard must additionally prove from the actuator receipt that its dangerous nominal pull stops while an independent policy perturbation on the same channel still has authority. No physical-task credit comes from these fixtures.
