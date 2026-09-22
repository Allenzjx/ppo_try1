# Actual recovery activation: fixed ticks 5992–6160

New headless run `164147...13764630c7a0468285bacb6aa3607ee8` versus old sealed deterministic video `152109...3cdf838115e44829b66ad9585ad92ba7`. All 169 new/old physical measurement rows were available. No target is substituted for measured velocity.

First difference: the gate is eligible in the **after-frame task at tick 6000** (P05 age 30 s); the first real changed dispatch is **6001**. Actual nominal, mapped nominal, raw policy response, REQUEST, native final targets and measured Full12 first differ from old at 6001. This is the new finite recovery nominal plus the frozen policy's response to changed observations, **not new PPO learning**; this is eval, with zero optimizer updates.

Original P05 held source wheels remain [0,0,0,0]. Recovery nominal becomes [.3,.3,.3,.3] rad/s at dispatch 6001. All 12 residual permissions remain one for all 169 audited ticks. No fresh authored wheel event exists at the 22 saved decision endpoints; 21 endpoints starting at 6000 are eligible.

Canonical order FL / FR / RL / RR; all values rad/s except raw Gaussian:

| Tick | Mapped N | Raw policy | Effective REQUEST | Final canonical target | Measured canonical angular velocity |
|---:|---|---|---|---|---|
| 6000 | 0 / 0 / 0 / 0 | −.877682 / .009383 / .030633 / −.292391 | −.705256 / .005630 / .030624 / −.170600 | −.705256 / .005630 / .030624 / −.170600 | −.705786 / −.254869 / .161450 / −.094755 |
| 6001 | .3 / .3 / .3 / .3 | −.877551 / .009430 / .030449 / −.292242 | −.705190 / .005658 / .030440 / −.170518 | −.405190 / .305658 / .330440 / .129482 | −.405679 / −.008079 / .333986 / .186417 |
| 6144 | .3 / .3 / .3 / .3 | −.891375 / .009606 / .034185 / −.304156 | −.712072 / .005764 / .034172 / −.177067 | −.412072 / .305764 / .334172 / .122933 | −.412528 / .123133 / .360812 / .173403 |
| 6160 | .3 / .3 / .3 / .3 | −.892329 / .009619 / .034291 / −.304582 | −.712542 / .005771 / .034277 / −.177300 | −.412542 / .305771 / .334277 / .122700 | −.413017 / .119326 / .361772 / .159191 |

The physical REQUEST equals the same-dispatch post-mapper wheel addition here; final targets also match saved `physical_observations.commanded_full12`. Native setter buffers verify wheel-axis signs [−,+,−,+] relative to canonical; at 6001 their actual native targets are [+.405190,+.305658,−.330440,+.129482]. Numeric native joint IDs and native measured angular velocities are not serialized, hence null—not assumed zero. Last writer is `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`; all mapping/setter audits passed.

From 6000→6160 (1.333 s), new FL front distance moves −33.902→−18.495 mm (+15.407 mm); gap 7.759→12.057 mm. Body displacement is [+16.274,−.458,+2.888] mm. Old trace over the same interval advances FL only +.00149 mm and body x −.04244 mm. FL remains AIR/not crossed/not placed at 6160; FR/RL/RR support the body at the sampled task endpoints.

Owner distinction: old source stop is legitimate and unchanged; the new explicitly versioned recovery advice adds the nominal rolling suggestion. PPO still cancels more than .3 rad/s on FL, leaving FL final negative; FR/RL/RR final positive and their measured responses differ under contact. No mask-loss claim and no task-success claim.
