"""Bounded immutable CP231936 evidence replay; no Torch, simulator, or writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from wlr50_clean.ppo.semantic_post_rr_front_prep_task import PostRRFrontPrepTask

SOURCE=ROOT/"runs/ppo_rr_capture_first_cp225280_v1/eval_CP231936_round1_DET_6b2d0df/source"
CAPS=[32.,36.,24.,112.,24.,36.,24.,36.,1.2,1.2,1.,.6]


def bounded_native(tick):
    """Binary seek on monotonic sealed JSONL; <=32 bounded line probes."""
    path=SOURCE/"capture_assist_ticks.jsonl"
    lo,hi=0,path.stat().st_size
    with path.open("rb") as stream:
        for _ in range(32):
            if hi-lo < 512000:
                break
            mid=(lo+hi)//2
            stream.seek(mid)
            stream.readline(200000)
            pos=stream.tell()
            line=stream.readline(200000)
            if not line.endswith(b"\n"):
                raise AssertionError("native evidence line exceeds bounded read")
            row=json.loads(line)
            found=row["episode_physics_tick"]
            if found==tick:
                return row
            if found<tick:
                lo=pos
            else:
                hi=mid
        stream.seek(lo)
        if lo:
            # lo is known line start, not an arbitrary midpoint.
            pass
        payload=stream.read(min(1024000,path.stat().st_size-lo))
        for line in payload.splitlines():
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["episode_physics_tick"]==tick:
                return row
    raise AssertionError(f"bounded source did not find native tick {tick}")


@unittest.skipUnless((SOURCE/"capture_assist_ticks.jsonl").exists(),"retained actual CP231936 evidence absent")
class RecordedPostRRTaskTests(unittest.TestCase):
    def test_actual_ack_raw_role_shapes_and_event_replay(self):
        path=SOURCE/"video_policy_decisions.jsonl"
        with path.open("rb") as stream:
            stream.seek(120807421)
            payload=stream.read(130971243-120807421)
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         "71229452a6baf8cee50888b9d37eebbdac449af2c0119aba02b06fe0d48f8339")
        decisions={r["end_tick"]:r for r in map(json.loads,payload.splitlines())}
        task=PostRRFrontPrepTask(phase_caps_full12={f"P{i:02}":CAPS for i in range(6,14)})
        result=[]
        for tick in (8288,8296,8304,8344,8352,8368,8376):
            step=decisions[tick]["step_info"]
            native=bounded_native(tick)
            self.assertEqual(native["episode_physics_tick"],tick)
            self.assertEqual(native["dispatch"]["drive_target_full12"],step["actual_drive_target_full12"])
            # This creates only a read-only frame-info envelope from recorded
            # same-tick facts, not a simulator snapshot or teacher trajectory.
            info=dict(semantic_task=deepcopy(step["semantic_task"]),
                drive_target_full12=native["dispatch"]["drive_target_full12"],
                atomic_ack=native["dispatch"],raw_observation=native,
                actuator_target_effect_audit=step["actuator_target_effect_audit"])
            out=task.observe(info)
            result.append(dict(tick=tick,active=out["post_rr_active"],
                contact=out["rr_contact_now"],grace=out["rr_transient_grace"],
                actual12=len(out["post_rr_measured_front"]["actual"]),
                receiver_geometry=out["post_rr_measured_front"]["receiver_geometry"],
                com=out["post_rr_measured_front"]["com"],
                fr_center=out["post_rr_measured_front"]["fr_center"],
                limitation=out["post_rr_prep_stall_reason"]))
            self.assertIsNotNone(out["post_rr_measured_front"]["receiver_geometry"])
            self.assertIsNotNone(out["post_rr_measured_front"]["com"])
            self.assertIsNotNone(out["post_rr_measured_front"]["fr_center"])
        self.assertFalse(result[0]["active"])
        self.assertTrue(result[1]["contact"])
        self.assertTrue(result[1]["active"])
        self.assertTrue(result[5]["grace"])
        self.assertFalse(result[6]["grace"])
        self.assertEqual(task.post_rr_touch_tick,8296)
        self.assertFalse(task.post_rr_prepared)
        self.assertEqual(task.snapshot()["post_rr_entry_provenance"]["ack_command_tick"],8475)
        print(json.dumps(dict(recorded_evidence_test="PASS",rows=result,
                             provenance=task.snapshot()["post_rr_entry_provenance"]),allow_nan=False))


if __name__=="__main__":
    unittest.main()
