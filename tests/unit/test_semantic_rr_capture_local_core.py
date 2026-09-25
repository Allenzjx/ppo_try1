"""Task wrapper tests use no simulator/Torch, and grant no physical credit."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from wlr50_clean.ppo.semantic_rr_capture_local import CaptureCore, interval_receipt, NAME
from test_semantic_rr_capture_local_task import observed


class Inner:
    def __init__(self):
        self.observation=(0.,)*422
        self.frame=None
        self.tick_observer=None
        self.done=True
        self.decision_count=0

    def reset(self, seed):
        self.frame=observed(0,crossed=False)
        self.done=False
        self.decision_count=0
        return self.observation

    def step(self, raw):
        for _ in range(8):
            before=self.frame
            self.frame=observed(before.physics_tick+1)
            self.tick_observer(before,self.frame,None)
        self.decision_count+=1
        return SimpleNamespace(terminated=False,info={'full_task_success':False},reward=99.)


def test_wrapper_preserves_422_and_explicit_absent_owner_state():
    inner=Inner(); core=CaptureCore(inner)
    obs=core.reset()
    assert len(obs)==448 and obs[:422]==inner.observation and obs[422:439]==(0.,)*17
    assert len(obs[439:])==9
    seen=[]
    core.tick_observer=lambda a,b,c: seen.append(b.physics_tick)
    step=core.step([0.]*12)
    assert seen==list(range(1,9))
    assert step.observation[439]==1.
    assert step.observation[447]==1.  # explicit same-attempt capture eligibility
    assert step.reward==0.  # activated inside a PREFIX decision, not credited
    step=core.step([0.]*12)
    assert step.info['local_reward']['on_policy_rr_sample'] is True
    assert step.reward!=99.  # global RR/RL objective is not mixed in
    assert step.terminated is False
    core.reset()
    assert core.observation[439]==0. and core.task.entry_gap_m is None


def test_partial_last_frame_is_quantization_not_extra_physics():
    result=interval_receipt(7979)
    assert result['experiment_id']==NAME
    assert result['frame_count']==998
    assert result['final_interval_physics_ticks']==3
    assert result['extra_physics_ticks']==0
    assert result['physical_duration_s']==7979/120
