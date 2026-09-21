import math
import pytest
from fl_probe_helpers import FiniteFLHold, quintic, trigger_ready


def test_hold_does_not_accumulate_HISTORY_or_change_other_channels():
    probe = FiniteFLHold((-1.5, 0.))
    probe.start(tick=2616, projected_residual=[.04, -.56]+[0.]*10)
    raw = [0.]*12
    for i in range(40):
        base = list(raw)
        base[0] = .9*raw[0]-.001
        base[1:] = [.2]*11
        raw, receipt = probe.action(base, caps=[18.,24.]+[24.]*10)
        assert raw[1:] == tuple(base[1:])
        assert 18*math.tanh(raw[0]) >= -1.46-1e-12
        if i >= 8:
            assert 18*math.tanh(raw[0]) == pytest.approx(-1.46)
            assert receipt['state'] == 'hold'
    assert not probe.complete
    for _ in range(20):
        raw, _ = probe.action([.2]*12, caps=[18.,24.]+[24.]*10)
    assert probe.complete and raw == pytest.approx([.2]*12)


@pytest.mark.parametrize('offsets', [(1.5,0.),(-1.5,1.)])
def test_both_signs_and_combo(offsets):
    p=FiniteFLHold(offsets);p.start(tick=8,projected_residual=[0.]*12)
    for _ in range(8):r, receipt=p.action([0.]*12,caps=[18.,24.]+[24.]*10)
    assert [18*math.tanh(r[0]),24*math.tanh(r[1])] == pytest.approx(offsets)
    assert receipt['interpolation'] == 1.


def test_capture_releases_without_further_descent_or_history_reset():
    p=FiniteFLHold((-1.5,0.));p.start(tick=0,projected_residual=[0.]*12)
    r,_=p.action([0.]*12,caps=[18.,24.]+[24.]*10)
    before=18*math.tanh(r[0])
    r,receipt=p.action([0.]*12,caps=[18.,24.]+[24.]*10,capture_or_leave=True)
    assert receipt['state']=='release' and before < 18*math.tanh(r[0]) <= 0


def test_trigger_requires_real_cross_uncaptured_current_AIR_and_endpoint():
    e={'physics_tick':2616,'valid':True,'termination_reason':None,
       'history':{'front_edge_crossed':{'FL':True},'placed':{'FL':False}},
       'current_legs':{'FL':{'air':True,'obstacle_pair_active':False,'within_top_xy':True,'ground_contact':False,'support':False}}}
    assert trigger_ready(phase='P05',endpoint=True,evaluator=e,tick=2616)
    assert not trigger_ready(phase='P05',endpoint=False,evaluator=e,tick=2616)
    assert not trigger_ready(phase='P05',endpoint=True,evaluator=e,tick=2624)
    e['history']['placed']['FL']=True
    assert not trigger_ready(phase='P05',endpoint=True,evaluator=e,tick=2616)


def test_smooth_profile():
    assert quintic(0)==0 and quintic(1)==1
    assert quintic(.001)<1e-7 and 1-quintic(.999)<1e-7
