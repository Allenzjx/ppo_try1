"""The original capture inventory is preserved; physics exceptions prohibited."""
import pytest
from wlr50_clean.ppo.semantic_cli import _frozen_media_revision

PATH = "src/wlr50_clean/infrastructure/video_capture.py"
OLD = "4de41b906d506cc98165c66e44afa767a9f8111a1483a09492a25df43346dbd6"
NEW = "6ec218b1a64344516aabeb5c32729314724ad6f175147f61086dfc38570c13ae"
EXPERIMENT = "rr_rl_timing_policy_learning_v1"

def test_exact_media_revision_preserves_actual_original_git_blob():
    result = _frozen_media_revision(PATH, OLD, NEW, EXPERIMENT)
    assert result["original_frozen_inventory_rewritten"] is False
    assert result["runtime_sha256"] == NEW

@pytest.mark.parametrize("path,old,new,experiment", [
    ("src/wlr50_clean/fsm/controller.py",OLD,NEW,EXPERIMENT),
    (PATH,"a"*64,NEW,EXPERIMENT),(PATH,OLD,"b"*64,EXPERIMENT),
    (PATH,OLD,NEW,"rr_capture_then_rl_transfer_v1")])
def test_no_blanket_frozen_exception(path,old,new,experiment):
    with pytest.raises(ValueError):
        _frozen_media_revision(path,old,new,experiment)

def test_unchanged_control_stays_strictly_verified():
    assert _frozen_media_revision("controller.py",OLD,OLD,EXPERIMENT) is None
