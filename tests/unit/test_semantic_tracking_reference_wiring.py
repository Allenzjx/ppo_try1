"""Selected profile and fail-closed reference wiring, without Isaac."""
from pathlib import Path

import pytest
import yaml

from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError
from wlr50_clean.ppo.semantic_backend import load_execution_profile, build_semantic_projector
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.semantic_tracking_reference import MODE
from wlr50_clean.ppo.actuator_target_effect import ActuatorTargetEffectError

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "configs/ppo_semantic_v3/execution_profile.yaml"
ZERO = (0.,)*12


def test_selected_profile_and_old_path_are_explicit():
    assert load_execution_profile(V3)["residual"]["tracking_reference_mode"] == MODE
    assert load_execution_profile(ROOT / "configs/ppo_semantic_v2/execution_profile.yaml")["residual"].get("tracking_reference_mode") is None


@pytest.mark.parametrize("mode,headroom", [(True, "same_tick_post_mapper_servo_margin_v1"),
    ("unknown", "same_tick_post_mapper_servo_margin_v1"), (MODE, None)])
def test_invalid_profile_reference_contract_rejected(tmp_path, mode, headroom):
    data = yaml.safe_load(V3.read_text(encoding="utf-8"))
    data["residual"].update(tracking_reference_mode=mode, policy_headroom_mode=headroom)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_execution_profile(path)


def test_direct_reference_dispatch_cannot_omit_headroom():
    from test_semantic_tracking_reference_dispatch import adapter
    a = adapter()
    with pytest.raises(RobotAdapterError):
        apply_semantic_residual(a, ZERO, physics_tick=1, tracking_servo_names=(),
            controller_bias_full12=ZERO, projected_residual_full12=ZERO,
            tracking_reference_mode=MODE, tracking_reference_bootstrap_tick=1)
    assert a.write_count == a.servo_target_mapper.feedback_tick == 1


def test_reference_setting_does_not_change_logical_projector_or_324_history(tmp_path):
    data = yaml.safe_load(V3.read_text(encoding="utf-8"))
    data["residual"].pop("tracking_reference_mode")
    path = tmp_path / "prior.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    current, prior = build_semantic_projector(V3), build_semantic_projector(path)
    for phase in ("P01", "P06", "P08", "P09", "P13"):
        kwargs = dict(state_id=phase, nominal_action_full12=ZERO,
                      reference_action_full12=ZERO, reference_delta_full12=ZERO,
                      previous_projected_residual_full12=(.5,)*8+(.015,)*4)
        assert current.project((.3,)*12, **kwargs) == prior.project((.3,)*12, **kwargs)


@pytest.mark.parametrize("field", ["physics_tick", "servo_tracking_feedback_sample_tick"])
def test_current_ack_bool_clock_is_not_an_integer_receipt(field):
    from test_semantic_tracking_reference_dispatch import adapter, checked_step, plan, audit
    a = adapter()
    context, before, ack, _ = checked_step(a, plan(ZERO), 1)
    ack[field] = True
    with pytest.raises(ActuatorTargetEffectError):
        audit(a, plan(ZERO), ack, before, context)
