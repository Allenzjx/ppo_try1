"""Selected new runtime paths and movie clocks; not a physics success test."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo.semantic_backend import load_execution_profile
from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
from wlr50_clean.ppo.semantic_supervisor import load_task_spec
from wlr50_clean.ppo.semantic_video import camera_for_experiment, task_interval_receipt, capture_assist_tick_evidence

EXPERIMENT = "p05_hip_only_continuation_v1"
ROOT = Path(__file__).resolve().parents[2]


def test_p05_version_paths_do_not_select_old_configuration():
    paths = cli.version_paths("v3", experiment_id=EXPERIMENT)
    assert paths == tuple(ROOT / kind / ("ppo_" + EXPERIMENT) for kind in ("runs", "outputs", "configs"))
    schema = load_semantic_observation_schema(paths[2] / "observation_schema.json")
    assert schema.dimension == 389
    assert schema.observation_layout == "role372_p05_capture_assist_v1"
    assert load_execution_profile(paths[2] / "execution_profile.yaml")["capture_assist_mode"] == EXPERIMENT
    assert load_task_spec(paths[2] / "stage_task_spec.yaml")["capture_continuation_semantics"] == EXPERIMENT


def test_p05_keeps_accepted_camera_and_full_finite_task_clock():
    assert camera_for_experiment(EXPERIMENT) == camera_for_experiment("task_conditioned_hip_wheel_v1")
    assert task_interval_receipt(24000, experiment_id=EXPERIMENT)["encoded_duration_s"] == 200.
    assert task_interval_receipt(6555, experiment_id=EXPERIMENT)["extra_post_frames"] == 0
    with pytest.raises(RuntimeError):
        task_interval_receipt(24001, experiment_id=EXPERIMENT)


def test_capture_tick_never_substitutes_permission_for_physical_placement():
    pending = {"fl_capture_pending": True, "allow_capture_continuation": True}
    frame = SimpleNamespace(physics_tick=3601, sim_time_s=3601/120., state_id="P06",
        nominal_action_full12=(0.,)*12, action_mask_full12=(1,)*12,
        info={"raw_observation": {}, "capture_assist": {"active": True},
              "semantic_task": {"capture_continuation": pending, "fl_capture_pending": True,
                  "physical_evaluator": {"history": {"placed": {"FL": False}},
                      "current_legs": {"FL": {"support": False, "air": True}}}},
              "atomic_ack": {"drive_target_full12": (1.,)*12}})
    row = capture_assist_tick_evidence(frame)
    assert row["phase"] == "P06" and row["fl_capture_pending"] is True
    assert row["placed_history"]["FL"] is False
    assert row["current_legs"]["FL"]["support"] is False
    assert row["dispatch"]["drive_target_full12"] == (1.,)*12


def test_parser_exposes_new_namespace_without_changing_evaluation_start():
    args = cli.parser().parse_args(["eval", "--run-dir", str(ROOT / "runs" / ("ppo_"+EXPERIMENT) / "test"),
        "--expected-head", "a"*40, "--semantic-version", "v3", "--experiment-id", EXPERIMENT,
        "--seed", "4001", "--mode", "semantic_prior_eval"])
    cli.validate_request(args)
    assert args.from_phase == "P01" and args.max_decisions == 3000
