"""CPU video-only regression: these tests are not a physical success claim."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo import semantic_video_cli as cli


class Recorder:
    def __init__(self, backend):
        self.backend = backend
        self.frames = []
        self.started = False

    def start(self):
        self.started = True
        return True

    def before_render(self, **kw):
        assert self.started
        self.frames.append({**kw, "completed_physics_steps": self.backend.steps})

    def after_render(self):
        pass

    def require_healthy(self):
        pass


class ResetBackend:
    def __init__(self):
        self.steps = self.writes = self.reads = self.renders = 0
        self.state = 0
        self.dispatches = []
        self._controller = None

    def _atomic_apply(self, adapter, command, **kwargs):
        self.writes += 1
        self.dispatches.append((tuple(command), copy.deepcopy(kwargs)))
        return {"physics_tick": kwargs["physics_tick"], "applied_full12": list(command),
                "drive_feedback_bias_requested_full12": list(kwargs["drive_feedback_bias_full12"])}

    def render_video_frame(self):
        self.renders += 1  # Rendering cannot advance this physical state.


def reset_core(backend, *, fail_tick=None, iterations=180, nonzero_tick=None):
    core = NS(backend=backend)
    def reset(*, seed):
        for tick in range(iterations):
            if tick == fail_tick:
                raise RuntimeError("native reset failed")
            command = video.ZERO12 if tick != nonzero_tick else (1.,) + video.ZERO12[1:]
            backend._atomic_apply(None, command, physics_tick=tick,
                tracking_servo_names=(), drive_feedback_bias_full12=video.ZERO12)
            backend.steps += 1
            backend.state += backend.steps * 17
        backend.reads += 1
        backend._controller = NS(physics_tick=0, state_id="P01", steps=1)
        core.frame = NS(physics_tick=0, state_id="P01", info={})
        core.observation = (backend.state,)
        core.decision_count, core.done = 0, False
    core.reset = reset
    return core


def test_settle_tail_matches_plain_reset_and_keeps_180_writes_steps_one_sensor_read():
    plain, captured = ResetBackend(), ResetBackend()
    reset_core(plain).reset(seed=4001)
    core = reset_core(captured)
    recorder, stream = Recorder(captured), io.StringIO()
    evidence = video.reset_with_existing_settle_tail(core, recorder, stream, seed=4001)
    assert captured.dispatches == plain.dispatches
    assert (captured.steps, captured.writes, captured.reads, captured.state) == (
        plain.steps, plain.writes, plain.reads, plain.state) == (180, 180, 1, 276930)
    assert captured._controller.steps == 1 and captured._controller.physics_tick == 0
    assert "_atomic_apply" not in vars(captured)
    assert evidence["extra_physics_ticks"] == evidence["extra_sensor_reads"] == 0
    assert [(row["sim_step"], row["completed_physics_steps"]) for row in recorder.frames] == [
        (index, 116 + index) for index in range(0, 65, 8)]
    assert [row["sim_time_s"] for row in recorder.frames] == [index/120 for index in range(0,65,8)]
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["physical_hold"]["physics_tick"] for row in rows] == list(range(116,180))
    assert all(not row["task_credit"] and not row["sensor_observation_sampled"] for row in rows)
    video.validate_existing_settle_evidence({"pre_action_source": "existing_reset_settle_tail",
        "extra_pre_action_physics_ticks": 0, "settle_capture_evidence": evidence}, rows)


@pytest.mark.parametrize("fault,match", [({"fail_tick":130}, "native reset failed"),
    ({"iterations":179}, "incomplete"), ({"iterations":181}, "reordered"),
    ({"nonzero_tick":117}, "zero-command")])
def test_settle_observer_failure_restores_existing_instance_override(fault, match):
    backend = ResetBackend()
    saved = backend._atomic_apply
    backend._atomic_apply = saved
    with pytest.raises((RuntimeError,video.SemanticVideoError), match=match):
        video.reset_with_existing_settle_tail(reset_core(backend, **fault), Recorder(backend),
                                              io.StringIO(), seed=4001)
    assert vars(backend)["_atomic_apply"] is saved


@pytest.mark.parametrize("kind", ["legacy", "semantic"])
def test_real_backend_reset_loop_observed_without_extra_dispatch_or_controller_refresh(kind, monkeypatch):
    from test_isaac_fsm_backend import FakeRuntime, _backend
    from test_semantic_backend import SemanticFrameStub
    from wlr50_clean.ppo import semantic_backend
    monkeypatch.setattr(semantic_backend, "_sha256_file", lambda path:"a"*64)
    def build():
        runtime = FakeRuntime()
        backend = _backend(runtime) if kind == "legacy" else semantic_backend.SemanticIsaacBackend(
            dependencies=runtime.dependencies(), controller_factory=lambda *args:SemanticFrameStub())
        core = NS(backend=backend)
        def reset(*, seed):
            core.frame = backend.reset(seed=seed, options={})
        core.reset = reset
        return runtime, backend, core
    plain_runtime, plain_backend, plain_core = build()
    plain_core.reset(seed=4001)
    runtime, backend, core = build()
    # The real render method remains in use; the recorder only observes it.
    frames=[]
    recorder=NS(start=lambda:True, before_render=lambda **kw:frames.append((kw,runtime.sim.step_count)),
                after_render=lambda:None, require_healthy=lambda:None)
    video.reset_with_existing_settle_tail(core,recorder,io.StringIO(),seed=4001)
    assert runtime.sim.step_count == plain_runtime.sim.step_count == 180
    assert runtime.adapter.write_count == plain_runtime.adapter.write_count == 180
    assert runtime.reader_count == plain_runtime.reader_count
    assert [event[0] for event in runtime.events if event[0] != "sim.render"] == [
        event[0] for event in plain_runtime.events]
    assert core.frame.physics_tick == plain_core.frame.physics_tick == 0
    assert backend._controller_frame.physics_tick == plain_backend._controller_frame.physics_tick == 0
    assert core.frame.info["raw_observation"] == plain_core.frame.info["raw_observation"]
    assert [steps for _,steps in frames] == list(range(116,181,8))


@pytest.mark.parametrize("role", ["A", "B", "C"])
def test_v3_backend_and_core_selection_keeps_A_actions_unchanged(role, monkeypatch):
    from wlr50_clean.ppo import isaac_fsm_backend, residual_direct_env, semantic_backend, semantic_env
    observed = []
    def constructor(kind):
        def build(*args, **kwargs):
            value = NS(kind=kind, args=args, kwargs=kwargs)
            observed.append(value)
            return value
        return build
    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", constructor("A_backend"))
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", constructor("A_core"))
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", constructor("semantic_backend"))
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", constructor("semantic_core"))
    core = cli.build_video_core(object(), role=role, semantic_version="v3")
    backend = core.args[0]
    assert backend.kwargs["audit_actuator_target_effect"] is True
    if role == "A":
        assert core.kind == "A_core" and backend.kind == "A_backend"
        assert backend.kwargs == {"audit_actuator_target_effect": True}
        assert core.kwargs == {"collect_trace": False}
    else:
        paths = video.video_configuration("v3")
        assert backend.kwargs["execution_profile"] == paths["execution_profile"]
        assert backend.kwargs["task_spec_path"] == paths["task_spec_path"]
        assert core.kwargs["action_config"] == paths["execution_profile"]
        assert core.kwargs["reward_config_path"] == paths["reward_config_path"]
        assert core.kwargs["observation_schema_path"] == paths["observation_schema_path"]


def test_default_v2_core_keeps_implicit_old_options(monkeypatch):
    from wlr50_clean.ppo import semantic_backend, semantic_env
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda app, **kwargs:NS(kwargs=kwargs))
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda backend, **kwargs:NS(backend=backend,kwargs=kwargs))
    core = cli.build_video_core(None, role="B", semantic_version="v2")
    assert core.backend.kwargs == {"audit_actuator_target_effect": True}
    assert core.kwargs == {"collect_trace": False}


def test_v3_configuration_replay_binds_task_quality_and_policy_files():
    paths = video.video_configuration("v3")
    source = {"semantic_version":"v3", "runtime_contract":{"semantic_version":"v3"},
              "evaluation_configuration": {key:video.file_record(path) for key,path in paths.items()}}
    assert video.validate_video_configuration(source) == paths
    source["evaluation_configuration"]["task_spec_path"] = video.file_record(video.video_configuration("v2")["task_spec_path"])
    with pytest.raises(video.SemanticVideoError, match="configuration"):
        video.validate_video_configuration(source)


@pytest.mark.parametrize("mutate", [lambda source:source.update(semantic_version="v2"),
    lambda source:source.update(extra_pre_action_physics_ticks=64),
    lambda source:source["settle_capture_evidence"].update(extra_sensor_reads=1)])
def test_v3_evidence_rejects_old_version_extra_warmup_or_hidden_reader(mutate):
    backend=ResetBackend(); stream=io.StringIO()
    evidence=video.reset_with_existing_settle_tail(reset_core(backend),Recorder(backend),stream,seed=4001)
    source={"semantic_version":"v3", "runtime_contract":{"semantic_version":"v3"},
        "evaluation_configuration":{key:video.file_record(path) for key,path in video.video_configuration("v3").items()},
        "pre_action_source":"existing_reset_settle_tail", "extra_pre_action_physics_ticks":0,
        "settle_capture_evidence":evidence}
    mutate(source)
    with pytest.raises(video.SemanticVideoError):
        video.validate_video_configuration(source)
        video.validate_existing_settle_evidence(source,[json.loads(line) for line in stream.getvalue().splitlines()])


@pytest.mark.parametrize("headroom_mode", [None, "same_tick_post_mapper_servo_margin_v1"])
def test_v3_post_hold_reuses_independent_residual_above_old_controller_limit(monkeypatch, headroom_mode):
    import torch
    from test_actuator_target_effect import _adapter
    from test_semantic_video import post_backend
    from test_semantic_residual_adapter import plan, dispatch
    from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
    adapter = _adapter()
    actuation = plan((15.,-15.)*4+(.3,)*4, controller=(.25,-.25)*4+(.02,)*4)
    for tick in range(1,49):
        ack = dispatch(adapter,actuation,tick)
    expected = adapter.robot._joint_pos_target_sim.clone()
    backend, _ = post_backend(monkeypatch)
    backend._adapter = adapter
    backend._policy_headroom_mode = headroom_mode
    backend._last_atomic_ack = actuation.annotate_ack(ack)
    backend._atomic_apply = lambda *args,**kwargs:IsaacFSMBackend._atomic_apply(backend,*args,**kwargs)
    result={"success":True,"termination_reason":None}
    for index in (1,2):
        row=video.common_post_success_tick(backend,NS(snapshot=result,observe=lambda raw:result),
                                          episode_ticks=31,post_index=index)
        assert torch.equal(adapter.robot._joint_pos_target_sim,expected)
        assert torch.count_nonzero(adapter.robot._joint_vel_target_sim) == 0
        assert row["atomic_ack"]["bounded_controller_bias_requested_full12"] == [.25,-.25]*4+[0.]*4
        assert row["atomic_ack"]["independent_policy_residual_requested_full12"] == [15.,-15.]*4+[0.]*4
        assert row["atomic_ack"].get("policy_headroom_mode") == headroom_mode
    assert adapter.write_count == 50
    assert adapter.robot.events == (["position.setter","velocity.setter","dispatch"]*48
        + ["position.setter","velocity.setter","dispatch","update"]*2)


@pytest.mark.parametrize("success", [False,True])
def test_comparison_titles_show_actual_task_outcome_never_unproven_improvement(success):
    source={"role":"C","success_candidate":success,"diagnostic_only":not success,
            "physical_episode":{"task_success":success}}
    title=video.comparison_title(source)
    assert "improved" not in title.lower()
    assert ("task SUCCESS" in title) is success
    assert ("NOT SUCCESS" in title) is not success


def test_comparison_pads_only_completed_shorter_source_without_stretching():
    left={"role":"A","episode_physics_ticks":1000,"success_candidate":True,
          "diagnostic_only":False,"physical_episode":{"task_success":True}}
    right={**left,"role":"C","episode_physics_ticks":1100}
    filters,frames=video.comparison_filter(left,right)
    assert frames == video.frames_for_episode(1100)
    assert "tpad=stop_mode=clone:stop=12" in filters
    assert "SOURCE COMPLETED - last frame held" in filters
    assert "PPO candidate - task SUCCESS" in filters
    assert "improved" not in filters and "setpts" not in filters
    assert "fps=" not in filters and "minterpolate" not in filters


def test_success_publication_filename_does_not_imply_improvement():
    assert video.publication_name_allowed("A","fsm_baseline_clean.mp4")
    assert video.publication_name_allowed("C","ppo_success_clean.mp4")
    assert not video.publication_name_allowed("C","ppo_improved_clean.mp4")


def test_video_powershell_selects_v3_directories_and_shared_process_lock():
    root=Path(__file__).resolve().parents[2]
    script=(root/"scripts/run_semantic_video.ps1").read_text()
    assert "[ValidateSet('v2','v3')][string]$SemanticVersion = 'v2'" in script
    assert "runs\\ppo_semantic_$SemanticVersion\\video_eval" in script
    assert "'--semantic-version',$SemanticVersion" in script
    assert "runs\\ppo_semantic_v2\\.single_process.lock" in script
