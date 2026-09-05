"""Independent A/B/C video capture; no optimizer or frozen-controller mutation.

All source files are immutable. Raw capture remains neutral/diagnostic until
physical success, real post-roll and independent decode/clock checks pass.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import uuid

from wlr50_clean.infrastructure.video_capture import (
    ActiveViewportVideoRecorder, find_ffmpeg, sha256_file, validate_mp4,
)
from wlr50_clean.evaluation.video_timeline import (
    decode_frame_timeline, load_viewport_frame_ledger, plan_action_window,
    verify_native_rate_output,
)
from .semantic_legacy_evaluation import (
    PhysicalEvaluationRecorder, measured_observation, physical_json,
)
from .semantic_supervisor import SemanticControllerAdapter, TaskEvaluator
from .semantic_training import verified_native_effect, write_json

HZ, FPS, STRIDE = 120, 15, 8
PRE_TICKS, POST_TICKS, MAX_FRAMES = 64, 184, 3000
CAMERA = {"eye_m": [1.45, -1.25, .8], "target_m": [.45, 0., .12]}
ROLES = {"A": "legacy_fsm_eval", "B": "semantic_prior_eval",
         "C": "semantic_residual_eval"}
ZERO12 = (0.,) * 12


class SemanticVideoError(RuntimeError):
    pass


class _PhysicalEndpoint(Exception):
    """Unwind only this capture's incomplete final decision, not the controller."""


def require(condition, message):
    if not condition:
        raise SemanticVideoError(message)


def jsonl(stream, row):
    stream.write(json.dumps(physical_json(row), allow_nan=False,
                            separators=(",", ":")) + "\n")
    stream.flush()


def frames_for_episode(episode_ticks):
    require(type(episode_ticks) is int and episode_ticks > 0, "bad episode tick count")
    return 1 + (PRE_TICKS + episode_ticks + POST_TICKS) // STRIDE


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def inside(root, record):
    path = Path(record["path"])
    # Reject Windows junctions as well as ordinary symlinks.
    for component in (path, *path.parents):
        metadata = component.lstat()
        require(not stat.S_ISLNK(metadata.st_mode)
                and not (getattr(metadata, "st_file_attributes", 0)
                         & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                "reparse/symlink video evidence")
    path = path.resolve(strict=True)
    require(path.is_relative_to(root.resolve()), "video evidence escaped its source")
    require(file_record(path) == dict(record), "video evidence hash/size changed")
    return path


def capture_frame(recorder, backend, global_tick):
    require(global_tick % STRIDE == 0, "render must use native 15 Hz grid")
    recorder.before_render(sim_step=global_tick, sim_time_s=global_tick / HZ)
    backend.render_video_frame()
    recorder.after_render()
    recorder.require_healthy()


def refresh_semantic_core_after_preroll(core):
    """Reset only episode-local sensing/controller/encoding, never physics.

    This is a video-only legal pre-episode initialization. A uses its existing
    legacy refresh method and never enters this function.
    """
    backend = core.backend
    require(core.decision_count == 0 and core.frame.physics_tick == 0
            and core.frame.state_id == "P01" and not core.done,
            "semantic pre-roll refresh requires fresh unstepped P01")
    generation = backend._committed_reset_generation
    reset_count = backend._reset_count
    previous_adapter = backend._adapter
    refreshed = backend.refresh_video_pre_action_frame()
    raw = refreshed.info["raw_observation"]
    from .semantic_backend import CONFIG_ROOT
    controller = SemanticControllerAdapter.from_paths(
        backend.fsm_path, backend.motion_contract_path,
        task_spec_path=CONFIG_ROOT / "stage_task_spec.yaml")
    controller_frame = controller.step(raw, sim_time_s=0.)
    require(controller_frame.state_id == "P01" and controller_frame.physics_tick == 0,
            "post-roll physical state did not initialize natural P01")
    backend._controller = controller
    backend._controller_frame = controller_frame
    frame = backend._build_authoritative_frame(raw, controller_frame, previous_frame=None)
    frame = replace(frame, info={**dict(frame.info),
        "video_pre_action_refresh": refreshed.info["video_pre_action_refresh"],
        "semantic_video_initialization": {
            "task_evaluator_recreated_from_current_tick_zero": True,
            "prefix_credit_imported": False, "physical_reset_performed": False}})
    backend._authoritative_frame = frame
    require(backend._adapter is previous_adapter
            and backend._committed_reset_generation == generation
            and backend._reset_count == reset_count, "pre-roll reset physics/mapper")
    from .semantic_observation import HISTORY_GROUPS, vector, semantic_task
    nominal = vector(frame.nominal_action_full12, 12, "refreshed nominal")
    drive = vector(frame.info["drive_target_full12"], 12, "refreshed actual drive")
    core.observation_builder.reset()
    core._history = {name: ZERO12 for name in HISTORY_GROUPS}
    core._history.update(previous_applied_full12=drive,
                        previous_previous_applied_full12=drive,
                        previous_nominal_full12=nominal)
    core._semantic_frame = core.observation_builder.build(frame, core._history)
    core.bridge.reset(state_id=frame.state_id, applied_action_full12=nominal)
    core.frame = frame
    core.observation = core.observation_schema.encode(core._semantic_frame.groups)
    core.trace = []
    core._episode_return = 0.
    core._transition_evidence_count = len(semantic_task(frame)["transition_evidence"])
    return tuple(core.observation), dict(frame.info)


def post_success_hold_receipt(ack):
    """Keep the realized episode's exact mapper/bias split, stopping wheels."""
    from .isaac_fsm_backend import _full12
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
    required = {"fsm_nominal_mapper_input_full12", "combined_post_mapper_bias_full12",
                "applied_full12", "drive_feedback_bias_requested_full12", "tracking_servo_names"}
    require(ack.get("ppo_actuation_contract") == "frozen_nominal_plus_post_mapper_residual.v1"
            and required <= ack.keys(), "post-roll lacks a verified last actuation receipt")
    nominal = _full12(ack["fsm_nominal_mapper_input_full12"], "last nominal mapper input")
    bias = _full12(ack["combined_post_mapper_bias_full12"], "last combined post-mapper bias")
    require(_full12(ack["applied_full12"], "last applied mapper command") == nominal
            and _full12(ack["drive_feedback_bias_requested_full12"], "last adapter bias") == bias,
            "post-roll receipt differs from the last actual adapter input")
    tracking = tuple(ack["tracking_servo_names"])
    require(len(set(tracking)) == len(tracking) and set(tracking) <= set(SERVO_ORDER),
            "post-roll receipt tracking names are invalid")
    return nominal[:8] + (0.,)*4, bias[:8] + (0.,)*4, tracking


def common_post_success_tick(backend, evaluator, *, episode_ticks, post_index):
    """One physical context tick authorized by the SAME independent evaluator.

    Does not change backend._done, any controller lifecycle/result, controller
    tick, episode tick or original controller object. Mapper/physics continue.
    """
    require(evaluator.snapshot.get("success") is True, "post-roll lacks common success")
    require(1 <= post_index <= POST_TICKS, "post-roll outside declared length")
    from .isaac_fsm_backend import _require_running, _validate_sensor_contract
    scene, adapter, reader = backend._scene, backend._adapter, backend._reader
    source = backend._controller_frame
    invariant = (id(backend._controller), id(source), backend._episode_tick,
                 backend._done, backend._controller.physics_tick)
    require(backend._episode_tick == episode_ticks, "episode tick changed after stop")
    _require_running(scene, "semantic physical video post-roll")
    previous_ack = dict(backend._last_atomic_ack)
    physical_tick = int(previous_ack["physics_tick"]) + 1
    # Repeat the last actual dispatch receipt, not the incoming controller
    # frame or _previous_action_full12 (which already contains PPO residual).
    # Retaining the nominal request prevents mapper compensation reset; the
    # exact combined bias keeps PPO strictly on its original post-mapper path.
    hold, bias, tracking = post_success_hold_receipt(previous_ack)
    ack = backend._atomic_apply(adapter, hold, physics_tick=physical_tick,
        tracking_servo_names=tracking, drive_feedback_bias_full12=bias)
    # The next hold must use the same receipt contract, not an unannotated ack.
    ack = {**ack,
        "ppo_actuation_contract": previous_ack["ppo_actuation_contract"],
        "fsm_nominal_mapper_input_full12": list(hold),
        "combined_post_mapper_bias_full12": list(bias),
        "video_post_success_hold": {"source_ack_physics_tick": previous_ack["physics_tick"],
            "servo_nominal_and_post_mapper_bias_preserved": True,
            "nominal_and_bias_wheels_explicitly_zero": True}}
    scene.sim.step(render=False)
    adapter.update_readback()
    logical_tick = backend._episode_sensor_tick_offset + episode_ticks + post_index
    raw = reader.read(physics_tick=logical_tick,
                      simulation_time_s=logical_tick/HZ,
                      commanded_full12=ack["drive_target_full12"])
    _validate_sensor_contract(raw, backend._dependencies.expected_contact_bodies,
                              require_finite=True)
    backend._raw_observation = raw
    backend._last_atomic_ack = ack
    backend._video_post_terminal_tick_count = post_index
    result = evaluator.observe(raw)
    require(invariant == (id(backend._controller), id(backend._controller_frame),
            backend._episode_tick, backend._done, backend._controller.physics_tick),
            "post-roll mutated controller or spoofed its terminal state")
    require(result.get("success") is True and result.get("termination_reason") is None,
            "physical post-success stability was lost")
    return {"post_index": post_index, "physical_command_tick": physical_tick,
            "task_credit": False, "root_state_write_count": 0,
            "controller_step_count": 0, "atomic_ack": ack,
            "raw_observation": measured_observation(raw),
            "physical_evaluation": result}


class EndpointObserver:
    """Common physical metrics first; exact-tick endpoint then native-grid render."""
    def __init__(self, physical, recorder, backend):
        self.physical, self.recorder, self.backend = physical, recorder, backend
        self.last_frame = None
        self.last_global_tick = 0
        self.reason = None

    def __call__(self, before, after, projection):
        self.physical.observe(before, after, projection)
        self.last_frame = after
        self.last_global_tick = PRE_TICKS + after.physics_tick
        if self.last_global_tick % STRIDE == 0:
            capture_frame(self.recorder, self.backend, self.last_global_tick)
        result = self.physical.evaluator.snapshot
        if result.get("success") is True or result.get("termination_reason") is not None:
            self.reason = "SUCCESS" if result.get("success") else result["termination_reason"]
            raise _PhysicalEndpoint
        if frames_for_episode(after.physics_tick) > MAX_FRAMES:
            self.reason = "VIDEO_FULL_CONTEXT_EXCEEDS_200S"
            raise _PhysicalEndpoint


def capture_semantic_video(core, *, role, seed, output_directory, contract,
                           policy_loader=None, recorder_factory=ActiveViewportVideoRecorder):
    """Capture one fresh P01 episode in one live process.

    policy_loader(refreshed_observation) returns (deterministic_action_callable,
    load_provenance, unchanged_model_assertion). It must use the official
    load_semantic_checkpoint; C without that loader is rejected. A/B use zero.
    """
    require(role in ROLES and seed == 4001, "wrong role or locked video seed")
    require((role == "C") == (policy_loader is not None), "wrong checkpoint role")
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    recorder = recorder_factory(root)
    backend = core.backend
    physical = None
    observer = None
    roll = None
    decisions = None
    recorder_manifest = None
    physical_summary = None
    load_provenance = None
    check_model = lambda: None
    issued_decisions, completed_decisions, partial_ticks = 0, 0, 0
    error = None
    reset_info = {}
    try:
        core.reset(seed=seed)
        require(core.frame.state_id == "P01" and core.frame.physics_tick == 0
                and core.decision_count == 0 and not core.done, "video did not reset to P01")
        reset_info = dict(core.frame.info)
        require(reset_info.get("reset_count") == 1
                and reset_info.get("reset_options") == {}
                and reset_info.get("training_phase_snapshot") is None,
                "video must own the first natural reset of a fresh process")
        camera = reset_info["locked_scene_snapshot"]["camera"]
        require(all(list(camera[key]) == value for key,value in CAMERA.items()),
                "camera differs from common A/B/C view")
        legacy_controller = backend._controller if role == "A" else None
        roll = (root/"physical_video_roll_ticks.jsonl").open("x", encoding="utf-8")
        decisions = (root/"video_policy_decisions.jsonl").open("x", encoding="utf-8")
        for _ in range(3):
            backend.render_video_frame()  # Shader warmup, no physics or encoded time.
        require(recorder.start(), "viewport capture did not start")
        capture_frame(recorder, backend, 0)
        for index in range(1, PRE_TICKS+1):
            hold = backend.advance_video_pre_action_tick()
            require(hold["root_state_write_count"] == 0, "pre-roll root write")
            jsonl(roll, {"kind": "pre_action", "global_tick": index, "task_credit": False,
                        "physical_hold": hold,
                        "raw_observation": measured_observation(backend.raw_observation)})
            if index % STRIDE == 0:
                capture_frame(recorder, backend, index)
        if role == "A":
            observation, refreshed = core.refresh_after_video_pre_action_hold()
            require(backend._controller is legacy_controller,
                    "legacy controller was replaced during video pre-roll")
        else:
            observation, refreshed = refresh_semantic_core_after_preroll(core)
        require(core.frame.physics_tick == 0 and core.frame.state_id == "P01",
                "pre-roll did not preserve logical P01 tick zero")
        reset_info = dict(refreshed)
        physical = PhysicalEvaluationRecorder(root)
        physical.start(core.frame)  # Prefix ticks cannot grant lift/cross/task credit.
        if policy_loader is not None:
            action, load_provenance, check_model = policy_loader(tuple(observation))
            require(load_provenance.get("checkpoint_loaded_and_verified") is True,
                    "C source lacks actual checkpoint load proof")
        else:
            action = lambda _observation, _decision: ZERO12
        observer = EndpointObserver(physical, recorder, backend)
        if role == "A":
            core.tick_callback = observer
        else:
            core.tick_observer = observer
        while not core.done:
            before_tick = backend._episode_tick
            phase = core.frame.state_id
            raw = tuple(float(value) for value in action(observation, issued_decisions))
            require(len(raw) == 12 and all(math.isfinite(value) for value in raw),
                    "policy did not return finite Full12")
            if role in ("A", "B"):
                require(raw == ZERO12, "baseline role emitted residual")
            if role == "A":
                backend.set_actuator_target_audit_request(phase, raw,
                    core.phase_actions.mask_for(phase))
            issued_decisions += 1
            try:
                step = core.step(raw)
            except _PhysicalEndpoint:
                partial_ticks = backend._episode_tick-before_tick
                jsonl(decisions, {"decision": issued_decisions, "request_phase": phase,
                    "raw_policy_action_full12": raw, "start_tick": before_tick,
                    "end_tick": backend._episode_tick, "physics_ticks": partial_ticks,
                    "environment_step_returned": False, "stop_reason": observer.reason})
                break
            completed_decisions += 1
            observation = tuple(step.observation)
            jsonl(decisions, {"decision": issued_decisions, "request_phase": phase,
                "raw_policy_action_full12": raw, "start_tick": before_tick,
                "end_tick": backend._episode_tick,
                "physics_ticks": backend._episode_tick-before_tick,
                "environment_step_returned": True, "step_info": step.info})
        endpoint = observer.last_frame
        require(endpoint is not None, "video had no episode physics")
        physical_summary = physical.summary()  # Metrics end before all post-roll.
        require(physical_summary["task_success"] is True, "episode did not meet common physical task")
        require(observer.reason == "SUCCESS", "video was stopped for another reason")
        require(frames_for_episode(endpoint.physics_tick) <= MAX_FRAMES,
                "full physical context would exceed 200 seconds")
        for index in range(1, POST_TICKS+1):
            row = common_post_success_tick(backend, physical.evaluator,
                episode_ticks=endpoint.physics_tick, post_index=index)
            global_tick = PRE_TICKS + endpoint.physics_tick + index
            jsonl(roll, {"kind": "post_success", "global_tick": global_tick, **row})
            if global_tick % STRIDE == 0:
                capture_frame(recorder, backend, global_tick)
        check_model()  # Actor/critic/optimizer/normalizer hashes must be unchanged.
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if physical is not None:
            physical.close()
        if roll is not None:
            roll.close()
        if decisions is not None:
            decisions.close()
        recorder_manifest = recorder.finalize()  # MUST precede SimulationApp.close.
    endpoint = None if observer is None else observer.last_frame
    if not recorder_manifest.get("valid"):
        error = error or "recorder/source full-decode failed"
    names = ("actual_viewport_video.mp4", "viewport_frame_ledger.jsonl",
             "viewport_buffer_video_manifest.json", "viewport_first_frame.png",
             "viewport_last_frame.png", "physical_video_roll_ticks.jsonl",
             "video_policy_decisions.jsonl", "physical_observations.jsonl",
             "native_tick_audit.jsonl", "stage_transition_evidence.jsonl",
             "phase_metrics.csv", "physics_quality_metrics.csv")
    payload = {"schema": "wlr50_clean.semantic_video_source.v1",
        "role": role, "mode": ROLES[role], "seed": seed, "runtime_contract": dict(contract),
        "capture_process_id": os.getpid(), "capture_process_instance_id": uuid.uuid4().hex,
        "fresh_process_single_episode": True, "episode_count": 1,
        "from_phase": "P01", "pre_action_ticks": PRE_TICKS,
        "requested_post_success_ticks": POST_TICKS,
        "performed_post_success_ticks": backend._video_post_terminal_tick_count,
        "episode_physics_ticks": None if endpoint is None else endpoint.physics_tick,
        "issued_policy_decisions": issued_decisions,
        "completed_environment_steps": completed_decisions,
        "interrupted_final_decision_ticks": partial_ticks,
        "camera": {**CAMERA, "resolution": [1280,720], "fps": FPS},
        "reset_evidence": {key: reset_info.get(key) for key in
            ("reset_count", "reset_options", "training_phase_snapshot",
             "video_pre_action_refresh", "semantic_video_initialization")},
        "checkpoint_load_provenance": load_provenance,
        "physical_episode": physical_summary, "optimizer_updates": 0,
        "success_candidate": error is None,
        "diagnostic_only": error is not None, "improved_claim": False,
        "source_acceptance_error": error,
        "stitched": False, "frame_interpolation": False, "speed_modified": False,
        "artifacts": {name: file_record(root/name) for name in names if (root/name).is_file()}}
    write_json(root/"semantic_video_source_manifest.json", payload)
    return payload


def validate_semantic_video_source(root, *, expected_role=None):
    """Independently replay physical task and bind source frames to native PTS."""
    root = Path(root).resolve(strict=True)
    source = json.loads((root/"semantic_video_source_manifest.json").read_text())
    require(source["schema"] == "wlr50_clean.semantic_video_source.v1", "source schema")
    require(source["role"] in ROLES and source["mode"] == ROLES[source["role"]], "source role")
    require(expected_role is None or source["role"] == expected_role, "role mismatch")
    require(source["success_candidate"] is True and source["diagnostic_only"] is False,
            "failed diagnostic is not publishable success")
    require(source["improved_claim"] is False and source["optimizer_updates"] == 0,
            "capture cannot establish PPO improvement or update weights")
    require(source["seed"] == 4001 and source["episode_count"] == 1
            and source["fresh_process_single_episode"] is True, "not fresh paired video source")
    require(source["camera"] == {**CAMERA, "resolution": [1280,720], "fps": FPS}, "camera mismatch")
    require(all(source[key] is False for key in ("stitched","frame_interpolation","speed_modified")),
            "source timeline was synthesized")
    require(source["pre_action_ticks"] == PRE_TICKS
            and source["performed_post_success_ticks"] == POST_TICKS, "incomplete real context")
    paths = {name: inside(root, record) for name,record in source["artifacts"].items()}
    endpoint = source["episode_physics_ticks"]
    require(frames_for_episode(endpoint) <= MAX_FRAMES, "video context exceeds 200 s")
    from .semantic_cli import runtime_contract
    require(runtime_contract(expected_head=source["runtime_contract"]["source_git_commit"])
            == source["runtime_contract"], "validator runtime/task spec differs from capture")
    if source["role"] == "C":
        require(source["checkpoint_load_provenance"]["checkpoint_loaded_and_verified"] is True,
                "C checkpoint load proof missing")
    evaluator = TaskEvaluator()
    count = 0
    with paths["physical_observations.jsonl"].open(encoding="utf-8") as stream:
        for count,line in enumerate(stream, 1):
            raw = json.loads(line)
            require(raw["physics_tick"] == count-1, "raw episode tick gap")
            evaluation = evaluator.observe(raw)
            require(count-1 == endpoint or not evaluation["success"], "episode continued after first success")
    require(count == endpoint+1 and evaluation["success"] is True
            and evaluation["termination_reason"] is None, "independent task replay did not succeed")
    decision_rows = [json.loads(line) for line in paths["video_policy_decisions.jsonl"].read_text().splitlines()]
    require(len(decision_rows) == source["issued_policy_decisions"], "decision ledger count")
    expected_start = 0
    completed_steps = 0
    for index, row in enumerate(decision_rows, 1):
        require(row["decision"] == index and row["start_tick"] == expected_start,
                "policy decision clock gap")
        require(1 <= row["physics_ticks"] <= STRIDE
                and row["end_tick"]-row["start_tick"] == row["physics_ticks"],
                "bad partial decision duration")
        if row["environment_step_returned"]:
            completed_steps += 1
        else:
            require(index == len(decision_rows) and row["stop_reason"] == "SUCCESS",
                    "non-final interrupted decision")
        expected_start = row["end_tick"]
    require(expected_start == endpoint and completed_steps == source["completed_environment_steps"],
            "decision ledger endpoint mismatch")
    require(decision_rows[-1]["physics_ticks"] == source["interrupted_final_decision_ticks"]
            and not decision_rows[-1]["environment_step_returned"], "partial endpoint was hidden")
    count = 0
    with paths["native_tick_audit.jsonl"].open(encoding="utf-8") as stream:
        for count,line in enumerate(stream, 1):
            row = json.loads(line); audit = row["native_audit"]
            require(row["episode_physics_tick"] == count, "native audit gap")
            verified_native_effect({"actuator_target_effect_audit": audit},
                                   tuple(audit["raw_policy_action_full12"]))
            if source["role"] in ("A", "B"):
                require(tuple(audit["raw_policy_action_full12"]) == ZERO12
                        and tuple(row["projected_residual_full12"]) == ZERO12,
                        "baseline source contains a residual")
            require(all(row[key] == 0 for key in (
                "in_episode_root_pose_writes","in_episode_root_velocity_writes",
                "in_episode_force_or_impulse_writes","in_episode_gravity_writes")), "forbidden state write")
    require(count == endpoint, "native audit count mismatch")
    roll_rows = [json.loads(line) for line in paths["physical_video_roll_ticks.jsonl"].read_text().splitlines()]
    pre = [row for row in roll_rows if row["kind"] == "pre_action"]
    post = [row for row in roll_rows if row["kind"] == "post_success"]
    require([row["global_tick"] for row in pre] == list(range(1,PRE_TICKS+1)), "pre-roll tick gap")
    require([row["global_tick"] for row in post] ==
            list(range(PRE_TICKS+endpoint+1, PRE_TICKS+endpoint+POST_TICKS+1)), "post-roll tick gap")
    for row in pre:
        require(row["task_credit"] is False and row["physical_hold"]["root_state_write_count"] == 0
                and tuple(row["physical_hold"]["applied_full12"]) == ZERO12,
                "pre-roll was not a real zero-command no-credit hold")
    for row in post:
        require(row["task_credit"] is False and row["root_state_write_count"] == 0
                and row["controller_step_count"] == 0, "post-roll changed task/control history")
        require(evaluator.observe(row["raw_observation"])["success"] is True,
                "independent post-roll replay lost physical stability")
    decoded = decode_frame_timeline(paths["actual_viewport_video.mp4"])
    ledger = load_viewport_frame_ledger(paths["viewport_frame_ledger.jsonl"])
    require(len(ledger) == frames_for_episode(endpoint), "encoded frame count mismatch")
    require([row.sim_step for row in ledger] == [STRIDE*i for i in range(len(ledger))],
            "frames were dropped/duplicated or a partial terminal tick was mislabeled")
    require(all(abs(row.sim_time_s-row.sim_step/HZ)<1e-10 for row in ledger), "ledger clock drift")
    window = plan_action_window(decoded, ledger,
        semantic_start_sim_s=PRE_TICKS/HZ, semantic_end_sim_s=(PRE_TICKS+endpoint)/HZ,
        expected_fps=FPS, requested_pre_roll_s=PRE_TICKS/HZ,
        requested_post_roll_s=1., maximum_preserved_roll_s=2.)
    require(window.is_full_source, "source context was silently trimmed")
    validation = validate_mp4(paths["actual_viewport_video.mp4"],
        expected_fps=FPS, expected_frame_count=len(ledger), maximum_duration_s=200.,
        require_sane_container_duration=False)
    require(validation["valid"] is True, "raw source did not fully decode")
    return source, window


def publication_name_allowed(role, name):
    # A requested baseline name is allowed only AFTER source validation below.
    # A physically successful C video is still a candidate, never improvement.
    names = {"A": {"fsm_original_baseline.mp4", "semantic_A_success.mp4"},
             "B": {"semantic_B_success.mp4"},
             "C": {"semantic_C_candidate.mp4"}}
    return name in names.get(role, set())


def publish_success_source(source_root, destination):
    source, window = validate_semantic_video_source(source_root)
    destination = Path(destination).resolve()
    require(publication_name_allowed(source["role"], destination.name),
            "only validated baseline/candidate names are supported; improvement is a separate decision")
    require(not destination.exists(), "refusing to overwrite published video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(source["artifacts"]["actual_viewport_video.mp4"]["path"])
    ffmpeg = find_ffmpeg()
    completed = subprocess.run([str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-copyts", "-start_at_zero", "-i", str(raw), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "copy", "-movflags", "+faststart", str(destination)],
        capture_output=True, text=True, errors="replace")
    require(completed.returncode == 0, completed.stderr[-2000:])
    validation = validate_mp4(destination, expected_fps=FPS,
        expected_frame_count=window.expected_frame_count, maximum_duration_s=200.,
        require_sane_container_duration=True)
    require(validation["valid"] is True, "published MP4 duration/codec/PTS failed")
    identity = verify_native_rate_output(decode_frame_timeline(destination), window,
        require_decoded_frame_identity=True, require_exact_pts_delta_identity=True)
    record = {"schema": "wlr50_clean.semantic_video_publication.v1",
        "source_manifest": file_record(Path(source_root)/"semantic_video_source_manifest.json"),
        "video": file_record(destination), "validation": validation,
        "source_identity": identity, "improved_claim": False, "single_episode": True,
        "publication_role": source["role"],
        "policy_claim_status": "candidate_not_improved" if source["role"]=="C" else "baseline_physical_success"}
    write_json(destination.with_suffix(".manifest.json"), record)
    return record
