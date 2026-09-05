"""N=8 semantic PPO kernel with honest whole-batch reset boundaries.

Physics is never advanced through eight SemanticEpisodeEnv.step calls. Each
row owns a semantic builder/bridge/reward history; one coordinator stages all
actions, advances the one physics scene, then consumes all resulting frames.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

from .phase_action_masks_v2 import PhaseTransitionBridge
from .semantic_backend import build_semantic_projector
from .semantic_env import ZERO12, PHYSICS_DT_S, WRITE_COUNTERS, _native_tick_evidence, _terminal_reason
from .semantic_observation import (
    HISTORY_GROUPS, SemanticObservationBuilder, SemanticNonFiniteObservation,
    load_semantic_observation_schema, semantic_task, vector,
)
from .semantic_reward import SemanticRewardCalculator, SemanticRewardSample, load_semantic_reward_config

BARRIER_REASON = "SEMANTIC_VECTOR_PEER_RESET"


class _SemanticRow:
    def __init__(self, *, schema, projector, reward):
        self.schema,self.projector,self.reward = schema,projector,reward
        self.builder = SemanticObservationBuilder(schema)
        self.bridge = PhaseTransitionBridge(projector)
        self.phase_decisions = Counter()
        self.total_decisions = 0
        self.total_ticks = 0

    def reset(self, frame):
        if _terminal_reason(frame) is not None:
            raise ValueError("row reset must be valid natural P01")
        self.builder.reset()
        nominal = vector(frame.nominal_action_full12,12,"reset nominal")
        drive = vector(frame.info["drive_target_full12"],12,"reset drive")
        self.history = dict.fromkeys(HISTORY_GROUPS,ZERO12)
        self.history.update(previous_applied_full12=drive,previous_previous_applied_full12=drive,
                            previous_nominal_full12=nominal)
        self.semantic = self.builder.build(frame,self.history)
        self.frame = frame
        self.bridge.reset(state_id=frame.state_id,applied_action_full12=nominal)
        self.episode_decisions = 0
        self.episode_return = 0.
        self.transition_count = len(semantic_task(frame).get("transition_evidence",()))
        return self.schema.encode(self.semantic.groups)

    def begin(self, raw):
        self.raw = vector(raw,12,"raw policy action")
        self.start_frame,self.start_semantic = self.frame,self.semantic
        self.samples,self.transitions,self.native_ticks = [],[],[]
        self.write_samples = {name:[] for name in WRITE_COUNTERS}
        self.reason = None
        self.finite_fallback = False

    def prepare(self):
        source = self.frame
        return self.bridge.project_tick(self.raw,state_id=source.state_id,
            nominal_action_full12=source.nominal_action_full12,
            reference_action_full12=source.nominal_action_full12,reference_delta_full12=ZERO12,
            runtime_action_mask_full12=source.action_mask_full12,
            safety=source.safety_projection,dt_s=PHYSICS_DT_S)

    def consume(self, current, bridged):
        source,before = self.frame,self.semantic
        if current.physics_tick!=source.physics_tick+1 or not math.isclose(
                current.sim_time_s-source.sim_time_s,PHYSICS_DT_S,rel_tol=0.,abs_tol=1e-8):
            raise ValueError("row physics clock did not advance once")
        projection = bridged.projection
        self.last_projection = projection
        self.reason = _terminal_reason(current)
        drive = vector(current.info["drive_target_full12"],12,"actual drive")
        nominal = vector(source.nominal_action_full12,12,"source nominal")
        residual = vector(projection.safe_projected_residual_full12,12,"projected residual")
        old = dict(self.history)
        self.history.update(previous_raw_full12=self.raw,previous_residual_full12=residual,
            previous_previous_residual_full12=old["previous_residual_full12"],
            previous_applied_full12=drive,previous_previous_applied_full12=old["previous_applied_full12"],
            previous_nominal_full12=nominal)
        try:
            after = self.builder.build(current,self.history)
        except SemanticNonFiniteObservation:
            if self.reason not in ("NAN_INF","PHYSICS_EXPLOSION"):
                raise
            after = before
            self.finite_fallback = True
        caps = tuple(a*b for a,b in zip(self.projector.config.scale_for(source.state_id),
                                        self.projector.config.physical_residual_scale_full12,strict=True))
        self.samples.append(SemanticRewardSample(before,after,PHYSICS_DT_S,nominal,
            old["previous_nominal_full12"],residual,old["previous_residual_full12"],
            drive,old["previous_applied_full12"],old["previous_previous_applied_full12"],caps))
        evidence = _native_tick_evidence(source,current,request_phase=self.start_frame.state_id,raw=self.raw,
            handoff_hold=bool(bridged.transition_metric and bridged.transition_metric.handoff_hold_used))
        if not evidence["verified"]:
            raise ValueError("one vector physics tick lacks verified row-bound native target evidence")
        self.native_ticks.append(evidence)
        for name in WRITE_COUNTERS:
            value = current.info.get(name)
            if type(value) is not int or value!=0:
                raise ValueError(f"missing or forbidden in-episode write evidence: {name}")
            self.write_samples[name].append(value)
        if bridged.transition_metric is not None:
            self.transitions.append(bridged.transition_metric.as_dict())
        self.frame,self.semantic = current,after
        self.total_ticks += 1

    def finish(self, row_index):
        # A peer reset is NOT passed as a task failure to the reward calculator.
        # Its final physical potential remains intact for correct PBRS/bootstrap.
        reward = self.reward.evaluate(self.start_semantic,self.semantic,self.samples,
            termination_reason=self.reason,task_success=self.reason=="SUCCESS")
        self.episode_return += reward["total"]
        self.episode_decisions += 1
        self.total_decisions += 1
        self.phase_decisions[self.start_frame.state_id] += 1
        all_transitions = semantic_task(self.frame).get("transition_evidence",())
        if len(all_transitions)<self.transition_count:
            raise ValueError("semantic transition history shrank during episode")
        new_transitions = list(all_transitions[self.transition_count:])
        self.transition_count = len(all_transitions)
        last = self.samples[-1]
        info = {"schema":"wlr50_clean.semantic_vector_decision.v1","env_index":row_index,
            "phase_id":self.start_frame.state_id,"end_phase_id":self.frame.state_id,
            "physics_tick":self.frame.physics_tick,"sim_time_s":self.frame.sim_time_s,
            "physics_ticks":len(self.samples),"decision_count":self.episode_decisions,
            "raw_policy_action_full12":self.raw,
            "nominal_action_full12":last.nominal,"projected_residual_full12":last.residual,
            "applied_action_full12":self.last_projection.applied_action_full12,
            "actual_drive_target_full12":last.actual_drive,
            "actuator_target_effect_audit":self.frame.info["actuator_target_effect_audit"],
            "actuator_target_effect_audit_ticks":list(self.native_ticks),
            "actuator_target_effect_audit_summary":{
                "physics_ticks":len(self.native_ticks),"verified_tick_count":len(self.native_ticks),
                "all_ticks_verified":True,
                "actual_native_effect_tick_count":sum(x["actual_native_effect"] for x in self.native_ticks),
                "own_phase_request_effect_tick_count":sum(x["own_phase_request_effect"] for x in self.native_ticks)},
            **{name:max(values) for name,values in self.write_samples.items()},
            "no_in_episode_state_writes_verified":True,
            "semantic_task":dict(semantic_task(self.frame)),
            "stage_transition_evidence":new_transitions,
            "phase_transition_action_jump":list(self.transitions),
            "reward":reward,"reward_breakdown":reward,
            "environment_reward":reward["total"],"termination_reason":self.reason,
            "task_success":self.reason=="SUCCESS","task_terminated":self.reason is not None,
            "episode_return":self.episode_return,"episode_elapsed_s":self.frame.sim_time_s,
            "terminal_observation_finite_fallback":self.finite_fallback,
            "terminal_bootstrap_allowed":self.reason is None}
        return self.schema.encode(self.semantic.groups),reward["total"],info


class SemanticVectorRslEnv:
    """Initial N=8/P01-only ABI. True task endings are distinct from peer resets."""
    num_envs = 8
    num_actions = 12
    max_episode_length = 3000

    def __init__(self, backend: Any, *, seeds=tuple(range(1001,1009)),
                 device="cuda:0", tick_observers=None):
        import torch
        if type(getattr(backend,"num_envs",None)) is not int or backend.num_envs!=8:
            raise ValueError("first semantic vector revision requires N=8")
        self.backend,self.device = backend,torch.device(device)
        self.seeds = tuple(seeds)
        if len(self.seeds)!=8 or any(type(x) is not int or x<0 for x in self.seeds):
            raise ValueError("eight nonnegative integer seed rows required")
        self.schema = load_semantic_observation_schema()
        config = load_semantic_reward_config()
        if self.schema.maximum_task_duration_s!=config.values["maximum_task_duration_s"]:
            raise ValueError("observation/reward task horizons differ")
        self.gamma = config.gamma
        self.rows = tuple(_SemanticRow(schema=self.schema,
            projector=build_semantic_projector(backend.execution_profile_path),
            reward=SemanticRewardCalculator(config)) for _ in range(8))
        self.tick_observers = (None,)*8 if tick_observers is None else tuple(tick_observers)
        if len(self.tick_observers)!=8 or any(observer is not None and not all(callable(getattr(observer,name,None)) for name in ("observe","reset","summary")) for observer in self.tick_observers):
            raise ValueError("eight optional observer objects with observe/reset/summary required")
        self._final_value_function = None
        self.episode_length_buf = torch.zeros(8,dtype=torch.long,device=self.device)
        self.completed_episodes = []
        self.peer_truncations = []
        self.total_decisions = 0
        self.reset_count = 0
        self.cfg = {"schema":"wlr50_clean.semantic_vector_env.v1","num_envs":8,
            "observation_dimension":self.schema.dimension,"physics_hz":120,"decision_hz":15,
            "reset_sampling":"P01_full_task_only","independent_subset_reset":False,
            "physical_backend":"one_scene_one_write_one_step",
            "task_deadline_bootstrap":False,
            "peer_reset_bootstrap":"gamma_times_critic_of_actual_final_observation_added_before_reset",
            "upstream_time_outs":False}
        self._reset_all()

    def _encode_batch(self, observations):
        import torch
        from tensordict import TensorDict
        tensor = torch.tensor(observations,dtype=torch.float32,device=self.device)
        if tensor.shape!=(8,self.schema.dimension) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("semantic batch observations changed shape or became nonfinite")
        return TensorDict({"policy":tensor,"critic":tensor.clone()},batch_size=[8],device=self.device)

    def _reset_all(self):
        batch = self.backend.reset_all(seeds=self.seeds)
        if len(batch.frames)!=8 or len({id(frame) for frame in batch.frames})!=8:
            raise ValueError("backend reset did not return eight distinct frames")
        for group in (self.backend.controllers,self.backend.readers):
            if len(group)!=8 or len({id(x) for x in group})!=8:
                raise ValueError("row controller/reader histories are shared")
        self._batch = batch
        for observer in self.tick_observers:
            if observer is not None:
                observer.reset()
        self._observations = self._encode_batch([row.reset(frame) for row,frame in zip(self.rows,batch.frames,strict=True)])
        self.episode_length_buf.zero_()
        self.reset_count += 1

    def bind_final_value_function(self, value_function: Callable, *, gamma: float):
        if not callable(value_function) or float(gamma)!=self.gamma:
            raise ValueError("final-observation critic and task reward gamma must agree")
        self._final_value_function = value_function

    def get_observations(self):
        return self._observations

    def step(self, actions):
        import torch
        if actions.shape!=(8,12) or not bool(torch.isfinite(actions).all()):
            raise ValueError("one finite raw Full12 action per row required")
        raw = tuple(tuple(float(x) for x in row) for row in actions.detach().cpu().tolist())
        for row,action in zip(self.rows,raw,strict=True):
            row.begin(action)
        self.backend.set_actuator_target_audit_requests(
            tuple(row.frame.state_id for row in self.rows),raw,
            tuple(row.projector.config.mask_for(row.frame.state_id) for row in self.rows))
        for _ in range(8):
            previous = self._batch
            projections = tuple(row.prepare() for row in self.rows)
            current = self.backend.step_physics_batch(tuple(x.projection.applied_action_full12 for x in projections))
            if len(current.frames)!=8 or current.physics_tick!=previous.physics_tick+1:
                raise ValueError("physical batch did not advance one tick")
            for name in ("global_physics_step_count","batched_articulation_write_count","exact_pair_capture_count"):
                if getattr(current,name)!=getattr(previous,name)+1:
                    raise ValueError(f"invalid one-step batch counter: {name}")
            for index,(row,frame,projection) in enumerate(zip(self.rows,current.frames,projections,strict=True)):
                source = row.frame
                row.consume(frame,projection)
                observer = self.tick_observers[index]
                if observer is not None:
                    observer.observe(source,frame,projection.projection)
            self._batch = current
            if any(row.reason is not None for row in self.rows):
                break
        finished = [row.finish(index) for index,row in enumerate(self.rows)]
        final_obs = self._encode_batch([x[0] for x in finished])
        environment_rewards = torch.tensor([x[1] for x in finished],device=self.device)
        rewards = environment_rewards.clone()
        infos = [x[2] for x in finished]
        task_done = torch.tensor([row.reason is not None for row in self.rows],dtype=torch.bool,device=self.device)
        barrier = bool(task_done.any())
        peers = ~task_done if barrier else torch.zeros(8,dtype=torch.bool,device=self.device)
        bootstrap = torch.zeros(8,dtype=rewards.dtype,device=self.device)
        if bool(peers.any()):
            if self._final_value_function is None:
                raise RuntimeError("bind live critic before any peer-reset transition")
            # Evaluate BEFORE reset, with unchanged rollout policy/critic weights.
            # Done=True severs GAE; folding gamma*V(final) into reward gives the
            # correct one-step target without RSL's pre-action timeout value.
            with torch.inference_mode():
                values = self._final_value_function(final_obs).detach().reshape(-1).to(self.device)
            if values.shape!=(8,) or not bool(torch.isfinite(values).all()):
                raise ValueError("final-observation critic produced invalid values")
            bootstrap[peers] = self.gamma*values[peers]
            rewards += bootstrap
        self.episode_length_buf += 1
        self.total_decisions += 8
        summaries = []
        for index,info in enumerate(infos):
            info.update(external_peer_reset_truncation=bool(peers[index]),
                bootstrap_credit=float(bootstrap[index]),ppo_storage_reward=float(rewards[index]),
                time_outs=False,bootstrap_source="actual_final_observation_before_reset" if peers[index] else None)
            if barrier:
                summary = {"env_index":index,"seed":self.seeds[index],
                    "policy_decisions":int(self.episode_length_buf[index]),
                    "duration_s":float(self.rows[index].frame.sim_time_s),
                    "task_terminated":bool(task_done[index]),"external_peer_reset_truncation":bool(peers[index]),
                    "termination_reason":info["termination_reason"] if task_done[index] else BARRIER_REASON,
                    "task_success":info["task_success"],"terminal_info":info}
                if self.tick_observers[index] is not None:
                    summary["quality_metrics"] = self.tick_observers[index].summary()
                summaries.append(summary)
                (self.completed_episodes if task_done[index] else self.peer_truncations).append(summary)
        extras = {"semantic_decisions":infos,"episode_summaries":summaries,
            "time_outs":torch.zeros(8,dtype=torch.bool,device=self.device),
            "true_task_terminations":task_done,"external_peer_reset_truncations":peers,
            "environment_rewards":environment_rewards,"peer_bootstrap_credit":bootstrap}
        if barrier:
            extras["terminal_observation"] = final_obs.clone()
            self._reset_all()
        else:
            self._observations = final_obs
        dones = torch.full((8,),barrier,dtype=torch.bool,device=self.device)
        return self.get_observations(),rewards,dones,extras

    def telemetry_summary(self):
        return {"policy_decisions":self.total_decisions,"num_envs":8,
            "reset_count":self.reset_count,"completed_episode_count":len(self.completed_episodes),
            "external_peer_truncation_count":len(self.peer_truncations),
            "success_count":sum(x["task_success"] for x in self.completed_episodes),
            "row_phase_decisions":[dict(row.phase_decisions) for row in self.rows],
            "row_physics_ticks":[row.total_ticks for row in self.rows],
            "sampling":"P01_only_synchronous_full_batch_reset",
            "peer_truncations_are_not_physical_failures":True}
