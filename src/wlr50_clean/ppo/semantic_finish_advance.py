"""Continue the accepted CP232960 package in its post-RL finish only.

Reuses the existing Isaac backend, atomic execution, RSL-RL update and video
loop. No teacher prefix is credited, no simulator state snapshot is injected,
and the entire accepted actor (including the learned RL head) is immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import traceback
from . import semantic_post_rr_front_prep as accepted
from . import semantic_rr_capture_local as old

ROOT=old.ROOT
NAME='ppo_finish_advance_then_home_v1'
CONFIG=ROOT/'configs'/NAME
OUTPUT=ROOT/'outputs'/NAME
SCHEMA='wlr50_clean.frozen_accepted490_finish_checkpoint.v1'
sha,write,line=old.sha,old.write,old.line


def settings():
    return json.loads((CONFIG/'local_training.json').read_text(encoding='utf-8'))


def source_metadata():
    cfg=settings(); path=ROOT/cfg['accepted_checkpoint']
    side=path.with_name(path.stem+'_manifest.json')
    if sha(path)!=cfg['accepted_checkpoint_sha256'] or sha(side)!=cfg['accepted_manifest_sha256']:
        raise ValueError('immutable accepted CP232960 checkpoint/manifest changed')
    meta=json.loads(side.read_text())
    if (not meta['save_load_round_trip'] or not meta['rollout_empty']
            or meta['observation_dimension']!=490
            or meta['runtime_contract']['local_contract']!=accepted.settings()):
        raise ValueError('accepted full490 recipe must stay unchanged')
    return path,meta


def contract(expected_head):
    result=accepted.contract(expected_head)
    _,meta=source_metadata()
    cfg=settings()
    if result['accepted_configuration']!=meta['runtime_contract']['accepted_configuration']:
        raise ValueError('accepted physical/control profiles changed')
    result.update(experiment_id=NAME,local_contract=cfg,
        accepted490_configuration=copy.deepcopy(meta['runtime_contract']['selected_configuration']),
        selected_configuration={'local_training.json':dict(path=(CONFIG/'local_training.json').relative_to(ROOT).as_posix(),
                                                         sha256=sha(CONFIG/'local_training.json'))})
    return result


def tensor_observation(values,device):
    import torch
    from tensordict import TensorDict
    value=torch.tensor([values],dtype=torch.float32,device=device)
    if value.shape!=(1,settings()['observation_dimension']) or not bool(torch.isfinite(value).all()):
        raise ValueError('finite accepted490 plus finish observation required')
    return TensorDict({'policy':value,'critic':value.clone()},batch_size=[1],device=device)


class FinishCore(accepted.PostRRCore):
    def _encode(self):
        original=super()._encode()
        self.observation=tuple(original)+tuple(self.task.finish_observation())
        if len(original)!=490 or len(self.observation)!=settings()['observation_dimension']:
            raise ValueError('accepted490 prefix or declared finish schema changed')
        return self.observation


def build_core(app):
    import yaml
    from .semantic_backend import SemanticIsaacBackend
    from .semantic_env import SemanticEpisodeEnv
    from .semantic_rr_capture_continuation_task import RRCaptureContinuationTaskConfig
    from .semantic_post_rr_front_prep_task import PostRRFrontPrepTask,PostRRFrontPrepTaskConfig
    from .semantic_finish_advance_task import FinishAdvanceTask,FinishAdvanceConfig
    from .semantic_finish_advance_source import controller_factory
    cfg=settings(); control=ROOT/cfg['accepted_control_config']
    prior_cfg=accepted.settings()
    rr_cfg=json.loads((control/'local_training.json').read_text())
    profile=yaml.safe_load((control/'execution_profile.yaml').read_text())
    prior_task=PostRRFrontPrepTask(PostRRFrontPrepTaskConfig(**prior_cfg['preparation']),
        continuation_config=RRCaptureContinuationTaskConfig(**rr_cfg['continuation_task']),
        phase_caps_full12=profile['residual']['phase_caps_full12'])
    task=FinishAdvanceTask(FinishAdvanceConfig(**cfg['requested_finish']),accepted_task=prior_task,
                          phase_caps_full12=profile['residual']['phase_caps_full12'])
    factory=controller_factory(task_spec_path=control/'stage_task_spec.yaml',
        read_local_state=lambda:task.accepted.accepted.snapshot(),read_post_rr_state=task.snapshot,
        finish_advance_config=cfg['requested_finish'])
    backend=SemanticIsaacBackend(app,audit_actuator_target_effect=True,
        execution_profile=control/'execution_profile.yaml',task_spec_path=control/'stage_task_spec.yaml',
        controller_factory=factory,continuation_source_pause=True)
    return FinishCore(SemanticEpisodeEnv(backend,collect_trace=False,
        action_config=control/'execution_profile.yaml',reward_config_path=control/'reward_config.yaml',
        observation_schema_path=control/'observation_schema.json'),task=task)


def make_runner(device,seed=1001,*,saved_configuration=None):
    import torch
    from types import SimpleNamespace
    from .rl_library_wrapper import construct_runner
    cfg=settings(); _,meta=source_metadata()
    configuration=copy.deepcopy(meta['runner_config'])
    configuration['device']=device
    old_actor=copy.deepcopy(configuration['actor'])
    configuration['actor']=dict(class_name='wlr50_clean.ppo.semantic_finish_advance_actor:SemanticFinishAdvanceMLPModel',
        hidden_dims=cfg['actor_hidden_dims'],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','init_std':.15,'std_type':'log'},
        observation_layout=cfg['observation_layout'],frozen_accepted_configuration=old_actor,
        initial_finish_std=cfg['active_initial_sigma_full12'])
    if saved_configuration is not None:
        actual=copy.deepcopy(saved_configuration); actual['device']=device
        checked=copy.deepcopy(actual); checked['actor'].pop('expected_accepted_state_sha256',None)
        if checked!=configuration:
            raise ValueError('saved finish runner configuration mismatch')
        configuration=actual
    initial=[0.]*cfg['observation_dimension']
    env=SimpleNamespace(num_envs=1,num_actions=12,cfg={'finish_task':NAME},device=device,
        get_observations=lambda:tensor_observation(initial,device))
    runner=construct_runner(env,configuration,log_dir=None)
    runner.alg.optimizer=torch.optim.Adam(list(runner.alg.actor.trainable_parameters())+
                                         list(runner.alg.critic.parameters()),lr=meta['learning_rate'])
    runner.alg.learning_rate=meta['learning_rate']
    runner._semantic_policy_version=runner.alg.actor.policy_version
    runner._semantic_front_replay=None
    runner.logger.writer=None
    runner.local_configuration=copy.deepcopy(configuration)
    return runner


def initialize_from_accepted(runner):
    import torch
    from .semantic_training import state_hash
    from .semantic_rr_capture_local_actor import tensor_state_sha256
    from .rl_library_wrapper import restore_training_rng_state
    path,meta=source_metadata()
    source=accepted.make_runner(runner.device,1001,saved_configuration=meta['runner_config'])
    accepted.load(source,path,meta['runtime_contract'])
    old_state=source.alg.save()
    digest=runner.alg.actor.load_accepted_state(old_state['actor_state_dict'],
        expected_sha256=tensor_state_sha256(old_state['actor_state_dict']))
    target_critic=runner.alg.critic.state_dict(); expanded={}; changed=[]
    for name,value in old_state['critic_state_dict'].items():
        target=target_critic[name]
        if value.shape==target.shape:
            expanded[name]=value.clone()
        elif value.ndim==2 and value.shape[1]==490 and target.shape==(value.shape[0],settings()['observation_dimension']):
            expanded[name]=torch.zeros_like(target); expanded[name][:,:490].copy_(value); changed.append(name)
        else:
            raise ValueError('unexpected finish critic migration shape')
    if len(changed)!=1:
        raise ValueError('one critic input expansion required')
    runner.alg.critic.load_state_dict(expanded,strict=True)
    # Keep actual critic Adam moments/steps and all hyperparameters. New finish
    # actor coordinates are new parameters, so their Adam state starts empty;
    # the entire old actor+Adam remains in the immutable source checkpoint.
    old_actor=list(source.alg.actor.trainable_parameters())
    new_actor=list(runner.alg.actor.trainable_parameters())
    old_critic=list(source.alg.critic.named_parameters())
    new_critic=list(runner.alg.critic.named_parameters())
    old_opt=old_state['optimizer_state_dict']; new_opt=runner.alg.optimizer.state_dict()
    old_ids=[x for g in old_opt['param_groups'] for x in g['params']]
    new_ids=[x for g in new_opt['param_groups'] for x in g['params']]
    if len(old_opt['param_groups'])!=1 or len(new_opt['param_groups'])!=1:
        raise ValueError('explicit single Adam group expected')
    groups=copy.deepcopy(old_opt['param_groups']); groups[0]['params']=new_ids
    migrated=dict(state={},param_groups=groups); moment_changes=[]
    for (old_name,op),(new_name,np),oi,ni in zip(old_critic,new_critic,old_ids[len(old_actor):],new_ids[len(new_actor):],strict=True):
        if old_name!=new_name or oi not in old_opt['state']:
            raise ValueError('actual trained critic Adam binding missing')
        state=copy.deepcopy(old_opt['state'][oi])
        for key,value in tuple(state.items()):
            if torch.is_tensor(value) and value.shape==op.shape and op.shape!=np.shape:
                bigger=value.new_zeros(np.shape); bigger[:,:490].copy_(value); state[key]=bigger
                moment_changes.append(new_name+'/'+key)
            elif torch.is_tensor(value) and value.ndim>0 and value.shape!=np.shape:
                raise ValueError('unexpected critic Adam tensor')
        migrated['state'][ni]=state
        collapsed=copy.deepcopy(state)
        if op.shape!=np.shape:
            for key,value in tuple(collapsed.items()):
                if torch.is_tensor(value) and value.shape==np.shape:
                    if torch.count_nonzero(value[:,490:]):
                        raise RuntimeError('new critic Adam columns not zero')
                    collapsed[key]=value[:,:490].clone()
        if state_hash(collapsed)!=state_hash(old_opt['state'][oi]):
            raise RuntimeError('critic Adam source moments/step changed')
    runner.alg.optimizer.load_state_dict(migrated)
    runner.alg.learning_rate=meta['learning_rate']
    runner.local_configuration['actor']['expected_accepted_state_sha256']=digest
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    restore_training_rng_state(meta['training_rng'],expected_seed=1001)
    lineage=dict(schema='wlr50_clean.finish_accepted490_migration.v1',
        accepted_complete_checkpoint=str(path),accepted_complete_sha256=sha(path),
        accepted_manifest_sha256=sha(path.with_name(path.stem+'_manifest.json')),
        accepted_tensor_sha256=digest,accepted_counts=copy.deepcopy(meta['counts']),
        accepted_state_hashes=copy.deepcopy(meta['state_hashes']),
        accepted_source_lineage=copy.deepcopy(meta['lineage']),
        accepted_actor_and_optimizer_preserved_in_source=True,
        new_finish_actor_Adam_initialized=True,critic_Adam_inherited=True,
        critic_zero_expanded=changed,critic_Adam_zero_expanded=moment_changes,
        Identity_normalizers_unchanged=True,RNG_preserved=True,actual_learning_rate=meta['learning_rate'],
        historical_AUX_updates=meta['historical_AUX_updates'],new_AUX_updates=0,
        old_rollout_reused=False,old_gain10_applied_again=False)
    counts=dict(finish_policy_decisions=0,finish_ppo_updates=0,finish_optimizer_steps=0,
        prefix_decisions=0,finish_opportunities=0,completed_episodes=0,full_successes=0,auxiliary_updates=0)
    return lineage,counts


def validate_runner(runner,runtime):
    cfg=settings()
    if (runtime['experiment_id']!=NAME or runtime['local_contract']!=cfg
            or runner.alg.actor.policy_version!=cfg['version']
            or runner.alg.actor.observation_dimension!=cfg['observation_dimension']):
        raise ValueError('finish runtime/actor/config mismatch')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)


def checkpoint_path(counts,runtime):
    n=counts['finish_policy_decisions']
    return OUTPUT/'checkpoints/history'/('checkpoint_CP%d_finish%06d_g%s.pt'%(
        settings()['historical_decision_origin']+n,n,runtime['source_git_commit'][:12]))


def save(runner,runtime,lineage,counts,*,source_run,publish_pointer=True):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import capture_training_rng_state
    validate_runner(runner,runtime)
    if counts['auxiliary_updates']!=0 or runner.alg.storage.step!=0 or runner.alg.transition.actions is not None:
        raise RuntimeError('complete update, empty rollout, AUX0 required')
    target=checkpoint_path(counts,runtime); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    payload=runner.alg.save(); hashes={k:state_hash(v) for k,v in payload.items()}
    infos=dict(schema=SCHEMA,runtime_contract=runtime,lineage=copy.deepcopy(lineage),counts=dict(counts),
        runner_config=copy.deepcopy(runner.local_configuration),state_hashes=hashes,
        learning_rate=runner.alg.learning_rate,training_rng=capture_training_rng_state(seed=1001),
        source_run=str(source_run),rollout_empty=True,front_FL_assist=True,rear_task_assists=False,
        public_preparation_module=True,public_finish_module=True,
        observation_dimension=settings()['observation_dimension'],local_auxiliary_events=[],
        historical_AUX_updates=lineage['historical_AUX_updates'])
    payload.update(infos=infos,iter=counts['finish_ppo_updates']); torch.save(payload,target)
    reloaded=torch.load(target,map_location=runner.device,weights_only=False)
    if reloaded['infos']!=infos or any(state_hash(reloaded[k])!=v for k,v in hashes.items()):
        raise RuntimeError('serialized finish package differs')
    runner.alg.load(reloaded,None,True); runner.alg.learning_rate=infos['learning_rate']
    validate_runner(runner,runtime)
    if any(state_hash(runner.alg.save()[k])!=v for k,v in hashes.items()):
        raise RuntimeError('strict finish state reload differs')
    manifest=target.with_name(target.stem+'_manifest.json')
    write(manifest,dict(infos,checkpoint=str(target),checkpoint_sha256=sha(target),save_load_round_trip=True))
    pointer=dict(checkpoint=str(target),checkpoint_sha256=sha(target),manifest=str(manifest),
                 manifest_sha256=sha(manifest),counts=dict(counts),evaluated=False)
    if publish_pointer:
        p=OUTPUT/'checkpoints/checkpoint_last_pointer.json'; write(p,pointer,replace=p.exists())
    return pointer


def load(runner,path,runtime):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import restore_training_rng_state
    path=Path(path).resolve(strict=True); side=path.with_name(path.stem+'_manifest.json')
    meta=json.loads(side.read_text())
    if (meta['schema']!=SCHEMA or meta['runtime_contract']!=runtime or meta['checkpoint_sha256']!=sha(path)
            or not meta['save_load_round_trip'] or not meta['rollout_empty'] or meta['counts']['auxiliary_updates']!=0):
        raise ValueError('finish checkpoint contract/hash/ledger mismatch')
    cfg=copy.deepcopy(meta['runner_config']); cfg['device']=runner.device
    current=copy.deepcopy(runner.local_configuration)
    current['actor'].setdefault('expected_accepted_state_sha256',cfg['actor']['expected_accepted_state_sha256'])
    if cfg!=current or meta['lineage']['accepted_complete_sha256']!=settings()['accepted_checkpoint_sha256']:
        raise ValueError('actual runner or accepted490 source binding changed')
    runner.alg.actor.expected_accepted_state_sha256=cfg['actor']['expected_accepted_state_sha256']
    data=torch.load(path,map_location=runner.device,weights_only=False)
    if any(meta.get(k)!=v for k,v in data['infos'].items()):
        raise ValueError('embedded finish metadata differs')
    runner.alg.load(data,None,True); runner.alg.learning_rate=meta['learning_rate']
    runner.local_configuration=cfg; validate_runner(runner,runtime)
    if any(state_hash(runner.alg.save()[k])!=v for k,v in meta['state_hashes'].items()):
        raise ValueError('reloaded actor/critic/Adam differs')
    restore_training_rng_state(meta['training_rng'],expected_seed=1001)
    runner.current_learning_iteration=meta['counts']['finish_ppo_updates']
    runner.checkpoint_load_provenance=dict(checkpoint=str(path),checkpoint_sha256=sha(path),
        manifest=str(side),manifest_sha256=sha(side),strict_actual_composite_load_verified=True,
        actor_critic_optimizer_hashes_verified=True,accepted_complete_CP232960_frozen=True)
    return copy.deepcopy(meta['lineage']),dict(meta['counts'])


def request(runner,observation,*,stochastic):
    from .semantic_finish_advance_actor import audited_finish_request
    return audited_finish_request(runner.alg.actor,observation,
        (lambda:runner.alg.act(observation)) if stochastic else
        (lambda:runner.alg.actor(observation,stochastic_output=False)),stochastic=stochastic)


def prefix(core,runner,stream,counts):
    import torch
    core.reset(seed=1001); start=counts['prefix_decisions']
    while not core.task.finish_active and not core.done:
        obs=tensor_observation(core.observation,runner.device)
        with torch.inference_mode(): raw,audit=request(runner,obs,stochastic=False)
        step=core.step(raw[0].cpu().tolist()); counts['prefix_decisions']+=1
        line(stream,dict(kind='frozen_complete_CP232960_prefix',PPO_credit=0,
            physical_tick=core.frame.physics_tick,phase=core.frame.state_id,policy_request=audit,
            local=core.task.snapshot()))
    if core.done:
        raise RuntimeError('accepted predecessor ended before legalRLfinish entry: '+str(step.info.get('termination_reason')))
    counts['finish_opportunities']+=1
    print(json.dumps(dict(finish_entry=core.task.snapshot(),uncredited_prefix_decisions=counts['prefix_decisions']-start)),flush=True)


def train(core,runner,runtime,lineage,counts,run,decisions):
    import torch
    from .semantic_training import audited_ppo_update,verified_native_effect,_stop_request
    if decisions<=0 or decisions%512 or settings()['rollout_length']!=512:
        raise ValueError('complete512 finish rollout required')
    runner.alg.train_mode(); (run/'rollouts').mkdir()
    with (run/'decisions.jsonl').open('x') as stream,(run/'updates.jsonl').open('x') as updates:
        for _ in range(decisions//512):
            phases,coverage=Counter(),Counter()
            for index in range(512):
                if core.done: prefix(core,runner,stream,counts)
                obs=tensor_observation(core.observation,runner.device); before=core.task.snapshot()
                if not core.task.finish_active or float(obs['policy'][0,490])!=1.:
                    raise RuntimeError('pre-RL prefix cannot enter finish PPO storage')
                with torch.inference_mode():
                    raw,audit=request(runner,obs,stochastic=True); selected=raw.detach().clone()
                    old_logp=runner.alg.transition.actions_log_prob.detach().clone()
                    step=core.step(raw[0].cpu().tolist())
                    issued=step.info.get('applied_raw_full12',raw[0].cpu().tolist())
                    if list(issued)!=raw[0].cpu().tolist(): raise RuntimeError('teacher cannot replace sampled actions')
                    verified_native_effect(step.info,tuple(issued))
                    nxt=tensor_observation(step.observation,runner.device)
                    done=torch.tensor([step.terminated],dtype=torch.bool,device=runner.device)
                    runner.alg.process_env_step(nxt,torch.tensor([step.reward],device=runner.device),done,
                                                {'time_outs':torch.zeros_like(done)})
                    if (not torch.equal(runner.alg.storage.actions[index],selected)
                            or not torch.equal(runner.alg.storage.actions_log_prob[index].view(-1),old_logp.view(-1))):
                        raise RuntimeError('actual raw sample/old logp changed')
                counts['finish_policy_decisions']+=1; phases[step.info['phase_id']]+=1
                requested=before.get('requested_finish',{})
                coverage[requested.get('mode','UNAVAILABLE')]+=1
                line(stream,dict(kind='post_RL_finish_on_policy',PPO_credit=1,
                    global_decision=settings()['historical_decision_origin']+counts['finish_policy_decisions'],
                    finish_decision=counts['finish_policy_decisions'],observation=obs['policy'][0].cpu().tolist(),
                    policy_request=audit,old_logp=old_logp.cpu().tolist(),before_local=before,step_info=step.info))
                if step.terminated:
                    counts['completed_episodes']+=1; counts['full_successes']+=int(step.info.get('full_task_success',False))
                if (index+1)%128==0:
                    print(json.dumps(dict(finish_collected=counts['finish_policy_decisions'],time_s=core.frame.sim_time_s,
                        phase=core.frame.state_id,finish=core.task.snapshot().get('requested_finish'))),flush=True)
            with torch.inference_mode(): runner.alg.compute_returns(nxt)
            storage=runner.alg.storage
            snapshot={k:getattr(storage,k).detach().cpu().clone() for k in
                ('actions','actions_log_prob','rewards','dones','values','returns','advantages')}
            snapshot['observations']={k:v.detach().cpu().clone() for k,v in storage.observations.items()}
            snapshot['distribution_params']=tuple(v.detach().cpu().clone() for v in storage.distribution_params)
            number=counts['finish_ppo_updates']+1
            torch.save(snapshot,run/'rollouts'/f'rollout_{number:04d}.pt')
            report=audited_ppo_update(runner,likelihood_audit_path=run/'rollouts'/f'likelihood_{number:04d}.json')
            validate_runner(runner,runtime); counts['finish_ppo_updates']+=1
            counts['finish_optimizer_steps']+=report['optimizer_steps']
            runner.current_learning_iteration=counts['finish_ppo_updates']
            report.update(counts=dict(counts),actual_phase_counts=dict(phases),actual_finish_mode_counts=dict(coverage),
                accepted_complete_CP232960_unchanged=True,AUX_updates=0)
            line(updates,report); pointer=save(runner,runtime,lineage,counts,source_run=run)
            print(json.dumps(dict(completed_update=number,checkpoint=pointer)),flush=True)
            stop=_stop_request(run,runtime)
            if stop is not None:
                write(run/'stop_after_update.accepted.json',dict(stop,checkpoint=pointer,rollout_empty=True)); break
    return pointer


def evaluate(core,runner,runtime,lineage,counts,run):
    cfg=settings()
    route=dict(validate_runner=validate_runner,auxiliary_ledger=[],settings=cfg,
        config_dir=ROOT/cfg['accepted_control_config'],tensor_observation=tensor_observation,request=request,
        control_contributions=dict(accepted_full_CP232960=lineage,public_preparation_module=True,
            public_finish_module=True,requested_finish=cfg['requested_finish'],new_AUX_updates=0,
            historical_AUX_updates=lineage['historical_AUX_updates'],finish_branch_counters=dict(counts),
            displayed_label='PPO + declared preparation + measured advance/home reference; complete CP232960 frozen; rear helpers OFF'))
    return old.evaluate(core,runner,runtime,lineage,counts,run,False,route=route)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('migrate','train','eval'))
    parser.add_argument('--expected-head',required=True); parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--decisions',type=int,default=512)
    parser.add_argument('--device',default='cuda:0'); args=parser.parse_args()
    runtime=contract(args.expected_head); run=args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME): raise ValueError('isolated finish namespace required')
    if args.mode!='migrate' and args.checkpoint is None: raise ValueError('saved composite required')
    if args.mode=='migrate' and args.checkpoint is not None: raise ValueError('migration source is the pinned accepted package')
    run.mkdir(parents=True,exist_ok=False)
    write(run/'run_manifest.started.json',dict(runtime_contract=runtime,mode=args.mode,checkpoint=str(args.checkpoint),
        started_utc=datetime.now(timezone.utc).isoformat()))
    app=None
    try:
        # Retain the accepted Windows native-extension import order: importing
        # tensordict's C extension after Kit may crash before any episode.
        import torch
        import tensordict
        from .rl_library_wrapper import seed_training_rngs
        if args.mode!='migrate':
            from isaaclab.app import AppLauncher
            app=AppLauncher(headless=args.mode=='train',enable_cameras=False).app; app.update()
        seed_training_rngs(1001)
        meta=json.loads(args.checkpoint.with_name(args.checkpoint.stem+'_manifest.json').read_text()) if args.checkpoint else None
        runner=make_runner(args.device,1001,saved_configuration=meta['runner_config'] if meta else None)
        lineage,counts=load(runner,args.checkpoint,runtime) if args.checkpoint else initialize_from_accepted(runner)
        if args.mode=='migrate': result=save(runner,runtime,lineage,counts,source_run=run)
        else:
            core=build_core(app)
            if args.mode=='train': result=train(core,runner,runtime,lineage,counts,run,args.decisions)
            else:
                runner.alg.eval_mode(); result=evaluate(core,runner,runtime,lineage,counts,run)
        write(run/'run_manifest.json',dict(lifecycle='COMPLETE',mode=args.mode,result=result))
        print(json.dumps(dict(run=str(run),lifecycle='COMPLETE')),flush=True)
    except BaseException:
        write(run/'failure.json',{'traceback':traceback.format_exc()}); raise
    finally:
        if app is not None: app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':
    main()
