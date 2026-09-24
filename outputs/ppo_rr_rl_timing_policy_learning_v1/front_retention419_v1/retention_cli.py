"""Finite offline ancestor-query AUX. Import is stdlib-only; no automatic fit.

Default invocation is a new source-bound CPU inspection. Execution needs an
explicit matching inspection, budget and unique output. No physical success is
implied; old rear rows are numeric invariance probes, not current-v3 coverage.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1'
BRANCH=OUT/'branches/ancestor220544_recapture_v2'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
ORIGIN=dict(zip(COUNTERS,(220544,1688,33760)))
MINIMUM=dict(zip(COUNTERS,(221568,1696,33920)))
HEAD='45862675a18a5bb2b75f0d4883e0cee2a3be24e3'
TEACHER=OUT/'checkpoints/history/checkpoint_rear_recapture_CP220544_g44219b4fdc4d.pt'
TEACHER_SHA='7bd9db99a70ab4a9bdf6c251c48f638d8dd21bdc9a854766c8b0ebbc7bb75382'
TEACHER_MANIFEST_SHA='85fed85652e1a45738785bf23316ce4130ad0012aaa6b4a52b442a989ce83e5a'
KERNEL_SHA='72c1e15d789a55eb2b27dd52edf78913460c972c685314b31415a46f53de10d3'
SELECTOR_SHA='70599c7fc5234ba3f131a2f7efcabc84eb09ccf1e2a9841ba810b3e948a2ad01'
REFERENCE=HERE/'front_retention419_zero_update_inspection.json'
REFERENCE_SHA='7c87a41c708092404afb0b297ec3043515f9bb8e531c1255855d3070703fcec7'
SCHEMA='wlr50_clean.rear419_ancestor_query_retention_receipt.v1'
LEDGER_SCHEMA='wlr50_clean.rear419_ancestor_query_front_retention_auxiliary.v1'
LEDGER_KEY='front_retention_auxiliary'
PARAMETERS=['actor.mlp.0.weight[:,0:2]']


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sidecar(path):return Path(path).with_name(Path(path).stem+'_manifest.json')


def output_path(path,*,checkpoint=False):
    path=Path(path).resolve()
    if checkpoint:
        require(path.parent==(BRANCH/'checkpoints/history').resolve()
            and path.name.startswith('checkpoint_aux_frontretention419_') and path.suffix=='.pt'
            and not sidecar(path).exists(),'unique same-branch AUX history destination required')
    else:
        require(path.is_relative_to(HERE.resolve()),'reports must stay in the new helper directory')
    require(not path.exists(),'output exists; never overwrite')
    return path


def validate_source_metadata(metadata,expected_head):
    require(expected_head==HEAD and metadata['runtime_contract']['source_git_commit']==HEAD
        and metadata['runtime_contract']['experiment_id']=='rr_rl_timing_policy_learning_v1'
        and metadata['policy_contract']['version']=='rr_rl_timing_policy_learning_history_v1'
        and metadata['policy_contract']['observation_dimension']==419
        and metadata.get('save_load_round_trip') is True,'requires exact current v3 full419 sealed source')
    require(all(type(metadata.get(k)) is int and metadata[k]>=MINIMUM[k] for k in COUNTERS)
        and metadata['global_policy_decisions']%128==0
        and metadata['optimizer_steps']==20*metadata['ppo_updates'],'source must retain completed PPO updates')
    route=metadata.get('checkpoint_output_routing',{})
    require(route.get('schema')=='wlr50_clean.checkpoint_output_routing.v1'
        and route.get('branch')==BRANCH.name and Path(route.get('output_root','')).resolve()==BRANCH.resolve()
        and route.get('main_latest_pointer_promotion') is False,'existing branch route required')
    require(metadata['rear_policy_timing_branch']['counter_origin']==ORIGIN
        and metadata['rear_policy_timing_branch_counts']=={k:metadata[k]-ORIGIN[k] for k in COUNTERS},
        'original branch origin/credit changed')
    require(metadata.get('rear_live_swing_migration',{}).get('schema')=='wlr50_clean.rear_live_swing_same419.v3',
        'v3 migration lineage required')


def source_binding(path,metadata,expected_sha,expected_manifest_sha,expected_head):
    path=Path(path).resolve(strict=True)
    validate_source_metadata(metadata,expected_head)
    require(path.parent==(BRANCH/'checkpoints/history').resolve(),'immutable same-branch source required')
    require(sha(path)==expected_sha==metadata['checkpoint_sha256'] and sha(sidecar(path))==expected_manifest_sha,
        'explicit source/sidecar SHA mismatch')
    pointer=BRANCH/'checkpoints/checkpoint_last_pointer.json'
    if pointer.exists():
        pointed=read(pointer); latest=read(pointed['manifest'])
        require(all(metadata[k]>=latest[k] for k in COUNTERS),'source is older than the branch latest completed update')
    return dict(path=str(path),sha256=expected_sha,manifest_sha256=expected_manifest_sha,
        counters={k:metadata[k] for k in COUNTERS},actor_sha256=metadata['actor_parameter_sha256'],
        runtime_contract_sha256=digest(metadata['runtime_contract']),output_route_sha256=digest(metadata['checkpoint_output_routing']))


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'candidate import failed')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load_numeric_tools():
    # Never called by stdlib-only metadata tests or module import.
    require(sha(HERE/'front_retention419.py')==KERNEL_SHA and sha(HERE/'inspect_zero_update.py')==SELECTOR_SHA,
        'immutable existing kernel/selector changed')
    kernel=load_module('_retention419_frozen_kernel',HERE/'front_retention419.py')
    selector=load_module('_retention419_frozen_selector',HERE/'inspect_zero_update.py')
    return kernel,selector


def decode_row(text):
    match=re.fullmatch(r'rollout_(\d{6}):flat_(\d{3}):global_(\d+)',text)
    require(match is not None,'invalid bound source row')
    update,index,global_step=map(int,match.groups())
    require(1689<=update<=1696 and 0<=index<128
        and global_step==220544+(update-1689)*128+index+1,'source row identity mismatch')
    return update,index


def load_query_data(kernel,selector,metadata,contract):
    """Reuse the fixed inspected row IDs; no selection search or PPO relabeling."""
    from wlr50_clean.ppo.semantic_rear_live_swing_migration import _previous_contract,validate_rear_live_swing_lineage
    torch=kernel.torch
    require(sha(REFERENCE)==REFERENCE_SHA,'original zero-update selection receipt changed')
    reference=read(REFERENCE)
    require(sha(TEACHER)==TEACHER_SHA and sha(sidecar(TEACHER))==TEACHER_MANIFEST_SHA,'teacher binding changed')
    teacher_metadata=read(sidecar(TEACHER))
    validate_rear_live_swing_lineage(metadata,contract)
    original_runtime=_previous_contract(contract,metadata['rear_live_swing_migration'])
    require(teacher_metadata['runtime_contract']==original_runtime
        and teacher_metadata['policy_contract']==metadata['policy_contract'],
        'teacher/data predecessor runtime or same419 policy mismatch')
    rows={};run_bindings={}
    for item in reference['source_rollouts']:
        path=Path(item['path']).resolve(strict=True)
        require(path.is_relative_to((ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1').resolve())
            and sha(path)==item['sha256'],'bound sealed rollout changed')
        run=path.parent.parent
        if str(run) not in run_bindings:
            manifest=read(run/'run_manifest.json')
            require(manifest['lifecycle']=='SUCCEEDED' and manifest['runtime_contract']==original_runtime,
                'historical data run is not sealed under its reviewed v2 runtime')
            run_bindings[str(run)]=sha(run/'run_manifest.json')
        rows[item['update']]=selector.rollout(path,metadata['policy_contract'],original_runtime)
    train_ids=reference['sets']['train']['row_ids'];validation_ids=reference['sets']['validation']['row_ids']
    holdout_ids=[row for group in reference['actual_same_input_invariance']['row_ids'].values() for row in group]
    require(len(train_ids)==33 and len(validation_ids)==32 and len(holdout_ids)==17
        and set(train_ids).isdisjoint(validation_ids),'fixed reviewed split changed')
    def selected(ids):return torch.stack([rows[u][i] for u,i in map(decode_row,ids)])
    train,validation,holdout=selected(train_ids),selected(validation_ids),selected(holdout_ids)
    require(bool((train[:,:2].sum(-1)==1).all()) and bool((validation[:,1]==1).all())
        and bool((holdout[:,:2]==0).all()),'selected phase responsibilities changed')
    require(bool((train[:,372:]==0).all()) and bool((validation[:,372:]==0).all()),
        'front states must retain reviewed WAIT/context/timing zeros for v2-v3 admission')
    # Teacher has no live role. Preserve the caller RNG around all construction/query work.
    rng=kernel.training.capture_training_rng_state(seed=metadata['seed'])
    try:
        teacher=selector.actor_copy(kernel,TEACHER)
        require(kernel.training.parameter_hash(teacher)==teacher_metadata['actor_parameter_sha256'],'teacher actor differs')
        tr,va=kernel.tensors(train),kernel.tensors(validation)
        with torch.no_grad():
            target=kernel.distribution(teacher,tr)['mean'].detach().clone()
            valid_target=kernel.distribution(teacher,va)['mean'].detach().clone()
            require(torch.equal(target,teacher(tr,stochastic_output=False))
                and torch.equal(valid_target,teacher(va,stochastic_output=False)),'query differs from official conditional mean')
    finally:
        kernel.training.restore_training_rng_state(rng,expected_seed=metadata['seed'])
    receipt=dict(schema='wlr50_clean.rear419_fixed_student_rows_ancestor_query.v1',
        reference_selection=dict(path=str(REFERENCE),sha256=REFERENCE_SHA),
        teacher=dict(path=str(TEACHER),sha256=TEACHER_SHA,manifest_sha256=TEACHER_MANIFEST_SHA),
        source_rollouts=reference['source_rollouts'],sealed_run_manifest_sha256=run_bindings,
        train_row_ids=train_ids,validation_row_ids=validation_ids,invariance_row_ids=holdout_ids,
        train_count=33,validation_count=32,invariance_count=17,actual_probe_phases=['P07','P09','P12'],
        no_real_P03_P06_or_P13_probe=True,train_validation_same_episode_correlated=True,
        teacher_query_semantics='offline_ancestor_conditional_raw_mean_on_exact_saved_student_HISTORY_not_executed_action_or_success_label',
        front_encoding_admission='exact v3 predecessor contract; P01/P02 with reviewed zero assist/context/timing tail; RL-only v3 change inactive',
        rear_probe_semantics='historical_v2_numeric_inputs_only_not_current_v3_physical_coverage',
        input_tensor_sha256=kernel.training.state_hash((train,validation,holdout)),
        target_tensor_sha256=kernel.training.state_hash((target,valid_target)),query_device='cpu',
        PPO_credit=0,teacher_deployed=False,physical_success_claimed=False)
    receipt['receipt_content_sha256']=digest(receipt)
    return dict(train_observations=train,validation_observations=validation,invariance_observations=holdout,
        train_raw_targets=target,validation_raw_targets=valid_target,receipt=receipt)


def fit_arguments(kernel,data,device):
    return (kernel.tensors(data['train_observations'],device=device),data['train_raw_targets'].to(device),
        kernel.tensors(data['validation_observations'],device=device),data['validation_raw_targets'].to(device),
        kernel.tensors(data['invariance_observations'],device=device))


def append_ledger(infos,*,report,report_binding,data_receipt,source,helpers,budget):
    accepted=report['accepted_auxiliary_updates'];attempted=report['attempted_auxiliary_optimizer_steps']
    require(type(accepted) is int and type(attempted) is int and 0<accepted<=attempted<=32,
        'only actually accepted finite AUX can be published')
    require(report['optimized_parameters']==PARAMETERS and report['optimized_scalar_count']==512
        and report['actor_parameter_sha256_before']!=report['actor_parameter_sha256_after']
        and all(report[k]==0 for k in ('PPO_decisions_added','PPO_updates_added','PPO_optimizer_steps_added')),
        'AUX parameter scope or zero PPO credit differs')
    result=deepcopy(infos);branch=result['rear_policy_timing_branch']
    require(branch['counter_origin']==ORIGIN,'original current419 branch origin changed')
    ledger=deepcopy(branch.get(LEDGER_KEY,dict(schema=LEDGER_SCHEMA,events=[])))
    require(ledger['schema']==LEDGER_SCHEMA and [e['event_index'] for e in ledger['events']]==list(range(1,len(ledger['events'])+1)),
        'front-retention ledger must be append-only and contiguous')
    event=dict(event_index=len(ledger['events'])+1,kind='offline_ancestor_query_P01_P02_phase_columns_not_PPO',
        source_checkpoint=source,teacher_checkpoint=data_receipt['teacher'],helpers_sha256=helpers,
        data_receipt_sha256=data_receipt['receipt_content_sha256'],fit_report=report_binding,budget=budget,
        accepted_auxiliary_updates=accepted,attempted_auxiliary_optimizer_steps=attempted,
        optimized_parameters=PARAMETERS,optimized_scalar_count=512,P01_P02_mean_and_log_sigma_may_change=True,
        same_input_P03_P13_Gaussian_unchanged=True,same_future_trajectory_claimed=False,
        targets_were_executed_actions=False,teacher_deployed=False,physical_success_claimed=False,
        PPO_counters_unchanged={k:infos[k] for k in COUNTERS},PPO_decisions_added=0,PPO_updates_added=0,PPO_optimizer_steps_added=0)
    ledger['events'].append(event)
    ledger['accepted_auxiliary_updates_total']=sum(e['accepted_auxiliary_updates'] for e in ledger['events'])
    ledger['attempted_auxiliary_optimizer_steps_total']=sum(e['attempted_auxiliary_optimizer_steps'] for e in ledger['events'])
    branch[LEDGER_KEY]=ledger
    restored=deepcopy(result);restored['rear_policy_timing_branch']=deepcopy(infos['rear_policy_timing_branch'])
    require(restored==infos,'new AUX ledger changed protected historical metadata')
    return result


def make_runner(kernel,metadata,device):
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    from wlr50_clean.ppo.semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY,REAR_POLICY_TIMING_OBSERVATION_LAYOUT
    runner,_=kernel.training.construct_semantic_runner(_shape_env(419,device),seed=metadata['seed'],device=device,
        policy_version=REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT,initialize_actor=False)
    require(runner._semantic_runner_config==metadata['runner_config'],'official source-device runner config changed')
    return runner


def save_auxiliary(kernel,runner,path,infos,*,source_path,source,report,fit_binding,data,helpers,budget):
    training=kernel.training;torch=kernel.torch
    require(runner.alg.storage.step==0 and runner.alg.transition.actions is None,'AUX requires empty rollout')
    require(training.parameter_hash(runner.alg.actor)==report['actor_parameter_sha256_after']
        and infos['actor_parameter_sha256']==report['actor_parameter_sha256_before'],'fit actor binding mismatch')
    before=torch.load(source_path,map_location='cpu',weights_only=False)['actor_state_dict']
    for name,value in runner.alg.actor.state_dict().items():
        actual=value.detach().cpu()
        require(torch.equal(actual[:,2:],before[name][:,2:]) if name=='mlp.0.weight' else torch.equal(actual,before[name]),
            'nonselected actor state changed: '+name)
    for key,value in (('critic_parameter_sha256',training.parameter_hash(runner.alg.critic)),
            ('optimizer_state_sha256',training.state_hash(runner.alg.optimizer.state_dict())),
            ('normalizer_state_sha256',training.state_hash(training._normalizers(runner)))):
        require(infos[key]==value,'protected state differs: '+key)
    require(infos['optimizer_learning_rate']==training.optimizer_learning_rate(runner)
        and infos['runner_config']==runner._semantic_runner_config
        and infos['training_rng_state']==training.capture_training_rng_state(seed=infos['seed']),'LR/device/full RNG changed')
    payload=append_ledger(infos,report=report,report_binding=fit_binding,data_receipt=data['receipt'],source=source,helpers=helpers,budget=budget)
    pointers=[root/'checkpoints'/name for root in (OUT,BRANCH)
        for name in ('checkpoint_last.pt','checkpoint_last_pointer.json','resume_state.json')]
    pointer_hashes={str(p):sha(p) if p.exists() else None for p in pointers}
    saved,manifest=training.save_semantic_checkpoint(runner,path,payload)
    fresh=make_runner(kernel,infos,infos['runner_config']['device'])
    loaded=training.load_semantic_checkpoint(fresh,saved,contract=infos['runtime_contract'],seed=infos['seed'])
    from wlr50_clean.ppo.semantic_rear_policy_timing_migration import validate_rear_policy_namespace
    validate_rear_policy_namespace(loaded,loaded['runtime_contract'],BRANCH,checkpoint_output_routing=loaded['checkpoint_output_routing'])
    for key in infos:
        if key.endswith(('_branch','_migration','_branch_counts')) and key!='rear_policy_timing_branch':
            require(loaded[key]==infos[key],'historical record changed: '+key)
    require(all(loaded[k]==infos[k] for k in COUNTERS)
        and loaded['rear_policy_timing_branch']==payload['rear_policy_timing_branch']
        and loaded['checkpoint_output_routing']==infos['checkpoint_output_routing']
        and training.parameter_hash(fresh.alg.actor)==report['actor_parameter_sha256_after']
        and training.state_hash(fresh.alg.optimizer.state_dict())==infos['optimizer_state_sha256']
        and training.parameter_hash(fresh.alg.critic)==infos['critic_parameter_sha256']
        and training.state_hash(training._normalizers(fresh))==infos['normalizer_state_sha256']
        and training.capture_training_rng_state(seed=infos['seed'])==infos['training_rng_state']
        and fresh.alg.storage.step==0 and fresh.alg.transition.actions is None,'official independent reload differs')
    require(pointer_hashes=={str(p):sha(p) if p.exists() else None for p in pointers},'AUX cannot promote any latest pointer')
    return dict(path=str(saved),sha256=sha(saved),manifest=str(manifest),manifest_sha256=sha(manifest),
        independent_official_reload_verified=True,latest_pointer_published=False,
        subsequent_real_PPO_ledger_carry_not_yet_verified=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--expected-manifest-sha256',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--execute-aux',action='store_true')
    parser.add_argument('--inspection-receipt',type=Path)
    parser.add_argument('--budget',type=Path)
    parser.add_argument('--aux-checkpoint',type=Path)
    args=parser.parse_args(argv);report_path=output_path(args.report)
    require(args.execute_aux or all(v is None for v in (args.inspection_receipt,args.budget,args.aux_checkpoint)),
        'read-only inspection cannot accept fit arguments')
    kernel,selector=load_numeric_tools();training=kernel.training;torch=kernel.torch
    from wlr50_clean.ppo import semantic_cli,semantic_migration
    from wlr50_clean.ppo.semantic_rear_policy_timing_migration import validate_rear_policy_namespace
    checkpoint=args.checkpoint.resolve(strict=True);metadata=semantic_migration.checkpoint_metadata(checkpoint)
    contract=semantic_cli.runtime_contract(expected_head=args.expected_head,semantic_version='v3',experiment_id='rr_rl_timing_policy_learning_v1')
    require(metadata['runtime_contract']==contract,'source/current runtime mismatch')
    source=source_binding(checkpoint,metadata,args.expected_source_sha256,args.expected_manifest_sha256,args.expected_head)
    validate_rear_policy_namespace(metadata,contract,BRANCH,checkpoint_output_routing=metadata['checkpoint_output_routing'])
    data=load_query_data(kernel,selector,metadata,contract)
    helpers={name:sha(HERE/name) for name in ('retention_cli.py','front_retention419.py','inspect_zero_update.py')}
    binding=dict(source_checkpoint=source,helpers_sha256=helpers,data_receipt_sha256=data['receipt']['receipt_content_sha256'],
        runtime_contract_sha256=digest(contract))
    result=dict(schema=SCHEMA,binding=binding,data_receipt=data['receipt'],PPO_decisions_added=0,
        PPO_updates_added=0,PPO_optimizer_steps_added=0,teacher_deployed=False,physical_success_claimed=False)
    if not args.execute_aux:
        require(not torch.cuda.is_available(),'read-only inspection requires CUDA hidden')
        rng=training.capture_training_rng_state(seed=metadata['seed'])
        try:
            actor=selector.actor_copy(kernel,checkpoint)
            require(training.parameter_hash(actor)==metadata['actor_parameter_sha256'],'student actor binding differs')
            inspection=kernel.inspect(actor,*fit_arguments(kernel,data,'cpu'))
        finally:training.restore_training_rng_state(rng,expected_seed=metadata['seed'])
        result.update(mode='read_only_current419_ancestor_query',auxiliary_updates=0,inspection=inspection,automatic_aux_enabled=False)
    else:
        require(all(v is not None for v in (args.inspection_receipt,args.budget,args.aux_checkpoint)),
            'fit requires explicit matching inspection, finite budget and unique AUX destination')
        prior=read(args.inspection_receipt)
        require(prior['mode']=='read_only_current419_ancestor_query' and prior['auxiliary_updates']==0
            and prior['binding']==binding,'stale or different source/helper/data inspection')
        budget=kernel.Budget(**read(args.budget));budget.validate()
        destination=output_path(args.aux_checkpoint,checkpoint=True)
        fit_path=output_path(report_path.with_name(report_path.stem+'_fit.json'))
        device=metadata['runner_config']['device']
        require(not str(device).startswith('cuda') or torch.cuda.is_available(),'source CUDA RNG topology must remain visible')
        runner=make_runner(kernel,metadata,device)
        infos=training.load_semantic_checkpoint(runner,checkpoint,contract=contract,seed=metadata['seed'])
        report=kernel.fit(runner,*fit_arguments(kernel,data,device),budget=budget,authorized=True)
        training.write_json(fit_path,report)
        fit_binding=dict(path=str(fit_path),sha256=sha(fit_path),content_sha256=digest(report))
        result.update(mode='explicit_finite419_ancestor_query_AUX_not_PPO',auxiliary_updates=report['accepted_auxiliary_updates'],
            fit_report=fit_binding,budget=read(args.budget),auxiliary_checkpoint=None,
            reviewed_inspection=dict(path=str(args.inspection_receipt.resolve()),sha256=sha(args.inspection_receipt)))
        if report['accepted_auxiliary_updates']:
            result['auxiliary_checkpoint']=save_auxiliary(kernel,runner,destination,infos,source_path=checkpoint,source=source,
                report=report,fit_binding=fit_binding,data=data,helpers=helpers,budget=read(args.budget))
    training.write_json(report_path,result)
    print(json.dumps(dict(report=str(report_path),mode=result['mode'],auxiliary_updates=result['auxiliary_updates'])))
    return result


if __name__=='__main__':main()
